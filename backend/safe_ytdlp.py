"""Isolated socket-guarded yt-dlp runner preventing redirects, manifests, segments, or external FFmpeg from reaching private/local IPs."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from backend.net_safety import _is_public_ip, validate_public_http_url

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

_NETWORK_PROTO_PREFIXES = (
    "http://",
    "https://",
    "rtsp://",
    "rtmp://",
    "rtmps://",
    "rtp://",
    "tcp://",
    "udp://",
    "ftp://",
    "ftps://",
    "mms://",
    "mmsh://",
    "hls+",
    "crypto+",
)

_ORIG_POPEN = subprocess.Popen


def install_public_only_socket_guard() -> None:
    """Wrap socket.getaddrinfo in this short-lived download process so every resolved IP must be public."""
    orig_getaddrinfo = socket.getaddrinfo

    def _guarded_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        infos = orig_getaddrinfo(host, port, family, type, proto, flags)
        for info in infos:
            sockaddr = info[4]
            if sockaddr and not _is_public_ip(sockaddr[0]):
                raise PermissionError(f"SSRF guard blocked non-public IP resolution: {host} -> {sockaddr[0]}")
        return infos

    socket.getaddrinfo = _guarded_getaddrinfo


def _inject_ffmpeg_file_whitelist(arg_list: List[str]) -> List[str]:
    """Strip any existing -protocol_whitelist and force file,pipe for ffprobe and before every ffmpeg -i input."""
    if not arg_list:
        return arg_list
    exe_name = Path(arg_list[0]).name.lower()
    if exe_name not in ("ffprobe", "ffprobe.exe", "ffmpeg", "ffmpeg.exe"):
        return arg_list

    # Strip any caller-supplied -protocol_whitelist option & value so it can never widen allowed protocols
    cleaned: List[str] = [arg_list[0]]
    i = 1
    while i < len(arg_list):
        if arg_list[i] == "-protocol_whitelist":
            i += 2
            continue
        cleaned.append(arg_list[i])
        i += 1

    if exe_name in ("ffprobe", "ffprobe.exe"):
        return [cleaned[0], "-protocol_whitelist", "file,pipe", *cleaned[1:]]

    out: List[str] = [cleaned[0]]
    for tok in cleaned[1:]:
        if tok == "-i":
            out.extend(["-protocol_whitelist", "file,pipe"])
        out.append(tok)
    return out


class _GuardedPopen(_ORIG_POPEN):
    """Subclass of subprocess.Popen (subclassable by yt_dlp.utils.Popen) blocking network URLs & enforcing FFmpeg file,pipe."""

    def __init__(self, args, *p_args, **kwargs):
        if isinstance(args, (str, bytes, os.PathLike)):
            raw_list = [os.fsdecode(args)]
            is_scalar = True
        else:
            raw_list = [os.fsdecode(a) for a in args]
            is_scalar = False

        for item in raw_list:
            s = item.strip().lower()
            if any(s.startswith(pref) or ("://" in s and pref in s) for pref in _NETWORK_PROTO_PREFIXES):
                raise PermissionError(f"SSRF guard blocked external subprocess network access: {s[:80]}")

        safe_args = args if is_scalar else _inject_ffmpeg_file_whitelist(raw_list)
        super().__init__(safe_args, *p_args, **kwargs)


def install_subprocess_network_guard() -> None:
    """Install subclass-compatible subprocess.Popen guard blocking network URLs & hardening FFmpeg/FFprobe calls."""
    subprocess.Popen = _GuardedPopen  # type: ignore[misc]


def disable_ytdlp_external_network_downloaders() -> None:
    """Hard-disable yt-dlp's FFmpeg/external network downloaders so all HTTP/HLS fetches stay inside Python's guarded socket layer."""
    try:
        from yt_dlp.downloader import external as ytdlp_external

        def _deny_can_download(*_a, **_kw) -> bool:
            return False

        def _deny_real_download(*_a, **_kw):
            raise PermissionError("External network downloader disabled by VoxPoly SSRF policy.")

        for attr in dir(ytdlp_external):
            cls = getattr(ytdlp_external, attr, None)
            if isinstance(cls, type) and issubclass(cls, ytdlp_external.ExternalFD):
                cls.can_download = classmethod(_deny_can_download)  # type: ignore[assignment]
                cls.real_download = _deny_real_download  # type: ignore[assignment]
    except Exception:
        pass


def make_cumulative_byte_cap_hook(max_bytes: int, max_mb: int):
    """Return a yt-dlp progress hook enforcing cumulative bytes across distinct tracks (video + audio) without double-counting .part files."""
    track_bytes: Dict[str, int] = {}

    def _enforce_byte_cap(status: dict) -> None:
        raw_name = str(status.get("filename") or status.get("tmpfilename") or "default")
        key = raw_name[:-5] if raw_name.endswith(".part") else raw_name
        downloaded = max(0, int(status.get("downloaded_bytes") or 0))
        total = max(0, int(status.get("total_bytes") or status.get("total_bytes_estimate") or 0))
        track_bytes[key] = max(track_bytes.get(key, 0), downloaded, total)
        if sum(track_bytes.values()) > max_bytes:
            raise ValueError(f"视频与音轨累计体积超过 {max_mb} MB 上限，已中止下载。")

    return _enforce_byte_cap


def main() -> int:
    if len(sys.argv) < 4:
        print(json.dumps({"ok": False, "error": "usage: safe_ytdlp <url> <out_dir> <max_mb>"}, ensure_ascii=True), flush=True)
        return 2

    url = sys.argv[1]
    out_dir = Path(sys.argv[2])
    max_mb = max(1, int(sys.argv[3]))
    max_bytes = max_mb * 1024 * 1024

    for k in _PROXY_ENV_VARS:
        os.environ.pop(k, None)

    try:
        clean_url = validate_public_http_url(url)
        install_public_only_socket_guard()
        install_subprocess_network_guard()

        import yt_dlp

        disable_ytdlp_external_network_downloaders()
        allow_generic = os.getenv("VOXPOLY_YTDLP_ALLOW_GENERIC", "").strip() == "1"
        extractors = ["all"] if allow_generic else ["default", "-generic"]

        out_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts = {
            "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(out_dir / "source.%(ext)s"),
            "noplaylist": True,
            "playlist_items": "1",
            "allowed_extractors": extractors,
            "hls_prefer_native": True,
            "socket_timeout": 20,
            "proxy": "",
            "quiet": True,
            "no_warnings": True,
            "max_filesize": max_bytes,
            "progress_hooks": [make_cumulative_byte_cap_hook(max_bytes, max_mb)],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            title = str((info or {}).get("title") or "在线视频")[:48]

        print(json.dumps({"ok": True, "title": title}, ensure_ascii=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
