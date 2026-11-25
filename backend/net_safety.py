"""Outbound URL validation & connection-pinned HTTP helpers to prevent SSRF and DNS rebinding."""
from __future__ import annotations

import concurrent.futures
import ipaddress
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.goog",
    "metadata",
    "instance-data",
}

_EXTRA_BLOCKED_V4 = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT (RFC 6598)
    ipaddress.ip_network("198.18.0.0/15"),   # Benchmarking (RFC 2544)
    ipaddress.ip_network("240.0.0.0/4"),     # Reserved
)

_EXTRA_BLOCKED_V6 = (
    ipaddress.ip_network("fec0::/10"),       # Deprecated IPv6 site-local (marked is_global in Python 3.12)
)

_DNS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="voxpoly_dns")


def _is_public_v4(ip: ipaddress.IPv4Address) -> bool:
    if any(ip in net for net in _EXTRA_BLOCKED_V4):
        return False
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv4Address):
        return _is_public_v4(ip)

    if ip.ipv4_mapped is not None:
        return _is_public_v4(ip.ipv4_mapped)
    if ip.sixtofour is not None and not _is_public_v4(ip.sixtofour):
        return False
    if ip.teredo is not None:
        t_server, t_client = ip.teredo
        if not _is_public_v4(t_server) or not _is_public_v4(t_client):
            return False
    if any(ip in net for net in _EXTRA_BLOCKED_V6):
        return False

    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_site_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def _resolve_host_bounded(host: str, port: int, timeout_sec: float = 5.0):
    if timeout_sec <= 0:
        raise TimeoutError(f"域名解析超时: {host}")
    fut = _DNS_EXECUTOR.submit(socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM)
    try:
        return fut.result(timeout=max(0.05, timeout_sec))
    except concurrent.futures.TimeoutError as exc:
        fut.cancel()
        raise TimeoutError(f"域名解析超时: {host}") from exc
    except socket.gaierror as exc:
        raise ValueError(f"无法解析域名 {host}: {exc}") from exc


def resolve_and_validate_public_url(url: str, dns_timeout_sec: float = 5.0) -> Tuple[str, str, str, int]:
    """Validate URL is public http(s) and return (clean_url, pinned_ip, hostname, port)."""
    clean = (url or "").strip()
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("仅支持 http:// 或 https:// 协议的公开网页链接。")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("链接缺少有效主机名。")
    if host in _BLOCKED_HOSTNAMES or host.endswith((".local", ".internal", ".localhost")):
        raise ValueError("出于安全限制，禁止访问本地或内网地址。")
    if parsed.username or parsed.password:
        raise ValueError("链接中不允许携带内嵌凭据。")

    port = parsed.port or (443 if scheme == "https" else 80)

    try:
        ip_obj = ipaddress.ip_address(host)
        if not _is_public_ip(str(ip_obj)):
            raise ValueError("出于安全限制，禁止访问内网、回环或云元数据 IP 地址。")
        return clean, str(ip_obj), host, port
    except ValueError as exc:
        if "安全限制" in str(exc):
            raise

    infos = _resolve_host_bounded(host, port, timeout_sec=dns_timeout_sec)
    if not infos:
        raise ValueError(f"域名 {host} 未返回可用网络地址。")

    resolved_ips = []
    for _, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        if not _is_public_ip(ip_str):
            raise ValueError("出于安全限制，目标域名解析到了内网、回环或保留地址，已拒绝访问。")
        if ip_str not in resolved_ips:
            resolved_ips.append(ip_str)

    return clean, resolved_ips[0], host, port


def validate_public_http_url(url: str) -> str:
    """Validate that a user-supplied URL is http(s) and resolves strictly to public internet IPs."""
    clean, _, _, _ = resolve_and_validate_public_url(url)
    return clean


def _build_pinned_url(original_url: str, pinned_ip: str, port: int) -> Tuple[str, str]:
    parsed = urlparse(original_url)
    scheme = parsed.scheme.lower()
    default_port = 443 if scheme == "https" else 80
    ip_host = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
    pinned_netloc = ip_host if port == default_port else f"{ip_host}:{port}"
    pinned_url = urlunparse((scheme, pinned_netloc, parsed.path or "/", parsed.params, parsed.query, ""))
    orig_host = (parsed.hostname or "").strip()
    if ":" in orig_host and not orig_host.startswith("["):
        orig_host = f"[{orig_host}]"
    host_header = orig_host if port == default_port else f"{orig_host}:{port}"
    return pinned_url, host_header


def _sni_extensions(clean_url: str, hostname: str) -> Dict[str, Any]:
    if urlparse(clean_url).scheme.lower() != "https":
        return {}
    try:
        return {"sni_hostname": hostname.encode("idna")}
    except Exception:
        return {"sni_hostname": hostname.encode("ascii", errors="ignore")}


def fetch_public_url_bytes(
    url: str,
    max_bytes: int = 5 * 1024 * 1024,
    max_redirects: int = 5,
    total_timeout_sec: float = 20.0,
    user_agent: str = "Mozilla/5.0 (compatible; VoxPolyStudio/3.0)",
    transport: httpx.BaseTransport | None = None,
) -> Tuple[str, bytes, str]:
    """Fetch public HTTP(S) content with fresh per-hop connection-pinned client, no env proxy, identity encoding, and hard wall-clock/byte caps."""
    deadline = time.monotonic() + total_timeout_sec
    cancel_ev = threading.Event()
    active_handles: List[Any] = []
    handle_lock = threading.Lock()
    result_box: Dict[str, Any] = {}

    def _worker() -> None:
        current_url = url.strip()
        try:
            for _ in range(max_redirects + 1):
                if cancel_ev.is_set():
                    return
                rem_dns = deadline - time.monotonic()
                if rem_dns <= 0:
                    raise TimeoutError("网页抓取超时。")

                clean_url, pinned_ip, hostname, port = resolve_and_validate_public_url(
                    current_url, dns_timeout_sec=min(rem_dns, 5.0)
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0 or cancel_ev.is_set():
                    raise TimeoutError("网页抓取超时。")

                pinned_url, host_header = _build_pinned_url(clean_url, pinned_ip, port)
                headers = {
                    "Host": host_header,
                    "User-Agent": user_agent,
                    "Accept-Encoding": "identity",
                }
                extensions = _sni_extensions(clean_url, hostname)
                timeout_cfg = httpx.Timeout(
                    connect=min(remaining, 8.0),
                    read=min(remaining, 10.0),
                    write=min(remaining, 5.0),
                    pool=min(remaining, 5.0),
                )

                client_kwargs: Dict[str, Any] = {
                    "trust_env": False,
                    "follow_redirects": False,
                    "timeout": timeout_cfg,
                }
                if transport is not None:
                    client_kwargs["transport"] = transport

                client = httpx.Client(**client_kwargs)
                with handle_lock:
                    active_handles.append(client)
                try:
                    req = client.build_request("GET", pinned_url, headers=headers, extensions=extensions)
                    resp = client.send(req, stream=True)
                    with handle_lock:
                        active_handles.append(resp)
                    try:
                        if cancel_ev.is_set() or time.monotonic() > deadline:
                            raise TimeoutError("网页抓取超时。")
                        if resp.status_code in (301, 302, 303, 307, 308):
                            loc = resp.headers.get("location", "").strip()
                            if not loc:
                                raise ValueError("目标网页返回了无效重定向。")
                            current_url = urljoin(clean_url, loc)
                            continue

                        resp.raise_for_status()
                        cl = resp.headers.get("content-length")
                        if cl and cl.isdigit() and int(cl) > max_bytes:
                            raise ValueError(f"目标网页体积超过 {max_bytes // (1024 * 1024)} MB 上限，已停止抓取。")

                        buf = bytearray()
                        if resp.is_stream_consumed:
                            raw_all = resp.content
                            if len(raw_all) > max_bytes:
                                raise ValueError(f"目标网页体积超过 {max_bytes // (1024 * 1024)} MB 上限，已停止抓取。")
                            buf.extend(raw_all)
                        else:
                            for chunk in resp.iter_raw(chunk_size=8192):
                                if cancel_ev.is_set() or time.monotonic() > deadline:
                                    raise TimeoutError("网页抓取超时。")
                                if len(buf) + len(chunk) > max_bytes:
                                    raise ValueError(f"目标网页体积超过 {max_bytes // (1024 * 1024)} MB 上限，已停止抓取。")
                                buf.extend(chunk)

                        if cancel_ev.is_set() or time.monotonic() > deadline:
                            raise TimeoutError("网页抓取超时。")
                        encoding = resp.encoding or "utf-8"
                        result_box["value"] = (clean_url, bytes(buf), encoding)
                        return
                    finally:
                        try:
                            resp.close()
                        except Exception:
                            pass
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass

            raise ValueError("目标网页重定向次数过多。")
        except BaseException as exc:
            result_box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=max(0.05, total_timeout_sec))
    if t.is_alive():
        cancel_ev.set()

        def _cleanup_async() -> None:
            with handle_lock:
                handles = list(active_handles)
            for h in handles:
                try:
                    h.close()
                except Exception:
                    pass

        threading.Thread(target=_cleanup_async, daemon=True).start()
        raise TimeoutError("网页抓取超过总时限，已强制中断。")

    if "error" in result_box:
        raise result_box["error"]
    if "value" not in result_box:
        raise TimeoutError("网页抓取超时。")
    return result_box["value"]


def post_public_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    total_timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """POST JSON to a validated public HTTP(S) endpoint with IP pinning, no env proxy, and no redirects."""
    clean_url, pinned_ip, hostname, port = resolve_and_validate_public_url(url, dns_timeout_sec=min(total_timeout_sec, 8.0))
    pinned_url, host_header = _build_pinned_url(clean_url, pinned_ip, port)
    req_headers = {"Host": host_header, **headers}
    extensions = _sni_extensions(clean_url, hostname)
    timeout_cfg = httpx.Timeout(connect=10.0, read=total_timeout_sec, write=15.0, pool=10.0)
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout_cfg) as client:
        req = client.build_request("POST", pinned_url, headers=req_headers, json=payload, extensions=extensions)
        resp = client.send(req)
        resp.raise_for_status()
        return resp.json()
