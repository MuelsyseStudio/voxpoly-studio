(() => {
  "use strict";

  const MASTERING_LABELS = {
    raw: "原始干声",
    podcast: "播客 -16 LUFS",
    shorts: "短视频 -14 LUFS",
    audiobook: "有声书 -18 LUFS",
  };

  const DEMO_PODCAST_SCRIPT = [
    "[主持人]: 欢迎收听今天的 VoxPoly 声音实验室！今天聊聊本地 AI 语音克隆为什么正在彻底改变个人内容创作。",
    "[嘉宾]: 没错，以前做多语种播客和视频配音，要么花高价找配音演员，要么按字符付昂贵的云端订阅费。",
    "[主持人]: 现在只需要一段十秒钟的清晰干声，就能在一台消费级显卡或 Colab 上输出播客级母带，还能自动锁住专业术语不翻车。",
    "[嘉宾]: 更棒的是，加上 OpenAI 兼容接口和 MCP 协议，连你的笔记软件和 AI 助手也能直接用你的声音朗读！",
  ].join("\n");

  const state = {
    accessToken: localStorage.getItem("voxpoly_access_token") || "",
    workspaceTab: "audio",
    languages: [],
    presets: [],
    cosyPresets: [],
    voices: [],
    jobs: [],
    podcasts: [],
    videos: [],
    glossary: [],
    features: {},
    enginesStatus: {},
    selectedEngineId: "qwen3",
    selectedVoiceId: null,
    selectedLangs: new Set(["zh", "en", "ja", "fr"]),
    scripts: {},
    mode: "icl",
    exaggeration: 0.5,
    cfgWeight: 0.35,
    cosyInstruct: "",
    activePresetId: null,
    activeJobId: null,
    activePodcastId: null,
    activeVideoId: null,
    loadedDoc: null,
    podSelectedLangs: new Set(["zh", "en"]),
    podSpeakerVoices: {},
    videoSelectedLangs: new Set(["en", "ja"]),
    videoSpeakerVoices: {},
    videoPreviewLang: "en",
    sourceDirty: false,
    scriptsCollapsed: false,
    pollTimer: null,
    podcastPollTimer: null,
    videoPollTimer: null,
    statusTimer: null,
    audioPlayers: new Map(),
    streamQueue: [],
    streamPlaying: false,
    streamAudio: new Audio(),
    streamPlayedSet: new Set(),
    mediaRecorder: null,
    micStream: null,
    micChunks: [],
    micBlob: null,
    micTimerId: null,
    micStartMs: 0,
    micAnimId: null,
    micAudioCtx: null,
  };

  const $ = (id) => document.getElementById(id);

  function showToast(msg, type = "default") {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    $("toastStack").appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s ease";
      setTimeout(() => el.remove(), 260);
    }, 3400);
  }

  function formatSec(sec) {
    if (!Number.isFinite(sec) || sec < 0) return "00:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function formatTenths(ms) {
    const totalTenths = Math.floor(ms / 100);
    const tenths = totalTenths % 10;
    const totalSec = Math.floor(totalTenths / 10);
    return `${String(Math.floor(totalSec / 60)).padStart(2, "0")}:${String(totalSec % 60).padStart(2, "0")}.${tenths}`;
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function parseKvText(raw) {
    const out = {};
    String(raw || "")
      .split(/[,，;\n]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .forEach((pair) => {
        const idx = pair.indexOf("=");
        if (idx > 0) {
          const k = pair.slice(0, idx).trim().toLowerCase();
          const v = pair.slice(idx + 1).trim();
          if (k && v) out[k] = v;
        }
      });
    return out;
  }

  async function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.accessToken) headers.set("X-Access-Token", state.accessToken);
    const resp = await fetch(url, { ...options, headers });
    let data = null;
    try { data = await resp.json(); } catch { data = null; }
    if (resp.status === 401) {
      $("tokenModal").hidden = false;
      throw new Error(data?.detail || "需要访问口令");
    }
    if (!resp.ok) throw new Error(data?.detail || data?.message || `请求失败 (${resp.status})`);
    return data;
  }

  function getActiveLanguages(engineId = state.selectedEngineId) {
    return state.languages.filter((l) => (l.engines || []).includes(engineId));
  }

  function estimateChunks(text) {
    const len = text.trim().length;
    if (!len) return 0;
    const cjk = (text.match(/[\u4e00-\u9fff\u3040-\u30ff]/g) || []).length;
    const limit = cjk / len > 0.25 ? 95 : 175;
    return Math.max(1, Math.ceil(len / limit));
  }

  function switchWorkspaceTab(tabName) {
    state.workspaceTab = tabName;
    document.querySelectorAll("[data-workspace-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.workspaceTab === tabName);
    });
    $("workspaceAudio").hidden = tabName !== "audio";
    $("workspacePodcast").hidden = tabName !== "podcast";
    $("workspaceVideo").hidden = tabName !== "video";
    $("workspaceTools").hidden = tabName !== "tools";
    if (tabName === "podcast") {
      renderPodcastSpeakersUI();
      const curPod = state.podcasts.find((p) => p.id === state.activePodcastId);
      if (curPod) renderActivePodcast(curPod);
    }
    if (tabName === "tools") {
      renderDeveloperSnippets();
    }
  }

  // ---------- Live Chunk-Streaming Player ("边生成边试听") ----------
  function resetStreamQueue() {
    state.streamQueue = [];
    state.streamPlayedSet.clear();
    state.streamPlaying = false;
    state.streamAudio.pause();
  }

  function enqueueStreamChunk(url) {
    if (state.streamPlayedSet.has(url)) return;
    state.streamPlayedSet.add(url);
    state.streamQueue.push(url);
    if (!state.streamPlaying) playNextStreamChunk();
  }

  function playNextStreamChunk() {
    if (!state.streamQueue.length) {
      state.streamPlaying = false;
      return;
    }
    state.streamPlaying = true;
    const nextUrl = state.streamQueue.shift();
    state.streamAudio.src = nextUrl;
    state.streamAudio.play().catch(() => { state.streamPlaying = false; });
  }

  state.streamAudio.addEventListener("ended", playNextStreamChunk);

  // ---------- Canvas Waveform ----------
  function drawWaveform(canvas, peaks, progress = 0, accentColor = "#0D6E6E") {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(80, Math.floor(rect.width || 180));
    const cssH = Math.max(32, Math.floor(rect.height || 40));
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const bars = Array.isArray(peaks) && peaks.length ? peaks : new Array(80).fill(0.18);
    const targetCount = Math.min(bars.length, Math.max(36, Math.floor(cssW / 4)));
    const sampled = [];
    for (let i = 0; i < targetCount; i++) {
      sampled.push(bars[Math.min(bars.length - 1, Math.floor((i / targetCount) * bars.length))]);
    }
    const count = sampled.length;
    const gap = 1.5;
    const barW = Math.max(1.5, (cssW - gap * (count - 1)) / count);
    const midY = cssH / 2;

    for (let i = 0; i < count; i++) {
      const val = Math.max(0.06, Math.min(1, Number(sampled[i]) || 0.1));
      const h = Math.max(3, val * (cssH - 8));
      ctx.fillStyle = i / count <= progress ? accentColor : "#CBD5E1";
      ctx.beginPath();
      ctx.roundRect(i * (barW + gap), midY - h / 2, barW, h, 1.5);
      ctx.fill();
    }
    if (progress > 0 && progress < 1) {
      const px = progress * cssW;
      ctx.strokeStyle = "#D92D20";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 2);
      ctx.lineTo(px, cssH - 2);
      ctx.stroke();
    }
  }

  function stopAllAudioExcept(exceptKey = null) {
    state.streamAudio.pause();
    state.streamPlaying = false;
    state.audioPlayers.forEach((entry, key) => {
      if (key !== exceptKey && !entry.audio.paused) {
        entry.audio.pause();
        entry.btn?.classList.remove("playing");
      }
    });
  }

  function bindWavePlayer({ key, url, peaks, canvas, btn, timecodeEl, durationSec }) {
    const prev = state.audioPlayers.get(key);
    if (prev && prev.url === url && prev.canvas === canvas) {
      drawWaveform(canvas, peaks, prev.audio.duration ? prev.audio.currentTime / prev.audio.duration : 0);
      return;
    }
    if (prev) prev.audio.pause();

    const audio = new Audio();
    audio.preload = "none";
    audio.src = url;
    const entry = { key, url, audio, peaks, canvas, btn, timecodeEl, durationSec: durationSec || 0 };
    state.audioPlayers.set(key, entry);

    const updateVisual = () => {
      const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : entry.durationSec;
      drawWaveform(canvas, peaks, dur > 0 ? Math.min(1, audio.currentTime / dur) : 0);
      if (timecodeEl) {
        timecodeEl.textContent = dur > 0 ? `${formatSec(audio.currentTime)} / ${formatSec(dur)}` : formatSec(audio.currentTime);
      }
    };
    audio.addEventListener("loadedmetadata", updateVisual);
    audio.addEventListener("timeupdate", updateVisual);
    audio.addEventListener("ended", () => {
      btn.classList.remove("playing");
      drawWaveform(canvas, peaks, 0);
      if (timecodeEl && entry.durationSec) timecodeEl.textContent = `00:00 / ${formatSec(entry.durationSec)}`;
    });

    btn.onclick = () => {
      if (audio.paused) {
        stopAllAudioExcept(key);
        audio.play().then(() => btn.classList.add("playing")).catch((e) => showToast(`播放失败: ${e.message}`, "error"));
      } else {
        audio.pause();
        btn.classList.remove("playing");
      }
    };
    canvas.onclick = (evt) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
      const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : entry.durationSec;
      if (dur > 0) {
        audio.currentTime = ratio * dur;
        updateVisual();
        if (audio.paused) {
          stopAllAudioExcept(key);
          audio.play().then(() => btn.classList.add("playing")).catch(() => {});
        }
      }
    };
    requestAnimationFrame(() => updateVisual());
  }

  // ---------- System & 3-Engine Status ----------
  async function loadSystemStatus(initial = false) {
    try {
      const data = await apiFetch("/api/system/status");
      const eng = data.engine;
      state.enginesStatus = eng.engines || {};
      state.features = data.features || {};
      const selEng = state.enginesStatus[state.selectedEngineId] || eng;

      const anyLoaded = Object.values(state.enginesStatus).some((e) => e.state === "ready") || data.asr?.loaded;

      const dot = $("engineDot");
      dot.className = "status-dot";
      if (selEng.state === "ready") {
        dot.classList.add("ready");
        $("engineStatusText").textContent = `${selEng.display_name} · ${selEng.precision || "就绪"}`;
        $("btnPreload").hidden = true;
      } else if (selEng.state === "loading" || selEng.state === "downloading") {
        dot.classList.add("busy");
        $("engineStatusText").textContent = selEng.state_detail || "模型加载中…";
        $("btnPreload").hidden = true;
      } else if (selEng.state === "error") {
        dot.classList.add("error");
        $("engineStatusText").textContent = "引擎异常";
        $("btnPreload").hidden = false;
      } else if (!selEng.installed) {
        $("engineStatusText").textContent = `${selEng.display_name} · 未安装`;
        $("btnPreload").hidden = true;
      } else {
        $("engineStatusText").textContent = `${selEng.display_name} · 按需加载`;
        $("btnPreload").hidden = false;
      }
      $("btnUnloadVram").hidden = !anyLoaded;

      if (eng.gpu?.available) {
        const shortGpu = eng.gpu.name.replace(/^NVIDIA\s+(Tesla\s+|GeForce\s+)?/i, "");
        const vramTxt = eng.gpu.vram_total_gb > 0 ? ` ${eng.gpu.vram_used_gb}/${eng.gpu.vram_total_gb}G` : "";
        $("gpuTelemetry").textContent = `${shortGpu}${vramTxt}`;
      } else {
        $("gpuTelemetry").textContent = "CPU 模式";
      }

      const setBadge = (eid, elId, fallbackTxt) => {
        const st = state.enginesStatus[eid];
        if (!st) return;
        const card = document.querySelector(`[data-select-engine="${eid}"]`);
        if (card) card.classList.toggle("unavailable", !st.installed);
        $(elId).textContent = !st.installed ? "未启用" : st.state === "ready" ? "已加载 · 就绪" : fallbackTxt;
      };
      const qv = (state.enginesStatus.qwen3?.variant || "1.7b").toUpperCase();
      setBadge("qwen3", "badgeEngineQwen", `${qv} · 10 语种`);
      setBadge("chatterbox", "badgeEngineChatterbox", "23 语种 · MIT");
      setBadge("cosyvoice3", "badgeEngineCosy", "方言与情绪指令");

      $("demucsStatusBadge").textContent = state.features.demucs_available
        ? "人声分离：Demucs htdemucs"
        : "人声分离：FFmpeg 智能闪避";
      $("ytdlpStatusBadge").textContent = state.features.yt_dlp_available
        ? "视频直抓：yt-dlp 就绪"
        : "视频直抓：需装 yt-dlp";
      $("ytdlpStatusBadge").className = `quality-badge ${state.features.yt_dlp_available ? "" : "fair"}`;

      const lsAvail = Boolean(state.features.lipsync_available);
      $("chkEnableLipsync").disabled = !lsAvail;
      $("lipsyncCheckRow").classList.toggle("disabled-row", !lsAvail);
      $("lipsyncLabel").textContent = lsAvail
        ? "启用 AI 唇形同步 Lip-Sync（MuseTalk GPU 已就绪）"
        : "启用 AI 唇形同步 Lip-Sync（需在 Colab L4/A100 Cell 2 勾选 MuseTalk）";

      if (data.asr?.configured_model_size) {
        $("asrOptionLabel").textContent = `上传后自动 Whisper ${data.asr.configured_model_size} 识别参考文本`;
      }

      if (initial) {
        state.languages = data.languages || [];
        state.presets = data.presets || [];
        state.cosyPresets = data.cosyvoice_presets || [];
        state.selectedLangs = new Set(data.default_selected_langs || ["zh", "en", "ja", "fr"]);
        if (data.settings?.mastering_preset) {
          $("audioMasteringSelect").value = data.settings.mastering_preset;
        }
        renderPresets();
        renderCosyPresets();
        syncEngineUI();
        renderPodcastLangChips();
        renderVideoLangChips();
        applyPreset(state.presets[0]?.id, false);
        fillSettingsModal(data.settings);
        renderDeveloperSnippets();
      }
    } catch (err) {
      if (!String(err.message).includes("口令")) {
        $("engineStatusText").textContent = "后端离线";
        $("engineDot").className = "status-dot error";
      }
    }
  }

  function renderCosyPresets() {
    const box = $("cosyInstructPresets");
    box.innerHTML = state.cosyPresets
      .map(
        (p) =>
          `<button type="button" class="preset-chip ${state.cosyInstruct === p.instruct ? "active" : ""}" data-cosy-preset="${escapeHtml(p.instruct)}">${escapeHtml(p.label)}</button>`
      )
      .join("");
    box.querySelectorAll("[data-cosy-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.cosyInstruct = btn.dataset.cosyPreset;
        $("cosyInstructInput").value = state.cosyInstruct;
        renderCosyPresets();
      });
    });
  }

  function syncEngineUI() {
    document.querySelectorAll("[data-select-engine]").forEach((el) => {
      el.classList.toggle("active", el.dataset.selectEngine === state.selectedEngineId);
    });
    const isQwen = state.selectedEngineId === "qwen3";
    const isCb = state.selectedEngineId === "chatterbox";
    const isCosy = state.selectedEngineId === "cosyvoice3";

    $("qwenModeBox").hidden = !isQwen;
    $("chatterboxParamBox").hidden = !isCb;
    $("cosyInstructBox").hidden = !isCosy;
    $("btnSelectAll23").hidden = !isCb;

    const allowed = new Set(getActiveLanguages().map((l) => l.code));
    state.selectedLangs.forEach((code) => { if (!allowed.has(code)) state.selectedLangs.delete(code); });
    if (state.selectedLangs.size === 0) state.selectedLangs.add("zh");

    $("langSectionLabel").textContent = isCb
      ? "目标语种（Chatterbox V3 原生支持 23 国语言）"
      : isCosy
      ? "目标语种（CosyVoice 3 支持 9 语种 + 18+ 中文方言）"
      : "目标语种（Qwen3-TTS 原生支持 10 国语言）";

    renderLangGrid();
    renderPerLangScripts();
    loadSystemStatus(false);
  }

  // ---------- Voices ----------
  async function loadVoices() {
    const data = await apiFetch("/api/voices");
    state.voices = data.voices || [];
    if (!state.selectedVoiceId || !state.voices.some((v) => v.id === state.selectedVoiceId)) {
      state.selectedVoiceId = state.voices[0]?.id || null;
    }
    renderVoiceList();
    renderActiveVoiceInspector();
    refreshVideoVoiceDropdown();
    renderPodcastSpeakersUI();
    renderActiveVideo();
    renderDeveloperSnippets();
  }

  function refreshVideoVoiceDropdown() {
    const sel = $("vidVoiceSelect");
    const prev = sel.value;
    sel.innerHTML =
      `<option value="auto_video">自动克隆原说话人声纹（推荐）</option>` +
      state.voices.map((v) => `<option value="${escapeHtml(v.id)}">声纹库：${escapeHtml(v.name)}</option>`).join("");
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  }

  function renderVoiceList() {
    const list = $("voiceList");
    $("voiceCountBadge").textContent = `${state.voices.length} 个`;
    if (!state.voices.length) {
      list.innerHTML = `<div class="empty-hint">暂无声纹档案，请在上方上传或录制 5–15 秒人声。</div>`;
      return;
    }
    list.innerHTML = state.voices
      .map((v) => {
        const active = v.id === state.selectedVoiceId ? "active" : "";
        const badge = v.is_demo ? "示范" : `${v.quality_score ?? "—"}分`;
        const delBtn = v.is_demo ? "" : `<button type="button" class="btn-del-mini" data-del-voice="${escapeHtml(v.id)}">删除</button>`;
        return `
          <div class="voice-item ${active}" data-select-voice="${escapeHtml(v.id)}">
            <div class="voice-item-info">
              <div class="voice-item-name">${escapeHtml(v.name)}</div>
              <div class="voice-item-meta">${v.duration_sec}s · ${Math.round((v.sample_rate || 24000) / 1000)}kHz · ${badge}</div>
            </div>
            ${delBtn}
          </div>
        `;
      })
      .join("");
    list.querySelectorAll("[data-select-voice]").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest("[data-del-voice]")) return;
        state.selectedVoiceId = el.dataset.selectVoice;
        renderVoiceList();
        renderActiveVoiceInspector();
      });
    });
    list.querySelectorAll("[data-del-voice]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await apiFetch(`/api/voices/${btn.dataset.delVoice}`, { method: "DELETE" });
          showToast("已删除声纹档案");
          await loadVoices();
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    });
  }

  function renderActiveVoiceInspector() {
    const voice = state.voices.find((v) => v.id === state.selectedVoiceId);
    const box = $("voiceInspector");
    if (!voice) { box.hidden = true; return; }
    box.hidden = false;
    $("activeVoiceName").textContent = voice.name;
    const qb = $("activeQualityBadge");
    qb.textContent = `${voice.quality_label || "就绪"} · ${voice.quality_score ?? 90}`;
    qb.className = `quality-badge ${voice.quality_grade || "good"}`;

    $("activeTelemetryRow").innerHTML = [
      `时长 ${voice.duration_sec}s`,
      `${Math.round((voice.sample_rate || 24000) / 1000)}kHz 单声道`,
      `电平 ${voice.rms_db} dB`,
      voice.detected_language && voice.detected_language !== "unknown" ? `原声 ${voice.detected_language.toUpperCase()}` : null,
    ].filter(Boolean).map((t) => `<span class="tele-chip">${escapeHtml(t)}</span>`).join("");

    const warns = [...(voice.warnings || []), ...(voice.asr_note ? [voice.asr_note] : [])];
    $("activeVoiceWarnings").hidden = warns.length === 0;
    if (warns.length) $("activeVoiceWarnings").innerHTML = warns.map((w) => `<div>• ${escapeHtml(w)}</div>`).join("");

    $("activeRefText").value = voice.ref_text || "";
    bindWavePlayer({
      key: `ref_${voice.id}`,
      url: voice.audio_url,
      peaks: voice.waveform_peaks,
      canvas: $("refWaveCanvas"),
      btn: $("btnPlayRef"),
      timecodeEl: $("refTimecode"),
      durationSec: voice.duration_sec,
    });
  }

  async function uploadAudioFile(file, customName = "") {
    if (!$("chkVoiceConsent").checked) {
      showToast("请先勾选「确认该录音为本人声音或已获明确授权」", "error");
      return;
    }
    const statusEl = $("uploadStatus");
    statusEl.hidden = false;
    statusEl.className = "inline-status";
    statusEl.textContent = "正在转码音频至 24kHz 并运行 Whisper 识别…";

    const fd = new FormData();
    fd.append("file", file, file.name || "recording.webm");
    fd.append("name", customName || $("uploadVoiceName").value.trim());
    fd.append("auto_transcribe", $("uploadAutoAsr").checked ? "true" : "false");

    try {
      const data = await apiFetch("/api/voices", { method: "POST", body: fd });
      state.selectedVoiceId = data.voice.id;
      $("uploadVoiceName").value = "";
      $("audioFileInput").value = "";
      statusEl.textContent = `声纹「${data.voice.name}」已入库（${data.voice.duration_sec}s · ${data.voice.quality_label}）`;
      await loadVoices();
      showToast(`声纹「${data.voice.name}」准备就绪`, "teal");
    } catch (err) {
      statusEl.className = "inline-status error";
      statusEl.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  // ---------- Mic Recorder ----------
  function drawIdleMicMeter() {
    const canvas = $("micMeterCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#DCE1E8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(12, canvas.height / 2);
    ctx.lineTo(canvas.width - 12, canvas.height / 2);
    ctx.stroke();
  }

  async function toggleMicRecording() {
    const btn = $("btnRecToggle");
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      showToast("当前浏览器不支持麦克风访问，请使用 HTTPS 打开或直接上传录音", "error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true } });
      state.micStream = stream;
      state.micChunks = [];
      state.micBlob = null;
      $("recSaveRow").hidden = true;
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      state.mediaRecorder = recorder;
      recorder.ondataavailable = (e) => { if (e.data?.size > 0) state.micChunks.push(e.data); };
      recorder.onstop = () => {
        clearInterval(state.micTimerId);
        cancelAnimationFrame(state.micAnimId);
        stream.getTracks().forEach((t) => t.stop());
        state.micAudioCtx?.close().catch(() => {});
        btn.classList.remove("recording");
        const elapsed = Date.now() - state.micStartMs;
        state.micBlob = new Blob(state.micChunks, { type: recorder.mimeType || "audio/webm" });
        if (elapsed < 1500) { $("recHint").textContent = "录音不足 1.5 秒，请重新录制"; drawIdleMicMeter(); return; }
        $("recHint").textContent = `录制完成（${(elapsed / 1000).toFixed(1)}s），输入名称后即可入库`;
        $("recSaveRow").hidden = false;
      };
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      state.micAudioCtx = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      audioCtx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.frequencyBinCount);
      const canvas = $("micMeterCanvas");
      const ctx = canvas.getContext("2d");
      const drawLive = () => {
        state.micAnimId = requestAnimationFrame(drawLive);
        analyser.getByteFrequencyData(buf);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const bars = 40;
        const step = Math.floor(buf.length / bars);
        const bw = (canvas.width - 20 - 3 * (bars - 1)) / bars;
        for (let i = 0; i < bars; i++) {
          const val = buf[i * step] / 255;
          const h = Math.max(4, val * (canvas.height - 14));
          ctx.fillStyle = val > 0.85 ? "#D92D20" : "#0D6E6E";
          ctx.beginPath();
          ctx.roundRect(10 + i * (bw + 3), (canvas.height - h) / 2, bw, h, 2);
          ctx.fill();
        }
      };
      drawLive();
      recorder.start(200);
      state.micStartMs = Date.now();
      btn.classList.add("recording");
      $("recHint").textContent = "正在录音… 建议朗读 8–12 秒后点击停止";
      state.micTimerId = setInterval(() => { $("recTimer").textContent = formatTenths(Date.now() - state.micStartMs); }, 100);
    } catch (err) {
      showToast(`无法打开麦克风: ${err.message}`, "error");
    }
  }

  // ==================== WORKSPACE 1: AUDIO & LONG-FORM ====================
  function renderPresets() {
    const bar = $("presetBar");
    bar.innerHTML = state.presets
      .map((p) => `<button type="button" class="preset-chip ${state.activePresetId === p.id ? "active" : ""}" data-preset="${escapeHtml(p.id)}" title="${escapeHtml(p.subtitle)}">${escapeHtml(p.title)}</button>`)
      .join("");
    bar.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset, true));
    });
  }

  function applyPreset(presetId, notify = true) {
    const preset = state.presets.find((p) => p.id === presetId);
    if (!preset) return;
    state.activePresetId = preset.id;
    state.sourceDirty = false;
    $("sourceScript").value = preset.source_text || "";
    updateSourceCount();
    state.scripts = { ...(preset.scripts || {}) };
    $("translateStatus").hidden = true;
    renderPresets();
    renderPerLangScripts();
    if (notify) showToast(`已载入「${preset.title}」`, "teal");
  }

  function updateSourceCount() {
    const txt = $("sourceScript").value.trim();
    const nChunks = estimateChunks(txt);
    $("sourceCharCount").textContent = nChunks > 1
      ? `${txt.length} 字 · 长文模式（自动切分约 ${nChunks} 段 + 批加速）`
      : `${txt.length} 字 · 单句模式`;
  }

  function renderLangGrid() {
    const grid = $("langGrid");
    const activeLangs = getActiveLanguages();
    grid.innerHTML = activeLangs
      .map((lang) => {
        const selected = state.selectedLangs.has(lang.code) ? "selected" : "";
        return `
          <button type="button" class="lang-chip ${selected}" data-lang="${escapeHtml(lang.code)}">
            <span class="lang-code-tag">${escapeHtml(lang.region_tag)}</span>
            <span class="lang-chip-labels">
              <span class="lang-zh">${escapeHtml(lang.label_zh)}</span>
              <span class="lang-native">${escapeHtml(lang.native_name)}</span>
            </span>
          </button>
        `;
      })
      .join("");
    grid.querySelectorAll("[data-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const code = btn.dataset.lang;
        if (state.selectedLangs.has(code)) {
          if (state.selectedLangs.size <= 1) { showToast("请至少保留一种目标语言"); return; }
          state.selectedLangs.delete(code);
        } else {
          state.selectedLangs.add(code);
        }
        renderLangGrid();
        renderPerLangScripts();
        updateGenerateButtonLabel();
      });
    });
    updateGenerateButtonLabel();
  }

  function updateGenerateButtonLabel() {
    $("btnGenerateLabel").textContent = `一键生成 ${state.selectedLangs.size} 国语音`;
  }

  function renderPerLangScripts() {
    const container = $("perLangScripts");
    if (state.scriptsCollapsed) {
      container.hidden = true;
      $("btnToggleScripts").textContent = "展开编辑区";
      return;
    }
    container.hidden = false;
    $("btnToggleScripts").textContent = "收起编辑区";
    const selectedList = getActiveLanguages().filter((l) => state.selectedLangs.has(l.code));
    container.innerHTML = selectedList
      .map((l) => `
        <div class="lang-script-card">
          <div class="lang-script-top">
            <span>${escapeHtml(l.label_zh)} · ${escapeHtml(l.native_name)}</span>
            <span>${escapeHtml(l.region_tag)}</span>
          </div>
          <textarea class="field-textarea compact" rows="3" data-lang-script="${escapeHtml(l.code)}" placeholder="点击「一键翻译所选语种」自动生成，或直接手动输入">${escapeHtml(state.scripts[l.code] || "")}</textarea>
        </div>
      `)
      .join("");
    container.querySelectorAll("[data-lang-script]").forEach((ta) => {
      ta.addEventListener("input", () => {
        state.scripts[ta.dataset.langScript] = ta.value;
        state.sourceDirty = false;
      });
    });
  }

  async function runOneClickTranslate(silent = false) {
    const src = $("sourceScript").value.trim();
    if (!src) { showToast("请先输入文案内容", "error"); return false; }
    const targetLangs = getActiveLanguages().filter((l) => state.selectedLangs.has(l.code)).map((l) => l.code);
    const btn = $("btnTranslate");
    const statusEl = $("translateStatus");
    btn.disabled = true;
    statusEl.hidden = false;
    statusEl.className = "inline-status";
    statusEl.textContent = `正在应用术语词典并并发翻译为 ${targetLangs.length} 种语言…`;
    try {
      const data = await apiFetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_text: src, target_langs: targetLangs }),
      });
      Object.entries(data.translations || {}).forEach(([k, v]) => { state.scripts[k] = v; });
      state.activePresetId = null;
      state.sourceDirty = false;
      renderPresets();
      renderPerLangScripts();
      statusEl.textContent = `翻译完成 · 引擎：${data.provider}${data.warnings?.length ? `（${data.warnings[0]}）` : ""}`;
      if (!silent) showToast(`已生成 ${targetLangs.length} 国台词`, "teal");
      return true;
    } catch (err) {
      statusEl.className = "inline-status error";
      statusEl.textContent = `翻译失败：${err.message}`;
      showToast(err.message, "error");
      return false;
    } finally {
      btn.disabled = false;
    }
  }

  async function startMultilingualGeneration() {
    if (!state.selectedVoiceId) { showToast("请先在左侧选择或上传参考人声", "error"); return; }
    const selSt = state.enginesStatus[state.selectedEngineId];
    if (selSt && !selSt.installed) { showToast(`${selSt.display_name} 尚未安装，请在安装脚本或 Colab Cell 2 勾选启用`, "error"); return; }

    const targetLangs = getActiveLanguages().filter((l) => state.selectedLangs.has(l.code)).map((l) => l.code);
    const missing = targetLangs.filter((code) => !(state.scripts[code] || "").trim());
    if (state.sourceDirty || missing.length > 0) {
      if (!$("sourceScript").value.trim()) { showToast("请先输入台词内容", "error"); return; }
      if (!(await runOneClickTranslate(true))) return;
    }

    const scriptsPayload = {};
    targetLangs.forEach((c) => { if ((state.scripts[c] || "").trim()) scriptsPayload[c] = state.scripts[c].trim(); });

    resetStreamQueue();
    const btn = $("btnGenerate");
    btn.disabled = true;
    try {
      const data = await apiFetch("/api/jobs/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          voice_id: state.selectedVoiceId,
          scripts: scriptsPayload,
          engine_id: state.selectedEngineId,
          mode: state.mode,
          exaggeration: state.exaggeration,
          cfg_weight: state.cfgWeight,
          instruct: $("cosyInstructInput").value.trim(),
          mastering_preset: $("audioMasteringSelect").value,
          source_text: $("sourceScript").value.trim(),
          preset_id: state.activePresetId,
          auto_reel: $("chkAutoReel").checked,
        }),
      });
      state.activeJobId = data.job.id;
      upsertJobLocal(data.job);
      renderHistory();
      renderJob(data.job);
      startPollingJob(data.job.id);
      showToast(`已启动 ${targetLangs.length} 语种声纹克隆任务`, "teal");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  function upsertJobLocal(job) {
    const idx = state.jobs.findIndex((j) => j.id === job.id);
    if (idx >= 0) state.jobs[idx] = job;
    else state.jobs.unshift(job);
  }

  function startPollingJob(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const data = await apiFetch(`/api/jobs/${jobId}`);
        upsertJobLocal(data.job);
        if ($("chkStreamPreview").checked && state.activeJobId === jobId) {
          const firstTrack = (data.job.tracks || [])[0];
          if (firstTrack?.status === "generating" && Array.isArray(firstTrack.chunk_urls) && firstTrack.total_chunks > 1) {
            firstTrack.chunk_urls.forEach((u) => enqueueStreamChunk(u));
          }
        }
        if (state.activeJobId === jobId) renderJob(data.job);
        renderHistory();
        loadSystemStatus(false);
        const terminal = ["completed", "partial", "failed"].includes(data.job.status);
        const anyRunning = (data.job.tracks || []).some((t) => t.status === "generating" || t.status === "pending");
        if (terminal && !anyRunning) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          if (data.job.status === "completed") showToast("全部多语种音轨生成完成！", "teal");
        }
      } catch {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    }, 1200);
  }

  async function loadJobs() {
    const data = await apiFetch("/api/jobs?limit=20");
    state.jobs = data.jobs || [];
    if (!state.activeJobId && state.jobs.length > 0) state.activeJobId = state.jobs[0].id;
    renderHistory();
    const cur = state.jobs.find((j) => j.id === state.activeJobId);
    if (cur) {
      renderJob(cur);
      if (["queued", "loading_model", "running"].includes(cur.status)) startPollingJob(cur.id);
    }
  }

  function renderJob(job) {
    if (!job) return;
    const statusMap = {
      queued: ["排队中", "running"],
      loading_model: ["装载 GPU 引擎…", "running"],
      running: ["正在克隆生成…", "running"],
      completed: ["全部生成完成", ""],
      partial: ["部分完成", "running"],
      failed: ["生成失败", "failed"],
    };
    const [statusLabel, statusCls] = statusMap[job.status] || [job.status, ""];
    $("jobMetaStrip").hidden = false;
    $("jobStatusBadge").textContent = statusLabel;
    $("jobStatusBadge").className = `job-status-badge ${statusCls}`;
    const mLbl = MASTERING_LABELS[job.mastering_preset] || "原始干声";
    $("jobMetaDesc").textContent = `引擎：${job.engine_name || job.engine_id} · 声纹：${job.voice_name} · 母带：${mLbl} · ${job.total_count} 语种`;

    const doneCount = (job.completed_count || 0) + (job.failed_count || 0);
    const pct = job.total_count ? Math.round((doneCount / job.total_count) * 100) : 0;
    const isBusy = ["queued", "loading_model", "running"].includes(job.status);
    $("jobProgressWrap").hidden = !isBusy;
    $("jobProgressFill").style.width = `${Math.max(8, pct)}%`;
    $("jobProgressLabel").textContent = `${doneCount} / ${job.total_count}`;

    const hasCompleted = (job.completed_count || 0) > 0;
    $("jobHeaderActions").hidden = !hasCompleted;
    if (hasCompleted) { $("btnDownloadZip").href = job.zip_url; $("btnBuildReel").disabled = isBusy; }

    const reelCard = $("reelCard");
    if (job.reel?.mp3_url) {
      reelCard.hidden = false;
      $("reelLangSummary").textContent = (job.reel.lang_codes || []).map((c) => c.toUpperCase()).join(" → ");
      $("reelDurationMetric").textContent = `${job.reel.duration_sec}s · 192k MP3`;
      $("btnDownloadReel").href = job.reel.mp3_url;
      $("btnDownloadReel").download = `VoxPoly_Reel_${job.id}.mp3`;
      bindWavePlayer({
        key: `reel_${job.id}_${job.reel.created_at || ""}`,
        url: `${job.reel.mp3_url}?t=${encodeURIComponent(job.reel.created_at || "1")}`,
        peaks: job.reel.waveform_peaks,
        canvas: $("reelWaveCanvas"),
        btn: $("btnPlayReel"),
        timecodeEl: $("reelTimecode"),
        durationSec: job.reel.duration_sec,
      });
    } else {
      reelCard.hidden = true;
    }

    const container = $("tracksList");
    container.innerHTML = (job.tracks || [])
      .map((t) => {
        const cls = t.status === "generating" ? "generating" : t.status === "failed" ? "failed" : "";
        let bodyHtml = "";
        if (t.status === "completed") {
          const chunkChip = t.total_chunks > 1 ? `<span class="tele-chip teal">长文 ${t.total_chunks} 段拼接</span>` : "";
          const mastChip = t.mastering_preset && t.mastering_preset !== "raw"
            ? `<span class="tele-chip teal">${escapeHtml(MASTERING_LABELS[t.mastering_preset] || t.mastering_preset)}</span>`
            : "";
          const wmChip = t.watermarked ? `<span class="tele-chip teal">隐形水印</span>` : "";
          bodyHtml = `
            <div class="wave-player">
              <button type="button" class="play-btn" data-play-track="${escapeHtml(t.lang_code)}" aria-label="播放 ${escapeHtml(t.lang_label)}">
                <svg class="icon-play" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg>
                <svg class="icon-pause" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
              </button>
              <canvas class="wave-canvas" data-wave-track="${escapeHtml(t.lang_code)}" width="120" height="40"></canvas>
              <span class="timecode" data-time-track="${escapeHtml(t.lang_code)}">00:00 / ${formatSec(t.duration_sec)}</span>
            </div>
            <div class="track-footer">
              <div class="track-metrics">
                <span class="tele-chip">时长 ${t.duration_sec}s</span>
                <span class="tele-chip">耗时 ${t.elapsed_sec}s</span>
                <span class="tele-chip">RTF ${t.rtf}×</span>
                ${chunkChip}${mastChip}${wmChip}
              </div>
              <div class="track-dl-group">
                <button type="button" class="btn-outline btn-sm" data-regen-track="${escapeHtml(t.lang_code)}">重新生成</button>
                <a class="btn-outline btn-sm" href="${escapeHtml(t.mp3_url)}" download="${escapeHtml(job.id)}_${escapeHtml(t.lang_code)}.mp3">MP3</a>
                <a class="btn-outline btn-sm" href="${escapeHtml(t.wav_url)}" download="${escapeHtml(job.id)}_${escapeHtml(t.lang_code)}.wav">WAV</a>
              </div>
            </div>
          `;
        } else if (t.status === "generating") {
          const readyCount = (t.ready_chunks || []).length;
          const totalC = t.total_chunks || 1;
          const previewBtn = t.chunk_urls?.length
            ? `<button type="button" class="btn-outline btn-sm" data-preview-chunk="${escapeHtml(t.chunk_urls[t.chunk_urls.length - 1])}">试听第 ${readyCount} 段</button>`
            : "";
          bodyHtml = `<div class="track-pending-bar"><span>GPU 正在合成 ${escapeHtml(t.lang_label)}${totalC > 1 ? `（第 ${Math.min(readyCount + 1, totalC)} / ${totalC} 段）` : ""}…</span>${previewBtn}</div>`;
        } else if (t.status === "failed") {
          bodyHtml = `<div class="track-pending-bar"><span>失败：${escapeHtml(t.error || "未知错误")}</span><button type="button" class="btn-outline btn-sm" data-regen-track="${escapeHtml(t.lang_code)}">重试</button></div>`;
        } else {
          bodyHtml = `<div class="track-pending-bar"><span>队列等待中（预计 ${t.total_chunks || 1} 段）…</span></div>`;
        }
        return `
          <article class="track-card ${cls}">
            <div class="track-top">
              <div class="track-lang-meta">
                <span class="track-code">${escapeHtml(t.region_tag || t.lang_code.toUpperCase())}</span>
                <span class="track-title">${escapeHtml(t.lang_label)}</span>
                <span class="track-native">${escapeHtml(t.native_name || "")}</span>
              </div>
              <span class="count-mono">${t.status === "completed" ? "24kHz PCM" : t.status}</span>
            </div>
            <div class="track-text">${escapeHtml(t.text)}</div>
            ${bodyHtml}
          </article>
        `;
      })
      .join("");

    (job.tracks || []).forEach((t) => {
      if (t.status === "completed") {
        const canvas = container.querySelector(`[data-wave-track="${t.lang_code}"]`);
        const btn = container.querySelector(`[data-play-track="${t.lang_code}"]`);
        const tc = container.querySelector(`[data-time-track="${t.lang_code}"]`);
        if (canvas && btn) {
          bindWavePlayer({
            key: `track_${job.id}_${t.lang_code}_${t.elapsed_sec || ""}`,
            url: `${t.mp3_url || t.wav_url}?v=${encodeURIComponent(job.updated_at || "1")}`,
            peaks: t.waveform_peaks,
            canvas,
            btn,
            timecodeEl: tc,
            durationSec: t.duration_sec,
          });
        }
      }
    });

    container.querySelectorAll("[data-preview-chunk]").forEach((btn) => {
      btn.addEventListener("click", () => {
        stopAllAudioExcept(null);
        state.streamAudio.src = btn.dataset.previewChunk;
        state.streamAudio.play().catch(() => {});
      });
    });

    container.querySelectorAll("[data-regen-track]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const code = btn.dataset.regenTrack;
        try {
          const data = await apiFetch(`/api/jobs/${job.id}/regenerate/${code}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: (state.scripts[code] || "").trim() || undefined,
              mode: state.mode,
              exaggeration: state.exaggeration,
              cfg_weight: state.cfgWeight,
              instruct: $("cosyInstructInput").value.trim(),
              mastering_preset: $("audioMasteringSelect").value,
            }),
          });
          upsertJobLocal(data.job);
          renderJob(data.job);
          startPollingJob(job.id);
          showToast(`正在重新合成 ${code.toUpperCase()} 音轨`);
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    });
  }

  function renderHistory() {
    const list = $("historyList");
    $("historyCount").textContent = `${state.jobs.length} 条记录`;
    if (!state.jobs.length) { list.innerHTML = `<div class="empty-hint">暂无历史批次</div>`; return; }
    list.innerHTML = state.jobs
      .map((j) => {
        const active = j.id === state.activeJobId ? "active" : "";
        const langs = (j.tracks || []).map((t) => t.region_tag || t.lang_code.toUpperCase()).join(" · ");
        const created = (j.created_at || "").replace("T", " ").slice(0, 16);
        const tag = j.engine_id === "chatterbox" ? "CB23" : j.engine_id === "cosyvoice3" ? "CV3" : "QW10";
        return `
          <div class="history-item ${active}" data-open-job="${escapeHtml(j.id)}">
            <div class="history-main">
              <div class="history-title">[${tag}] ${escapeHtml(j.voice_name)} · ${escapeHtml(langs)}</div>
              <div class="history-sub">${created} · ${j.completed_count}/${j.total_count} 完成</div>
            </div>
            <button type="button" class="btn-del-mini" data-del-job="${escapeHtml(j.id)}">删除</button>
          </div>
        `;
      })
      .join("");
    list.querySelectorAll("[data-open-job]").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest("[data-del-job]")) return;
        state.activeJobId = el.dataset.openJob;
        renderHistory();
        renderJob(state.jobs.find((j) => j.id === state.activeJobId));
      });
    });
    list.querySelectorAll("[data-del-job]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/jobs/${btn.dataset.delJob}`, { method: "DELETE" });
        state.jobs = state.jobs.filter((j) => j.id !== btn.dataset.delJob);
        if (state.activeJobId === btn.dataset.delJob) state.activeJobId = state.jobs[0]?.id || null;
        renderHistory();
        if (state.activeJobId) renderJob(state.jobs.find((j) => j.id === state.activeJobId));
      });
    });
  }

  // ==================== WORKSPACE 2: PODCAST & DOCUMENT IMPORTER ====================
  function parseLocalDialogueSpeakers(text) {
    const speakerRe = /^\s*(?:\[([^\]]{1,24})\]|【([^】]{1,24})】|([A-Za-z\u4e00-\u9fff0-9_\-]{1,18}))\s*[:：]\s*(.+)$/;
    const speakers = [];
    let turns = 0;
    let lastSpk = "主持人";
    for (const raw of String(text || "").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) continue;
      const m = line.match(speakerRe);
      let spk = lastSpk;
      if (m) {
        spk = (m[1] || m[2] || m[3] || "").trim();
        lastSpk = spk;
      }
      if (!speakers.includes(spk)) speakers.push(spk);
      turns += 1;
    }
    return { turns, speakers };
  }

  function renderPodcastSpeakersUI() {
    const { turns, speakers } = parseLocalDialogueSpeakers($("podScriptInput").value);
    $("podTurnStats").textContent = `${turns} 轮对话 · ${speakers.length} 位角色`;
    const grid = $("podSpeakerVoicesGrid");
    if (!speakers.length) {
      grid.innerHTML = `<div class="empty-hint">在上方输入 <code>[主持人]: 台词</code> 格式即可自动识别角色。</div>`;
      return;
    }
    grid.innerHTML = speakers
      .map((spk, idx) => {
        if (!state.podSpeakerVoices[spk] && state.voices.length) {
          state.podSpeakerVoices[spk] = state.voices[idx % state.voices.length].id;
        }
        const curVid = state.podSpeakerVoices[spk] || (state.voices[0]?.id || "");
        const opts = state.voices.length
          ? state.voices
              .map((v) => `<option value="${escapeHtml(v.id)}" ${v.id === curVid ? "selected" : ""}>${escapeHtml(v.name)}</option>`)
              .join("")
          : `<option value="">请先在左侧上传声纹</option>`;
        return `
          <div class="speaker-assign-item">
            <span class="speaker-tag">${escapeHtml(spk)}</span>
            <select class="field-input" data-pod-spk="${escapeHtml(spk)}">${opts}</select>
          </div>
        `;
      })
      .join("");
    grid.querySelectorAll("[data-pod-spk]").forEach((sel) => {
      sel.addEventListener("change", () => {
        state.podSpeakerVoices[sel.dataset.podSpk] = sel.value;
      });
    });
  }

  function renderPodcastLangChips() {
    const engId = $("podEngineSelect").value || "qwen3";
    const langs = getActiveLanguages(engId);
    const allowed = new Set(langs.map((l) => l.code));
    state.podSelectedLangs.forEach((c) => { if (!allowed.has(c)) state.podSelectedLangs.delete(c); });
    if (state.podSelectedLangs.size === 0) state.podSelectedLangs.add("zh");

    const box = $("podLangChips");
    box.innerHTML = langs
      .map((l) => `<button type="button" class="preset-chip ${state.podSelectedLangs.has(l.code) ? "active" : ""}" data-pod-lang="${escapeHtml(l.code)}">${escapeHtml(l.region_tag)} ${escapeHtml(l.label_zh)}</button>`)
      .join("");
    box.querySelectorAll("[data-pod-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const c = btn.dataset.podLang;
        if (state.podSelectedLangs.has(c)) {
          if (state.podSelectedLangs.size <= 1) return;
          state.podSelectedLangs.delete(c);
        } else {
          state.podSelectedLangs.add(c);
        }
        renderPodcastLangChips();
      });
    });
  }

  function renderLoadedDocument(doc) {
    state.loadedDoc = doc;
    $("docLoadedBox").hidden = false;
    $("docLoadedTitle").textContent = doc.title || "已导入文档";
    $("docLoadedStats").textContent = `${(doc.source_type || "doc").toUpperCase()} · ${doc.chapters.length} 章节 · ${doc.total_chars} 字`;
    const sel = $("docChapterSelect");
    const opts = [
      `<option value="all">整篇全文（${doc.total_chars} 字）</option>`,
      ...doc.chapters.map((c, idx) => `<option value="${idx}">第 ${idx + 1} 章：${escapeHtml(c.title)}（${c.char_count} 字）</option>`),
    ];
    sel.innerHTML = opts.join("");
    sel.value = doc.chapters.length === 1 ? "0" : "all";
    $("docChapterPreview").value = doc.chapters.length === 1 ? doc.chapters[0].text : doc.full_text;
  }

  async function importDocumentFile(file) {
    const st = $("docImportStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = `正在解析文档「${file.name}」并自动提取章节…`;
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const data = await apiFetch("/api/documents/import-file", { method: "POST", body: fd });
      $("docFileInput").value = "";
      renderLoadedDocument(data.document);
      st.textContent = `导入成功：共提取 ${data.document.chapters.length} 章节，总计 ${data.document.total_chars} 字。`;
      showToast(`已导入《${data.document.title}》`, "teal");
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  async function importDocumentUrl() {
    const url = $("docUrlInput").value.trim();
    if (!url) { showToast("请输入网页文章链接", "error"); return; }
    const st = $("docImportStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = "正在抓取网页正文并自动清洗排版…";
    try {
      const data = await apiFetch("/api/documents/import-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      renderLoadedDocument(data.document);
      st.textContent = `抓取完成：《${data.document.title}》（${data.document.total_chars} 字）`;
      showToast("网页文章提取成功！", "teal");
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  async function adaptDocToPodcastScript() {
    const text = $("docChapterPreview").value.trim();
    if (text.length < 20) { showToast("请先导入至少 20 字的文章正文", "error"); return; }
    const btn = $("btnAdaptToPodcast");
    btn.disabled = true;
    try {
      const data = await apiFetch("/api/podcasts/script-from-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_text: text,
          host_a: $("podHostA").value.trim() || "主持人",
          host_b: $("podHostB").value.trim() || "嘉宾",
          target_lang: [...state.podSelectedLangs][0] || "zh",
        }),
      });
      $("podScriptInput").value = data.script_text;
      if (state.loadedDoc?.title) $("podTitleInput").value = `${state.loadedDoc.title.slice(0, 32)} · 双人播客`;
      renderPodcastSpeakersUI();
      showToast(`剧本生成完成（${data.turns.length} 轮对话 · ${data.provider}）`, "teal");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  function upsertPodcastLocal(pod) {
    const idx = state.podcasts.findIndex((p) => p.id === pod.id);
    if (idx >= 0) state.podcasts[idx] = pod;
    else state.podcasts.unshift(pod);
  }

  async function loadPodcasts() {
    const data = await apiFetch("/api/podcasts?limit=20");
    state.podcasts = data.podcasts || [];
    if (!state.activePodcastId && state.podcasts.length) state.activePodcastId = state.podcasts[0].id;
    renderPodcastHistory();
    const cur = state.podcasts.find((p) => p.id === state.activePodcastId);
    if (cur) {
      renderActivePodcast(cur);
      if (["queued", "loading_model", "running"].includes(cur.status)) pollPodcastJob(cur.id);
    }
  }

  async function startPodcastGeneration() {
    const scriptText = $("podScriptInput").value.trim();
    if (!scriptText) { showToast("请先输入多角色对话剧本", "error"); return; }
    if (!state.voices.length) { showToast("声纹库为空，请先在左侧上传参考录音", "error"); return; }
    const st = $("podGenStatus");
    st.hidden = false;
    st.className = "inline-status compact-inline";
    st.textContent = "正在翻译多角色剧本并启动双人播客合成…";
    $("btnGeneratePodcast").disabled = true;
    try {
      const data = await apiFetch("/api/podcasts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: $("podTitleInput").value.trim() || "AI 多角色播客",
          script_text: scriptText,
          speaker_voices: state.podSpeakerVoices,
          target_langs: [...state.podSelectedLangs],
          engine_id: $("podEngineSelect").value,
          mode: state.mode,
          mastering_preset: $("podMasteringSelect").value,
          add_ambient_bgm: $("chkPodAmbientBgm").checked,
        }),
      });
      upsertPodcastLocal(data.podcast);
      state.activePodcastId = data.podcast.id;
      renderPodcastHistory();
      renderActivePodcast(data.podcast);
      pollPodcastJob(data.podcast.id);
      st.textContent = `播客任务已启动（${data.podcast.speakers.join(" & ")} · ${data.podcast.tracks.length} 语种）`;
      showToast("双人播客任务已启动", "teal");
    } catch (err) {
      st.className = "inline-status compact-inline error";
      st.textContent = err.message;
      showToast(err.message, "error");
    } finally {
      $("btnGeneratePodcast").disabled = false;
    }
  }

  function pollPodcastJob(podcastId) {
    if (state.podcastPollTimer) clearInterval(state.podcastPollTimer);
    state.podcastPollTimer = setInterval(async () => {
      try {
        const data = await apiFetch(`/api/podcasts/${podcastId}`);
        upsertPodcastLocal(data.podcast);
        if (state.activePodcastId === podcastId) renderActivePodcast(data.podcast);
        renderPodcastHistory();
        if (["completed", "partial", "failed"].includes(data.podcast.status)) {
          clearInterval(state.podcastPollTimer);
          state.podcastPollTimer = null;
          if (data.podcast.status === "completed") showToast("多语种双人播客合成完成！", "teal");
        }
      } catch {
        clearInterval(state.podcastPollTimer);
        state.podcastPollTimer = null;
      }
    }, 1500);
  }

  function renderActivePodcast(pod) {
    const box = $("podcastActiveTracks");
    if (!pod) return;
    const anyDone = (pod.tracks || []).some((t) => t.status === "completed");
    $("btnPodDownloadZip").hidden = !anyDone;
    if (anyDone) $("btnPodDownloadZip").href = pod.zip_url;

    box.innerHTML = (pod.tracks || [])
      .map((t) => {
        if (t.status === "completed") {
          const mastTxt = MASTERING_LABELS[pod.mastering_preset] || pod.mastering_preset;
          return `
            <article class="track-card">
              <div class="track-top">
                <div class="track-lang-meta">
                  <span class="track-code">${escapeHtml(t.region_tag || t.lang_code.toUpperCase())}</span>
                  <span class="track-title">${escapeHtml(pod.title)} · ${escapeHtml(t.lang_label)}版</span>
                </div>
                <span class="count-mono">${(pod.speakers || []).join(" / ")}</span>
              </div>
              <div class="wave-player">
                <button type="button" class="play-btn teal" data-play-pod="${escapeHtml(t.lang_code)}" aria-label="播放播客">
                  <svg class="icon-play" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg>
                  <svg class="icon-pause" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                </button>
                <canvas class="wave-canvas" data-wave-pod="${escapeHtml(t.lang_code)}" width="120" height="40"></canvas>
                <span class="timecode" data-time-pod="${escapeHtml(t.lang_code)}">00:00 / ${formatSec(t.duration_sec)}</span>
              </div>
              <div class="track-footer">
                <div class="track-metrics">
                  <span class="tele-chip">时长 ${t.duration_sec}s</span>
                  <span class="tele-chip">耗时 ${t.elapsed_sec}s</span>
                  <span class="tele-chip teal">${escapeHtml(mastTxt)}</span>
                  ${pod.add_ambient_bgm ? `<span class="tele-chip teal">氛围 BGM</span>` : ""}
                  ${t.watermarked ? `<span class="tele-chip teal">隐形水印</span>` : ""}
                </div>
                <div class="track-dl-group">
                  <a class="btn-teal btn-sm" href="${escapeHtml(t.mp3_url)}" download="${escapeHtml(pod.id)}_${escapeHtml(t.lang_code)}.mp3">下载 MP3</a>
                  <a class="btn-outline btn-sm" href="${escapeHtml(t.wav_url)}" download="${escapeHtml(pod.id)}_${escapeHtml(t.lang_code)}.wav">无损 WAV</a>
                </div>
              </div>
            </article>
          `;
        }
        if (t.status === "failed") {
          return `<article class="track-card failed"><div class="track-pending-bar">${escapeHtml(t.lang_label)} 失败：${escapeHtml(t.error || "未知错误")}</div></article>`;
        }
        return `<article class="track-card generating"><div class="track-pending-bar">正在逐轮合成 ${escapeHtml(t.lang_label)} 播客（第 ${t.done_turns || 0} / ${t.total_turns || 0} 轮对话）…</div></article>`;
      })
      .join("");

    (pod.tracks || []).forEach((t) => {
      if (t.status === "completed") {
        const canvas = box.querySelector(`[data-wave-pod="${t.lang_code}"]`);
        const btn = box.querySelector(`[data-play-pod="${t.lang_code}"]`);
        const tc = box.querySelector(`[data-time-pod="${t.lang_code}"]`);
        if (canvas && btn) {
          bindWavePlayer({
            key: `pod_${pod.id}_${t.lang_code}_${t.elapsed_sec || ""}`,
            url: `${t.mp3_url}?v=${encodeURIComponent(pod.updated_at || "1")}`,
            peaks: t.waveform_peaks,
            canvas,
            btn,
            timecodeEl: tc,
            durationSec: t.duration_sec,
          });
        }
      }
    });
  }

  function renderPodcastHistory() {
    const list = $("podcastHistoryList");
    $("podcastHistoryCount").textContent = `${state.podcasts.length} 期`;
    if (!state.podcasts.length) { list.innerHTML = `<div class="empty-hint">暂无历史播客</div>`; return; }
    list.innerHTML = state.podcasts
      .map((p) => `
        <div class="history-item ${p.id === state.activePodcastId ? "active" : ""}" data-open-pod="${escapeHtml(p.id)}">
          <div class="history-main">
            <div class="history-title">${escapeHtml(p.title)}</div>
            <div class="history-sub">${(p.speakers || []).join(" & ")} · ${(p.tracks || []).filter((t) => t.status === "completed").length}/${(p.tracks || []).length} 语种完成</div>
          </div>
          <button type="button" class="btn-del-mini" data-del-pod="${escapeHtml(p.id)}">删除</button>
        </div>
      `)
      .join("");
    list.querySelectorAll("[data-open-pod]").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest("[data-del-pod]")) return;
        state.activePodcastId = el.dataset.openPod;
        renderPodcastHistory();
        renderActivePodcast(state.podcasts.find((p) => p.id === state.activePodcastId));
      });
    });
    list.querySelectorAll("[data-del-pod]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/podcasts/${btn.dataset.delPod}`, { method: "DELETE" });
        state.podcasts = state.podcasts.filter((p) => p.id !== btn.dataset.delPod);
        if (state.activePodcastId === btn.dataset.delPod) state.activePodcastId = state.podcasts[0]?.id || null;
        renderPodcastHistory();
        if (state.activePodcastId) renderActivePodcast(state.podcasts.find((p) => p.id === state.activePodcastId));
      });
    });
  }

  // ==================== WORKSPACE 3: VIDEO DUBBING STUDIO V3 ====================
  function renderVideoLangChips() {
    const engId = $("vidEngineSelect").value || "qwen3";
    const langs = getActiveLanguages(engId);
    const allowed = new Set(langs.map((l) => l.code));
    state.videoSelectedLangs.forEach((c) => { if (!allowed.has(c)) state.videoSelectedLangs.delete(c); });
    if (state.videoSelectedLangs.size === 0) state.videoSelectedLangs.add("en");

    const box = $("vidLangChips");
    box.innerHTML = langs
      .map((l) => `<button type="button" class="preset-chip ${state.videoSelectedLangs.has(l.code) ? "active" : ""}" data-vid-lang="${escapeHtml(l.code)}">${escapeHtml(l.region_tag)} ${escapeHtml(l.label_zh)}</button>`)
      .join("");
    box.querySelectorAll("[data-vid-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const c = btn.dataset.vidLang;
        if (state.videoSelectedLangs.has(c)) {
          if (state.videoSelectedLangs.size <= 1) return;
          state.videoSelectedLangs.delete(c);
        } else {
          state.videoSelectedLangs.add(c);
        }
        renderVideoLangChips();
        renderActiveVideo();
      });
    });
  }

  async function loadVideos() {
    const data = await apiFetch("/api/videos");
    state.videos = data.videos || [];
    if (!state.activeVideoId && state.videos.length) state.activeVideoId = state.videos[0].id;
    renderVideoHistory();
    renderActiveVideo();
  }

  function upsertVideoLocal(v) {
    const idx = state.videos.findIndex((x) => x.id === v.id);
    if (idx >= 0) state.videos[idx] = v;
    else state.videos.unshift(v);
  }

  async function uploadAndAnalyzeVideo(file) {
    const st = $("videoUploadStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = "正在上传视频、分离人声与 BGM、运行 Whisper 断句并自动识别多说话人，请稍候…";
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const data = await apiFetch("/api/videos/analyze", { method: "POST", body: fd });
      upsertVideoLocal(data.video);
      state.activeVideoId = data.video.id;
      state.videoSpeakerVoices = {};
      $("videoFileInput").value = "";
      const spkCount = (data.video.speakers || []).length || 1;
      st.textContent = `解析完成：${data.video.segments.length} 句台词 · 检测到 ${spkCount} 位说话人 · ${data.video.bgm_method}`;
      renderVideoHistory();
      renderActiveVideo();
      showToast("视频解析与说话人声纹提取完成！", "teal");
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  async function importVideoFromUrl() {
    const url = $("videoUrlInput").value.trim();
    if (!url) { showToast("请粘贴视频链接", "error"); return; }
    const st = $("videoUploadStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = "正在通过 yt-dlp 拉取在线视频并自动分离人声与识别时间轴，请耐心等待…";
    $("btnImportVideoUrl").disabled = true;
    try {
      const data = await apiFetch("/api/videos/import-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      upsertVideoLocal(data.video);
      state.activeVideoId = data.video.id;
      state.videoSpeakerVoices = {};
      $("videoUrlInput").value = "";
      const spkCount = (data.video.speakers || []).length || 1;
      st.textContent = `在线视频解析完成：《${data.video.title}》· ${data.video.segments.length} 句 · ${spkCount} 位说话人`;
      renderVideoHistory();
      renderActiveVideo();
      showToast("在线视频导入解析成功！", "teal");
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    } finally {
      $("btnImportVideoUrl").disabled = false;
    }
  }

  function renderActiveVideo() {
    const video = state.videos.find((v) => v.id === state.activeVideoId);
    const stage = $("videoProjectStage");
    if (!video) { stage.hidden = true; return; }
    stage.hidden = false;
    const spkList = video.speakers || [];
    $("videoTitle").textContent = video.title || video.original_filename;
    $("videoMetaStats").textContent = `${video.width}×${video.height} · ${video.duration_sec}s · ${video.segments.length} 句 · ${spkList.length || 1} 人 · ${video.bgm_method}`;
    if ($("sourceVideoPlayer").getAttribute("data-src") !== video.source_video_url) {
      $("sourceVideoPlayer").setAttribute("data-src", video.source_video_url);
      $("sourceVideoPlayer").src = video.source_video_url;
    }

    // Multi-speaker voice mapping UI
    const spkBox = $("vidSpeakersBox");
    if (spkList.length >= 1) {
      spkBox.hidden = false;
      $("vidSpeakersLabel").textContent = spkList.length > 1
        ? `声学聚类检测到 ${spkList.length} 位说话人（已各提取原声参考，可按角色单独指定声纹）`
        : "说话人声纹映射（默认自动克隆视频原说话人声音）";
      $("vidSpeakersGrid").innerHTML = spkList
        .map((sp) => {
          const cur = state.videoSpeakerVoices[sp.speaker] || "auto_video";
          const opts =
            `<option value="auto_video" ${cur === "auto_video" ? "selected" : ""}>自动克隆 ${escapeHtml(sp.speaker)} 原声 (${sp.segment_count}句)</option>` +
            state.voices.map((v) => `<option value="${escapeHtml(v.id)}" ${cur === v.id ? "selected" : ""}>声纹库：${escapeHtml(v.name)}</option>`).join("");
          return `
            <div class="speaker-assign-item">
              <span class="speaker-tag">${escapeHtml(sp.speaker)}</span>
              <select class="field-input" data-vid-spk="${escapeHtml(sp.speaker)}">${opts}</select>
            </div>
          `;
        })
        .join("");
      $("vidSpeakersGrid").querySelectorAll("[data-vid-spk]").forEach((sel) => {
        sel.addEventListener("change", () => {
          state.videoSpeakerVoices[sel.dataset.vidSpk] = sel.value;
        });
      });
    } else {
      spkBox.hidden = true;
    }

    const chosen = [...state.videoSelectedLangs];
    if (!chosen.includes(state.videoPreviewLang)) state.videoPreviewLang = chosen[0] || "en";
    const sel = $("vidPreviewLangSelect");
    sel.innerHTML = chosen
      .map((code) => {
        const l = state.languages.find((x) => x.code === code);
        return `<option value="${escapeHtml(code)}" ${code === state.videoPreviewLang ? "selected" : ""}>校对译文：${escapeHtml(l ? l.label_zh : code)}</option>`;
      })
      .join("");

    const trList = (video.translations || {})[state.videoPreviewLang] || [];
    $("vidSegmentsList").innerHTML = (video.segments || [])
      .map((s, idx) => `
        <div class="seg-row">
          <div class="seg-time">
            <span>${formatSec(s.start)} → ${formatSec(s.end)} (${s.duration_sec}s)</span>
            <span class="speaker-tag">${escapeHtml(s.speaker || "SPEAKER_1")}</span>
          </div>
          <div class="seg-orig">${escapeHtml(s.text)}</div>
          <textarea class="field-textarea compact" rows="2" data-vid-seg-idx="${idx}" placeholder="点击「一键等时长翻译字幕」自动生成">${escapeHtml(trList[idx] || "")}</textarea>
        </div>
      `)
      .join("");

    $("vidSegmentsList").querySelectorAll("[data-vid-seg-idx]").forEach((ta) => {
      ta.addEventListener("input", () => {
        const i = Number(ta.dataset.vidSegIdx);
        if (!video.translations) video.translations = {};
        if (!video.translations[state.videoPreviewLang]) video.translations[state.videoPreviewLang] = video.segments.map(() => "");
        video.translations[state.videoPreviewLang][i] = ta.value;
      });
    });

    const outPanel = $("vidOutputsPanel");
    const outputs = video.outputs || [];
    if (!outputs.length) {
      outPanel.hidden = true;
    } else {
      outPanel.hidden = false;
      $("btnVideoDownloadZip").href = video.zip_url;
      $("vidOutputsGrid").innerHTML = outputs
        .map((o) => {
          if (o.status === "completed") {
            const lsBadge = o.lipsync_applied ? `<span class="tele-chip teal">Lip-Sync 已同步</span>` : "";
            return `
              <div class="vid-out-card">
                <div class="track-top">
                  <div class="track-lang-meta">
                    <span class="track-code">${escapeHtml(o.region_tag)}</span>
                    <span class="track-title">${escapeHtml(o.lang_label)} 译配成片</span>
                    ${lsBadge}
                  </div>
                  <span class="count-mono">${o.mp4_size_mb} MB · 耗时 ${o.elapsed_sec}s</span>
                </div>
                <video controls playsinline src="${escapeHtml(o.mp4_url)}?v=${encodeURIComponent(video.updated_at || "1")}"></video>
                <div class="track-dl-group">
                  <a class="btn-teal btn-sm" href="${escapeHtml(o.mp4_url)}" download="${escapeHtml(video.title)}_${escapeHtml(o.lang_code)}.mp4">下载 MP4</a>
                  <a class="btn-outline btn-sm" href="${escapeHtml(o.bilingual_srt_url)}" download="bilingual_${escapeHtml(o.lang_code)}.srt">双语 SRT</a>
                  ${o.ass_url ? `<a class="btn-outline btn-sm" href="${escapeHtml(o.ass_url)}" download="sub_${escapeHtml(o.lang_code)}.ass">ASS 样式字幕</a>` : ""}
                  <a class="btn-outline btn-sm" href="${escapeHtml(o.mp3_url)}" download="dub_${escapeHtml(o.lang_code)}.mp3">纯音频 MP3</a>
                </div>
              </div>
            `;
          }
          if (o.status === "failed") {
            return `<div class="vid-out-card"><div class="track-pending-bar">${escapeHtml(o.lang_label)} 失败：${escapeHtml(o.error || "未知错误")}</div></div>`;
          }
          return `<div class="vid-out-card"><div class="track-pending-bar">正在合成 ${escapeHtml(o.lang_label)} 译配视频（第 ${o.done_segments || 0}/${o.total_segments || 0} 句）…</div></div>`;
        })
        .join("");
    }
  }

  function renderVideoHistory() {
    const list = $("videoHistoryList");
    $("videoHistoryCount").textContent = `${state.videos.length} 条记录`;
    if (!state.videos.length) { list.innerHTML = `<div class="empty-hint">暂无视频译配记录</div>`; return; }
    list.innerHTML = state.videos
      .map((v) => `
        <div class="history-item ${v.id === state.activeVideoId ? "active" : ""}" data-open-video="${escapeHtml(v.id)}">
          <div class="history-main">
            <div class="history-title">${escapeHtml(v.title)} (${v.duration_sec}s)</div>
            <div class="history-sub">${v.segments.length} 句 · ${(v.speakers || []).length || 1} 人 · ${(v.outputs || []).filter((o) => o.status === "completed").length} 语种成片</div>
          </div>
          <button type="button" class="btn-del-mini" data-del-video="${escapeHtml(v.id)}">删除</button>
        </div>
      `)
      .join("");
    list.querySelectorAll("[data-open-video]").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest("[data-del-video]")) return;
        state.activeVideoId = el.dataset.openVideo;
        state.videoSpeakerVoices = {};
        renderVideoHistory();
        renderActiveVideo();
      });
    });
    list.querySelectorAll("[data-del-video]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/videos/${btn.dataset.delVideo}`, { method: "DELETE" });
        state.videos = state.videos.filter((v) => v.id !== btn.dataset.delVideo);
        if (state.activeVideoId === btn.dataset.delVideo) state.activeVideoId = state.videos[0]?.id || null;
        renderVideoHistory();
        renderActiveVideo();
      });
    });
  }

  async function translateActiveVideo(silent = false) {
    const video = state.videos.find((v) => v.id === state.activeVideoId);
    if (!video) return false;
    const langs = [...state.videoSelectedLangs];
    const st = $("videoDubStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = `正在为 ${video.segments.length} 条时间轴字幕执行多语种等时长翻译（自动应用术语词典）…`;
    try {
      const data = await apiFetch(`/api/videos/${video.id}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_langs: langs }),
      });
      upsertVideoLocal(data.video);
      renderActiveVideo();
      st.textContent = `翻译完成 · ${data.provider}`;
      if (!silent) showToast("视频多语种字幕翻译完成", "teal");
      return true;
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
      return false;
    }
  }

  async function startVideoDubbing() {
    const video = state.videos.find((v) => v.id === state.activeVideoId);
    if (!video) return;
    const langs = [...state.videoSelectedLangs];
    const missing = langs.filter((c) => !video.translations?.[c]?.length);
    if (missing.length > 0 && !(await translateActiveVideo(true))) return;

    const st = $("videoDubStatus");
    st.hidden = false;
    st.className = "inline-status";
    st.textContent = "已提交多语种视频译配任务，GPU 正在逐句克隆、母带处理并混音回填画面…";
    try {
      const data = await apiFetch(`/api/videos/${video.id}/dub`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_langs: langs,
          translations: video.translations || {},
          engine_id: $("vidEngineSelect").value,
          voice_id: $("vidVoiceSelect").value,
          speaker_voices: state.videoSpeakerVoices,
          mode: state.mode,
          exaggeration: state.exaggeration,
          cfg_weight: state.cfgWeight,
          instruct: $("cosyInstructInput").value.trim(),
          mastering_preset: $("vidMasteringSelect").value,
          burn_subtitles: $("chkBurnSubtitles").checked,
          subtitle_style: $("vidSubtitleStyleSelect").value,
          enable_lipsync: $("chkEnableLipsync").checked && !$("chkEnableLipsync").disabled,
          bgm_volume: parseFloat($("vidBgmSlider").value),
        }),
      });
      upsertVideoLocal(data.video);
      renderActiveVideo();
      pollVideoJob(video.id);
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  function pollVideoJob(videoId) {
    if (state.videoPollTimer) clearInterval(state.videoPollTimer);
    state.videoPollTimer = setInterval(async () => {
      try {
        const data = await apiFetch(`/api/videos/${videoId}`);
        upsertVideoLocal(data.video);
        if (state.activeVideoId === videoId) renderActiveVideo();
        renderVideoHistory();
        if (["completed", "partial", "failed"].includes(data.video.dub_status)) {
          clearInterval(state.videoPollTimer);
          state.videoPollTimer = null;
          const st = $("videoDubStatus");
          st.hidden = false;
          st.className = data.video.dub_status === "failed" ? "inline-status error" : "inline-status";
          st.textContent = data.video.dub_status === "completed" ? "全部多语种译配视频生成完成！" : `视频译配状态：${data.video.dub_status}`;
          if (data.video.dub_status === "completed") showToast("多语种译配视频已就绪！", "teal");
        }
      } catch {
        clearInterval(state.videoPollTimer);
        state.videoPollTimer = null;
      }
    }, 1500);
  }

  // ==================== WORKSPACE 4: GLOSSARY, WATERMARK & DEVELOPER API ====================
  async function loadGlossary() {
    const data = await apiFetch("/api/glossary");
    state.glossary = data.terms || [];
    renderGlossary();
  }

  function renderGlossary() {
    $("glossaryCountBadge").textContent = `${state.glossary.length} 条术语`;
    const list = $("glossaryList");
    if (!state.glossary.length) {
      list.innerHTML = `<div class="empty-hint">暂无术语词条，可在上方添加需要锁定的品牌名或专业缩写。</div>`;
      return;
    }
    list.innerHTML = state.glossary
      .map((item) => {
        const chips = [];
        if (item.keep_original) chips.push(`<span class="tele-chip teal">锁定不翻译</span>`);
        const ovs = Object.entries(item.lang_overrides || {});
        if (ovs.length) chips.push(`<span class="tele-chip">译名：${escapeHtml(ovs.map(([k, v]) => `${k}=${v}`).join(", "))}</span>`);
        const prons = Object.entries(item.tts_pronunciation || {});
        if (prons.length) chips.push(`<span class="tele-chip teal">读音：${escapeHtml(prons.map(([k, v]) => `${k}=${v}`).join(", "))}</span>`);
        return `
          <div class="glossary-item">
            <div class="glossary-main">
              <div class="glossary-top">
                <span class="glossary-term">${escapeHtml(item.term)}</span>
                ${chips.join("")}
              </div>
              ${item.notes ? `<div class="glossary-notes">${escapeHtml(item.notes)}</div>` : ""}
            </div>
            <button type="button" class="btn-del-mini" data-del-gloss="${escapeHtml(item.id)}">删除</button>
          </div>
        `;
      })
      .join("");
    list.querySelectorAll("[data-del-gloss]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          const data = await apiFetch(`/api/glossary/${btn.dataset.delGloss}`, { method: "DELETE" });
          state.glossary = data.terms || [];
          renderGlossary();
          showToast("已删除词条");
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    });
  }

  async function saveGlossaryTerm() {
    const term = $("glossTerm").value.trim();
    if (!term) { showToast("请输入术语原文", "error"); return; }
    try {
      const data = await apiFetch("/api/glossary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          term,
          keep_original: $("glossKeepOrig").checked,
          lang_overrides: parseKvText($("glossLangOverrides").value),
          tts_pronunciation: parseKvText($("glossTtsPron").value),
          notes: $("glossNotes").value.trim(),
        }),
      });
      state.glossary = data.terms || [];
      $("glossTerm").value = "";
      $("glossLangOverrides").value = "";
      $("glossTtsPron").value = "";
      $("glossNotes").value = "";
      renderGlossary();
      showToast(`已保存术语「${term}」`, "teal");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  async function verifyWatermarkFile(file) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const res = await apiFetch("/api/watermark/verify", { method: "POST", body: fd });
      const card = $("wmResultCard");
      card.hidden = false;
      card.className = `wm-result-card ${res.detected ? "detected" : ""}`;
      $("wmVerdictBadge").textContent = res.detected ? "检测到 AI 水印" : "未检测到水印";
      $("wmFileMeta").textContent = res.filename || file.name;
      $("wmVerdictText").textContent = res.verdict;
      $("wmMetricsRow").innerHTML = [
        `置信度 ${res.confidence}%`,
        `相关 Z-Score ${res.z_score}`,
        `有效检测帧 ${res.active_frames}`,
      ].map((t) => `<span class="tele-chip">${escapeHtml(t)}</span>`).join("");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  function renderDeveloperSnippets() {
    const origin = window.location.origin;
    const sampleVoice = state.selectedVoiceId || state.voices[0]?.id || "demo_official";
    $("snippetOpenAI").textContent =
`# Base URL: ${origin}/v1
curl -X POST ${origin}/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"model":"voxpoly-qwen3-1.7b","voice":"${sampleVoice}","input":"你好，这是通过 OpenAI 兼容接口实时合成的本人声音。","response_format":"mp3"}' \\
  --output speech.mp3`;

    $("snippetMcp").textContent = JSON.stringify(
      {
        mcpServers: {
          voxpoly: {
            command: "python",
            args: ["-m", "backend.mcp_server"],
            env: { VOXPOLY_API_URL: origin },
          },
        },
      },
      null,
      2
    );

    $("snippetCli").textContent =
`# 命令行一键多语种合成
python -m backend.cli speak --voice ${sampleVoice} --text "Hello VoxPoly V3" --langs zh,en,ja --mastering podcast --out ./out

# 批量把整个文件夹的 PDF / EPUB / Markdown 转为有声书
python -m backend.cli batch ./books --voice ${sampleVoice} --langs zh,en --mastering audiobook --out ./audiobooks

# Docker GPU 一键启动
docker compose up -d`;
  }

  // ---------- Settings ----------
  function fillSettingsModal(s) {
    if (!s) return;
    $("setProvider").value = s.translation_provider || "auto";
    $("setApiBase").value = s.llm_api_base || "";
    $("setModel").value = s.llm_model || "deepseek-chat";
    $("setApiKey").value = "";
    $("setApiKey").placeholder = s.has_llm_api_key ? `已保存 (${s.llm_api_key_masked})，留空保持不变` : "sk-...";
    $("setQwenVariant").value = s.qwen_variant || "1.7b";
    $("setMasteringPreset").value = s.mastering_preset || "raw";
    $("setIdleMinutes").value = String(s.idle_unload_minutes ?? 0);
    $("setEnableBatch").checked = s.enable_batch_accel !== false;
    $("setEnableWatermark").checked = s.enable_watermark !== false;
  }

  async function saveSettings() {
    try {
      const updated = await apiFetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          translation_provider: $("setProvider").value,
          llm_api_base: $("setApiBase").value.trim(),
          llm_model: $("setModel").value.trim(),
          llm_api_key: $("setApiKey").value.trim(),
          qwen_variant: $("setQwenVariant").value,
          mastering_preset: $("setMasteringPreset").value,
          idle_unload_minutes: Number($("setIdleMinutes").value || 0),
          enable_batch_accel: $("setEnableBatch").checked,
          enable_watermark: $("setEnableWatermark").checked,
        }),
      });
      fillSettingsModal(updated);
      $("audioMasteringSelect").value = updated.mastering_preset || "raw";
      await loadSystemStatus(false);
      $("settingsModal").hidden = true;
      showToast("设置已保存", "teal");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  // ---------- Event Wiring ----------
  function initEvents() {
    document.querySelectorAll("[data-workspace-tab]").forEach((btn) => {
      btn.addEventListener("click", () => switchWorkspaceTab(btn.dataset.workspaceTab));
    });

    document.querySelectorAll("[data-capture-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.captureTab;
        document.querySelectorAll("[data-capture-tab]").forEach((b) => {
          b.classList.toggle("active", b === btn);
          b.setAttribute("aria-selected", String(b === btn));
        });
        $("paneUpload").hidden = tab !== "upload";
        $("paneRecord").hidden = tab !== "record";
        if (tab === "record") drawIdleMicMeter();
      });
    });

    document.querySelectorAll("[data-select-engine]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedEngineId = btn.dataset.selectEngine;
        syncEngineUI();
      });
    });

    $("cosyInstructInput").addEventListener("input", (e) => { state.cosyInstruct = e.target.value; renderCosyPresets(); });
    $("rngExag").addEventListener("input", (e) => { state.exaggeration = parseFloat(e.target.value); $("valExag").textContent = state.exaggeration.toFixed(2); });
    $("rngCfg").addEventListener("input", (e) => { state.cfgWeight = parseFloat(e.target.value); $("valCfg").textContent = state.cfgWeight.toFixed(2); });

    // Voice upload & mic
    const dz = $("dropzone");
    const fileInput = $("audioFileInput");
    dz.addEventListener("click", () => fileInput.click());
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) uploadAudioFile(e.dataTransfer.files[0]); });
    fileInput.addEventListener("change", () => { if (fileInput.files?.[0]) uploadAudioFile(fileInput.files[0]); });

    $("btnRecToggle").addEventListener("click", toggleMicRecording);
    $("btnSaveRecording").addEventListener("click", async () => {
      if (!state.micBlob) return;
      const file = new File([state.micBlob], "mic_recording.webm", { type: state.micBlob.type || "audio/webm" });
      await uploadAudioFile(file, $("recVoiceName").value.trim() || "麦克风录制声纹");
      $("recVoiceName").value = "";
      $("recSaveRow").hidden = true;
      $("recTimer").textContent = "00:00.0";
      drawIdleMicMeter();
    });

    $("btnSaveRefText").addEventListener("click", async () => {
      if (!state.selectedVoiceId) return;
      await apiFetch(`/api/voices/${state.selectedVoiceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref_text: $("activeRefText").value.trim() }),
      });
      await loadVoices();
      showToast("参考文本已保存", "teal");
    });

    $("btnReTranscribe").addEventListener("click", async () => {
      if (!state.selectedVoiceId) return;
      const btn = $("btnReTranscribe");
      btn.disabled = true; btn.textContent = "识别中…";
      try {
        const data = await apiFetch(`/api/voices/${state.selectedVoiceId}/transcribe`, { method: "POST" });
        await loadVoices();
        showToast(`识别完成（${data.asr.model_size} · ${data.asr.elapsed_sec}s）`, "teal");
      } catch (err) { showToast(err.message, "error"); }
      finally { btn.disabled = false; btn.textContent = "AI 自动识别"; }
    });

    // Audio workspace events
    $("sourceScript").addEventListener("input", () => {
      updateSourceCount();
      state.activePresetId = null;
      state.sourceDirty = true;
      $("translateStatus").hidden = false;
      $("translateStatus").className = "inline-status warn";
      $("translateStatus").textContent = "原句已修改：可点击「一键翻译」预览译文，或直接点击生成自动翻译并合成。";
      renderPresets();
    });
    $("btnSelectTop4").addEventListener("click", () => { state.selectedLangs = new Set(["zh", "en", "ja", "fr"]); renderLangGrid(); renderPerLangScripts(); });
    $("btnSelectCore10").addEventListener("click", () => { state.selectedLangs = new Set(getActiveLanguages().slice(0, 10).map((l) => l.code)); renderLangGrid(); renderPerLangScripts(); });
    $("btnSelectAll23").addEventListener("click", () => { state.selectedLangs = new Set(getActiveLanguages().map((l) => l.code)); renderLangGrid(); renderPerLangScripts(); });
    $("btnToggleScripts").addEventListener("click", () => { state.scriptsCollapsed = !state.scriptsCollapsed; renderPerLangScripts(); });
    document.querySelectorAll("[data-mode]").forEach((btn) => {
      btn.addEventListener("click", () => { state.mode = btn.dataset.mode; document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b === btn)); });
    });
    $("btnTranslate").addEventListener("click", () => runOneClickTranslate(false));
    $("btnGenerate").addEventListener("click", startMultilingualGeneration);
    $("btnBuildReel").addEventListener("click", async () => {
      if (!state.activeJobId) return;
      const data = await apiFetch(`/api/jobs/${state.activeJobId}/reel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      upsertJobLocal(data.job); renderJob(data.job); showToast("串烧已重新合成", "teal");
    });

    // Document & Podcast events
    const ddz = $("docDropzone");
    const dInput = $("docFileInput");
    ddz.addEventListener("click", () => dInput.click());
    ddz.addEventListener("dragover", (e) => { e.preventDefault(); ddz.classList.add("dragover"); });
    ddz.addEventListener("dragleave", () => ddz.classList.remove("dragover"));
    ddz.addEventListener("drop", (e) => { e.preventDefault(); ddz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) importDocumentFile(e.dataTransfer.files[0]); });
    dInput.addEventListener("change", () => { if (dInput.files?.[0]) importDocumentFile(dInput.files[0]); });
    $("btnImportDocUrl").addEventListener("click", importDocumentUrl);

    $("docChapterSelect").addEventListener("change", () => {
      if (!state.loadedDoc) return;
      const val = $("docChapterSelect").value;
      $("docChapterPreview").value = val === "all" ? state.loadedDoc.full_text : (state.loadedDoc.chapters[Number(val)]?.text || "");
    });

    $("btnSendToAudiobook").addEventListener("click", () => {
      const txt = $("docChapterPreview").value.trim();
      if (!txt) { showToast("请先导入文档正文", "error"); return; }
      $("sourceScript").value = txt.slice(0, 8000);
      $("audioMasteringSelect").value = "audiobook";
      state.activePresetId = null;
      state.sourceDirty = true;
      updateSourceCount();
      renderPresets();
      switchWorkspaceTab("audio");
      showToast("已送入长文有声书台（自动切至 -18 LUFS 有声书母带）", "teal");
    });

    $("btnAdaptToPodcast").addEventListener("click", adaptDocToPodcastScript);
    $("btnLoadPodcastDemo").addEventListener("click", () => {
      $("podScriptInput").value = DEMO_PODCAST_SCRIPT;
      renderPodcastSpeakersUI();
      showToast("已载入双人播客示例剧本", "teal");
    });
    $("podScriptInput").addEventListener("input", renderPodcastSpeakersUI);
    $("podEngineSelect").addEventListener("change", renderPodcastLangChips);
    $("btnGeneratePodcast").addEventListener("click", startPodcastGeneration);

    // Video dubbing events
    const vdz = $("videoDropzone");
    const vInput = $("videoFileInput");
    vdz.addEventListener("click", () => vInput.click());
    vdz.addEventListener("dragover", (e) => { e.preventDefault(); vdz.classList.add("dragover"); });
    vdz.addEventListener("dragleave", () => vdz.classList.remove("dragover"));
    vdz.addEventListener("drop", (e) => { e.preventDefault(); vdz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) uploadAndAnalyzeVideo(e.dataTransfer.files[0]); });
    vInput.addEventListener("change", () => { if (vInput.files?.[0]) uploadAndAnalyzeVideo(vInput.files[0]); });
    $("btnImportVideoUrl").addEventListener("click", importVideoFromUrl);
    $("vidEngineSelect").addEventListener("change", renderVideoLangChips);
    $("vidPreviewLangSelect").addEventListener("change", (e) => { state.videoPreviewLang = e.target.value; renderActiveVideo(); });
    $("vidBgmSlider").addEventListener("input", (e) => { $("valBgmVol").textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`; });
    $("btnVideoTranslate").addEventListener("click", () => translateActiveVideo(false));
    $("btnVideoDub").addEventListener("click", startVideoDubbing);

    // Glossary, Watermark & API copy events
    $("btnSaveGlossary").addEventListener("click", saveGlossaryTerm);
    const wdz = $("wmDropzone");
    const wInput = $("wmFileInput");
    wdz.addEventListener("click", () => wInput.click());
    wdz.addEventListener("dragover", (e) => { e.preventDefault(); wdz.classList.add("dragover"); });
    wdz.addEventListener("dragleave", () => wdz.classList.remove("dragover"));
    wdz.addEventListener("drop", (e) => { e.preventDefault(); wdz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) verifyWatermarkFile(e.dataTransfer.files[0]); });
    wInput.addEventListener("change", () => { if (wInput.files?.[0]) verifyWatermarkFile(wInput.files[0]); });

    document.querySelectorAll("[data-copy-target]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const txt = $(btn.dataset.copyTarget)?.textContent || "";
        try {
          await navigator.clipboard.writeText(txt);
          showToast("代码片段已复制到剪贴板", "teal");
        } catch {
          showToast("复制失败，请手动选中复制", "error");
        }
      });
    });

    // System controls
    $("btnPreload").addEventListener("click", async () => {
      await apiFetch("/api/system/preload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ engine_id: state.selectedEngineId }) });
      showToast("正在后台装载所选引擎至 GPU 显存…", "teal");
      setTimeout(() => loadSystemStatus(false), 600);
    });
    $("btnUnloadVram").addEventListener("click", async () => {
      await apiFetch("/api/system/unload", { method: "POST" });
      await loadSystemStatus(false);
      showToast("已释放全部 GPU 显存", "teal");
    });

    // Modals
    $("btnOpenSettings").addEventListener("click", async () => { try { fillSettingsModal(await apiFetch("/api/settings")); } catch {} $("settingsModal").hidden = false; });
    $("btnCloseSettings").addEventListener("click", () => ($("settingsModal").hidden = true));
    $("btnCancelSettings").addEventListener("click", () => ($("settingsModal").hidden = true));
    $("settingsModal").addEventListener("click", (e) => { if (e.target === $("settingsModal")) $("settingsModal").hidden = true; });
    $("btnSaveSettings").addEventListener("click", saveSettings);

    $("btnSubmitToken").addEventListener("click", async () => {
      state.accessToken = $("accessTokenInput").value.trim();
      localStorage.setItem("voxpoly_access_token", state.accessToken);
      $("tokenModal").hidden = true;
      await bootData();
    });

    window.addEventListener("resize", () => {
      state.audioPlayers.forEach((entry) => {
        const dur = Number.isFinite(entry.audio.duration) && entry.audio.duration > 0 ? entry.audio.duration : entry.durationSec;
        drawWaveform(entry.canvas, entry.peaks, dur > 0 ? Math.min(1, entry.audio.currentTime / dur) : 0);
      });
    });
  }

  async function bootData() {
    await loadSystemStatus(true);
    await Promise.all([loadVoices(), loadJobs(), loadPodcasts(), loadVideos(), loadGlossary()]);
    if (!$("podScriptInput").value.trim()) {
      $("podScriptInput").value = DEMO_PODCAST_SCRIPT;
    }
    renderPodcastSpeakersUI();
  }

  async function boot() {
    initEvents();
    drawIdleMicMeter();
    await bootData();
    state.statusTimer = setInterval(() => loadSystemStatus(false), 8000);
  }

  boot();
})();
