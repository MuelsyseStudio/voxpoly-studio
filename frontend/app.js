(() => {
  "use strict";

  const state = {
    accessToken: localStorage.getItem("voxpoly_access_token") || "",
    workspaceTab: "audio",
    languages: [],
    presets: [],
    cosyPresets: [],
    voices: [],
    jobs: [],
    videos: [],
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
    activeVideoId: null,
    videoSelectedLangs: new Set(["en", "ja"]),
    videoPreviewLang: "en",
    sourceDirty: false,
    scriptsCollapsed: false,
    pollTimer: null,
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
    state.streamAudio.play().catch(() => {
      state.streamPlaying = false;
    });
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
      const selEng = state.enginesStatus[state.selectedEngineId] || eng;

      const anyLoaded =
        Object.values(state.enginesStatus).some((e) => e.state === "ready") || data.asr?.loaded;

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
        const vramTxt = eng.gpu.vram_total_gb > 0 ? ` · ${eng.gpu.vram_used_gb}/${eng.gpu.vram_total_gb} GB` : "";
        $("gpuTelemetry").textContent = `${shortGpu} (${eng.gpu.recommended_precision})${vramTxt}`;
      } else {
        $("gpuTelemetry").textContent = "CPU 模式";
      }

      // Badges for all 3 engines
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

      $("demucsStatusBadge").textContent = data.features?.demucs_available
        ? "人声分离：Demucs htdemucs (无损保留 BGM)"
        : "人声分离：FFmpeg 智能闪避 BGM（Colab 自动启用 Demucs）";

      if (data.asr?.configured_model_size) {
        $("asrOptionLabel").textContent = `上传后自动 Whisper ${data.asr.configured_model_size} 识别参考文本`;
      }

      if (initial) {
        state.languages = data.languages || [];
        state.presets = data.presets || [];
        state.cosyPresets = data.cosyvoice_presets || [];
        state.selectedLangs = new Set(data.default_selected_langs || ["zh", "en", "ja", "fr"]);
        renderPresets();
        renderCosyPresets();
        syncEngineUI();
        renderVideoLangChips();
        applyPreset(state.presets[0]?.id, false);
        fillSettingsModal(data.settings);
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
    state.selectedLangs.forEach((code) => {
      if (!allowed.has(code)) state.selectedLangs.delete(code);
    });
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
  }

  function refreshVideoVoiceDropdown() {
    const sel = $("vidVoiceSelect");
    const prev = sel.value;
    sel.innerHTML =
      `<option value="auto_video">自动克隆视频原说话人声纹（推荐）</option>` +
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

  // ---------- Audio Scripting & Generation ----------
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
    statusEl.textContent = `正在并发翻译为 ${targetLangs.length} 种语言…`;
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
    if (selSt && !selSt.installed) { showToast(`${selSt.display_name} 尚未安装，请在 Colab Cell 2 勾选启用`, "error"); return; }

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
        // Auto-stream first track's finished chunks if user enabled "边生成边试听"
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
    $("jobMetaDesc").textContent = `引擎：${job.engine_name || job.engine_id} · 声纹：${job.voice_name} · ${job.total_count} 语种${job.instruct ? ` · 指令「${job.instruct.slice(0, 18)}」` : ""}`;

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
                ${chunkChip}${wmChip}
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
          const previewBtn =
            t.chunk_urls?.length
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

  // ==================== VIDEO DUBBING STUDIO ====================
  function renderVideoLangChips() {
    const engId = $("vidEngineSelect").value || "qwen3";
    const langs = getActiveLanguages(engId);
    const allowed = new Set(langs.map((l) => l.code));
    state.videoSelectedLangs.forEach((c) => { if (!allowed.has(c)) state.videoSelectedLangs.delete(c); });
    if (state.videoSelectedLangs.size === 0) state.videoSelectedLangs.add("en");

    const box = $("vidLangChips");
    box.innerHTML = langs
      .map((l) => {
        const active = state.videoSelectedLangs.has(l.code) ? "active" : "";
        return `<button type="button" class="preset-chip ${active}" data-vid-lang="${escapeHtml(l.code)}">${escapeHtml(l.region_tag)} ${escapeHtml(l.label_zh)}</button>`;
      })
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
    st.textContent = "正在上传视频、分离人声与背景音乐（Demucs/FFmpeg）并用 Whisper 提取时间轴字幕，请稍候…";
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const data = await apiFetch("/api/videos/analyze", { method: "POST", body: fd });
      upsertVideoLocal(data.video);
      state.activeVideoId = data.video.id;
      $("videoFileInput").value = "";
      st.textContent = `解析完成：共识别 ${data.video.segments.length} 条语音字幕 · BGM 分离模式：${data.video.bgm_method}`;
      renderVideoHistory();
      renderActiveVideo();
      showToast("视频人声分离与字幕提取完成！", "teal");
    } catch (err) {
      st.className = "inline-status error";
      st.textContent = err.message;
      showToast(err.message, "error");
    }
  }

  function renderActiveVideo() {
    const video = state.videos.find((v) => v.id === state.activeVideoId);
    const stage = $("videoProjectStage");
    if (!video) { stage.hidden = true; return; }
    stage.hidden = false;
    $("videoTitle").textContent = video.title || video.original_filename;
    $("videoMetaStats").textContent = `${video.width}×${video.height} · ${video.duration_sec}s · ${video.segments.length} 句 · ${video.bgm_method}`;
    if ($("sourceVideoPlayer").getAttribute("data-src") !== video.source_video_url) {
      $("sourceVideoPlayer").setAttribute("data-src", video.source_video_url);
      $("sourceVideoPlayer").src = video.source_video_url;
    }

    // Preview language selector for segment editing
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
          <div class="seg-time">${formatSec(s.start)} → ${formatSec(s.end)} (${s.duration_sec}s)</div>
          <div class="seg-orig">${escapeHtml(s.text)}</div>
          <textarea class="field-textarea compact" rows="2" data-vid-seg-idx="${idx}" placeholder="点击「一键等时长翻译字幕」自动生成">${escapeHtml(trList[idx] || "")}</textarea>
        </div>
      `)
      .join("");

    $("vidSegmentsList").querySelectorAll("[data-vid-seg-idx]").forEach((ta) => {
      ta.addEventListener("input", () => {
        const i = Number(ta.dataset.vidSegIdx);
        if (!video.translations) video.translations = {};
        if (!video.translations[state.videoPreviewLang]) {
          video.translations[state.videoPreviewLang] = video.segments.map(() => "");
        }
        video.translations[state.videoPreviewLang][i] = ta.value;
      });
    });

    // Outputs
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
            return `
              <div class="vid-out-card">
                <div class="track-top">
                  <div class="track-lang-meta">
                    <span class="track-code">${escapeHtml(o.region_tag)}</span>
                    <span class="track-title">${escapeHtml(o.lang_label)} 译配成片</span>
                  </div>
                  <span class="count-mono">${o.mp4_size_mb} MB · 耗时 ${o.elapsed_sec}s</span>
                </div>
                <video controls playsinline src="${escapeHtml(o.mp4_url)}?v=${encodeURIComponent(video.updated_at || "1")}"></video>
                <div class="track-dl-group">
                  <a class="btn-teal btn-sm" href="${escapeHtml(o.mp4_url)}" download="${escapeHtml(video.title)}_${escapeHtml(o.lang_code)}.mp4">下载 MP4</a>
                  <a class="btn-outline btn-sm" href="${escapeHtml(o.bilingual_srt_url)}" download="bilingual_${escapeHtml(o.lang_code)}.srt">双语 SRT</a>
                  <a class="btn-outline btn-sm" href="${escapeHtml(o.srt_url)}" download="sub_${escapeHtml(o.lang_code)}.srt">单语 SRT</a>
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
            <div class="history-sub">${v.segments.length} 句 · ${(v.outputs || []).filter((o) => o.status === "completed").length} 语种成片</div>
          </div>
          <button type="button" class="btn-del-mini" data-del-video="${escapeHtml(v.id)}">删除</button>
        </div>
      `)
      .join("");
    list.querySelectorAll("[data-open-video]").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest("[data-del-video]")) return;
        state.activeVideoId = el.dataset.openVideo;
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
    st.textContent = `正在为 ${video.segments.length} 条时间轴字幕执行多语种等时长翻译…`;
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
    st.textContent = "已提交多语种视频译配任务，GPU 正在逐句克隆并混音回填画面…";
    try {
      const data = await apiFetch(`/api/videos/${video.id}/dub`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_langs: langs,
          translations: video.translations || {},
          engine_id: $("vidEngineSelect").value,
          voice_id: $("vidVoiceSelect").value,
          mode: state.mode,
          exaggeration: state.exaggeration,
          cfg_weight: state.cfgWeight,
          instruct: $("cosyInstructInput").value.trim(),
          burn_subtitles: $("chkBurnSubtitles").checked,
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

  // ==================== WATERMARK VERIFY ====================
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

  // ---------- Settings ----------
  function fillSettingsModal(s) {
    if (!s) return;
    $("setProvider").value = s.translation_provider || "auto";
    $("setApiBase").value = s.llm_api_base || "";
    $("setModel").value = s.llm_model || "deepseek-chat";
    $("setApiKey").value = "";
    $("setApiKey").placeholder = s.has_llm_api_key ? `已保存 (${s.llm_api_key_masked})，留空保持不变` : "sk-...";
    $("setQwenVariant").value = s.qwen_variant || "1.7b";
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
          idle_unload_minutes: Number($("setIdleMinutes").value || 0),
          enable_batch_accel: $("setEnableBatch").checked,
          enable_watermark: $("setEnableWatermark").checked,
        }),
      });
      fillSettingsModal(updated);
      await loadSystemStatus(false);
      $("settingsModal").hidden = true;
      showToast("设置已保存", "teal");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  // ---------- Event Wiring ----------
  function initEvents() {
    // Top workspace tabs
    document.querySelectorAll("[data-workspace-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.workspaceTab = btn.dataset.workspaceTab;
        document.querySelectorAll("[data-workspace-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        $("workspaceAudio").hidden = state.workspaceTab !== "audio";
        $("workspaceVideo").hidden = state.workspaceTab !== "video";
        $("workspaceWatermark").hidden = state.workspaceTab !== "watermark";
      });
    });

    // Capture tabs
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

    // 3-Engine selector
    document.querySelectorAll("[data-select-engine]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedEngineId = btn.dataset.selectEngine;
        syncEngineUI();
      });
    });

    $("cosyInstructInput").addEventListener("input", (e) => {
      state.cosyInstruct = e.target.value;
      renderCosyPresets();
    });
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

    // Audio script & lang controls
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
    $("btnSelectCore10").addEventListener("click", () => {
      state.selectedLangs = new Set(getActiveLanguages().slice(0, 10).map((l) => l.code));
      renderLangGrid(); renderPerLangScripts();
    });
    $("btnSelectAll23").addEventListener("click", () => {
      state.selectedLangs = new Set(getActiveLanguages().map((l) => l.code));
      renderLangGrid(); renderPerLangScripts();
    });
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

    // Video dubbing events
    const vdz = $("videoDropzone");
    const vInput = $("videoFileInput");
    vdz.addEventListener("click", () => vInput.click());
    vdz.addEventListener("dragover", (e) => { e.preventDefault(); vdz.classList.add("dragover"); });
    vdz.addEventListener("dragleave", () => vdz.classList.remove("dragover"));
    vdz.addEventListener("drop", (e) => { e.preventDefault(); vdz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) uploadAndAnalyzeVideo(e.dataTransfer.files[0]); });
    vInput.addEventListener("change", () => { if (vInput.files?.[0]) uploadAndAnalyzeVideo(vInput.files[0]); });
    $("vidEngineSelect").addEventListener("change", renderVideoLangChips);
    $("vidPreviewLangSelect").addEventListener("change", (e) => { state.videoPreviewLang = e.target.value; renderActiveVideo(); });
    $("vidBgmSlider").addEventListener("input", (e) => { $("valBgmVol").textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`; });
    $("btnVideoTranslate").addEventListener("click", () => translateActiveVideo(false));
    $("btnVideoDub").addEventListener("click", startVideoDubbing);

    // Watermark Lab events
    const wdz = $("wmDropzone");
    const wInput = $("wmFileInput");
    wdz.addEventListener("click", () => wInput.click());
    wdz.addEventListener("dragover", (e) => { e.preventDefault(); wdz.classList.add("dragover"); });
    wdz.addEventListener("dragleave", () => wdz.classList.remove("dragover"));
    wdz.addEventListener("drop", (e) => { e.preventDefault(); wdz.classList.remove("dragover"); if (e.dataTransfer?.files?.[0]) verifyWatermarkFile(e.dataTransfer.files[0]); });
    wInput.addEventListener("change", () => { if (wInput.files?.[0]) verifyWatermarkFile(wInput.files[0]); });

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
    await Promise.all([loadVoices(), loadJobs(), loadVideos()]);
  }

  async function boot() {
    initEvents();
    drawIdleMicMeter();
    await bootData();
    state.statusTimer = setInterval(() => loadSystemStatus(false), 8000);
  }

  boot();
})();
