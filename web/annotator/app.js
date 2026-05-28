const state = {
  mode: "annotate",
  media: [],
  filtered: [],
  currentIndex: -1,
  current: null,
  boxes: [],
  drawing: false,
  start: null,
  cursor: null,
  naturalWidth: 0,
  naturalHeight: 0,
  scale: 1,
  frameTime: null,
  baseCanvas: document.createElement("canvas"),
  sourceStatsByName: new Map(),
  liveRegistryCameras: [],
  liveLocalCameras: [],
  liveLocalScanRunning: false,
  liveSelectedMediaPath: "",
  liveEvents: [],
  liveRunning: false,
  liveRecording: false,
  liveModelPath: "",
};

const els = {
  notificationTray: document.getElementById("notificationTray"),
  annotateTabButton: document.getElementById("annotateTabButton"),
  liveTabButton: document.getElementById("liveTabButton"),
  annotationTab: document.getElementById("annotationTab"),
  liveTab: document.getElementById("liveTab"),
  annotationView: document.getElementById("annotationView"),
  liveView: document.getElementById("liveView"),
  folderInput: document.getElementById("folderInput"),
  projectInput: document.getElementById("projectInput"),
  splitSelect: document.getElementById("splitSelect"),
  scanButton: document.getElementById("scanButton"),
  kindFilter: document.getElementById("kindFilter"),
  mediaCount: document.getElementById("mediaCount"),
  mediaList: document.getElementById("mediaList"),
  rawFileCount: document.getElementById("rawFileCount"),
  rawVideoCount: document.getElementById("rawVideoCount"),
  rawImageCount: document.getElementById("rawImageCount"),
  annotationStatsRows: document.getElementById("annotationStatsRows"),
  statsUpdatedAt: document.getElementById("statsUpdatedAt"),
  refreshStatsButton: document.getElementById("refreshStatsButton"),
  video: document.getElementById("videoPlayer"),
  image: document.getElementById("imagePreview"),
  canvas: document.getElementById("annotationCanvas"),
  currentName: document.getElementById("currentName"),
  boxCount: document.getElementById("boxCount"),
  status: document.getElementById("statusText"),
  captureButton: document.getElementById("captureButton"),
  saveButton: document.getElementById("saveButton"),
  negativeButton: document.getElementById("negativeButton"),
  undoButton: document.getElementById("undoButton"),
  clearButton: document.getElementById("clearButton"),
  prevButton: document.getElementById("prevButton"),
  nextButton: document.getElementById("nextButton"),
  videoButton: document.getElementById("videoButton"),
  playButton: document.getElementById("playButton"),
  pauseButton: document.getElementById("pauseButton"),
  backButton: document.getElementById("backButton"),
  forwardButton: document.getElementById("forwardButton"),
  liveCameraSelect: document.getElementById("liveCameraSelect"),
  liveScanLocalButton: document.getElementById("liveScanLocalButton"),
  liveSourceInput: document.getElementById("liveSourceInput"),
  liveConfInput: document.getElementById("liveConfInput"),
  liveFpsInput: document.getElementById("liveFpsInput"),
  liveFrameSkipInput: document.getElementById("liveFrameSkipInput"),
  livePresetSelect: document.getElementById("livePresetSelect"),
  liveImageSizeInput: document.getElementById("liveImageSizeInput"),
  liveDeviceSelect: document.getElementById("liveDeviceSelect"),
  liveRecordInput: document.getElementById("liveRecordInput"),
  liveStartButton: document.getElementById("liveStartButton"),
  liveStopButton: document.getElementById("liveStopButton"),
  liveVideoPreview: document.getElementById("liveVideoPreview"),
  liveImagePreview: document.getElementById("liveImagePreview"),
  liveStream: document.getElementById("liveStream"),
  livePlaceholder: document.getElementById("livePlaceholder"),
  liveTitle: document.getElementById("liveTitle"),
  liveStatus: document.getElementById("liveStatus"),
  liveRefreshEventsButton: document.getElementById("liveRefreshEventsButton"),
  liveEventsUpdatedAt: document.getElementById("liveEventsUpdatedAt"),
  liveEventList: document.getElementById("liveEventList"),
};

const ctx = els.canvas.getContext("2d");
const baseCtx = state.baseCanvas.getContext("2d");

init();

async function init() {
  const defaults = await getJson("/api/defaults");
  els.folderInput.value = defaults.folder;
  els.projectInput.value = defaults.project_dir;
  state.liveModelPath = defaults.live_model || "data_store/models/trained/yolov8n_drone_best.pt";
  els.liveSourceInput.value = "0";
  bindEvents();
  await loadLiveCameras();
  scanLocalCameras({ automatic: true });
  await scanFolder();
  await refreshLiveEvents();
  window.setInterval(refreshLiveEvents, 5000);
}

function bindEvents() {
  els.annotateTabButton.addEventListener("click", () => showTab("annotate"));
  els.liveTabButton.addEventListener("click", () => showTab("live"));
  els.scanButton.addEventListener("click", scanFolder);
  els.folderInput.addEventListener("change", refreshStats);
  els.projectInput.addEventListener("change", refreshStats);
  els.refreshStatsButton.addEventListener("click", refreshStats);
  els.kindFilter.addEventListener("change", renderMediaList);
  els.captureButton.addEventListener("click", captureCurrentFrame);
  els.videoButton.addEventListener("click", showVideo);
  els.playButton.addEventListener("click", playVideo);
  els.pauseButton.addEventListener("click", () => els.video.pause());
  els.backButton.addEventListener("click", () => seekVideo(-1));
  els.forwardButton.addEventListener("click", () => seekVideo(1));
  els.liveCameraSelect.addEventListener("change", onLiveCameraChanged);
  els.liveScanLocalButton.addEventListener("click", scanLocalCameras);
  els.liveSourceInput.addEventListener("input", onLiveCustomSourceInput);
  els.livePresetSelect.addEventListener("change", applyLivePreset);
  [els.liveFpsInput, els.liveFrameSkipInput, els.liveImageSizeInput].forEach((input) => {
    input.addEventListener("input", markLivePresetCustom);
  });
  els.liveDeviceSelect.addEventListener("change", stopLiveDetectionForSourceChange);
  els.liveStartButton.addEventListener("click", startLiveDetection);
  els.liveStopButton.addEventListener("click", stopLiveDetection);
  els.liveRefreshEventsButton.addEventListener("click", refreshLiveEvents);
  els.liveStream.addEventListener("load", () => {
    if (state.liveRunning) setLiveStatus("Streaming");
  });
  els.liveStream.addEventListener("error", () => {
    if (state.liveRunning) setLiveStatus("Stream stopped or unavailable");
  });
  els.liveVideoPreview.addEventListener("loadedmetadata", () => setLiveStatus(`Video preview ready (${formatTime(els.liveVideoPreview.duration)})`));
  els.liveVideoPreview.addEventListener("error", () => setLiveStatus("Browser cannot play this video preview."));
  els.liveImagePreview.addEventListener("load", () => setLiveStatus("Image preview ready"));
  els.liveImagePreview.addEventListener("error", () => setLiveStatus("Image preview unavailable"));
  els.video.addEventListener("error", () => setStatus("Browser cannot play this video codec. Convert to H.264 MP4 or use screenshots/images."));
  els.video.addEventListener("loadedmetadata", () => setStatus(`Video ready (${formatTime(els.video.duration)})`));
  els.saveButton.addEventListener("click", () => saveAnnotation(false));
  els.negativeButton.addEventListener("click", () => saveAnnotation(true));
  els.undoButton.addEventListener("click", () => {
    state.boxes.pop();
    draw();
  });
  els.clearButton.addEventListener("click", () => {
    state.boxes = [];
    draw();
  });
  els.prevButton.addEventListener("click", () => selectRelative(-1));
  els.nextButton.addEventListener("click", () => selectRelative(1));

  els.canvas.addEventListener("pointerdown", onPointerDown);
  els.canvas.addEventListener("pointermove", onPointerMove);
  els.canvas.addEventListener("pointerup", onPointerUp);
  window.addEventListener("resize", draw);
  window.addEventListener("keydown", onKeyDown);
}

function showTab(name) {
  const live = name === "live";
  state.mode = live ? "live" : "annotate";
  els.liveTab.classList.toggle("active", live);
  els.annotationTab.classList.toggle("active", !live);
  els.liveView.classList.toggle("active", live);
  els.annotationView.classList.toggle("active", !live);
  els.liveTabButton.classList.toggle("active", live);
  els.annotateTabButton.classList.toggle("active", !live);
  renderMediaList();
  if (live) {
    const selectedMedia = selectedLiveMediaItem();
    if (selectedMedia && !state.liveRunning) {
      showLiveMediaPreview(selectedMedia);
    } else if (!state.liveRunning) {
      showLivePlaceholder();
    }
  } else {
    draw();
  }
}

async function loadLiveCameras() {
  try {
    const result = await getJson("/api/live/cameras");
    state.liveRegistryCameras = result.cameras || [];
    renderLiveCameraSelect();
    if (state.liveRegistryCameras.length) {
      setLiveStatus(`${state.liveRegistryCameras.length} cameras loaded`);
    }
  } catch (_error) {
    setLiveStatus("Camera registry unavailable");
  }
}

function renderLiveCameraSelect() {
  const previousValue = els.liveCameraSelect.value;
  els.liveCameraSelect.innerHTML = `<option value="">Custom source</option>`;

  if (state.liveRegistryCameras.length) {
    const group = document.createElement("optgroup");
    group.label = "Configured cameras";
    state.liveRegistryCameras.forEach((camera) => {
      const option = document.createElement("option");
      option.value = `camera:${camera.id}`;
      option.dataset.streamKind = "camera";
      option.dataset.source = camera.id;
      const model = camera.model ? ` · ${camera.model}` : "";
      option.textContent = `${camera.name} (${camera.address || camera.id})${model}`;
      option.disabled = camera.enabled === false;
      group.appendChild(option);
    });
    els.liveCameraSelect.appendChild(group);
  }

  if (state.liveLocalCameras.length) {
    const group = document.createElement("optgroup");
    group.label = "Local cameras";
    state.liveLocalCameras.forEach((camera) => {
      const option = document.createElement("option");
      option.value = `source:${camera.source}`;
      option.dataset.streamKind = "source";
      option.dataset.source = camera.source;
      const size = camera.width && camera.height ? ` · ${camera.width}x${camera.height}` : "";
      option.textContent = `${camera.name}${size}`;
      group.appendChild(option);
    });
    els.liveCameraSelect.appendChild(group);
  }

  if ([...els.liveCameraSelect.options].some((option) => option.value === previousValue)) {
    els.liveCameraSelect.value = previousValue;
  }
}

async function scanLocalCameras(options = {}) {
  if (state.liveLocalScanRunning) return;
  const automatic = options && options.automatic === true;
  const scanLabel = automatic ? "Auto-scanning local cameras" : "Scanning local cameras";
  try {
    state.liveLocalScanRunning = true;
    els.liveScanLocalButton.disabled = true;
    els.liveScanLocalButton.textContent = "Scanning...";
    els.liveScanLocalButton.setAttribute("aria-busy", "true");
    setLiveStatus(`${scanLabel}...`);
    showNotification(`${scanLabel} started`, "info", 2500);
    const result = await getJson("/api/live/local-cameras?max_index=5");
    if (result.error) {
      setLiveStatus(result.error);
      showNotification(result.error, "error", 5000);
      return;
    }
    state.liveLocalCameras = result.cameras || [];
    renderLiveCameraSelect();
    const status = state.liveLocalCameras.length
      ? `${state.liveLocalCameras.length} local cameras found`
      : "No local cameras found";
    setLiveStatus(status);
    showNotification(`${scanLabel} finished. ${status}.`, state.liveLocalCameras.length ? "success" : "info", 4000);
    if (automatic && hasGenericLocalCameraNames(state.liveLocalCameras) && options.retryNames !== false) {
      window.setTimeout(() => scanLocalCameras({ automatic: true, retryNames: false }), 1500);
    }
  } catch (_error) {
    setLiveStatus("Local camera scan failed");
    showNotification("Local camera scan failed", "error", 5000);
  } finally {
    state.liveLocalScanRunning = false;
    els.liveScanLocalButton.disabled = false;
    els.liveScanLocalButton.textContent = "Scan Local";
    els.liveScanLocalButton.removeAttribute("aria-busy");
  }
}

function hasGenericLocalCameraNames(cameras) {
  return cameras.length > 0 && cameras.some((camera) => /^Local camera \d+$/.test(String(camera.name || "")));
}

function onLiveCameraChanged() {
  const selected = selectedLiveSourceOption();
  if (!selected) return;
  state.liveSelectedMediaPath = "";
  const source = selected.dataset.source || "";
  if (selected.dataset.streamKind === "camera") {
    els.liveSourceInput.value = `camera:${source}`;
  } else if (selected.dataset.streamKind === "source") {
    els.liveSourceInput.value = source;
  }
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLivePlaceholder();
  els.liveTitle.textContent = liveSourceLabel();
}

function selectLiveMediaSource(item) {
  state.liveSelectedMediaPath = item.path;
  els.liveCameraSelect.value = "";
  els.liveSourceInput.value = item.path;
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLiveMediaPreview(item);
}

function onLiveCustomSourceInput() {
  els.liveCameraSelect.value = "";
  state.liveSelectedMediaPath = "";
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLivePlaceholder();
  els.liveTitle.textContent = liveSourceLabel();
}

function startLiveDetection() {
  if (state.liveRunning) stopLiveDetection({ restorePreview: false });
  const url = liveStreamUrl();
  hideLiveMediaPreview();
  els.liveStream.src = url;
  els.liveStream.style.display = "block";
  els.livePlaceholder.style.display = "none";
  state.liveRunning = true;
  state.liveRecording = els.liveRecordInput.checked;
  els.liveStartButton.disabled = true;
  els.liveRecordInput.disabled = true;
  els.liveTitle.textContent = liveSourceLabel();
  setLiveStatus(state.liveRecording ? "Starting stream and recording..." : "Starting stream...");
  if (state.liveRecording) {
    showNotification("Recording started", "info", 2500);
  }
  window.setTimeout(refreshLiveEvents, 500);
}

function stopLiveDetection(options = {}) {
  const restorePreview = options.restorePreview !== false;
  const wasRecording = state.liveRecording;
  state.liveRunning = false;
  state.liveRecording = false;
  els.liveStream.removeAttribute("src");
  els.liveStream.style.display = "none";
  els.liveStartButton.disabled = false;
  els.liveRecordInput.disabled = false;
  if (restorePreview) {
    const selectedMedia = selectedLiveMediaItem();
    if (selectedMedia) {
      showLiveMediaPreview(selectedMedia);
    } else {
      showLivePlaceholder();
    }
  } else {
    showLivePlaceholder();
  }
  setLiveStatus("Stopped");
  if (wasRecording) {
    showNotification("Recording stopped. Refreshing media list...", "success", 4000);
    window.setTimeout(scanFolder, 1500);
  }
  window.setTimeout(refreshLiveEvents, 500);
}

function stopLiveDetectionForSourceChange() {
  if (!state.liveRunning) return;
  stopLiveDetection();
  setLiveStatus("Source changed. Press Start.");
  els.liveTitle.textContent = liveSourceLabel();
}

function liveStreamUrl() {
  const params = new URLSearchParams();
  const selected = selectedLiveSourceOption();
  if (selected && selected.dataset.streamKind === "camera") {
    params.set("camera", selected.dataset.source || "");
  } else if (selected && selected.dataset.streamKind === "source") {
    params.set("source", selected.dataset.source || "0");
  } else if (state.liveSelectedMediaPath) {
    params.set("source", state.liveSelectedMediaPath);
  } else {
    params.set("source", els.liveSourceInput.value || "0");
  }
  params.set("model", state.liveModelPath || "data_store/models/trained/yolov8n_drone_best.pt");
  params.set("conf", els.liveConfInput.value || "0.5");
  params.set("max_fps", els.liveFpsInput.value || "5");
  params.set("frame_skip", els.liveFrameSkipInput.value || "0");
  params.set("imgsz", els.liveImageSizeInput.value || "640");
  if (els.liveDeviceSelect.value) {
    params.set("device", els.liveDeviceSelect.value);
  }
  if (els.liveRecordInput.checked) {
    params.set("record", "1");
    params.set("record_dir", els.folderInput.value);
    params.set("record_max_mb", "30");
  }
  params.set("_", Date.now().toString());
  return `/api/live/stream?${params.toString()}`;
}

function applyLivePreset() {
  const presets = {
    balanced: { fps: "5", skip: "0", imageSize: "640" },
    fast: { fps: "12", skip: "2", imageSize: "416" },
    quality: { fps: "3", skip: "0", imageSize: "960" },
  };
  const preset = presets[els.livePresetSelect.value];
  if (!preset) return;
  els.liveFpsInput.value = preset.fps;
  els.liveFrameSkipInput.value = preset.skip;
  els.liveImageSizeInput.value = preset.imageSize;
  stopLiveDetectionForSourceChange();
}

function markLivePresetCustom() {
  els.livePresetSelect.value = "custom";
  stopLiveDetectionForSourceChange();
}

function liveSourceLabel() {
  const selected = selectedLiveSourceOption();
  if (selected) return selected.textContent;
  const selectedMedia = state.media.find((item) => item.path === state.liveSelectedMediaPath);
  if (selectedMedia) return `${selectedMedia.relative || selectedMedia.name} · ${selectedMedia.kind}`;
  return els.liveSourceInput.value || "Custom source";
}

function selectedLiveSourceOption() {
  const selected = els.liveCameraSelect.selectedOptions[0];
  return selected && selected.value ? selected : null;
}

function selectedLiveMediaItem() {
  return state.media.find((item) => item.path === state.liveSelectedMediaPath) || null;
}

function showLiveMediaPreview(item) {
  els.liveStream.removeAttribute("src");
  els.liveStream.style.display = "none";
  els.livePlaceholder.style.display = "none";
  els.liveTitle.textContent = `${item.relative || item.name} · ${item.kind}`;
  const url = mediaUrl(item.path, mediaVersion(item));
  if (item.kind === "video") {
    els.liveImagePreview.removeAttribute("src");
    els.liveImagePreview.style.display = "none";
    els.liveVideoPreview.style.display = "block";
    if (els.liveVideoPreview.src !== new URL(url, window.location.href).href) {
      els.liveVideoPreview.src = url;
      els.liveVideoPreview.load();
    }
    setLiveStatus("Video preview loading...");
  } else {
    els.liveVideoPreview.pause();
    els.liveVideoPreview.removeAttribute("src");
    els.liveVideoPreview.style.display = "none";
    els.liveImagePreview.style.display = "block";
    els.liveImagePreview.src = url;
    setLiveStatus("Image preview loading...");
  }
}

function hideLiveMediaPreview() {
  els.liveVideoPreview.pause();
  els.liveVideoPreview.removeAttribute("src");
  els.liveVideoPreview.style.display = "none";
  els.liveImagePreview.removeAttribute("src");
  els.liveImagePreview.style.display = "none";
}

function showLivePlaceholder() {
  hideLiveMediaPreview();
  els.liveStream.removeAttribute("src");
  els.liveStream.style.display = "none";
  els.livePlaceholder.style.display = "flex";
}

async function scanFolder() {
  setStatus("Scanning...");
  const result = await postJson("/api/scan", { folder: els.folderInput.value });
  if (result.error) {
    setStatus(result.error);
    return;
  }
  state.media = result.media || [];
  state.currentIndex = -1;
  state.current = null;
  if (!state.media.some((item) => item.path === state.liveSelectedMediaPath)) {
    state.liveSelectedMediaPath = "";
    showLivePlaceholder();
  }
  renderMediaList();
  await refreshStats();
  if (state.filtered.length) {
    if (state.mode === "live") {
      selectLiveMediaSource(state.filtered[0]);
    } else {
      selectMedia(0);
    }
  } else {
    setStatus("No media found");
  }
}

function renderMediaList() {
  const kind = els.kindFilter.value;
  state.filtered = filteredMedia(kind);
  els.mediaCount.textContent = `${state.filtered.length} files`;
  const live = state.mode === "live";
  renderMediaItems(els.mediaList, state.filtered, {
    isActive: (item) => live
      ? state.liveSelectedMediaPath === item.path
      : state.current && state.current.path === item.path,
    onSelect: (item, index) => {
      if (live) {
        selectLiveMediaSource(item);
      } else {
        selectMedia(index);
      }
    },
  });
}

function renderMediaItems(container, items, options) {
  container.innerHTML = "";
  items.forEach((item, index) => {
    const stats = sourceStatsFor(item);
    const button = document.createElement("button");
    button.className = "mediaItem";
    if (options.isActive(item, index)) button.classList.add("active");
    button.innerHTML = `
      <span class="mediaIdentity">
        <span class="mediaName">${escapeHtml(item.name)}</span>
        <span class="mediaMeta">${item.kind} · ${formatBytes(item.size)}</span>
      </span>
      ${mediaStatsMarkup(stats)}
    `;
    button.addEventListener("click", () => options.onSelect(item, index));
    container.appendChild(button);
  });
}

function filteredMedia(kind) {
  return state.media.filter((item) => kind === "all" || item.kind === kind);
}

function selectRelative(delta) {
  if (!state.filtered.length) return;
  const next = Math.max(0, Math.min(state.filtered.length - 1, state.currentIndex + delta));
  selectMedia(next);
}

function selectMedia(index) {
  state.currentIndex = index;
  state.current = state.filtered[index];
  state.boxes = [];
  state.naturalWidth = 0;
  state.naturalHeight = 0;
  state.frameTime = null;
  els.currentName.textContent = state.current ? state.current.relative : "No media selected";
  renderMediaList();

  if (state.current.kind === "video") {
    loadVideo(state.current);
  } else {
    loadImage(state.current);
  }
}

function loadVideo(item) {
  const url = mediaUrl(item.path, mediaVersion(item));
  els.image.style.display = "none";
  els.canvas.style.display = "none";
  els.video.style.display = "block";
  els.video.src = url;
  els.video.load();
  setStatus("Video ready");
}

function loadImage(item) {
  const url = mediaUrl(item.path, mediaVersion(item));
  els.video.pause();
  els.video.removeAttribute("src");
  els.video.style.display = "none";
  els.image.onload = () => {
    state.baseCanvas.width = els.image.naturalWidth;
    state.baseCanvas.height = els.image.naturalHeight;
    baseCtx.drawImage(els.image, 0, 0);
    state.naturalWidth = els.image.naturalWidth;
    state.naturalHeight = els.image.naturalHeight;
    state.frameTime = null;
    els.image.style.display = "none";
    els.canvas.style.display = "block";
    state.boxes = [];
    draw();
    setStatus("Image ready");
  };
  els.image.src = url;
}

function captureCurrentFrame() {
  if (!state.current) return;
  if (state.current.kind === "image") {
    draw();
    return;
  }
  if (!els.video.videoWidth || !els.video.videoHeight) {
    setStatus("Video frame not ready");
    return;
  }
  state.baseCanvas.width = els.video.videoWidth;
  state.baseCanvas.height = els.video.videoHeight;
  baseCtx.drawImage(els.video, 0, 0, state.baseCanvas.width, state.baseCanvas.height);
  state.naturalWidth = state.baseCanvas.width;
  state.naturalHeight = state.baseCanvas.height;
  state.frameTime = els.video.currentTime;
  els.video.pause();
  els.video.style.display = "none";
  els.canvas.style.display = "block";
  state.boxes = [];
  draw();
  setStatus(`Captured ${formatTime(els.video.currentTime)}`);
}

function showVideo() {
  if (!state.current || state.current.kind !== "video") return;
  els.canvas.style.display = "none";
  els.video.style.display = "block";
  setStatus("Video ready");
}

async function playVideo() {
  if (!state.current || state.current.kind !== "video") return;
  showVideo();
  try {
    await els.video.play();
  } catch (_error) {
    setStatus("Click inside the video area, then press Play again.");
  }
}

function seekVideo(deltaSeconds) {
  if (!state.current || state.current.kind !== "video") return;
  els.video.currentTime = clamp(els.video.currentTime + deltaSeconds, 0, els.video.duration || 0);
  showVideo();
}

function draw() {
  if (!state.naturalWidth || !state.naturalHeight) {
    updateBoxCount();
    return;
  }

  const viewer = els.canvas.closest(".viewer");
  const maxWidth = Math.max(320, viewer.clientWidth - 24);
  const maxHeight = Math.max(240, viewer.clientHeight - 24);
  state.scale = Math.min(maxWidth / state.naturalWidth, maxHeight / state.naturalHeight, 1);
  const displayWidth = Math.round(state.naturalWidth * state.scale);
  const displayHeight = Math.round(state.naturalHeight * state.scale);
  els.canvas.width = displayWidth;
  els.canvas.height = displayHeight;
  els.canvas.style.width = `${displayWidth}px`;
  els.canvas.style.height = `${displayHeight}px`;

  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.drawImage(state.baseCanvas, 0, 0, els.canvas.width, els.canvas.height);

  state.boxes.forEach((box) => drawBox(box, "#30bced"));
  if (state.drawing && state.start && state.cursor) {
    drawBox(normaliseBox({ x1: state.start.x, y1: state.start.y, x2: state.cursor.x, y2: state.cursor.y }), "#f25f5c");
  }
  updateBoxCount();
}

function drawBox(box, color) {
  const scaleX = els.canvas.width / state.naturalWidth;
  const scaleY = els.canvas.height / state.naturalHeight;
  const x = box.x1 * scaleX;
  const y = box.y1 * scaleY;
  const w = (box.x2 - box.x1) * scaleX;
  const h = (box.y2 - box.y1) * scaleY;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.font = "13px system-ui";
  ctx.fillText("drone", x + 4, Math.max(15, y - 5));
}

function onPointerDown(event) {
  if (!state.naturalWidth) return;
  event.preventDefault();
  els.canvas.setPointerCapture(event.pointerId);
  state.drawing = true;
  state.start = eventToImagePoint(event);
  state.cursor = state.start;
}

function onPointerMove(event) {
  if (!state.drawing) return;
  event.preventDefault();
  state.cursor = eventToImagePoint(event);
  draw();
}

function onPointerUp(event) {
  if (!state.drawing || !state.start) return;
  event.preventDefault();
  state.cursor = eventToImagePoint(event);
  const box = normaliseBox({ x1: state.start.x, y1: state.start.y, x2: state.cursor.x, y2: state.cursor.y });
  if (box.x2 - box.x1 > 3 && box.y2 - box.y1 > 3) {
    state.boxes.push(box);
  }
  state.drawing = false;
  state.start = null;
  state.cursor = null;
  if (els.canvas.hasPointerCapture(event.pointerId)) {
    els.canvas.releasePointerCapture(event.pointerId);
  }
  draw();
}

function eventToImagePoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) * (els.canvas.width / rect.width);
  const canvasY = (event.clientY - rect.top) * (els.canvas.height / rect.height);
  const x = clamp(canvasX * (state.naturalWidth / els.canvas.width), 0, state.naturalWidth - 1);
  const y = clamp(canvasY * (state.naturalHeight / els.canvas.height), 0, state.naturalHeight - 1);
  return { x, y };
}

async function saveAnnotation(negative) {
  if (!state.current || !state.naturalWidth || !state.naturalHeight) {
    setStatus("Capture or load an image first");
    return;
  }
  const boxes = negative ? [] : state.boxes;
  const payload = {
    project_dir: els.projectInput.value,
    class_name: "drone",
    split: els.splitSelect.value,
    source_path: state.current.path,
    media_kind: state.current.kind,
    frame_time: state.current.kind === "video" ? state.frameTime : null,
    image_width: state.naturalWidth,
    image_height: state.naturalHeight,
    boxes,
    image_data: state.baseCanvas.toDataURL("image/jpeg", 0.92),
  };
  const result = await postJson("/api/save", payload);
  if (result.error) {
    setStatus(result.error);
    return;
  }
  setStatus(`Saved ${result.image_id} (${result.box_count} boxes)`);
  await refreshStats();
}

function onKeyDown(event) {
  if (event.target && ["INPUT", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === "ArrowRight") selectRelative(1);
  if (event.key === "ArrowLeft") selectRelative(-1);
  if (event.key === "Backspace") {
    state.boxes.pop();
    draw();
  }
  if (event.key === "s") saveAnnotation(false);
  if (event.key === "0") saveAnnotation(true);
}

async function getJson(url) {
  const response = await fetch(url);
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

async function refreshStats() {
  try {
    els.refreshStatsButton.disabled = true;
    const stats = await postJson("/api/stats", {
      folder: els.folderInput.value,
      project_dir: els.projectInput.value,
    });
    renderStats(stats);
  } catch (_error) {
    renderStats(null);
  } finally {
    els.refreshStatsButton.disabled = false;
  }
}

async function refreshLiveEvents() {
  try {
    els.liveRefreshEventsButton.disabled = true;
    const result = await getJson("/api/live/events?limit=50");
    state.liveEvents = result.events || [];
    renderLiveEvents();
  } catch (_error) {
    els.liveEventsUpdatedAt.textContent = "Events unavailable";
  } finally {
    els.liveRefreshEventsButton.disabled = false;
  }
}

function renderLiveEvents() {
  els.liveEventsUpdatedAt.textContent = `Updated ${formatClock(new Date().toISOString())}`;
  if (!state.liveEvents.length) {
    els.liveEventList.innerHTML = `<div class="emptyState">No events yet</div>`;
    return;
  }

  els.liveEventList.innerHTML = state.liveEvents.map((event) => liveEventMarkup(event)).join("");
}

function liveEventMarkup(event) {
  const type = String(event.event_type || "event");
  const source = shortSource(event.source || "");
  const best = event.best_track || {};
  const image = event.image_path
    ? `<img class="eventThumb" src="${mediaUrl(event.image_path)}" alt="">`
    : "";
  const details = eventDetailText(type, event, best);
  const imageClass = image ? "" : " noImage";
  return `
    <article class="eventItem ${eventClass(type)}${imageClass}">
      ${image}
      <div class="eventBody">
        <div class="eventTopline">
          <strong>${escapeHtml(eventTitle(type))}</strong>
          <span>${escapeHtml(formatClock(event.timestamp))}</span>
        </div>
        <div class="eventDetails">${escapeHtml(details)}</div>
        <div class="eventSource">${escapeHtml(source)}</div>
      </div>
    </article>
  `;
}

function eventTitle(type) {
  if (type === "drone_detected") return "Drone detected";
  if (type === "recording_started") return "Recording started";
  if (type === "recording_saved") return "Recording saved";
  if (type === "recording_skipped") return "Recording skipped";
  return type.replaceAll("_", " ");
}

function eventClass(type) {
  if (type === "drone_detected") return "alert";
  if (type === "error") return "error";
  return "";
}

function eventDetailText(type, event, best) {
  if (type === "drone_detected") {
    const confidence = Number(best.confidence || 0);
    const trackId = best.track_id ? ` #${best.track_id}` : "";
    return `${best.label || "drone"}${trackId} ${confidence.toFixed(2)} · frame ${formatInteger(event.frame_index)}`;
  }
  if (type === "stop") {
    return `${event.reason || "stopped"} · ${formatInteger(event.frames_seen)} frames · ${formatInteger(event.detection_events)} detections`;
  }
  if (type === "start") {
    const settings = event.settings || {};
    return `${event.source_kind || "source"} · ${settings.device || "auto"} · ${settings.image_size || "img"}px`;
  }
  if (type === "recording_started") {
    return `${event.max_size_mb || 30} MB max segment`;
  }
  if (type === "recording_saved") {
    return `${formatBytes(Number(event.size_bytes || 0))} · ${shortSource(event.recording_path || "")}`;
  }
  return String(event.message || event.reason || "");
}

function shortSource(value) {
  const text = String(value || "");
  if (!text) return "";
  const parts = text.split("/");
  return parts.length > 2 ? parts.slice(-2).join("/") : text;
}

function renderStats(stats) {
  const raw = stats && stats.raw ? stats.raw : {};
  els.statsUpdatedAt.textContent = stats && stats.generated_at
    ? `Updated ${formatClock(stats.generated_at)}`
    : "Stats unavailable";
  els.rawFileCount.textContent = formatInteger(raw.files);
  els.rawVideoCount.textContent = formatInteger(raw.videos);
  els.rawImageCount.textContent = formatInteger(raw.images);

  const annotationStats = stats && stats.annotations ? stats.annotations : {};
  const splits = annotationStats.splits || {};
  const rows = ["train", "val"].map((split) => annotationSplitRow(split, splits[split] || {}));
  els.annotationStatsRows.innerHTML = rows.join("");

  const sources = annotationStats.sources || [];
  state.sourceStatsByName = sourceStatsMap(sources);
  renderMediaList();
}

function annotationSplitRow(split, stats) {
  return `
    <tr>
      <th>${escapeHtml(split)}</th>
      <td>${formatInteger(stats.total)}</td>
      <td>${formatInteger(stats.positive)}</td>
      <td>${formatInteger(stats.negative)}</td>
    </tr>
  `;
}

function sourceStatsMap(sources) {
  const statsByName = new Map();
  sources.forEach((source) => {
    const sourceName = source.source || "";
    statsByName.set(sourceKey(sourceName), source);
  });
  return statsByName;
}

function sourceStatsFor(item) {
  if (!item) return null;
  return state.sourceStatsByName.get(sourceKey(item.name))
    || state.sourceStatsByName.get(sourceKey(item.relative));
}

function mediaStatsMarkup(stats) {
  if (!stats) {
    return `<span class="mediaStats"><span class="mediaStat empty">0 annotated frames</span></span>`;
  }
  return `
    <span class="mediaStats">
      ${compactStat(stats.frames, "frames")}
      ${compactStat(stats.positive, "positive")}
      ${compactStat(stats.negative, "negative")}
      ${compactStat(stats.boxes, "boxes")}
    </span>
  `;
}

function compactStat(value, label) {
  return `
    <span class="mediaStat">
      <strong>${formatInteger(value)}</strong>
      <span>${escapeHtml(label)}</span>
    </span>
  `;
}

function sourceKey(value) {
  return String(value || "").toLowerCase();
}

function mediaUrl(path, version = "") {
  const base = `/api/media?path=${encodeURIComponent(path)}`;
  if (!version) return base;
  return `${base}&v=${encodeURIComponent(String(version))}`;
}

function mediaVersion(item) {
  if (!item) return "";
  return `${item.mtime || ""}-${item.size || ""}`;
}

function normaliseBox(box) {
  return {
    x1: Math.min(box.x1, box.x2),
    y1: Math.min(box.y1, box.y2),
    x2: Math.max(box.x1, box.x2),
    y2: Math.max(box.y1, box.y2),
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateBoxCount() {
  els.boxCount.textContent = `${state.boxes.length} boxes`;
}

function setStatus(text) {
  els.status.textContent = text;
}

function setLiveStatus(text) {
  els.liveStatus.textContent = text;
}

function showNotification(message, type = "info", timeout = 3500) {
  if (!els.notificationTray) return;
  const toast = document.createElement("div");
  toast.className = `notificationToast ${type}`;
  toast.textContent = message;
  els.notificationTray.prepend(toast);
  while (els.notificationTray.children.length > 4) {
    els.notificationTray.lastElementChild.remove();
  }
  if (timeout > 0) {
    window.setTimeout(() => toast.remove(), timeout);
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatInteger(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString() : "0";
}

function formatClock(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "now";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
