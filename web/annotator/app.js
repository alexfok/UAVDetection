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
  liveSelectedMediaPaths: [],
  selectedLiveEventIds: new Set(),
  liveEvents: [],
  liveRunning: false,
  liveRecording: false,
  liveStreamJobs: [],
  liveRecordingChecks: new Map(),
  liveRecordingPollTimer: null,
  liveModelPath: "",
  trainingJob: null,
};

const els = {
  notificationTray: document.getElementById("notificationTray"),
  annotateTabButton: document.getElementById("annotateTabButton"),
  liveTabButton: document.getElementById("liveTabButton"),
  trainingTabButton: document.getElementById("trainingTabButton"),
  annotationTab: document.getElementById("annotationTab"),
  liveTab: document.getElementById("liveTab"),
  trainingTab: document.getElementById("trainingTab"),
  annotationView: document.getElementById("annotationView"),
  liveView: document.getElementById("liveView"),
  trainingView: document.getElementById("trainingView"),
  folderInput: document.getElementById("folderInput"),
  projectInput: document.getElementById("projectInput"),
  splitSelect: document.getElementById("splitSelect"),
  scanButton: document.getElementById("scanButton"),
  kindFilter: document.getElementById("kindFilter"),
  mediaCount: document.getElementById("mediaCount"),
  mediaList: document.getElementById("mediaList"),
  removeSelectedMediaButton: document.getElementById("removeSelectedMediaButton"),
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
  liveCameraProfileSelect: document.getElementById("liveCameraProfileSelect"),
  liveClearCamerasButton: document.getElementById("liveClearCamerasButton"),
  liveClearMediaButton: document.getElementById("liveClearMediaButton"),
  liveScanLocalButton: document.getElementById("liveScanLocalButton"),
  liveSourceInput: document.getElementById("liveSourceInput"),
  liveConfInput: document.getElementById("liveConfInput"),
  livePreviewFpsInput: document.getElementById("livePreviewFpsInput"),
  liveDetectionFpsInput: document.getElementById("liveDetectionFpsInput"),
  liveFrameSkipInput: document.getElementById("liveFrameSkipInput"),
  livePresetSelect: document.getElementById("livePresetSelect"),
  liveImageSizeInput: document.getElementById("liveImageSizeInput"),
  livePreviewSizeSelect: document.getElementById("livePreviewSizeSelect"),
  liveJpegQualityInput: document.getElementById("liveJpegQualityInput"),
  liveDeviceSelect: document.getElementById("liveDeviceSelect"),
  liveRecordInput: document.getElementById("liveRecordInput"),
  liveRecordLabelsInput: document.getElementById("liveRecordLabelsInput"),
  liveStartButton: document.getElementById("liveStartButton"),
  liveStopButton: document.getElementById("liveStopButton"),
  liveVideoPreview: document.getElementById("liveVideoPreview"),
  liveImagePreview: document.getElementById("liveImagePreview"),
  liveStream: document.getElementById("liveStream"),
  liveStreamGrid: document.getElementById("liveStreamGrid"),
  livePlaceholder: document.getElementById("livePlaceholder"),
  liveTitle: document.getElementById("liveTitle"),
  liveStatus: document.getElementById("liveStatus"),
  liveRefreshEventsButton: document.getElementById("liveRefreshEventsButton"),
  liveRemoveEventsButton: document.getElementById("liveRemoveEventsButton"),
  liveEventsUpdatedAt: document.getElementById("liveEventsUpdatedAt"),
  liveEventList: document.getElementById("liveEventList"),
  trainingScopeSelect: document.getElementById("trainingScopeSelect"),
  trainingFromDateInput: document.getElementById("trainingFromDateInput"),
  trainingToDateInput: document.getElementById("trainingToDateInput"),
  trainingEpochsInput: document.getElementById("trainingEpochsInput"),
  trainingImageSizeInput: document.getElementById("trainingImageSizeInput"),
  trainingBatchInput: document.getElementById("trainingBatchInput"),
  trainingDeviceSelect: document.getElementById("trainingDeviceSelect"),
  trainingPrepareOnlyInput: document.getElementById("trainingPrepareOnlyInput"),
  trainingStartButton: document.getElementById("trainingStartButton"),
  trainingStopButton: document.getElementById("trainingStopButton"),
  trainingRefreshButton: document.getElementById("trainingRefreshButton"),
  trainingStatusText: document.getElementById("trainingStatusText"),
  trainingUpdatedAt: document.getElementById("trainingUpdatedAt"),
  trainingStateValue: document.getElementById("trainingStateValue"),
  trainingDetailValue: document.getElementById("trainingDetailValue"),
  trainingDatasetValue: document.getElementById("trainingDatasetValue"),
  trainingRangeValue: document.getElementById("trainingRangeValue"),
  trainingOutputValue: document.getElementById("trainingOutputValue"),
  trainingElapsedValue: document.getElementById("trainingElapsedValue"),
  trainingProgressLabel: document.getElementById("trainingProgressLabel"),
  trainingProgressText: document.getElementById("trainingProgressText"),
  trainingProgressFill: document.getElementById("trainingProgressFill"),
  trainingLogPath: document.getElementById("trainingLogPath"),
  trainingLog: document.getElementById("trainingLog"),
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
  onTrainingScopeChanged();
  await loadLiveCameras();
  await loadLocalCameras();
  await scanFolder();
  await refreshLiveEvents();
  await refreshTrainingStatus();
  window.setInterval(refreshLiveEvents, 5000);
  window.setInterval(refreshTrainingStatus, 3000);
}

function bindEvents() {
  els.annotateTabButton.addEventListener("click", () => showTab("annotate"));
  els.liveTabButton.addEventListener("click", () => showTab("live"));
  els.trainingTabButton.addEventListener("click", () => showTab("training"));
  els.scanButton.addEventListener("click", scanFolder);
  els.folderInput.addEventListener("change", refreshStats);
  els.projectInput.addEventListener("change", refreshStats);
  els.refreshStatsButton.addEventListener("click", refreshStats);
  els.kindFilter.addEventListener("change", renderMediaList);
  els.removeSelectedMediaButton.addEventListener("click", removeSelectedMedia);
  els.captureButton.addEventListener("click", captureCurrentFrame);
  els.videoButton.addEventListener("click", showVideo);
  els.playButton.addEventListener("click", playVideo);
  els.pauseButton.addEventListener("click", () => els.video.pause());
  els.backButton.addEventListener("click", () => seekVideo(-1));
  els.forwardButton.addEventListener("click", () => seekVideo(1));
  els.liveCameraSelect.addEventListener("change", onLiveCameraChanged);
  els.liveCameraProfileSelect.addEventListener("change", stopLiveDetectionForSourceChange);
  els.liveClearCamerasButton.addEventListener("click", () => clearLiveCameraSelection());
  els.liveClearMediaButton.addEventListener("click", () => clearLiveMediaSelection());
  els.liveScanLocalButton.addEventListener("click", scanLocalCameras);
  els.liveSourceInput.addEventListener("input", onLiveCustomSourceInput);
  els.livePresetSelect.addEventListener("change", applyLivePreset);
  [els.livePreviewFpsInput, els.liveDetectionFpsInput, els.liveFrameSkipInput, els.liveImageSizeInput, els.liveJpegQualityInput].forEach((input) => {
    input.addEventListener("input", markLivePresetCustom);
  });
  els.livePreviewSizeSelect.addEventListener("change", markLivePresetCustom);
  els.liveDeviceSelect.addEventListener("change", stopLiveDetectionForSourceChange);
  els.liveStartButton.addEventListener("click", startLiveDetection);
  els.liveStopButton.addEventListener("click", stopLiveDetection);
  els.liveRefreshEventsButton.addEventListener("click", refreshLiveEvents);
  els.liveRemoveEventsButton.addEventListener("click", removeSelectedEvents);
  els.liveEventList.addEventListener("change", onLiveEventSelectionChanged);
  els.trainingScopeSelect.addEventListener("change", onTrainingScopeChanged);
  els.trainingFromDateInput.addEventListener("change", renderTrainingSelection);
  els.trainingToDateInput.addEventListener("change", renderTrainingSelection);
  els.trainingPrepareOnlyInput.addEventListener("change", updateTrainingControls);
  els.trainingStartButton.addEventListener("click", startTrainingJob);
  els.trainingStopButton.addEventListener("click", stopTrainingJob);
  els.trainingRefreshButton.addEventListener("click", refreshTrainingStatus);
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
  const training = name === "training";
  state.mode = live ? "live" : training ? "training" : "annotate";
  els.liveTab.classList.toggle("active", live);
  els.trainingTab.classList.toggle("active", training);
  els.annotationTab.classList.toggle("active", !live && !training);
  els.liveView.classList.toggle("active", live);
  els.trainingView.classList.toggle("active", training);
  els.annotationView.classList.toggle("active", !live && !training);
  els.liveTabButton.classList.toggle("active", live);
  els.trainingTabButton.classList.toggle("active", training);
  els.annotateTabButton.classList.toggle("active", !live && !training);
  renderMediaList();
  if (live) {
    const selectedMedia = selectedLiveMediaItems();
    if (selectedMedia.length && !state.liveRunning) {
      showLiveMediaSelection();
    } else if (!state.liveRunning) {
      showLivePlaceholder();
    }
  } else if (training) {
    refreshTrainingStatus();
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
  const previousValues = new Set([...els.liveCameraSelect.selectedOptions].map((option) => option.value));
  els.liveCameraSelect.innerHTML = "";

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

  [...els.liveCameraSelect.options].forEach((option) => {
    option.selected = previousValues.has(option.value);
  });
}

async function loadLocalCameras() {
  try {
    const result = await getJson("/api/live/local-cameras?max_index=5");
    if (result.error) {
      setLiveStatus(result.error);
      return;
    }
    state.liveLocalCameras = result.cameras || [];
    renderLiveCameraSelect();
    if (result.source === "cache" && state.liveLocalCameras.length) {
      setLiveStatus(`${state.liveLocalCameras.length} local cameras loaded from cache`);
    } else if (result.source === "scanning") {
      setLiveStatus("Local camera discovery is running in the background");
    } else if (result.source === "missing") {
      setLiveStatus("Local camera cache missing. Use Scan Local.");
    }
  } catch (_error) {
    setLiveStatus("Local camera cache unavailable");
  }
}

async function scanLocalCameras() {
  if (state.liveLocalScanRunning) return;
  const scanLabel = "Scanning local cameras";
  try {
    state.liveLocalScanRunning = true;
    els.liveScanLocalButton.disabled = true;
    els.liveScanLocalButton.textContent = "Scanning...";
    els.liveScanLocalButton.setAttribute("aria-busy", "true");
    setLiveStatus(`${scanLabel}...`);
    showNotification(`${scanLabel} started`, "info", 2500);
    const result = await getJson("/api/live/local-cameras?max_index=5&refresh=1");
    if (result.error) {
      setLiveStatus(result.error);
      showNotification(result.error, "error", 5000);
      return;
    }
    if (result.source === "scanning") {
      setLiveStatus(result.message || "Local camera discovery is already running");
      showNotification("Local camera discovery is already running", "info", 4000);
      return;
    }
    state.liveLocalCameras = result.cameras || [];
    renderLiveCameraSelect();
    const status = state.liveLocalCameras.length
      ? `${state.liveLocalCameras.length} local cameras found`
      : "No local cameras found";
    setLiveStatus(status);
    showNotification(`${scanLabel} finished. ${status}.`, state.liveLocalCameras.length ? "success" : "info", 4000);
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

function onLiveCameraChanged() {
  const selected = selectedLiveSourceOptions();
  if (!selected.length) {
    updateLiveMediaSourceInput();
    renderMediaList();
    stopLiveDetectionForSourceChange();
    if (selectedLiveMediaItems().length) {
      showLiveMediaSelection();
    } else {
      showLivePlaceholder();
    }
    els.liveTitle.textContent = liveSourceLabel();
    return;
  }
  clearLiveMediaSelection({ silent: true });
  els.liveSourceInput.value = selected.length === 1
    ? liveOptionSourceValue(selected[0])
    : `${selected.length} cameras selected`;
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLivePlaceholder();
  els.liveTitle.textContent = liveSourceLabel();
}

function toggleLiveMediaSource(item) {
  clearLiveCameraSelection({ silent: true });
  const existingIndex = state.liveSelectedMediaPaths.indexOf(item.path);
  if (existingIndex >= 0) {
    state.liveSelectedMediaPaths.splice(existingIndex, 1);
  } else {
    state.liveSelectedMediaPaths.push(item.path);
  }
  updateLiveMediaSourceInput();
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLiveMediaSelection();
}

function onLiveCustomSourceInput() {
  clearLiveCameraSelection({ silent: true });
  clearLiveMediaSelection({ silent: true });
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLivePlaceholder();
  els.liveTitle.textContent = liveSourceLabel();
}

function startLiveDetection() {
  if (state.liveRunning) stopLiveDetection({ restorePreview: false });
  const jobs = liveStreamJobs();
  if (!jobs.length) {
    setLiveStatus("Choose a source first");
    return;
  }
  const recordingRequested = els.liveRecordInput.checked;
  if (recordingRequested) {
    jobs.forEach((job) => {
      job.clientRunId = makeClientRunId();
    });
  }
  hideLiveMediaPreview();
  renderLiveStreamGrid(jobs);
  els.livePlaceholder.style.display = "none";
  state.liveRunning = true;
  state.liveRecording = recordingRequested;
  state.liveStreamJobs = jobs;
  els.liveStartButton.disabled = true;
  els.liveRecordInput.disabled = true;
  els.liveRecordLabelsInput.disabled = true;
  els.liveTitle.textContent = liveSourceLabel();
  setLiveStatus(state.liveRecording ? "Starting stream and recording..." : "Starting stream...");
  if (state.liveRecording) {
    showNotification("Recording requested. Waiting for backend confirmation...", "info", 5000);
    startRecordingStatusWatch(jobs);
  } else {
    stopRecordingStatusWatch();
  }
  window.setTimeout(refreshLiveEvents, 500);
}

function stopLiveDetection(options = {}) {
  const restorePreview = options.restorePreview !== false;
  const wasRecording = state.liveRecording;
  state.liveRunning = false;
  state.liveRecording = false;
  state.liveStreamJobs = [];
  stopRecordingStatusWatch();
  els.liveStream.removeAttribute("src");
  els.liveStream.style.display = "none";
  clearLiveStreamGrid();
  els.liveStartButton.disabled = false;
  els.liveRecordInput.disabled = false;
  els.liveRecordLabelsInput.disabled = false;
  if (restorePreview) {
    const selectedMedia = selectedLiveMediaItems();
    if (selectedMedia.length) {
      showLiveMediaSelection();
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

function liveStreamJobs() {
  const selected = selectedLiveSourceOptions();
  if (selected.length) {
    return selected.map((option) => ({
      key: option.value,
      label: option.textContent.trim(),
      params: liveStreamParamsForOption(option),
    }));
  }
  const mediaItems = selectedLiveMediaItems();
  if (mediaItems.length) {
    return mediaItems.map((item) => ({
      key: `media:${item.path}`,
      label: `${item.relative || item.name} · ${item.kind}`,
      params: { source: item.path },
    }));
  }
  const source = els.liveSourceInput.value.trim();
  if (!source) return [];
  return [
    {
      key: `source:${source}`,
      label: source,
      params: { source },
    },
  ];
}

function liveStreamParamsForOption(option) {
  const source = option.dataset.source || "";
  if (option.dataset.streamKind === "camera") return { camera: source };
  return { source: source || "0" };
}

function liveOptionSourceValue(option) {
  const source = option.dataset.source || "";
  if (option.dataset.streamKind === "camera") return `camera:${source}`;
  return source;
}

function liveStreamUrl(job) {
  const params = new URLSearchParams();
  Object.entries(job.params || {}).forEach(([key, value]) => {
    params.set(key, value);
  });
  if (job.clientRunId) {
    params.set("client_run_id", job.clientRunId);
  }
  params.set("model", state.liveModelPath || "data_store/models/trained/yolov8n_drone_best.pt");
  params.set("conf", els.liveConfInput.value || "0.3");
  params.set("preview_fps", els.livePreviewFpsInput.value || "4");
  params.set("detect_fps", els.liveDetectionFpsInput.value || "2");
  params.set("max_fps", els.livePreviewFpsInput.value || "4");
  params.set("frame_skip", els.liveFrameSkipInput.value || "0");
  params.set("imgsz", els.liveImageSizeInput.value || "960");
  const previewSize = livePreviewSize();
  params.set("max_width", String(previewSize.width));
  params.set("max_height", String(previewSize.height));
  params.set("quality", els.liveJpegQualityInput.value || "85");
  if (job.params && job.params.camera) {
    params.set("camera_profile", els.liveCameraProfileSelect.value || "main");
  }
  if (els.liveDeviceSelect.value) {
    params.set("device", els.liveDeviceSelect.value);
  }
  if (els.liveRecordInput.checked) {
    params.set("record", "1");
    params.set("record_dir", els.folderInput.value);
    params.set("record_max_mb", "30");
    params.set("record_name_suffix", liveRecordSuffix(job));
    if (els.liveRecordLabelsInput.checked) {
      params.set("record_labels", "1");
    }
  }
  params.set("_", Date.now().toString());
  return `/api/live/stream?${params.toString()}`;
}

function renderLiveStreamGrid(jobs) {
  clearLiveStreamGrid();
  els.liveStreamGrid.style.display = "grid";
  jobs.forEach((job) => {
    const tile = document.createElement("section");
    tile.className = "liveStreamTile";
    tile.dataset.streamKey = job.key;
    if (job.clientRunId) tile.dataset.clientRunId = job.clientRunId;

    const header = document.createElement("div");
    header.className = "liveStreamTileHeader";
    const title = document.createElement("strong");
    title.textContent = job.label || "Live source";
    const stateText = document.createElement("span");
    stateText.textContent = "Starting";
    header.append(title, stateText);

    const image = document.createElement("img");
    image.className = "liveStreamImage";
    image.alt = "";
    image.addEventListener("load", () => {
      stateText.textContent = job.clientRunId ? "Streaming · confirming recording" : "Streaming";
      if (state.liveRunning) setLiveStatus(`${job.label || "Source"} streaming`);
    });
    image.addEventListener("error", () => {
      stateText.textContent = "Unavailable";
      if (state.liveRunning) setLiveStatus(`${job.label || "Source"} unavailable`);
    });
    image.src = liveStreamUrl(job);

    tile.append(header, image);
    els.liveStreamGrid.appendChild(tile);
  });
}

function clearLiveStreamGrid() {
  [...els.liveStreamGrid.querySelectorAll("img")].forEach((image) => image.removeAttribute("src"));
  els.liveStreamGrid.innerHTML = "";
  els.liveStreamGrid.style.display = "none";
}

function liveRecordSuffix(job) {
  if (job.params && job.params.camera) return `camera_${job.params.camera}`;
  const source = String((job.params && job.params.source) || "");
  if (/^\d+$/.test(source)) return `camera_${source}`;
  try {
    const url = new URL(source);
    if (url.hostname) return `source_${url.hostname}`;
  } catch (_error) {
    // Not a URL; fall back to the file/source basename.
  }
  const basename = source.split(/[\\/]/).filter(Boolean).pop() || job.label || "source";
  return `source_${basename}`;
}

function makeClientRunId() {
  const random = Math.random().toString(36).slice(2, 10);
  return `live_${Date.now()}_${random}`;
}

function startRecordingStatusWatch(jobs) {
  stopRecordingStatusWatch();
  const checks = jobs
    .filter((job) => job.clientRunId)
    .map((job) => [
      job.clientRunId,
      {
        label: job.label || "Live source",
        startedAt: Date.now(),
        status: "pending",
      },
    ]);
  state.liveRecordingChecks = new Map(checks);
  scheduleRecordingStatusPoll(400);
}

function stopRecordingStatusWatch() {
  if (state.liveRecordingPollTimer) {
    window.clearTimeout(state.liveRecordingPollTimer);
    state.liveRecordingPollTimer = null;
  }
  state.liveRecordingChecks.clear();
}

function scheduleRecordingStatusPoll(delayMs = 800) {
  if (!state.liveRunning || !state.liveRecording || !state.liveRecordingChecks.size) return;
  if (state.liveRecordingPollTimer) window.clearTimeout(state.liveRecordingPollTimer);
  state.liveRecordingPollTimer = window.setTimeout(pollRecordingStatus, delayMs);
}

async function pollRecordingStatus() {
  state.liveRecordingPollTimer = null;
  if (!state.liveRunning || !state.liveRecording || !state.liveRecordingChecks.size) return;
  try {
    const result = await getJson("/api/live/events?limit=200");
    const events = Array.isArray(result.events) ? result.events.slice().reverse() : [];
    for (const event of events) {
      const runId = String(event.client_run_id || "");
      const check = state.liveRecordingChecks.get(runId);
      if (!check) continue;
      handleRecordingStatusEvent(runId, check, event);
    }
    expirePendingRecordingChecks();
  } catch (_error) {
    setLiveStatus("Recording status check failed");
  }
  if (state.liveRecordingChecks.size) {
    const hasPending = [...state.liveRecordingChecks.values()].some((check) => check.status === "pending");
    scheduleRecordingStatusPoll(hasPending ? 800 : 2500);
  }
}

function handleRecordingStatusEvent(runId, check, event) {
  const type = String(event.event_type || "");
  if (type === "recording_started" && check.status === "pending") {
    check.status = "confirmed";
    updateRecordingTileStatus(runId, "Streaming · recording");
    const path = shortSource(event.recording_path || event.recording_dir || "");
    const mode = event.labels ? "Labeled recording" : "Raw recording";
    showNotification(`${mode} in progress${path ? `: ${path}` : ""}`, "success", 6000);
    setLiveStatus(`${check.label} recording in progress`);
    return;
  }
  if (type === "recording_failed" || type === "recording_skipped" || isRecordingErrorEvent(event)) {
    check.status = "failed";
    updateRecordingTileStatus(runId, "Streaming · recording failed");
    showNotification(`${check.label}: ${event.message || event.reason || "recording failed"}`, "error", 8000);
    setLiveStatus(`${check.label} recording failed`);
    state.liveRecordingChecks.delete(runId);
    return;
  }
  if (type === "stop" && check.status === "pending") {
    check.status = "failed";
    updateRecordingTileStatus(runId, "Stopped · recording not confirmed");
    showNotification(`${check.label}: stream stopped before recording was confirmed`, "error", 8000);
    state.liveRecordingChecks.delete(runId);
  }
}

function expirePendingRecordingChecks() {
  const now = Date.now();
  for (const [runId, check] of state.liveRecordingChecks.entries()) {
    if (check.status !== "pending") continue;
    if (now - check.startedAt < 12000) continue;
    check.status = "timeout";
    updateRecordingTileStatus(runId, "Streaming · recording unconfirmed");
    showNotification(`${check.label}: recording confirmation not received`, "error", 8000);
    setLiveStatus(`${check.label} recording confirmation not received`);
    state.liveRecordingChecks.delete(runId);
  }
}

function isRecordingErrorEvent(event) {
  if (String(event.event_type || "") !== "error") return false;
  return /record/i.test(String(event.message || event.reason || ""));
}

function updateRecordingTileStatus(runId, text) {
  const tile = [...els.liveStreamGrid.querySelectorAll(".liveStreamTile")].find(
    (candidate) => candidate.dataset.clientRunId === runId,
  );
  const stateText = tile ? tile.querySelector(".liveStreamTileHeader span") : null;
  if (stateText) stateText.textContent = text;
}

function applyLivePreset() {
  const presets = {
    balanced: { previewFps: "6", detectionFps: "2", skip: "0", imageSize: "640", previewSize: "1280x720", quality: "75" },
    fast: { previewFps: "10", detectionFps: "1.5", skip: "0", imageSize: "416", previewSize: "854x480", quality: "65" },
    quality: { previewFps: "4", detectionFps: "2", skip: "0", imageSize: "960", previewSize: "1920x1080", quality: "85" },
  };
  const preset = presets[els.livePresetSelect.value];
  if (!preset) return;
  els.livePreviewFpsInput.value = preset.previewFps;
  els.liveDetectionFpsInput.value = preset.detectionFps;
  els.liveFrameSkipInput.value = preset.skip;
  els.liveImageSizeInput.value = preset.imageSize;
  els.livePreviewSizeSelect.value = preset.previewSize;
  els.liveJpegQualityInput.value = preset.quality;
  stopLiveDetectionForSourceChange();
}

function markLivePresetCustom() {
  els.livePresetSelect.value = "custom";
  stopLiveDetectionForSourceChange();
}

function livePreviewSize() {
  const raw = els.livePreviewSizeSelect.value || "1280x720";
  const [width, height] = raw.split("x").map((value) => Number.parseInt(value, 10));
  return {
    width: Number.isFinite(width) ? width : 1280,
    height: Number.isFinite(height) ? height : 720,
  };
}

function onTrainingScopeChanged() {
  const isDateRange = els.trainingScopeSelect.value === "date-range";
  els.trainingFromDateInput.disabled = !isDateRange;
  els.trainingToDateInput.disabled = !isDateRange;
  renderTrainingSelection();
  updateTrainingControls();
}

function trainingPayload() {
  const scope = els.trainingScopeSelect.value || "since-last";
  const payload = {
    dataset_scope: scope,
    project_dir: els.projectInput.value,
    model: state.liveModelPath || "data_store/models/trained/yolov8n_drone_best.pt",
    output_model: state.liveModelPath || "data_store/models/trained/yolov8n_drone_best.pt",
    epochs: els.trainingEpochsInput.value || "25",
    imgsz: els.trainingImageSizeInput.value || "640",
    batch: els.trainingBatchInput.value || "8",
    device: els.trainingDeviceSelect.value || "",
    prepare_only: els.trainingPrepareOnlyInput.checked,
  };
  if (scope === "date-range") {
    payload.from_date = els.trainingFromDateInput.value;
    payload.to_date = els.trainingToDateInput.value;
  }
  return payload;
}

async function startTrainingJob() {
  const payload = trainingPayload();
  if (payload.dataset_scope === "date-range" && !payload.from_date && !payload.to_date) {
    setTrainingStatus("Choose a date range");
    showNotification("Training date range needs From, To, or both", "error", 4500);
    return;
  }

  els.trainingStartButton.disabled = true;
  setTrainingStatus(payload.prepare_only ? "Preparing dataset..." : "Starting training...");
  try {
    const result = await postJson("/api/training/start", payload);
    if (result.error) {
      setTrainingStatus(result.error);
      showNotification(result.error, "error", 6000);
      renderTrainingStatus(result);
      return;
    }
    renderTrainingStatus(result);
    showNotification(payload.prepare_only ? "Dataset preparation started" : "Training started", "info", 3000);
  } catch (_error) {
    setTrainingStatus("Training request failed");
    showNotification("Training request failed", "error", 5000);
  } finally {
    updateTrainingControls();
  }
}

async function stopTrainingJob() {
  try {
    setTrainingStatus("Stopping training...");
    const result = await postJson("/api/training/stop", {});
    renderTrainingStatus(result);
  } catch (_error) {
    setTrainingStatus("Stop request failed");
  }
}

async function refreshTrainingStatus() {
  try {
    const result = await getJson("/api/training/status");
    renderTrainingStatus(result);
  } catch (_error) {
    setTrainingStatus("Training status unavailable");
  }
}

function renderTrainingStatus(result) {
  const job = result && result.job ? result.job : null;
  const progress = result && result.progress ? result.progress : { current: 0, total: 0, percent: 0 };
  state.trainingJob = job;
  const running = Boolean(result && result.running);
  const status = job ? String(job.status || result.status || "unknown") : "idle";
  const percent = Number(progress.percent || 0);

  els.trainingStatusText.textContent = statusText(status);
  els.trainingUpdatedAt.textContent = `Updated ${formatClock(new Date().toISOString())}`;
  els.trainingStateValue.textContent = statusText(status);
  els.trainingStateValue.className = `trainingState ${status}`;
  els.trainingDetailValue.textContent = jobDetailText(job);
  els.trainingProgressText.textContent = progress.total
    ? `${formatInteger(progress.current)} / ${formatInteger(progress.total)} epochs · ${percent.toFixed(percent % 1 ? 1 : 0)}%`
    : `${percent.toFixed(0)}%`;
  els.trainingProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  els.trainingLog.textContent = result && result.log ? result.log : "No training output yet.";
  els.trainingLogPath.textContent = job && job.log_path ? job.log_path : "No log file";
  if (state.mode === "training" && running) {
    els.trainingLog.scrollTop = els.trainingLog.scrollHeight;
  }

  if (job) {
    els.trainingDatasetValue.textContent = datasetScopeLabel(job.dataset_scope || els.trainingScopeSelect.value);
    els.trainingRangeValue.textContent = jobRangeText(job);
    els.trainingOutputValue.textContent = shortSource(job.output_model || "yolov8n_drone_best.pt");
    els.trainingElapsedValue.textContent = `Elapsed ${formatDuration(Number(job.elapsed_seconds || 0))}`;
  } else {
    renderTrainingSelection();
  }
  updateTrainingControls(Boolean(running));
}

function renderTrainingSelection() {
  const scope = els.trainingScopeSelect.value || "since-last";
  els.trainingDatasetValue.textContent = datasetScopeLabel(scope);
  if (scope === "date-range") {
    const from = els.trainingFromDateInput.value || "any start";
    const to = els.trainingToDateInput.value || "any end";
    els.trainingRangeValue.textContent = `${from} to ${to}`;
  } else if (scope === "all") {
    els.trainingRangeValue.textContent = "Uses every reviewed annotation in the project";
  } else {
    els.trainingRangeValue.textContent = "Uses annotations saved after the previous training cutoff";
  }
  els.trainingOutputValue.textContent = shortSource(state.liveModelPath || "yolov8n_drone_best.pt");
}

function updateTrainingControls(running = state.trainingJob && ["running", "starting", "stopping"].includes(state.trainingJob.status)) {
  els.trainingStartButton.disabled = Boolean(running);
  els.trainingStopButton.disabled = !running;
  els.trainingStartButton.textContent = els.trainingPrepareOnlyInput.checked ? "Prepare" : "Train";
}

function setTrainingStatus(text) {
  els.trainingStatusText.textContent = text;
}

function statusText(status) {
  const text = String(status || "idle").replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function jobDetailText(job) {
  if (!job) return "No training job yet";
  const mode = job.prepare_only ? "prepare only" : `${formatInteger(job.epochs)} epochs`;
  const device = job.device || "auto";
  return `${mode} · ${formatInteger(job.imgsz)}px · batch ${formatInteger(job.batch)} · ${device}`;
}

function jobRangeText(job) {
  const scope = String(job.dataset_scope || "");
  if (scope === "date-range") {
    return `${job.from_date || "any start"} to ${job.to_date || "any end"}`;
  }
  if (scope === "all") return "Every reviewed annotation";
  return "After previous training metadata/model timestamp";
}

function datasetScopeLabel(scope) {
  if (scope === "all") return "All annotations";
  if (scope === "date-range") return "Date range";
  return "Since last training";
}

function liveSourceLabel() {
  const selected = selectedLiveSourceOptions();
  if (selected.length === 1) return selected[0].textContent;
  if (selected.length > 1) return `${selected.length} selected camera sources`;
  const selectedMedia = selectedLiveMediaItems();
  if (selectedMedia.length === 1) return `${selectedMedia[0].relative || selectedMedia[0].name} · ${selectedMedia[0].kind}`;
  if (selectedMedia.length > 1) return `${selectedMedia.length} selected media sources`;
  return els.liveSourceInput.value || "Custom source";
}

function selectedLiveSourceOptions() {
  return [...els.liveCameraSelect.selectedOptions].filter((option) => option.value && !option.disabled);
}

function selectedLiveSourceOption() {
  return selectedLiveSourceOptions()[0] || null;
}

function clearLiveCameraSelection(options = {}) {
  [...els.liveCameraSelect.options].forEach((option) => {
    option.selected = false;
  });
  if (options.silent) return;
  onLiveCameraChanged();
}

function selectedLiveMediaItem() {
  return selectedLiveMediaItems()[0] || null;
}

function selectedLiveMediaItems() {
  return state.liveSelectedMediaPaths
    .map((path) => state.media.find((item) => item.path === path))
    .filter(Boolean);
}

function clearLiveMediaSelection(options = {}) {
  state.liveSelectedMediaPaths = [];
  if (options.silent) return;
  updateLiveMediaSourceInput();
  renderMediaList();
  stopLiveDetectionForSourceChange();
  showLivePlaceholder();
}

function updateLiveMediaSourceInput() {
  const items = selectedLiveMediaItems();
  if (items.length === 1) {
    els.liveSourceInput.value = items[0].path;
  } else if (items.length > 1) {
    els.liveSourceInput.value = `${items.length} media sources selected`;
  } else {
    els.liveSourceInput.value = "";
  }
}

function showLiveMediaSelection() {
  const selectedMedia = selectedLiveMediaItems();
  if (selectedMedia.length === 1) {
    showLiveMediaPreview(selectedMedia[0]);
    return;
  }
  showLivePlaceholder();
  els.liveTitle.textContent = liveSourceLabel();
  if (selectedMedia.length > 1) {
    setLiveStatus(`${selectedMedia.length} media sources selected`);
  }
}

function showLiveMediaPreview(item) {
  els.liveStream.removeAttribute("src");
  els.liveStream.style.display = "none";
  clearLiveStreamGrid();
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
  clearLiveStreamGrid();
  els.livePlaceholder.style.display = "flex";
}

async function scanFolder(options = {}) {
  setStatus("Scanning...");
  const result = await postJson("/api/scan", { folder: els.folderInput.value });
  if (result.error) {
    setStatus(result.error);
    return;
  }
  state.media = result.media || [];
  state.currentIndex = -1;
  state.current = null;
  const availablePaths = new Set(state.media.map((item) => item.path));
  const previousMediaSelectionCount = state.liveSelectedMediaPaths.length;
  state.liveSelectedMediaPaths = state.liveSelectedMediaPaths.filter((path) => availablePaths.has(path));
  if (previousMediaSelectionCount && !state.liveSelectedMediaPaths.length) {
    updateLiveMediaSourceInput();
    showLivePlaceholder();
  }
  renderMediaList();
  await refreshStats();
  const autoSelect = options.autoSelect !== false;
  if (!state.filtered.length) {
    setStatus("No media found");
    return;
  }
  if (!autoSelect) {
    setStatus("Ready");
    return;
  }
  if (state.mode === "live") {
    if (!selectedLiveSourceOptions().length && !state.liveSelectedMediaPaths.length) {
      toggleLiveMediaSource(state.filtered[0]);
    }
  } else {
    selectMedia(0);
  }
}

function renderMediaList() {
  const kind = els.kindFilter.value;
  state.filtered = filteredMedia(kind);
  els.mediaCount.textContent = `${state.filtered.length} files`;
  const live = state.mode === "live";
  renderMediaItems(els.mediaList, state.filtered, {
    isActive: (item) => live
      ? state.liveSelectedMediaPaths.includes(item.path)
      : state.current && state.current.path === item.path,
    onSelect: (item, index) => {
      if (live) {
        toggleLiveMediaSource(item);
      } else {
        selectMedia(index);
      }
    },
  });
  updateRemoveMediaButton();
}

function selectedMediaForRemoval() {
  if (state.mode === "live") return selectedLiveMediaItems();
  return state.current ? [state.current] : [];
}

function updateRemoveMediaButton() {
  const selectedCount = selectedMediaForRemoval().length;
  els.removeSelectedMediaButton.disabled = selectedCount === 0;
  els.removeSelectedMediaButton.textContent = selectedCount > 1
    ? `Remove ${selectedCount}`
    : "Remove Selected";
}

async function removeSelectedMedia() {
  const items = selectedMediaForRemoval();
  if (!items.length) return;
  const label = items.length === 1 ? items[0].name : `${items.length} media files`;
  if (!window.confirm(`Move ${label} to data_store/trash/media?`)) return;
  if (state.liveRunning) stopLiveDetection({ restorePreview: false });

  els.removeSelectedMediaButton.disabled = true;
  try {
    const result = await postJson("/api/media/remove", {
      folder: els.folderInput.value,
      paths: items.map((item) => item.path),
    });
    if (result.error) {
      showNotification(result.error, "error", 6000);
      return;
    }
    clearRemovedMediaSelections(items.map((item) => item.path));
    await scanFolder({ autoSelect: false });
    const removed = (result.removed || []).length;
    const failed = (result.failed || []).length;
    showNotification(
      failed ? `Moved ${removed}; ${failed} failed` : `Moved ${removed} media file${removed === 1 ? "" : "s"} to trash`,
      failed ? "error" : "success",
      5000,
    );
  } catch (_error) {
    showNotification("Media removal failed", "error", 5000);
  } finally {
    updateRemoveMediaButton();
  }
}

function clearRemovedMediaSelections(paths) {
  const removed = new Set(paths);
  state.liveSelectedMediaPaths = state.liveSelectedMediaPaths.filter((path) => !removed.has(path));
  if (state.current && removed.has(state.current.path)) {
    state.current = null;
    state.currentIndex = -1;
    state.boxes = [];
    state.naturalWidth = 0;
    state.naturalHeight = 0;
    state.frameTime = null;
    els.currentName.textContent = "No media selected";
    els.video.pause();
    els.video.removeAttribute("src");
    els.video.style.display = "none";
    els.image.removeAttribute("src");
    els.image.style.display = "none";
    els.canvas.style.display = "none";
    setStatus("Media removed");
    draw();
  }
  updateLiveMediaSourceInput();
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
    const eventIds = new Set(state.liveEvents.map((event) => event.event_id).filter(Boolean));
    state.selectedLiveEventIds = new Set([...state.selectedLiveEventIds].filter((eventId) => eventIds.has(eventId)));
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
    updateRemoveEventsButton();
    return;
  }

  els.liveEventList.innerHTML = state.liveEvents.map((event) => liveEventMarkup(event)).join("");
  updateRemoveEventsButton();
}

function liveEventMarkup(event) {
  const type = String(event.event_type || "event");
  const eventId = String(event.event_id || "");
  const source = liveEventSourceText(event);
  const best = event.best_track || {};
  const checked = eventId && state.selectedLiveEventIds.has(eventId) ? " checked" : "";
  const image = event.image_path
    ? `<img class="eventThumb" src="${mediaUrl(event.image_path)}" alt="">`
    : "";
  const details = eventDetailText(type, event, best);
  const imageClass = image ? "" : " noImage";
  return `
    <article class="eventItem ${eventClass(type)}${imageClass}">
      <label class="eventSelect" title="Select event">
        <input type="checkbox" data-event-id="${escapeHtml(eventId)}"${checked}>
      </label>
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

function onLiveEventSelectionChanged(event) {
  const input = event.target.closest("input[data-event-id]");
  if (!input) return;
  const eventId = input.dataset.eventId;
  if (!eventId) return;
  if (input.checked) {
    state.selectedLiveEventIds.add(eventId);
  } else {
    state.selectedLiveEventIds.delete(eventId);
  }
  updateRemoveEventsButton();
}

function updateRemoveEventsButton() {
  const selectedCount = state.selectedLiveEventIds.size;
  els.liveRemoveEventsButton.disabled = selectedCount === 0;
  els.liveRemoveEventsButton.textContent = selectedCount > 1
    ? `Remove ${selectedCount}`
    : "Remove Selected";
}

async function removeSelectedEvents() {
  const eventIds = [...state.selectedLiveEventIds];
  if (!eventIds.length) return;
  const label = eventIds.length === 1 ? "1 event" : `${eventIds.length} events`;
  if (!window.confirm(`Remove ${label} from the live event log?`)) return;

  els.liveRemoveEventsButton.disabled = true;
  try {
    const result = await postJson("/api/live/events/remove", { event_ids: eventIds });
    if (result.error) {
      showNotification(result.error, "error", 6000);
      return;
    }
    state.selectedLiveEventIds.clear();
    await refreshLiveEvents();
    const removed = (result.removed || []).length;
    const failed = (result.failed || []).length;
    showNotification(
      failed ? `Removed ${removed}; ${failed} failed` : `Removed ${removed} event${removed === 1 ? "" : "s"}`,
      failed ? "error" : "success",
      5000,
    );
  } catch (_error) {
    showNotification("Event removal failed", "error", 5000);
  } finally {
    updateRemoveEventsButton();
  }
}

function liveEventSourceText(event) {
  const source = shortSource(event.source || "");
  const sourceId = String(event.source_id || "");
  if (sourceId && source && sourceId !== source) return `${sourceId} · ${source}`;
  return sourceId || source;
}

function eventTitle(type) {
  if (type === "drone_detected") return "Drone detected";
  if (type === "drone_in_frame") return "Drone entered frame";
  if (type === "drone_out_frame") return "Drone left frame";
  if (type === "recording_started") return "Recording started";
  if (type === "recording_saved") return "Recording saved";
  if (type === "recording_skipped") return "Recording skipped";
  if (type === "recording_failed") return "Recording failed";
  if (type === "detector_ready") return "Detector ready";
  if (type === "source_reconnect") return "Source reconnect";
  if (type === "source_reconnected") return "Source reconnected";
  return type.replaceAll("_", " ");
}

function eventClass(type) {
  if (type === "drone_detected" || type === "drone_in_frame") return "alert";
  if (type === "error" || type === "recording_failed") return "error";
  return "";
}

function eventDetailText(type, event, best) {
  if (type === "drone_detected" || type === "drone_in_frame") {
    const confidence = Number(best.confidence || 0);
    const trackId = best.track_id ? ` #${best.track_id}` : "";
    return `${best.label || "drone"}${trackId} ${confidence.toFixed(2)} · frame ${formatInteger(event.frame_index)}`;
  }
  if (type === "drone_out_frame") {
    const duration = Number(event.duration_seconds || 0);
    const absence = Number(event.absence_seconds || 0);
    return `last seen frame ${formatInteger(event.last_seen_frame_index)} · ${duration.toFixed(1)}s visible · ${absence.toFixed(1)}s absent`;
  }
  if (type === "stop") {
    return `${event.reason || "stopped"} · ${formatInteger(event.frames_seen)} frames · ${formatInteger(event.detection_events)} detections`;
  }
  if (type === "start") {
    const settings = event.settings || {};
    return `${event.source_kind || "source"} · ${settings.device || "auto"} · ${settings.image_size || "img"}px`;
  }
  if (type === "recording_started") {
    return `${event.labels ? "labeled demo" : "raw"} · ${event.max_size_mb || 30} MB max segment`;
  }
  if (type === "recording_saved") {
    return `${event.labels ? "labeled demo" : "raw"} · ${formatBytes(Number(event.size_bytes || 0))} · ${shortSource(event.recording_path || "")}`;
  }
  if (type === "recording_failed") {
    return String(event.message || event.reason || "recording failed");
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

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds || 0)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  if (hours) return `${hours}h ${minutes}m ${rest}s`;
  if (minutes) return `${minutes}m ${rest}s`;
  return `${rest}s`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
