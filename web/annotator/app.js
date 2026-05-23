const state = {
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
};

const els = {
  folderInput: document.getElementById("folderInput"),
  projectInput: document.getElementById("projectInput"),
  splitSelect: document.getElementById("splitSelect"),
  scanButton: document.getElementById("scanButton"),
  kindFilter: document.getElementById("kindFilter"),
  mediaCount: document.getElementById("mediaCount"),
  mediaList: document.getElementById("mediaList"),
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
};

const ctx = els.canvas.getContext("2d");
const baseCtx = state.baseCanvas.getContext("2d");

init();

async function init() {
  const defaults = await getJson("/api/defaults");
  els.folderInput.value = defaults.folder;
  els.projectInput.value = defaults.project_dir;
  bindEvents();
  await scanFolder();
}

function bindEvents() {
  els.scanButton.addEventListener("click", scanFolder);
  els.kindFilter.addEventListener("change", renderMediaList);
  els.captureButton.addEventListener("click", captureCurrentFrame);
  els.videoButton.addEventListener("click", showVideo);
  els.playButton.addEventListener("click", playVideo);
  els.pauseButton.addEventListener("click", () => els.video.pause());
  els.backButton.addEventListener("click", () => seekVideo(-1));
  els.forwardButton.addEventListener("click", () => seekVideo(1));
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
  renderMediaList();
  if (state.filtered.length) {
    selectMedia(0);
  } else {
    setStatus("No media found");
  }
}

function renderMediaList() {
  const kind = els.kindFilter.value;
  state.filtered = state.media.filter((item) => kind === "all" || item.kind === kind);
  els.mediaCount.textContent = `${state.filtered.length} files`;
  els.mediaList.innerHTML = "";

  state.filtered.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = "mediaItem";
    if (state.current && state.current.path === item.path) button.classList.add("active");
    button.innerHTML = `
      <span class="mediaName">${escapeHtml(item.name)}</span>
      <span class="mediaMeta">${item.kind} · ${formatBytes(item.size)}</span>
    `;
    button.addEventListener("click", () => selectMedia(index));
    els.mediaList.appendChild(button);
  });
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
  const url = mediaUrl(item.path);
  els.image.style.display = "none";
  els.canvas.style.display = "none";
  els.video.style.display = "block";
  els.video.src = url;
  els.video.load();
  setStatus("Video ready");
}

function loadImage(item) {
  const url = mediaUrl(item.path);
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

  const viewer = document.querySelector(".viewer");
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
  const x = box.x1 * state.scale;
  const y = box.y1 * state.scale;
  const w = (box.x2 - box.x1) * state.scale;
  const h = (box.y2 - box.y1) * state.scale;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.font = "13px system-ui";
  ctx.fillText("drone", x + 4, Math.max(15, y - 5));
}

function onPointerDown(event) {
  if (!state.naturalWidth) return;
  state.drawing = true;
  state.start = eventToImagePoint(event);
  state.cursor = state.start;
}

function onPointerMove(event) {
  if (!state.drawing) return;
  state.cursor = eventToImagePoint(event);
  draw();
}

function onPointerUp(event) {
  if (!state.drawing || !state.start) return;
  state.cursor = eventToImagePoint(event);
  const box = normaliseBox({ x1: state.start.x, y1: state.start.y, x2: state.cursor.x, y2: state.cursor.y });
  if (box.x2 - box.x1 > 3 && box.y2 - box.y1 > 3) {
    state.boxes.push(box);
  }
  state.drawing = false;
  state.start = null;
  state.cursor = null;
  draw();
}

function eventToImagePoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  const x = clamp(((event.clientX - rect.left) / rect.width) * state.naturalWidth, 0, state.naturalWidth - 1);
  const y = clamp(((event.clientY - rect.top) / rect.height) * state.naturalHeight, 0, state.naturalHeight - 1);
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

function mediaUrl(path) {
  return `/api/media?path=${encodeURIComponent(path)}`;
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
