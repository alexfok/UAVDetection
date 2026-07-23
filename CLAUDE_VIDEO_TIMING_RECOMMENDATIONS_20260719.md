# UAVDetection video timing — architecture recommendations

Prepared: 2026-07-19
Author: Claude (Sonnet 4.6), commissioned by Alex Fok
Input document: `CLAUDE_VIDEO_TIMING_CONTEXT_20260719.md`
Base commit reviewed: `950550a11af27a609c6a3eb7db986133ebf8dfd2`

---

## 1. Root-cause confirmation

The root-cause model in the context document is correct and complete.

**Core failure:** category confusion between wall-clock time and frame-production cadence.

- `cv2.VideoWriter` declared at 16 FPS interprets every `write()` call as consuming exactly `1/16 s` of encoded time.
- The server loop produced ~9 FPS (8.8–9.1 FPS from segment measurements).
- Encoded duration = `frame_count / declared_fps` ≈ `550 / 16` ≈ 34 s; wall time was 61 s.
- Segment rollover on wall time was correct; the container timeline was wrong.
- No capture timestamp was ever propagated to the muxer, so it had no way to know the actual cadence.

**Key clarification:** the problem is not that 9 FPS is "too slow" — it is that the declared FPS did not match the achieved FPS. If 9 FPS had been declared and achieved, recording would have been correct.

The CFR slot-fill mitigation (commit `bf10dd2`) is a valid symptom suppressor: it keeps `frame_count / declared_fps ≈ elapsed wall time` at the cost of repeated pixels. It does not restore lost temporal resolution. That tradeoff is appropriate for the stated goal (truthful duration), provided the duplication rate and CPU impact remain bounded (concern #2 in the context document).

---

## 2. Target architecture

Clock ownership at each boundary:

```
Camera RTSP stream
  │
  │  [packet PTS — camera clock]
  ▼
Source reader  (cap.read())
  │
  │  capture_ts = time.monotonic()  ← set ONCE here, immediately after read()
  ▼
LatestFrameStore ──────────────────────────────► Recorder (owns its own timeline)
  │                                                │
  │  frame + capture_ts sampled                   │  PTS = capture_ts − session_start
  ▼                                                │  cv2.VideoWriter at measured record_fps
Preview / Detection loop                           ▼
  │                                          MP4 segment (duration ≈ wall time)
  │  JPEG → TLS multipart
  ▼
Browser <img>  (MJPEG, real-time)     Browser <video>  (encoded cadence, correct)
                                      + requestVideoFrameCallback overlay canvas
```

**Invariants:**
- `capture_ts` is set once, by the thread calling `cap.read()`, immediately after the call returns.
- The recorder's slot calculation uses `capture_ts` from the frame, never a new `monotonic()` call at write time.
- `preview_fps` and `record_fps` are separate parameters with separate semantics.
- Browser `<video>` + `requestVideoFrameCallback` already delegates encoded timeline to the codec — preserve this.

---

## 3. Minimal patch plan

**Scope:** stay on OpenCV, no new dependencies, deployable in hours.
**Goal:** make CFR duplication correct and bounded; decouple `record_fps` from `preview_fps`.

### Changes

1. **Attach `capture_ts` at acquisition.** Set `time.monotonic()` immediately after `cap.read()`. Propagate it through `LatestFrameCapture` (add a field).

2. **Use frame's `capture_ts` in `StreamRecorder.write()`.** Replace the current `time.monotonic()` call inside `write()` with the frame's own timestamp. Eliminates jitter from delivery delay.

3. **Separate `record_fps` from `preview_fps`.** Accept `record_fps` in the recorder constructor. Derive it from `cap.get(cv2.CAP_PROP_FPS)` at session start, with a measured-FPS fallback (median of first 30 inter-frame intervals). Do not use the UI Preview FPS value as the recording FPS.

4. **Cap catch-up writes.** Limit `frames_due` per `write()` call to `int(record_fps * 2)`. If a stall exceeds 2 s worth of frames, log it and accept the gap rather than emitting a burst.

5. **Fill final tail on close.** On `StreamRecorder.close(stop_ts)`, fill slots from the last written PTS to `stop_ts`. Fixes concern #5 (final tail gap).

6. **Log per-segment diagnostics.** After each close: declared FPS, measured FPS (`frame_count / actual_duration`), frame count, duplicate count, stop reason. Allows validation without a media probe.

### Tradeoffs on Jetson ARM64

| Aspect | Assessment |
|---|---|
| New dependencies | None |
| Build/deploy step | None beyond normal code deploy |
| Duration accuracy | ±`1/record_fps` (e.g., ±62 ms at 16 FPS) |
| Catch-up CPU spike | Eliminated by the cap |
| Frame content | Still downstream of server loop (detection, resize); loop stall = no new frames |
| Camera FPS reliability | `CAP_PROP_FPS` may return 0 or wrong value — fallback is required (see §6) |

---

## 4. Robust patch plan

**Scope:** FFmpeg/GStreamer parallel recorder, independent of inference/preview loop.
**Goal:** preserve camera timing regardless of inference load.

### Architecture change

Add a `RawSegmentRecorder` thread that reads from the camera independently and writes via:
- **Stream-copy (preferred for H.264/H.265 RTSP):** `ffmpeg -rtsp_transport tcp -i $URL -c copy -f segment -segment_time 60 -segment_list playlist.m3u8 out%03d.mp4` — preserves original PTS, zero re-encode cost.
- **Transcode with passthrough PTS (fallback):** `ffmpeg -i pipe:0 -vsync passthrough` fed decoded frames via stdin or PyAV (`av.open()`).

Write a sidecar JSON at segment close: `{start_monotonic, stop_monotonic, start_wall, frame_count, declared_fps, measured_fps, drop_count, dup_count, stop_reason}`.

### Tradeoffs on Jetson ARM64

| Aspect | Assessment |
|---|---|
| New dependencies | FFmpeg (likely already installed); PyAV (`pip install av`, aarch64 wheel available) |
| Two RTSP connections | Risk: some cameras limit concurrent clients or add jitter to the second session — **must be tested** |
| CPU/GPU | Stream-copy avoids re-encode, saving inference budget |
| Crash recovery | Recorder subprocess needs a watchdog thread; systemd `Restart` does not cover subprocesses |
| PTS accuracy | Original camera timestamps preserved end-to-end |
| Deployment complexity | Higher; subprocess lifecycle management required |

### When to prefer robust

- Recording is used as independent evidence (after-action review, operational log).
- Inference load pushes preview loop below 5 FPS regularly.
- Timestamp accuracy better than `±1/fps` is required.

---

## 5. Specific code changes

### `scripts/annotation_server.py`

#### `LatestFrameCapture` (around L817–L853) — add `capture_ts`

```python
LatestFrameCapture = namedtuple(
    'LatestFrameCapture', ['frame', 'index', 'capture_ts']
)

# acquisition loop, immediately after cap.read():
ret, frame = cap.read()
ts = time.monotonic()
latest = LatestFrameCapture(frame=frame, index=idx, capture_ts=ts)
```

#### `StreamRecorder.__init__` (L2448–) — accept `record_fps`, add cap and counters

```python
def __init__(self, path, record_fps, frame_size, ...):
    self._fps = record_fps          # not preview_fps
    self._max_catchup_frames = int(record_fps * 2)
    self._dup_count = 0
    self._last_frame = None
    self._segment_start_ts = None
    self._last_capture_ts = None
    self._frames_written = 0
```

#### `StreamRecorder.write()` (L2586–L2591 and surrounding) — use `capture_ts`, cap catch-up

```python
def write(self, frame, capture_ts):
    if self._segment_start_ts is None:
        self._segment_start_ts = capture_ts
    if self._last_frame is None:
        self._last_frame = frame

    frames_due = (
        int((capture_ts - self._segment_start_ts) * self._fps) + 1
        - self._frames_written
    )
    frames_due = min(frames_due, self._max_catchup_frames)

    if frames_due <= 0:
        self._last_frame = frame
        return

    for _ in range(frames_due - 1):
        self._writer.write(self._last_frame)
        self._frames_written += 1
        self._dup_count += 1

    self._writer.write(frame)
    self._frames_written += 1
    self._last_frame = frame
    self._last_capture_ts = capture_ts
```

#### `StreamRecorder.close()` — fill final tail, log diagnostics

```python
def close(self, stop_ts=None):
    if stop_ts is not None and self._last_frame is not None and self._segment_start_ts is not None:
        tail = (
            int((stop_ts - self._segment_start_ts) * self._fps) + 1
            - self._frames_written
        )
        tail = min(max(tail, 0), self._max_catchup_frames)
        for _ in range(tail):
            self._writer.write(self._last_frame)
            self._frames_written += 1
            self._dup_count += 1
    self._writer.release()
    self._log_summary()

def _log_summary(self):
    if not self._frames_written or not self._segment_start_ts:
        return
    actual_dur = self._frames_written / self._fps
    logger.info(
        "Segment closed: frames=%d declared_fps=%.2f duration=%.2fs "
        "dup_count=%d (%.1f%%)",
        self._frames_written, self._fps, actual_dur,
        self._dup_count, 100.0 * self._dup_count / self._frames_written,
    )
```

#### Recorder instantiation (L930–L950) — derive `record_fps` from camera

```python
cap_fps = cap.get(cv2.CAP_PROP_FPS)
if not (1.0 < cap_fps < 120.0):
    cap_fps = None  # unreliable; will measure

# measure fallback: median inter-frame interval over first 30 frames
if cap_fps is None:
    intervals = []
    prev_ts = time.monotonic()
    for _ in range(30):
        cap.read()
        now = time.monotonic()
        intervals.append(now - prev_ts)
        prev_ts = now
    intervals.sort()
    cap_fps = 1.0 / intervals[len(intervals) // 2]
    logger.info("Measured camera FPS: %.2f", cap_fps)

record_fps = cap_fps
recorder = StreamRecorder(path, record_fps=record_fps, ...)
```

#### Main loop (L1049–L1063) — pass `capture_ts`

```python
if recorder:
    frame_to_record = annotated_frame if labeled_mode else latest.frame
    recorder.write(frame_to_record, capture_ts=latest.capture_ts)
```

On segment close, pass `stop_ts=time.monotonic()` to `recorder.close()`.

### `web/annotator/index.html` (L194–L199)

Add an optional `record_fps` input (empty = auto) alongside Preview FPS and Detect FPS. When empty, omit the parameter so the server uses its auto-detected value.

### `web/annotator/app.js` (L821–L856)

Include `record_fps` in the recording query string only when the user has set it explicitly; otherwise omit it.

---

## 6. Test matrix

| # | Test | Location | Acceptance criterion |
|---|---|---|---|
| T1 | CFR slot fill: normal rate | `test_annotation_server_helpers.py` | `frames / fps` within `±0.5/fps` of elapsed time |
| T2 | CFR slot fill: stall 3 s | new helper test | catch-up per call ≤ `2 * fps`; total frames still ≈ correct |
| T3 | Final tail on close | new helper test | `close(stop_ts)` after 0.5 s gap → file duration ≥ `stop_ts − start_ts − 1/fps` |
| T4 | `capture_ts` vs. call-time drift | new helper test | inject frame with `capture_ts` 50 ms before `write()` call; slot uses `capture_ts` |
| T5 | Real container duration | new integration test | write 10 s at 8 FPS declared as 8 FPS; `ffprobe duration` ∈ `[9.875, 10.125]` s |
| T6 | VFR fixture overlay mapping | new integration test | synthetic VFR file; `mediaTime * fps + 1` index drifts ≤ 1 frame over 60 s |
| T7 | Multi-object overlay identity | new JS/integration test | two objects with swapping detector order; no box interpolates across unrelated IDs |
| T8 | Analysis latency | new timed test | 60 s file at 3-detect-FPS cadence; analysis completes < 5 s on Jetson |
| T9 | Segment boundary continuity | new integration test | last PTS of segment N + first PTS of segment N+1 delta ≤ `1.5/fps` |
| T10 | Duplicate-frame CPU backpressure | new load test | 5 s stall → catch-up write ≤ 200 ms wall time; no OOM |

### Jetson on-device validation protocol

```bash
# 1. Record a 120 s live session → 2 segments.

# 2. Check each segment duration:
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1 segment_N.mp4

# Sum of durations must equal 120 ± 2 s.

# 3. Cross-check frame count vs. declared FPS:
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames,r_frame_rate segment_N.mp4
# nb_read_frames / r_frame_rate must match ffprobe duration within 1/fps.

# 4. Check duplication rate in server log:
grep "Segment closed" ~/UAVDetection/uav-detection.log
# dup_count / frames_written should be < 30% under normal inference load.

# 5. Overlay alignment: open each segment in the UI, pause at 3 random timestamps,
# confirm detection box is within 1 frame of where it appears in the raw video.
```

---

## 7. High-risk assumptions — verify on live Jetson before implementation

| # | Assumption | How to verify | Fallback |
|---|---|---|---|
| A1 | `cap.get(CAP_PROP_FPS)` returns a valid value for the Roni camera | Log value; compare with measured median inter-frame interval | Always use measured-FPS fallback (see §5 instantiation code) |
| A2 | `cap.get(CAP_PROP_POS_MSEC)` is monotonic and accurate post-decode | Log first 30 values; check for zeros or non-advancing values | Use `time.monotonic()` at acquire time (minimal patch already does this) |
| A3 | Camera tolerates two simultaneous RTSP connections | Open second connection in test mode; check for frame drops or auth errors on either | Robust path may need a GStreamer `tee` or a single-consumer demux instead |
| A4 | Catch-up cap of `2 * fps` covers worst observed stall | Log max stall duration over 10 min live inference | Raise cap, or accept time gap and log it clearly |
| A5 | OpenCV codec fallback lands on H.264, not `mp4v` | Check `Segment closed` log for codec selected | Force `avc1` / fail loudly if unavailable |
| A6 | `requestVideoFrameCallback` ticks are fine enough to distinguish frames at 8.9 FPS | Open corrected file in browser; log `mediaTime` at each callback; confirm 111 ms resolution | Clamp frame index rounding and add ±1-frame tolerance in overlay lookup |

---

## 8. Recommendation

**Implement the minimal patch first** (1–2 days, no new dependencies, directly testable):

1. `capture_ts` field on `LatestFrameCapture`
2. `record_fps` derived from camera, separate from `preview_fps`
3. `write()` uses frame's `capture_ts`
4. Catch-up cap of `2 * record_fps`
5. Tail fill on `close(stop_ts)`
6. Per-segment log line

Validate with `ffprobe` duration checks over 2–3 real sessions and the Jetson protocol above. If the catch-up cap causes visible frame gaps under inference load, or if independent timestamped recording becomes a requirement, pursue the robust FFmpeg parallel-recorder path using `ffmpeg -c copy` for stream-copy (PyAV for fallback). Verify dual RTSP connection behavior before committing to that design.

Do not modify or delete existing `data_store` artifacts until the chosen design is agreed and the first validated recording session is confirmed good.
