# UAVDetection video timing consultation context

Prepared: 2026-07-19 (Asia/Jerusalem)
Purpose: self-contained technical handoff for consultation with Claude about live recording duration, prerecorded playback cadence, and detection-overlay timing.

## What I want reviewed

Please review the evidence and current implementation below and recommend a robust timing architecture for both:

1. Recording a live RTSP/IP-camera source on the Jetson without shortening or lengthening elapsed time.
2. Playing prerecorded files in the browser at their encoded cadence while keeping detector overlays and voice alerts aligned.

The most useful response would:

- distinguish capture time, processing time, encoded presentation time (PTS), and browser media time;
- assess the current fixed-rate frame-duplication mitigation;
- identify any remaining races, drift, frame/overlay alignment errors, and resource risks;
- compare a minimal safe fix with a stronger design (for example, direct FFmpeg/GStreamer recording separated from preview/inference);
- propose deterministic unit/integration tests and an on-Jetson validation protocol;
- avoid modifying or deleting `data_store` artifacts until a design is agreed.

## Executive summary

There were two related but distinct classes of timing failure:

1. **Bad recording timeline:** a live camera session ran for about 61 seconds per segment, but each MP4 contained only about 540–554 frames while declaring 16 FPS. The files therefore play in about 34 seconds. Their wall-clock content is compressed to roughly 1.77x–1.84x speed.
2. **Unstable browser preview/playback:** prerecorded clips and remote MJPEG previews went through a server processing/encoding/TLS path whose delivery cadence could freeze, burst, catch up, drop frames, or play much slower than the encoded video.

The current code at commit `950550a11af27a609c6a3eb7db986133ebf8dfd2` contains two major mitigations:

- `StreamRecorder` writes a CFR file and duplicates the last frame to fill wall-clock frame slots when processing is slower than the requested recording FPS.
- Live MJPEG uses the browser's native `<img>` multipart decoder, while prerecorded files (when not re-recording) use native HTML `<video>` playback plus a separate precomputed-detection canvas overlay driven by `requestVideoFrameCallback`.

These mitigations are deployed on the live Jetson. They do not repair the timestamps in already-created source MP4s. Corrected comparison copies were generated separately by deriving an approximate real duration from the camera's burned-in timestamps and re-encoding at approximately 8.8–9.1 FPS.

## Concrete evidence: problematic and corrected files

All files below exist on both the Mac workspace and the current Jetson under `~/UAVDetection/data_store/raw_data/Roni/`. They are runtime data and are intentionally not tracked by Git.

### Best A/B example

- Problematic original: [`record_1107_15-57_camera_ip_camera_196_10.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_10.mp4)
- Timestamp-corrected comparison: [`demo_corrected_record_1107_15-57_camera_ip_camera_196_10.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_10.mp4)
- Measurement/correction manifest: [`demo_corrected_record_1107_15-57_segments_10-16.json`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_segments_10-16.json)

Jetson browser links (self-signed HTTPS and normal UI authentication are required):

- [Problematic segment 10 on Jetson](https://192.168.100.197:8765/api/media?path=%2Fhome%2Fubuntu%2FUAVDetection%2Fdata_store%2Fraw_data%2FRoni%2Frecord_1107_15-57_camera_ip_camera_196_10.mp4)
- [Corrected segment 10 on Jetson](https://192.168.100.197:8765/api/media?path=%2Fhome%2Fubuntu%2FUAVDetection%2Fdata_store%2Fraw_data%2FRoni%2Fdemo_corrected_record_1107_15-57_camera_ip_camera_196_10.mp4)
- [Correction manifest on Jetson](https://192.168.100.197:8765/api/media?path=%2Fhome%2Fubuntu%2FUAVDetection%2Fdata_store%2Fraw_data%2FRoni%2Fdemo_corrected_record_1107_15-57_segments_10-16.json)

### Complete affected sequence

| Segment | Problematic original | Corrected comparison | Burned-in span | Frames | Declared FPS | Encoded duration | Corrected FPS | Corrected duration |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 10 | [`..._10.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_10.mp4) | [`corrected_..._10.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_10.mp4) | 62 s | 547 | 16.0 | 34.1875 s | 8.823 | 61.9971 s |
| 11 | [`..._11.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_11.mp4) | [`corrected_..._11.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_11.mp4) | 61 s | 552 | 16.0 | 34.5000 s | 9.050 | 60.9945 s |
| 12 | [`..._12.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_12.mp4) | [`corrected_..._12.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_12.mp4) | 62 s | 549 | 16.0 | 34.3125 s | 8.855 | 61.9989 s |
| 13 | [`..._13.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_13.mp4) | [`corrected_..._13.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_13.mp4) | 61 s | 551 | 16.0 | 34.4375 s | 9.033 | 60.9986 s |
| 14 | [`..._14.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_14.mp4) | [`corrected_..._14.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_14.mp4) | 61 s | 554 | 16.0 | 34.6250 s | 9.082 | 60.9998 s |
| 15 | [`..._15.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_15.mp4) | [`corrected_..._15.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_15.mp4) | 61 s | 553 | 16.0 | 34.5625 s | 9.066 | 60.9971 s |
| 16 | [`..._16.mp4`](data_store/raw_data/Roni/record_1107_15-57_camera_ip_camera_196_16.mp4) | [`corrected_..._16.mp4`](data_store/raw_data/Roni/demo_corrected_record_1107_15-57_camera_ip_camera_196_16.mp4) | 61 s | 540 | 16.0 | 33.7500 s | 8.852 | 61.0032 s |

The correction is a forensic workaround, not recovery of missing temporal detail. It changes playback rate so the file duration matches the visible clock span; it cannot recreate frames that were never recorded. The “ground truth” duration is also approximate because it was read from whole-second burned-in timestamps.

Useful integrity checks for segment 10:

```text
original SHA-256:  f7f0927a37a1e891d1b2047e3526a29634b8648b1be2fcebd3f1671451fe5a3e
corrected SHA-256: 63a9501a353af3a3fdd6b5b1f94b0d99f51b8390c468678fe75b4057a70568da
manifest SHA-256:  f7ca386392199efe1f6e96d480120e3c51755ce07b0ce456a627cb71ae77736b
```

## Root cause of the original shortened recordings

The old recorder used OpenCV `VideoWriter` as a constant-frame-rate encoder:

```python
writer = cv2.VideoWriter(path, fourcc, requested_preview_fps, frame_size)
```

It then wrote exactly one frame whenever the server's combined preview/detection loop produced one. A fixed-FPS `VideoWriter` does not know how much wall time elapsed between calls and this code supplied no per-frame PTS. If the loop produced approximately 9 frames/sec but the writer declared 16 FPS, 60 seconds of capture produced about 540 frames and an encoded duration of `540 / 16 = 33.75` seconds.

The old segmenter nevertheless rolled over on wall time (`time.monotonic() - segment_started_at >= 60`). That explains the diagnostic signature: files were started about a minute apart and the burned-in clock advances about a minute, while their container duration is only about 34 seconds.

The loop itself included source acquisition, resizing, detector/tracker state, drawing, recording, JPEG encoding, and a TLS multipart socket write. Any of those could make its achieved cadence lower than the requested/declared recording FPS. Live inference is asynchronous now, but recording is still downstream of the preview loop rather than an independent source-recording pipeline.

## Current recording implementation

Primary path: [`scripts/annotation_server.py`](scripts/annotation_server.py)

- Request parsing and recording settings: [`scripts/annotation_server.py#L765`](scripts/annotation_server.py#L765) through `#L791`.
- Capture/source initialization and effective preview FPS: [`scripts/annotation_server.py#L817`](scripts/annotation_server.py#L817) through `#L853`.
- `StreamRecorder` receives the requested `preview_fps`, not a measured camera FPS: [`scripts/annotation_server.py#L930`](scripts/annotation_server.py#L930) through `#L950`.
- Main loop cadence and latest-frame acquisition: [`scripts/annotation_server.py#L968`](scripts/annotation_server.py#L968) through `#L990`.
- Async live detection and rendering: [`scripts/annotation_server.py#L1016`](scripts/annotation_server.py#L1016) through `#L1048`.
- Recording happens after drawing and before JPEG preview encoding: [`scripts/annotation_server.py#L1049`](scripts/annotation_server.py#L1049) through `#L1063`.
- Recorder implementation: [`scripts/annotation_server.py#L2448`](scripts/annotation_server.py#L2448) through `#L2583`.
- Wall-clock CFR slot calculation: [`scripts/annotation_server.py#L2586`](scripts/annotation_server.py#L2586) through `#L2591`.

Current mitigation, introduced by commit [`bf10dd2`](https://github.com/alexfok/UAVDetection/commit/bf10dd2033f4fe517b1f28d383eccfa0f82d86a4):

```python
frames_due = int((now - segment_started_at) * fps) + 1 - frames_written

if frames_due <= 0:
    # Do not overrun the CFR timeline.
    cache_current_frame_and_return()
else:
    # Fill missed CFR slots with the last observed frame.
    write_last_frame(frames_due - 1)
    write_current_frame()
```

This should keep `frame_count / declared_fps` close to segment wall time even when processing is late. It trades time compression for repeated frames. That is preferable for truthful duration, but motion can still visibly freeze when the upstream loop stalls.

Recording-related UI/API details:

- Current UI default is Preview FPS `12`, Detect FPS `3`: [`web/annotator/index.html#L194`](web/annotator/index.html#L194) through `#L199`.
- Browser query construction (`record=1`, directory, 30 MB maximum, label mode): [`web/annotator/app.js#L821`](web/annotator/app.js#L821) through `#L856`.
- Server hard cap is 30 MiB, with rollover targeted near 28 MiB: [`scripts/annotation_server.py#L37`](scripts/annotation_server.py#L37) through `#L38`.
- Rollover can be caused by 60 seconds, frame-size change, or estimated/file size; therefore a segment is not guaranteed to represent exactly 60 seconds.
- Writer codec fallback order is `avc1`, `H264`, `VP90`, `VP80`, `mp4v`, then `MJPG`: [`scripts/annotation_server.py#L2516`](scripts/annotation_server.py#L2516) through `#L2556`.
- Raw mode records the resized latest frame. Labeled mode records the same frame with current UI overlays. Neither mode is a direct copy/remux of the camera's encoded RTSP packets.

## Current playback implementation

There are now separate paths for live sources and prerecorded files.

### Live RTSP/camera preview

The browser assigns `/api/live/stream?...` directly to an `<img>` element. Its native multipart MJPEG decoder consumes the stream:

- Branch and image setup: [`web/annotator/app.js#L859`](web/annotator/app.js#L859) through `#L916`.
- Server pacing: [`scripts/annotation_server.py#L852`](scripts/annotation_server.py#L852) and [`scripts/annotation_server.py#L968`](scripts/annotation_server.py#L968).
- One flushed TLS/socket write per multipart frame: [`scripts/annotation_server.py#L1082`](scripts/annotation_server.py#L1082) through `#L1098`.

This replaced a JavaScript JPEG buffering/decoding implementation that exhibited freeze/catch-up cycles with remote 1080p frames. Relevant commits are [`37581ca`](https://github.com/alexfok/UAVDetection/commit/37581ca), [`0a6ce3b`](https://github.com/alexfok/UAVDetection/commit/0a6ce3b), [`f92c0b0`](https://github.com/alexfok/UAVDetection/commit/f92c0b0), and [`ee78de1`](https://github.com/alexfok/UAVDetection/commit/ee78de1).

### Prerecorded file playback with detections

When the selected source is a video file and recording is **not** checked, the browser uses a native `<video>` element. Before playback starts, `/api/live/file-analysis` scans the file, runs detector samples, and returns detections keyed by 1-based decoded frame index:

- UI branch: [`web/annotator/app.js#L878`](web/annotator/app.js#L878) through `#L893`.
- Analysis request and native-video setup: [`web/annotator/app.js#L928`](web/annotator/app.js#L928) through `#L995`.
- Media-time-to-frame mapping: `frame = mediaTime * fps + 1` at [`web/annotator/app.js#L953`](web/annotator/app.js#L953) through `#L963`.
- Detection interpolation/expiry: [`web/annotator/app.js#L997`](web/annotator/app.js#L997) through `#L1026`.
- Backend analysis endpoint: [`scripts/annotation_server.py#L1118`](scripts/annotation_server.py#L1118) through `#L1208`.
- Detection scan/cache keyed by path, stat, model, cadence, skip, and size: [`scripts/annotation_server.py#L1841`](scripts/annotation_server.py#L1841) through `#L1890`.
- Byte-range media serving used by `<video>`: [`scripts/annotation_server.py#L1230`](scripts/annotation_server.py#L1230) through `#L1265`.

This architecture was introduced by current commit [`950550a`](https://github.com/alexfok/UAVDetection/commit/950550a11af27a609c6a3eb7db986133ebf8dfd2). It correctly delegates encoded playback cadence and buffering to the browser. However, it faithfully follows the bad 34-second container timeline of an old problematic file; the browser cannot infer that its content originally spanned 61 seconds.

If recording is checked while playing a file, the native-video branch is bypassed. The file is decoded through the server MJPEG/recorder path instead. That case needs explicit review because it has different timing semantics from ordinary native playback.

## Remaining concerns for review

These are hypotheses/risks, not all confirmed defects:

1. **Recording remains coupled to preview cadence.** The raw recorder receives frames after resize and server-loop pacing. A direct RTSP recorder could preserve camera timestamps/packets independently of detection and browser delivery.
2. **CFR duplication is a mitigation, not source timing.** It makes duration honest but repeats stale pixels. Long stalls can create many duplicate frames in one call and cause additional CPU/I/O pressure.
3. **No explicit capture timestamp travels with a frame.** `LatestFrameCapture` exposes a token/frame/index, but `StreamRecorder.write()` timestamps receipt with a new `time.monotonic()` call.
4. **OpenCV `VideoWriter` cannot assign arbitrary PTS here.** The design assumes CFR and derives duration solely from frame count and declared FPS.
5. **Close behavior does not fill through the final stop instant.** Slots are filled only on `write()`. The final segment can end slightly before the user's stop time if there was a last-frame gap.
6. **Native overlay mapping assumes constant FPS.** `mediaTime * OpenCV_average_fps + 1` can drift for variable-frame-rate files, unusual edit lists, or inaccurate FPS metadata.
7. **Overlay interpolation pairs detections by list index, not track identity.** With multiple objects or changing detector order, boxes may interpolate between unrelated detections.
8. **Analysis is front-loaded.** The whole video is decoded/inferred before native playback begins. Large files can produce long startup latency and retain a sizable per-process detection cache.
9. **The analysis and media requests are separate.** A file can theoretically change between analysis and playback. The cache key notices future analysis changes, but the response does not give the browser a content/version identity to bind both requests.
10. **Preview FPS is overloaded.** It controls server delivery pacing and is also passed as recording FPS, although those are conceptually different policies.
11. **Detection/voice semantics use sampled frames.** The UI interpolates overlays and keeps file voice state active for a one-second grace interval. This should be checked against desired alert entry/exit timing.
12. **Corrected legacy files are approximate CFR reinterpretations.** Their 8.8–9.1 FPS values are inferred from whole-second timestamps, not original packet PTS.

## Suggested architectural decision to evaluate

A likely stronger separation of concerns is:

```text
RTSP/camera packets or timestamped decoded frames
            |
            +--> recorder (own clock/PTS, segment muxer, no detector dependency)
            |
            +--> latest-frame sampler --> detector/tracker --> browser preview/alerts
```

For RTSP, Claude should assess whether the recorder should:

- stream-copy/remux the camera video with FFmpeg/GStreamer when compatible;
- transcode using source PTS when remux is not viable;
- deliberately generate CFR from timestamped decoded frames, with metrics for dropped/duplicated frames;
- store sidecar segment metadata with monotonic start/end, wall-clock start/end, frame/packet counts, declared/observed rates, drops, duplicates, source ID, and stop reason.

The minimal-change alternative is to retain OpenCV but separate `record_fps` from `preview_fps`, carry capture timestamps into the recorder, cap catch-up writes, fill the final tail on close, and log timing diagnostics. Please compare the operational reliability and deployment burden of both approaches on Jetson ARM64.

## Tests that currently exist

- Wall-clock slot helper test: [`tests/test_annotation_server_helpers.py#L34`](tests/test_annotation_server_helpers.py#L34) through `#L38`.
- Stable file-preview interval helper test: [`tests/test_annotation_server_helpers.py#L29`](tests/test_annotation_server_helpers.py#L29) through `#L32`.
- Native live MJPEG structure test: [`tests/test_ui_integrity.py#L165`](tests/test_ui_integrity.py#L165) through `#L174`.
- Native prerecorded video/overlay structure test: [`tests/test_ui_integrity.py#L176`](tests/test_ui_integrity.py#L176) through `#L192`.

Local verification performed while preparing this handoff:

```text
Python compile: scripts/annotation_server.py and app/sources.py passed
HTML parse: web/annotator/index.html passed
Focused tests: 20 tests passed
```

Gaps in current testing:

- no simulated recorder test that writes a real container and verifies duration with a media probe;
- no jitter/stall test over a full segment boundary;
- no assertion for final duration on close;
- no end-to-end RTSP recording test with a timestamped synthetic source;
- no browser test measuring native playback wall duration and overlay alignment;
- no VFR fixture;
- no multi-object overlay identity/interpolation test;
- no performance/backpressure test for a long analysis or a large duplicate-frame catch-up.

No new live recording was started while preparing this document because that would create device state. The current code therefore has structural/unit validation and live served-code verification, but this handoff does not claim a fresh end-to-end duration test of the mitigation.

## Repository and deployment state (verified 2026-07-19)

Mac workspace:

```text
/Users/afok/Library/CloudStorage/OneDrive-NVIDIACorporation/Private/UAVDetection
```

Git state:

```text
local HEAD:   950550a11af27a609c6a3eb7db986133ebf8dfd2
origin/main:  aac2f02327f171795e57df8b4a20e8fc6dfcbf98
divergence:   local HEAD is 16 commits ahead of origin/main
user change:  README.md is modified and must be preserved
```

The timing/playback work is deployed but not yet represented by `origin/main`. Do not assume a fresh Git clone contains it.

Jetson access:

```text
reachable host:       192.168.100.197
unreachable/timeout:  192.168.100.199
SSH:                  ssh -o BatchMode=yes ubuntu@192.168.100.197
architecture:         aarch64
install directory:    /home/ubuntu/UAVDetection (~/UAVDetection)
web UI:               https://192.168.100.197:8765
service:              uav-detection.service
service state:        active/running
service user:         ubuntu
restart policy:       always
observed PID:          63667 (ephemeral; re-check it)
installed manifest:   commit 950550a11af27a609c6a3eb7db986133ebf8dfd2, artifact_kind code-only
```

The served `/static/app.js` was verified to contain all three current markers:

```text
image.src = liveStreamUrl(job)
startAnalyzedFilePlayback
requestVideoFrameCallback
```

Default Jetson media folder returned by `/api/defaults`:

```text
/home/ubuntu/UAVDetection/data_store/raw_data/Roni
```

Do not print or move camera credentials. They live only in the Jetson-local file:

```text
~/UAVDetection/data_store/system_config/annotation_server.env
```

## Codex skills and safe operational workflows

The two relevant local skill instructions are:

- Deployment: `/Users/afok/.codex/skills/jetson-deploy/SKILL.md`
- State/data synchronization: `/Users/afok/.codex/skills/sync-uav-jetson-state/SKILL.md`

### Code deployment summary

Use a clean temporary worktree at the intended commit so the user's modified `README.md` or untracked local artifacts are not packaged accidentally:

```bash
commit=$(git rev-parse --short HEAD)
git worktree add --detach /private/tmp/uav_deploy_worktree_$commit $commit
cd /private/tmp/uav_deploy_worktree_$commit
python3 scripts/build_deployment.py \
  /Users/afok/Library/CloudStorage/OneDrive-NVIDIACorporation/Private/UAVDetection/data_store/deployment_artifacts \
  --name UAVDetection_deploy_$(date +%Y%m%d)_jetson_$commit \
  --format tar.gz \
  --force
```

Inspect `DEPLOYMENT_MANIFEST.json`; require `artifact_kind: code-only` and the intended clean commit. Transfer/stage the tarball, then run on the Jetson from its extracted directory:

```bash
python3 scripts/deploy.py preflight --target jetson --install-dir ~/UAVDetection
python3 scripts/deploy.py upgrade --target jetson --install-dir ~/UAVDetection --skip-deps
```

The upgrade must preserve `~/UAVDetection/data_store`. Noninteractive sudo may fail only at service installation/restart even after the code copy succeeded. First try `sudo -n systemctl restart uav-detection.service`. If sudo is unavailable, verify that the existing service runs as `ubuntu` with `Restart=always`; only then can terminating its current process allow systemd to restart it without editing the unit.

After deployment, verify the manifest commit, service status/PID, served JS markers/APIs, and run:

```bash
cd ~/UAVDetection
.venv_cuda/bin/python scripts/run_regression.py --code --skip-js
```

### Recording/data synchronization summary

`data_store` is not Git content. The safe direction is:

```text
Jetson ~/UAVDetection/data_store
    -> Mac repo-local data_store/
    -> Google Drive uavdrive:current
```

Use `scripts/sync_uav_state.sh preflight`, then `pull`, `compare`, `drive`, and `verify`. The pull is backup-preserving and must not use `--delete`. Drive publication must use additive `rclone copy`, not delete-capable `rclone sync`. Preserve exact paths so example links and event metadata remain valid.

## Relevant commit chronology

Newest first:

| Commit | Purpose |
|---|---|
| [`950550a`](https://github.com/alexfok/UAVDetection/commit/950550a11af27a609c6a3eb7db986133ebf8dfd2) | Native prerecorded `<video>` playback with separately analyzed, frame-aligned overlay |
| [`ee78de1`](https://github.com/alexfok/UAVDetection/commit/ee78de1) | Use native browser MJPEG preview playback |
| [`f92c0b0`](https://github.com/alexfok/UAVDetection/commit/f92c0b0) | Keep live preview on real-time cadence |
| [`6affe83`](https://github.com/alexfok/UAVDetection/commit/6affe83) | Hide stale boxes between analyzed frames |
| [`0c38078`](https://github.com/alexfok/UAVDetection/commit/0c38078) | Precompute aligned detections for demo clips |
| [`650867d`](https://github.com/alexfok/UAVDetection/commit/650867d) | Honor file detection cadence without stale frames |
| [`918998d`](https://github.com/alexfok/UAVDetection/commit/918998d) | Align demo playback and detection frames |
| [`0a6ce3b`](https://github.com/alexfok/UAVDetection/commit/0a6ce3b) | Add JavaScript buffering for remote preview (later replaced) |
| [`37581ca`](https://github.com/alexfok/UAVDetection/commit/37581ca) | Reduce remote multipart/JPEG bursts |
| [`bf10dd2`](https://github.com/alexfok/UAVDetection/commit/bf10dd2033f4fe517b1f28d383eccfa0f82d86a4) | Add stable prerecorded cadence and wall-clock CFR recording slot fill |
| [`2671d69`](https://github.com/alexfok/UAVDetection/commit/2671d69) | Earlier live demo playback smoothing |

Because several interim approaches were intentionally superseded, review the current tree and final architecture rather than treating every commit as cumulative design guidance.

## Requested output from Claude

Please return:

1. A concise diagnosis confirming or correcting the root-cause model.
2. A recommended target architecture, including clock/timestamp ownership at each boundary.
3. A minimal patch plan and a robust patch plan, with tradeoffs specific to Jetson ARM64/offline deployment.
4. Specific code-level changes by function/file.
5. A test matrix with measurable acceptance criteria, such as maximum duration error, allowed drops/duplicates, overlay timing error, and behavior across stalls/segment rollover.
6. Any high-risk assumptions that need measurement on the live Jetson before implementation.
