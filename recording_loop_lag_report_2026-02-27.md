# Recording Loop Lag Investigation (Jetson Thor)

Date: 2026-02-27

## Scope

Investigate root causes of `Record loop is running slower (...)` warnings during `lerobot-record` for DK1 bimanual recording.

The baseline symptom in your run:
- 3 cameras, `640x480@30`, `dataset.fps=30`
- `vcodec=auto` selecting `h264_nvenc`
- repeated loop drops, including severe dips (down to single-digit Hz in previous run sample)

Reference tuning guidance: [LeRobot streaming video troubleshooting](https://huggingface.co/docs/lerobot/main/en/streaming_video_encoding#5-troubleshooting)

## Experiment Setup

All tests were run with:
- Same robot + teleop ports
- Same camera devices
- `display_data=false`
- `dataset.push_to_hub=false`
- isolated local dataset roots per run (`/tmp/ledream-exp/ds_*`)
- `tegrastats` captured at 1s interval in parallel

Logs are in:
- `/tmp/ledream-exp/logs/exp*_record.log`
- `/tmp/ledream-exp/logs/exp*_tegrastats.log`

## Hypotheses and Checks

### H1) Software encoding is causing the lag
Check:
- Use `--dataset.vcodec=auto` and verify selected codec.

Result:
- Confirmed auto-selected `h264_nvenc` in runs (`exp1`, `exp2b`, `exp3`, `exp4`, `exp5`).
- Lag warnings still present.

Conclusion:
- SW encode is **not** the primary cause anymore.
- HW encode helps, but does not eliminate jitter.

### H2) Target FPS is too aggressive for end-to-end pipeline
Checks:
- `exp1`: 3 cams, cam fps 30, dataset fps 30
- `exp2b`: 3 cams, cam fps 30, dataset fps 20

Results:
- `exp1`: 11 warnings, min warning speed `9.6 Hz`
- `exp2b`: 1 warning, min warning speed `15.6 Hz` (target now 20)

Conclusion:
- Strong evidence that **30 Hz loop budget is too tight** on this setup.
- Lowering loop target significantly stabilizes recording.

### H3) Camera ingest path (multi-camera capture/USB/decode) is a bottleneck
Checks:
- `exp3`: 1 camera at 30 fps
- `exp1`: 3 cameras at 30 fps
- `exp5`: 3 cameras with `backend=V4L2` and `fourcc=MJPG`

Results:
- `exp3`: still warnings (5), but less severe than `exp1`.
- `exp1`: 11 warnings, severe dips.
- `exp5` (MJPG): still 11 warnings; severe floor improved somewhat (min ~15.5), but not solved.

Conclusion:
- Moving from 1 cam -> 3 cams increases instability, so multi-camera path contributes.
- MJPG alone does not resolve core issue.

### H4) Streaming encoder threads are the dominant issue
Check:
- `exp4`: same 3-cam@30 workload but `streaming_encoding=false`.

Result:
- Warnings persist (8 warnings).

Conclusion:
- Encoder pipeline contributes some overhead, but **not the main root cause** of loop misses.

### H5) Global CPU saturation is causing starvation
Check:
- `tegrastats` summaries over runs.

Observed:
- Some cores briefly hit 90-100% in all runs.
- Not all cores are uniformly saturated continuously.
- Warnings still occur even when average multi-core utilization is moderate.

Conclusion:
- Not a simple "all-CPU pegged" scenario.
- More consistent with **single-thread / bursty scheduling bottlenecks** in capture/control path (and possibly camera/driver timing jitter).

## Quantitative Summary

- `exp1` (3 cam, 30/30, streaming on): 11 warnings, min `9.6 Hz`
- `exp2b` (3 cam, cam 30, dataset 20): 1 warning, min `15.6 Hz`
- `exp3` (1 cam, 30/30): 5 warnings, min `16.0 Hz`
- `exp4` (3 cam, 30/30, streaming off): 8 warnings, min `15.8 Hz`
- `exp5` (3 cam, 30/30, MJPG+V4L2): 11 warnings, min `15.5 Hz`

## Most Likely Root Cause

Primary:
- **Loop deadline pressure at 30 Hz with 3-camera pipeline**, even with NVENC.

Contributing:
- Multi-camera capture jitter and scheduling bursts (likely in camera read/decode + control loop interaction).
- Encoder mode affects severity, but is not the root driver after switching to NVENC.

Not primary:
- Push-to-hub/upload (tests were local and still reproduced).
- Pure software video encoding (NVENC confirmed active).

## Practical Recommendation

For reliability-first recording on this platform:
- Keep `vcodec=auto` (confirmed selecting NVENC).
- Run at `dataset.fps=20` (or 25 if you want to test middle ground).
- Keep camera fps at 30 if device rejects 20.
- Keep `display_data=false`.

If you want deeper root-cause isolation later:
- Add one experiment at `dataset.fps=25` (same setup) to find stable upper bound.
- Profile camera read latencies per camera in code (time spent per capture call).
- Pin record process / tune scheduler priority if acceptable for your runtime.

