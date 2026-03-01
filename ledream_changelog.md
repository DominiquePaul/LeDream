# LeDream Changelog

This file documents changes made in the LeDream fork relative to upstream
[huggingface/lerobot](https://github.com/huggingface/lerobot). It serves as a
reference for contributors and AI agents working on this codebase.

**Upstream remote:** `upstream` -> `https://github.com/huggingface/lerobot.git`
**Fork diverged from:** `975dcad9` (Feat: add OpenArm Mini teleoperator #3022)

## Upstream Sync

To pull in new upstream changes:

```bash
git fetch upstream
git tag pre-sync-$(date +%Y-%m-%d)   # safety tag
git merge upstream/main               # resolve conflicts, then commit
```

## Custom Files (not in upstream)

These files are entirely ours and will never conflict with upstream merges:

| File | Purpose |
|------|---------|
| `configs/*.json` | Training config files (ACT, DP, pi05) for `make train` |
| `scripts/birest.py` | Move bimanual arms to rest pose |
| `scripts/snapshot.py` | Capture camera snapshots |
| `scripts/check_dataset_indices.py` | Validate dataset episode indices |
| `scripts/check_recording_gaps.py` | Detect timing gaps in recordings |
| `scripts/repair_dataset_episodes_metadata.py` | Fix broken episode metadata |
| `ledream_changelog.md` | This file |
| `lerobot_readme.md` | Original LeRobot README (moved from README.md) |
| `recording_loop_lag_report_2026-02-27.md` | Investigation report on recording lag |
| `trlc-dk1` | Git submodule for DK1 hardware config |

## Modified Upstream Files

These files were changed from their upstream versions. **Pay attention during
upstream merges** -- conflicts are likely here.

### `src/lerobot/scripts/lerobot_record.py`

- Added `_make_progress_bar()` for an ASCII progress bar during recording
  (shows elapsed time, total time, Hz)
- Added `phase_label` parameter to `record_loop()` to distinguish Recording vs
  Reset phases in the progress bar
- Added policy action debug logging (logs action stats once per second during
  recording to help diagnose policy issues)
- Added `_collect_numeric_values()` helper to extract floats from nested
  action dicts/tensors for debug logging
- Changed "no action source" warning to log only once instead of every frame
- Added `precise_sleep` and timestamp update in the no-action-source branch so
  reset/control windows end correctly

### `src/lerobot/utils/control_utils.py`

- Replaced the headless environment warning (which disabled all keyboard input)
  with a **stdin-based single-key control loop** that works over SSH/tmux:
  - `Enter`/`n` = finish current loop
  - `r` = rerecord episode
  - `q` = discard and stop
  - `h` = help
- Uses `tty.setcbreak()` for immediate keypress detection without waiting for
  newline, with proper terminal attribute restoration on exit
- Runs in a daemon thread so it doesn't block recording

### `Makefile`

Heavily extended from upstream's minimal test-only Makefile:
- Added robot port/camera defaults for bimanual DK1 setup
- Added targets: `biteleop`, `birecord`, `birest`, `snapshot`, `teleop-arms`,
  `record-arms`
- Added job/checkpoint management: `list-jobs`, `list-checkpoints`, `upload-latest`
- Added training workflow: `list-configs`, `train`, `train-cmd`
- Added quick eval: `eval` (with `EPHEMERAL` flag)

### `README.md`

Replaced with project-specific documentation. Original moved to `lerobot_readme.md`.

### `.gitignore`

Minor additions for project-specific files.

## Change Log

### 2026-03-01

- Added training config system (`configs/*.json`) with `make train`, `make
  train-cmd`, and `make list-configs`
- Added `make eval MODEL=...` for quick real-robot evaluation with `EPHEMERAL`
  cleanup option
- Established naming convention: `{dataset}_{policy}[-{variant}]-{checkpoint}`
- Moved `dominique_notes.md` content into `README.md` with full restructure
- Added `make list-jobs`, `make list-checkpoints`, `make upload-latest` for
  checkpoint management
- Removed `cam-tuner`, `birecord3o`, `birecord2o` targets and `cam_tuner.py`
- Switched `huggingface-cli` to `hf` in upload commands
- Disabled all GitHub Actions CI/CD workflows (not deleted, just disabled)

### 2026-02-28

- Added `make birest` target and `scripts/birest.py`
- Added training notes and commands documentation

### 2026-02-25

- Initial fork setup from huggingface/lerobot
- Added Makefile configuration for bimanual teleoperation and data recording
- Added progress bar to recording loop (`lerobot_record.py`)
- Added headless stdin keyboard controls (`control_utils.py`)
- Added `scripts/snapshot.py` for camera snapshots
- Added DK1 hardware submodule (`trlc-dk1`)
