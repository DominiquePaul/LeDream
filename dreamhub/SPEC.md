# DreamHub - Application Specification

Living document tracking all decided features, architecture, and design decisions.

## Overview
DreamHub is an internal web app for Dream Machines (dream-machines.eu) to manage and visualize robotics experiments, datasets, and models. Currently focused on robotics data (may expand in the future).

**Live URL**: https://hub.dream-machines.eu
**Repo location**: `research-os/` subdirectory of LeDream repo, branch `research-os`

## Branding
- **Name**: DreamHub
- **Style**: Dark theme, DM Mono font for logo, aligned with dream-machines.eu aesthetic
- **Domain**: `hub.dream-machines.eu` (custom domain on Vercel, A record → 76.76.21.21)

## Architecture

### Current (v2 — Supabase)
- **Frontend**: Next.js 16 with App Router, TypeScript, Tailwind CSS
- **Database**: Supabase (Postgres) — project `tgmgiovecbqzbrqgvfoh` (us-east-2)
- **Auth**: Email + password via Supabase Auth, RLS on all tables
- **HF Reads**: Client-side (public HF API, no token needed)
- **HF Writes**: Server-side API routes (`/api/hf/rename`, `/api/hf/delete`) — HF token in env vars
- **Deployment**: Vercel (static pages + serverless API routes)
- **Multi-device**: Full sync across browsers/devices via Supabase
- **Multi-user**: Row-level security (currently single-tenant, all authenticated users share data)

## Data Model

### Core Entities
- **Hypothesis**: Central entity. Status: exploring/active/confirmed/rejected
- **Experiment**: Tests a hypothesis, links to datasets and models. Status: planned/in-progress/completed/failed
- **Dataset**: Synced from HuggingFace or planned locally. Has metadata, tags, visualizer links. Status: planned/synced
- **Model**: Synced from HuggingFace. Grouped by base name, with iteration/checkpoint tracking
- **Evaluation**: Rollout log entry. Links to model, checkpoint, eval dataset, experiment. Outcome: success/partial/failure
- **Tag**: Categorized as task/robot/status/custom

### Dataset Metadata
- Collection conditions (how variation was handled)
- Teleoperator instructions
- Known issues (e.g., "missing IR filter")
- Notes (free text)
- Estimated hours
- Episode count
- Tags (task type, robot form, manual/bimanual)

### Model Metadata
- Policy type (ACT, SmolVLA, Diffusion, YOLO)
- Description
- Iterations with: version, config JSON, notes, trainedOn dataset links, HF link, date
- Tags

### Default Tags
**Task**: chess, pcb-placement, grasping, object-interaction
**Robot**: SO-100, Phosphobot, single-manual, bimanual
**Policy**: ACT, SmolVLA, Diffusion, YOLO
**Status**: evaluation, training-data

## Pages / Views

### Dashboard
- Stat cards: datasets, models, checkpoints, experiments, total hours, total episodes
- Data hours by tag (progress bars)
- Episodes by tag (progress bars)
- Experiment status breakdown
- Active hypotheses (clickable)
- Recently updated datasets
- All stat cards link to relevant pages
- **Planned**: summary widgets for recent evaluations, planned collections pending, sync status

### Experiments
- Hypothesis-centered accordion view
- Full CRUD for hypotheses (title, description, status)
- Full inline editing for experiments (name, notes, results, status)
- Dataset/model linking via search+select dropdowns
- Linked assets shown as colored pills with HF links and visualizer icons

### Relational Graph
- React Flow interactive canvas
- Node types: hypothesis (purple), experiment (amber), dataset (blue), model (green)
- Auto-layout: hypotheses left, experiments center, assets right
- Click-to-navigate to relevant page
- Mini-map, zoom controls, pan
- **Planned**: evaluation datasets shown as dataset nodes (they are a type of dataset)

### Datasets
- Search + tag filter
- Expandable detail view with full metadata editing
- Tag assignment (toggle UI)
- LeRobot visualizer links (Eye icon)
- HuggingFace links
- Rename on HuggingFace (via API)
- Unified delete: removes from both DreamHub and HuggingFace, requires typing `permanently delete [name]`
- Unlabelled dataset notification banner with filter toggle
- "Plan Collection" form: create placeholder datasets (status=planned) with metadata, auto-matched by name during HF sync
- "Push to HF" button: writes tags + metadata to HF README
- "Related Models" section: shows models with iterations trained on this dataset
- Hours displayed as `Xh Ym` format

### Evaluations
- Log individual rollouts: model, checkpoint, outcome (success/partial/failure), notes, date
- Link eval_ datasets (HF datasets with video recordings)
- Per-model success rate stats with progress bars
- Filter by model
- **Planned**: auto-suggest eval datasets based on model name matching
- **Planned**: mobile-optimized quick-log (larger tap targets, swipe gestures)

### Models
- Search + tag filter
- Expandable detail view
- Description editing
- Tag assignment (toggle UI)
- Iteration list with: version, date, notes editing, config JSON editing, trainedOn links
- Show/hide config toggle

### Tags
- Create/edit/delete tags
- Category grouping (task, robot, status, custom)
- Color picker (12 presets)
- Usage count display

## HuggingFace Integration

### Sync (Read)
- Auto-sync on first load if never synced
- Manual sync via sidebar button
- Fetches from `https://huggingface.co/api/datasets?author=dopaul&limit=200`
- Fetches from `https://huggingface.co/api/models?author=dopaul&limit=200`
- Groups models by base name (strips checkpoint suffixes like `_20k`, `-002000`)
- Infers metadata from names: task type, robot form, episode counts, hours

### Write Operations (via server-side API routes)
- **Rename dataset**: `POST /api/hf/rename` → proxies to HF `POST /api/repos/move`
- **Delete dataset**: `POST /api/hf/delete` → proxies to HF `DELETE /api/repos/delete`
- **Push metadata**: `POST /api/hf/push-metadata` → writes tags + metadata sections to HF README via commit API
- HF token stored server-side (`HF_TOKEN` env var), never exposed to browser
- Auth verified via Supabase session token

### Bidirectional Metadata Sync
- **Pull (during sync)**: READMEs fetched in parallel for all datasets, parsed for YAML tags and markdown sections (collection conditions, teleoperator instructions, known issues, notes). Only fills empty fields (doesn't overwrite local edits).
- **Push (manual)**: "Push to HF" button generates README with YAML front matter and markdown sections, commits to HF repo.
- Auto-detect `trainedOn` relationships: dataset names matched as prefixes of model iteration version names.

## Data Management
- **Export**: Download all data as timestamped JSON backup
- **Import**: Restore from JSON backup file
- **Reset**: Clear all data with confirmation, triggers re-sync

## Responsive Design
- Mobile: hamburger menu sidebar, fixed top bar with sync button
- All pages: responsive grids, reduced padding, proper text sizing
- Touch-friendly tap targets

## HuggingFace User
- Username: `dopaul`
- Token: Write-capable, stored in Vercel env vars (`HF_TOKEN`)

## Supabase Database Schema
### Main Tables
- `tags` (id, name, color, category, created_at)
- `datasets` (id, hf_id, name, description, status, collection_conditions, teleoperator_instructions, known_issues, notes, estimated_hours, episode_count, hf_url, downloads, last_modified, created_at)
- `models` (id, hf_id, name, description, policy_type, hf_url, downloads, last_modified, created_at)
- `model_iterations` (id, model_id FK, version, config jsonb, notes, hf_id, hf_url, created_at)
- `hypotheses` (id, title, description, status, created_at, updated_at)
- `experiments` (id, name, hypothesis_id FK, status, notes, results, created_at, updated_at)
- `evaluations` (id, experiment_id FK nullable, model_id FK, iteration_id FK nullable, eval_dataset_id FK nullable, outcome, notes, created_at)
- `app_settings` (key PK, value, updated_at)

### Junction Tables (many-to-many)
- `dataset_tags`, `model_tags`, `hypothesis_tags` — entity-to-tag relationships
- `experiment_datasets`, `experiment_models` — experiment asset links
- `iteration_datasets` — model iteration trainedOn links

All foreign keys use ON DELETE CASCADE.
