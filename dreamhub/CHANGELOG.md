# DreamHub Changelog

## 2026-03-04 (session 3)

### Features
- **Supabase migration** — Schema, RLS policies, auth (email+password), frontend refactor from localStorage to Supabase
- **HF token security** — Token stored server-side only, all HF write ops go through API routes
- **HF metadata push (bidirectional)** — "Push to HF" button writes tags + metadata to README; during sync, READMEs are fetched and parsed to fill empty metadata fields
- **Unlabelled dataset notification** — Banner shows count of untagged datasets, toggle to filter
- **Auto-detect dataset-model relationships** — `trainedOn` auto-detected by name prefix matching; "Related Models" section on dataset detail view
- **Data collection log** — "Plan Collection" creates placeholder datasets (status=planned), auto-matched by name during HF sync
- **Evaluation tracking** — New `/evaluations` page: log rollouts (model, checkpoint, outcome, notes), link eval_ datasets, per-model success rate stats
- **Unified delete** — Deleting a dataset removes from both DreamHub and HuggingFace; requires typing `permanently delete [name]` to confirm
- **Hours display** — Shown as `5h3m` instead of decimal
- **Custom domain** — `hub.dream-machines.eu` active

### Infra
- Renamed Vercel project and npm package to "dreamhub"
- Added admin user via Supabase auth
- Fixed GoTrue NULL token columns issue (requires empty strings, not NULLs)

---

## 2026-03-03 (session 2)

### Features
- **DreamHub rebrand** — Renamed from "Research OS" to "DreamHub"
- **HF rename/delete** — Server-side API routes for renaming and deleting HF datasets
- **New default tags** — Task, robot, status, and policy type tags auto-created during sync
- **SPEC.md** — Architecture documentation added

### Data Management
- Export: download all data as timestamped JSON backup
- Import: restore from JSON backup
- Reset: clear all data and re-sync from HuggingFace

### Dashboard
- Episodes by Tag breakdown alongside Hours by Tag
- Experiment Status breakdown (planned/in-progress/completed/failed)
- Checkpoints stat card showing total model iterations
- Clickable stat cards navigate to relevant pages

### Experiments Page
- Full experiment editing (name, notes, results, status)
- Dataset and model linking via search dropdowns

### Datasets Page
- Tag assignment UI with colored toggle buttons

### Models Page
- Tag assignment, description editing, iteration notes, config JSON editing
- trainedOn display for linked training datasets

### Graph Page
- Click-to-navigate on nodes

---

## 2026-03-03 (session 1)

- Full Next.js 16 app with 6 pages: Dashboard, Experiments, Graph, Datasets, Models, Tags
- Client-side architecture using localStorage + direct HuggingFace API calls
- Auto-sync: fetches all datasets and models from HF user `dopaul`
- Seed data: 4 hypotheses with 7 experiments pre-populated and linked to real HF assets
- Deployed to Vercel
- Fixed `.gitignore` blocking `research-os/src/lib/` (root `lib/` pattern)
- Fixed HF sync crash: Models API `createdAt`/`lastModified` fallback
