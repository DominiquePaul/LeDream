# Research OS - Application Specification

Living document tracking all decided features, architecture, and design decisions.

## Overview
Research OS is a personal "second brain" for robotics research at Dream Machines. It helps track papers, maintain reading notes with interactive visualizations, discover new research, identify promising authors, and use curated paper collections as context for LLM-assisted reasoning.

**Repo location**: `research-os/` subdirectory of dreammachines monorepo
**Live URL**: `research.dream-machines.eu` (pending Vercel project creation — see QUESTIONS.md)
**GitHub**: `DominiquePaul/dreammachines` (main branch)

## Architecture

### Stack
- **Frontend**: Next.js (App Router), TypeScript, MDX for paper pages
- **Database**: Supabase (Postgres) — same project as DreamHub (`tgmgiovecbqzbrqgvfoh`), shared auth, separate tables with `research_` prefix
- **Embeddings**: pgvector on Supabase for semantic paper search and capsule context selection
- **Auth**: Shared with DreamHub via Supabase Auth. Password protected (internal tool).
- **Deployment**: Vercel, separate project from DreamHub. Custom domain: `research.dream-machines.eu`
- **External APIs**: Semantic Scholar (free) for citation counts, author affiliations, references, discovery
- **LLM**: Claude API for summaries, capsule synthesis, in-app chat, and on-demand paper analysis
- **MCP Server**: Exposes Research OS data to Claude Code and other MCP clients

### Content Format: MDX + DB Hybrid
- **Paper notes (prose)**: Stored in Supabase, editable in-browser via lightweight markdown editor
- **Interactive widgets**: React components in code (`src/components/widgets/`), imported into MDX pages
- **MDX pages**: Live in repo (`src/app/papers/[slug]/page.mdx`), combine DB-fetched notes with embedded widget components
- **Dual editing**: Notes editable in-browser (writes to DB) OR via IDE/Claude Code (edits MDX files). Both paths coexist.

### MCP Integration
Research OS exposes an MCP server with tools:
- `get_capsule_context(tag)` — assembled context for a tag group
- `get_paper_notes(slug)` — rich notes for a single paper
- `search_papers(query)` — semantic search across paper embeddings
- `list_papers_by_tag(tag)` — all papers in a tag group

This allows Claude Code, claude.ai, and the in-app chat to all consume the same data layer.

## Data Model

### Supabase Tables (research_ prefix)

**research_papers**
- id, slug, title, year, arxiv_id, abstract, one_liner, category
- embedding (vector) — for semantic search
- created_at, updated_at

**research_authors**
- id, name, affiliation, affiliation_country
- Populated from Semantic Scholar at paper add time

**research_paper_authors** (join)
- paper_id, author_id

**research_tags**
- id, name, color, description

**research_paper_tags** (join)
- paper_id, tag_id

**research_notes**
- id, paper_id, content (markdown), granularity (oneliner | rich)
- Rich notes grow over time as LLM conversations discover new information
- embedding (vector) — for semantic clustering

**research_capsule_syntheses**
- id, tag_id, synthesis_text, generated_at
- Auto-generated summary of how papers in the tag group relate

**research_citations**
- id, citing_paper_id, cited_paper_id, is_in_collection

**research_discovery_queue**
- id, paper_title, arxiv_id, citing_paper_ids (which papers in collection it cites)
- status: unread | to-read | dismissed | added
- discovered_at

**research_paper_edges** (lineage graph)
- source_paper_id, target_paper_id

### Note Granularity
Two levels:
- **One-liners** (~10 words): Always included in capsule context. Already exist for all 74 papers.
- **Rich notes** (unlimited length, grows over time): Per-paper detailed notes. The LLM can spawn subagents to read original papers and augment these notes with new information discovered during conversations. Notes become a living cache that gets richer with every interaction.

### Capsule Context Assembly
For a tag group, context is assembled as:
1. One-liners for ALL papers in the group
2. Capsule synthesis (auto-generated relationship summary)
3. Rich notes for all papers (modern LLMs handle 200K tokens; 44 papers x 2 pages each = ~60K tokens, well within limits)
4. On-demand: subagent reads original paper PDF/arxiv for questions not covered by existing notes, then saves findings back to rich notes for future reuse

## External Data Integration

### Semantic Scholar API (free, no payment)
- **Citation counts & velocity**: Weekly cron refresh, batched (e.g., 20 papers/run, full rotation in ~4 weeks)
- **Author affiliations**: Fetched at paper add time, stored in research_authors
- **References**: Which papers cite which, cross-referenced with collection
- **Discovery feed**: Daily cron checks for new papers citing papers in collection. Results go to research_discovery_queue with status tracking (unread -> to-read -> added/dismissed)

### Smart Rate Limiting
Semantic Scholar allows 100 req/sec (free). Cron jobs batch requests across runs so the full collection refreshes over multiple days rather than all at once. No risk of hitting limits.

### Cron Schedule
- **Weekly**: Citation count refresh (batched across collection)
- **Daily**: New paper discovery (check for new citations of collection papers)
- **On add**: Author affiliation lookup

## Pages / Views

### Paper Page (`/papers/[slug]`)
- MDX page with embedded interactive widgets
- In-browser note editor (markdown, writes to DB)
- Paper metadata: authors, year, category, tags, citation count
- Related papers (from lineage graph + semantic similarity)
- Link to arxiv/original paper

### Collection (`/`)
- All papers with search, tag filter, category filter
- Semantic search powered by pgvector embeddings
- Mini dashboard: paper count, reading velocity, coverage by category

### Lineage Graph (`/lineage`)
- Existing dagre DAG visualization (preserve and enhance)
- Click-to-navigate to paper pages

### Capsules (`/capsules/[tag]`)
- Papers grouped by tag
- Capsule synthesis (auto-generated)
- "Export context" button — copies assembled context for pasting into external LLM
- "Open chat" — in-app LLM conversation with capsule context pre-loaded

### Discovery Feed (`/discover`)
- New papers citing collection papers (from daily cron)
- Status: unread / to-read / dismissed / added
- One-click add to collection

### Author Network (`/authors`)
- Authors across collection, grouped by affiliation
- Filter: "European university, appears in 3+ papers"
- Affiliation data from Semantic Scholar

### In-App Chat (`/chat`)
- Chat UI calling Claude API
- Select capsule/tag to load as context
- Subagent capability: can read original papers and augment notes
- Conversation history stored in DB

## Planned Features (by phase)

### Phase 1: Foundation — DONE (2026-03-05)
- [x] Supabase tables created (research_papers, research_authors, research_paper_authors, research_tags, research_paper_tags, research_notes, research_paper_edges, research_capsule_syntheses, research_discovery_queue) with RLS
- [x] pgvector extension enabled
- [x] 65 papers + 76 edges + 52 authors migrated from papers.json to Supabase
- [x] Next.js 16 + React 19 + Tailwind v4 + Supabase client
- [x] Auth gate (shared Supabase Auth with DreamHub)
- [x] Collection home page with search, category filters, tag filters
- [x] Paper detail page (`/papers/[slug]`) with metadata, tag assignment, markdown note editor (edit/preview, Ctrl+S)
- [x] Tag management page (`/tags`) — create/delete tags
- [x] Capsule view (`/capsules/[tagId]`) with "Export Context for LLM" button
- [x] Responsive sidebar with categories, tags, visualization links
- [x] All 20 existing interactive viz pages preserved at `/methods/*` with MathJax
- [x] Lineage graph preserved at `/lineage` (still uses static JSON — migration deferred)
- [ ] Deploy to Vercel (manual step required — see QUESTIONS.md)

### Phase 2: External Data
- Semantic Scholar integration (citation counts, author data, references)
- Cron jobs for weekly citation refresh + daily discovery
- Discovery queue with to-read list
- Author network page

### Phase 3: LLM & Context
- Paper embeddings via pgvector
- Capsule context assembly + export
- Rich note augmentation (subagents read papers, save findings)
- MCP server for Claude Code integration
- Semantic search across collection

### Phase 4: Intelligence
- In-app LLM chat with capsule context
- Citation velocity visualization
- Semantic similarity clustering (auto-suggest tags)
- "Confirmed learnings" detection — findings validated across multiple papers
- Mini reading velocity dashboard

### Phase 5: DreamHub Bridge (future)
- Link papers to DreamHub experiments ("I read this, here's my reproduction")
- Cross-app LLM context (experiments + papers in same conversation)

## Existing Components Worth Preserving
- **Interactive visualizations**: Canvas-based animations, slider controls, toggle buttons for each paper's key concepts. Core differentiator.
- **Lineage graph**: DAG visualization of paper relationships using dagre layout.
- **MathJax integration**: LaTeX rendering via better-react-mathjax.
- **CoreIdeasAndFeatures**: Expandable accordion for paper key ideas with detail text.

## Paper Data (migrated to Supabase)
65 papers tracked across categories:
- **rl**: DQN, DDPG, PPO, SAC, BCQ, BEAR, CQL, RLHF, RLPD, IQL, TD3, GRPO, DPO, etc.
- **wm (world models)**: World Models, PlaNet, Dreamer, DreamerV2, DreamerV3, V-JEPA, V-JEPA 2, TD-MPC
- **robotics**: Robot Controllers, FAST, SERL, ACT, pi0, DayDreamer
- **technique**: LayerNorm, VAE, CMA-ES, BYOL, MAE, ViT, RoPE, STE, BPE

20 papers with existing interactive visualization pages: DQN, DDPG, PPO, SAC, BCQ, BEAR, CQL, RLHF, RLPD, IQL, LayerNorm, World Models, PlaNet, Dreamer, DreamerV2, DreamerV3, V-JEPA, V-JEPA 2, Robot Controllers, FAST

## Business Context
- Dream Machines is a robotics company (dream-machines.eu)
- Edge is identifying promising research and replicating/combining results into products
- Author identification serves as hiring intelligence (European university + 3+ papers = interesting)
- Paper collections feed into technical decision-making
- Shared Supabase with DreamHub enables future cross-app reasoning
