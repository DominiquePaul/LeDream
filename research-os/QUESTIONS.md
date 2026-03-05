# Research OS - Open Questions

Rolling log of open questions. Resolved items go to SPEC.md (decisions).

---

## Resolved (moved to SPEC.md)

- Q1: Content format -> **MDX** with DB-stored prose notes
- Q2: Database -> **Supabase**, same project as DreamHub, shared auth, `public` schema with `research_` prefix (same as DreamHub tables — keeps join option open)
- Q3: Deployment -> **research.dream-machines.eu**, separate Vercel project, password protected
- Q4/Q5: Editing -> **Hybrid**: in-browser markdown editor (textarea + preview, Ctrl+S to save) for notes in DB, plus MDX files editable via IDE/Claude Code
- Q6: Context granularity -> **Two levels**: one-liners (always) + rich notes (unlimited, grow over time). Plus capsule synthesis per tag group.
- Q7: LLM conversations -> **All three**: in-app chat (Phase 4), context export (Phase 3), MCP server for Claude Code (Phase 3)
- Q8: External APIs -> **Semantic Scholar** (free). Smart batched cron to stay within limits.
- Q9: Refresh -> Weekly citations, daily discovery, on-add author affiliation. Cron jobs on Vercel Pro.
- Q11: Research frontier -> Dropped
- Q12: DreamHub experiment links -> Deferred to Phase 5
- Q14: Semantic clustering -> Yes, via embeddings (Phase 4)
- Q15: Reading velocity dashboard -> Yes, mini version (Phase 4)
- Q16: Mobile -> Responsive design for mobile + tablet throughout

---

## Action Required: Deploy to Vercel

The app is built, committed, and pushed to `main`. You need to create the Vercel project manually:

1. **Go to** https://vercel.com/new
2. **Import** the `DominiquePaul/dreammachines` GitHub repo
3. **Set Root Directory** to `research-os`
4. **Framework Preset**: Next.js (auto-detected)
5. **Environment Variables** — add these:
   - `NEXT_PUBLIC_SUPABASE_URL` = `https://tgmgiovecbqzbrqgvfoh.supabase.co`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = (same key as DreamHub — check DreamHub Vercel env vars)
6. **Deploy**
7. **Add Domain**: Go to Project Settings > Domains > Add `research.dream-machines.eu`
8. **DNS**: Vercel will show the exact DNS record needed. Add it on GoDaddy.

Note: I couldn't create the Vercel project programmatically (no CLI auth token, and the MCP doesn't have a create-project tool).

---

## Open Questions (for next sprint)

1. **Lineage graph: migrate to Supabase data?**
   Currently the lineage graph (`/lineage`) reads from the static `papers.json`. The same data is now in Supabase (`research_papers` + `research_paper_edges`). Should we migrate the graph to fetch from Supabase, or is the static approach fine for now?

2. **Existing visualization pages: when to start MDX migration?**
   The 20 interactive viz pages (`/methods/*`) still work as standalone React components. You said MDX sounds fine for migration. Should we start converting in the next sprint, or wait until you've used the new system for a while?

3. **Next sprint priorities: Phase 2 (Semantic Scholar) or polish Phase 1?**
   Phase 1 is functional. Before moving to Phase 2, is there anything in Phase 1 you want improved (UX, styling, additional fields, paper CRUD from UI, etc.)?
