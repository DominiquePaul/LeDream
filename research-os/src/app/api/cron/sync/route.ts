import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import {
  getPaperByArxivId,
  getPaperByDOI,
  getPaperById,
  searchPaperByTitle,
} from "@/lib/semantic-scholar";
import { getServerSupabase } from "@/lib/supabase-server";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Cron-triggered sync: runs S2 sync + AI resolve in one pass.
 * Secured by CRON_SECRET header (Vercel cron sets this automatically).
 */
export async function GET(request: Request) {
  // Verify cron secret (Vercel sends this automatically for cron jobs)
  const authHeader = request.headers.get("authorization");
  const cronSecret = process.env.CRON_SECRET;
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Use service role key for cron (no user session)
  const supabase = getServerSupabase();

  const log: string[] = [];
  let synced = 0;
  let resolved = 0;
  let notFound = 0;
  let errors = 0;

  // ── Phase 1: Standard S2 sync (arxiv_id papers first) ──
  const { data: withArxiv } = await supabase
    .from("research_papers")
    .select("id, title, arxiv_id, semantic_scholar_id, citation_count")
    .is("semantic_scholar_id", null)
    .not("arxiv_id", "is", null)
    .order("updated_at", { ascending: true })
    .limit(15);

  const { data: withoutArxiv } = await supabase
    .from("research_papers")
    .select("id, title, arxiv_id, semantic_scholar_id, citation_count")
    .is("semantic_scholar_id", null)
    .is("arxiv_id", null)
    .order("updated_at", { ascending: true })
    .limit(5);

  const toSync = [...(withArxiv || []), ...(withoutArxiv || [])];

  for (const paper of toSync) {
    try {
      let s2Paper = null;

      if (paper.arxiv_id) {
        await delay(150);
        s2Paper = await getPaperByArxivId(paper.arxiv_id);
      }

      if (!s2Paper) {
        await delay(1500);
        s2Paper = await searchPaperByTitle(paper.title);
      }

      if (!s2Paper) {
        await supabase
          .from("research_papers")
          .update({ updated_at: new Date().toISOString() })
          .eq("id", paper.id);
        log.push(`sync:not_found: ${paper.title}`);
        notFound++;
        continue;
      }

      await updatePaper(supabase, paper.id, paper.citation_count, s2Paper);
      await syncAuthors(supabase, paper.id, s2Paper);
      synced++;
      log.push(`sync:ok: ${paper.title}`);
    } catch (err) {
      errors++;
      log.push(`sync:error: ${paper.title}: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }

  // ── Phase 2: AI resolve for remaining unresolved papers ──
  const { data: unresolved } = await supabase
    .from("research_papers")
    .select("id, title, arxiv_id, year, abstract")
    .is("semantic_scholar_id", null)
    .order("updated_at", { ascending: true })
    .limit(5);

  if (unresolved?.length && process.env.ANTHROPIC_API_KEY) {
    const anthropic = new Anthropic();

    for (const paper of unresolved) {
      try {
        const msg = await anthropic.messages.create({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 300,
          messages: [
            {
              role: "user",
              content: `I need to find this paper on Semantic Scholar. Provide the most likely identifiers.

Title: "${paper.title}"
${paper.year ? `Year: ${paper.year}` : ""}

Reply ONLY with a JSON object (no markdown):
{
  "arxiv_id": "YYMM.NNNNN or null",
  "doi": "full DOI string or null",
  "s2_id": "Semantic Scholar paper ID (40-char hex) or null",
  "alt_title": "exact title as it appears on Semantic Scholar, or null"
}`,
            },
          ],
        });

        const text = msg.content[0].type === "text" ? msg.content[0].text : "";
        let parsed: Record<string, string | null>;
        try {
          parsed = JSON.parse(text.trim());
        } catch {
          log.push(`resolve:parse_error: ${paper.title}`);
          errors++;
          continue;
        }

        let s2Paper = null;

        if (parsed.arxiv_id) {
          await delay(150);
          s2Paper = await getPaperByArxivId(parsed.arxiv_id);
          if (s2Paper) {
            await supabase
              .from("research_papers")
              .update({ arxiv_id: parsed.arxiv_id })
              .eq("id", paper.id);
          }
        }
        if (!s2Paper && parsed.s2_id) {
          await delay(150);
          s2Paper = await getPaperById(parsed.s2_id);
        }
        if (!s2Paper && parsed.doi) {
          await delay(150);
          s2Paper = await getPaperByDOI(parsed.doi);
        }
        if (!s2Paper && parsed.alt_title) {
          await delay(1500);
          s2Paper = await searchPaperByTitle(parsed.alt_title);
        }

        if (s2Paper) {
          await updatePaper(supabase, paper.id, null, s2Paper);
          resolved++;
          log.push(`resolve:ok: ${paper.title}`);
        } else {
          await supabase
            .from("research_papers")
            .update({ updated_at: new Date().toISOString() })
            .eq("id", paper.id);
          notFound++;
          log.push(`resolve:not_found: ${paper.title}`);
        }
      } catch (err) {
        errors++;
        log.push(`resolve:error: ${paper.title}: ${err instanceof Error ? err.message : "unknown"}`);
      }
    }
  }

  // ── Phase 3: Refresh citation counts for already-synced papers ──
  const { data: toRefresh } = await supabase
    .from("research_papers")
    .select("id, title, semantic_scholar_id, citation_count")
    .not("semantic_scholar_id", "is", null)
    .order("updated_at", { ascending: true })
    .limit(10);

  let refreshed = 0;
  for (const paper of toRefresh || []) {
    try {
      await delay(150);
      const s2Paper = await getPaperById(paper.semantic_scholar_id);
      if (s2Paper) {
        const updates: Record<string, unknown> = {
          citation_count: s2Paper.citationCount || 0,
          updated_at: new Date().toISOString(),
        };
        if (paper.citation_count && s2Paper.citationCount) {
          updates.citation_velocity = s2Paper.citationCount - paper.citation_count;
        }
        await supabase.from("research_papers").update(updates).eq("id", paper.id);
        refreshed++;
      }
    } catch {
      // non-critical, skip
    }
  }

  // Save sync log
  const summary = { synced, resolved, refreshed, notFound, errors, timestamp: new Date().toISOString() };
  await supabase.from("research_sync_log").insert({
    action: "cron_sync",
    summary: JSON.stringify(summary),
    details: JSON.stringify(log),
  });

  return NextResponse.json(summary);
}

async function updatePaper(
  supabase: ReturnType<typeof getServerSupabase>,
  paperId: string,
  existingCitationCount: number | null,
  s2Paper: {
    paperId: string;
    citationCount?: number;
    url?: string;
    abstract?: string;
    authors?: { authorId: string; name: string; affiliations?: string[]; hIndex?: number; paperCount?: number; homepage?: string }[];
  }
) {
  const updates: Record<string, unknown> = {
    semantic_scholar_id: s2Paper.paperId,
    citation_count: s2Paper.citationCount || 0,
    semantic_scholar_url: s2Paper.url || null,
    updated_at: new Date().toISOString(),
  };

  if (existingCitationCount && s2Paper.citationCount) {
    updates.citation_velocity = s2Paper.citationCount - existingCitationCount;
  }

  if (s2Paper.abstract) {
    const { data: current } = await supabase
      .from("research_papers")
      .select("abstract")
      .eq("id", paperId)
      .single();
    if (!current?.abstract) {
      updates.abstract = s2Paper.abstract;
    }
  }

  await supabase.from("research_papers").update(updates).eq("id", paperId);
}

async function syncAuthors(
  supabase: ReturnType<typeof getServerSupabase>,
  _paperId: string,
  s2Paper: {
    authors?: { authorId: string; name: string; affiliations?: string[]; hIndex?: number; paperCount?: number; homepage?: string }[];
  }
) {
  if (!s2Paper.authors?.length) return;

  for (const s2Author of s2Paper.authors.slice(0, 5)) {
    if (!s2Author.authorId) continue;

    const lastName = s2Author.name.split(" ").pop() || s2Author.name;
    const { data: dbAuthors } = await supabase
      .from("research_authors")
      .select("id, semantic_scholar_id")
      .ilike("name", `%${lastName}%`);

    const match = dbAuthors?.find(
      (a) => !a.semantic_scholar_id || a.semantic_scholar_id === s2Author.authorId
    );

    if (match) {
      const authorUpdates: Record<string, unknown> = {
        semantic_scholar_id: s2Author.authorId,
      };
      if (s2Author.affiliations?.[0]) authorUpdates.affiliation = s2Author.affiliations[0];
      if (s2Author.hIndex) authorUpdates.h_index = s2Author.hIndex;
      if (s2Author.paperCount) authorUpdates.paper_count = s2Author.paperCount;
      if (s2Author.homepage) authorUpdates.homepage_url = s2Author.homepage;

      await supabase.from("research_authors").update(authorUpdates).eq("id", match.id);
    }
  }
}
