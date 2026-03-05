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
 * AI-assisted resolver for papers that failed standard S2 sync.
 * Uses Claude to figure out the correct identifier for each paper.
 */
export async function POST(request: Request) {
  const supabase = getServerSupabase(request.headers.get("authorization"));

  // Find papers that still don't have S2 IDs
  const { data: unresolved } = await supabase
    .from("research_papers")
    .select("id, title, arxiv_id, year, abstract")
    .is("semantic_scholar_id", null)
    .order("updated_at", { ascending: true })
    .limit(10);

  if (!unresolved?.length) {
    return NextResponse.json({ message: "All papers resolved!", resolved: 0 });
  }

  const anthropic = new Anthropic();
  const results: { id: string; title: string; status: string; method?: string }[] = [];

  for (const paper of unresolved) {
    try {
      // Step 1: If it has arxiv_id, try direct lookup first
      if (paper.arxiv_id) {
        await delay(150);
        const s2 = await getPaperByArxivId(paper.arxiv_id);
        if (s2) {
          await updatePaper(supabase, paper.id, s2);
          results.push({ id: paper.id, title: paper.title, status: "resolved", method: "arxiv" });
          continue;
        }
      }

      // Step 2: Ask Claude to identify the paper and provide lookup info
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
  "alt_title": "exact title as it appears on Semantic Scholar, or null",
  "confidence": "high/medium/low"
}`,
          },
        ],
      });

      const text = msg.content[0].type === "text" ? msg.content[0].text : "";
      let parsed: {
        arxiv_id?: string | null;
        doi?: string | null;
        s2_id?: string | null;
        alt_title?: string | null;
        confidence?: string;
      };
      try {
        parsed = JSON.parse(text.trim());
      } catch {
        results.push({ id: paper.id, title: paper.title, status: "parse_error" });
        continue;
      }

      // Step 3: Try each identifier Claude suggested
      let s2Paper = null;

      if (parsed.arxiv_id) {
        await delay(150);
        s2Paper = await getPaperByArxivId(parsed.arxiv_id);
        if (s2Paper) {
          // Also save the arxiv_id we discovered
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
        await updatePaper(supabase, paper.id, s2Paper);
        results.push({
          id: paper.id,
          title: paper.title,
          status: "resolved",
          method: parsed.arxiv_id && s2Paper ? "ai_arxiv" : parsed.s2_id ? "ai_s2id" : parsed.doi ? "ai_doi" : "ai_title",
        });
      } else {
        // Bump updated_at so it goes to back of queue
        await supabase
          .from("research_papers")
          .update({ updated_at: new Date().toISOString() })
          .eq("id", paper.id);
        results.push({ id: paper.id, title: paper.title, status: "not_found" });
      }
    } catch (err) {
      results.push({
        id: paper.id,
        title: paper.title,
        status: "error",
      });
    }
  }

  return NextResponse.json({
    resolved: results.filter((r) => r.status === "resolved").length,
    notFound: results.filter((r) => r.status === "not_found").length,
    errors: results.filter((r) => r.status === "error" || r.status === "parse_error").length,
    total: unresolved.length,
    results,
  });
}

async function updatePaper(
  supabase: ReturnType<typeof getServerSupabase>,
  paperId: string,
  s2Paper: { paperId: string; citationCount?: number; url?: string; abstract?: string }
) {
  const updates: Record<string, unknown> = {
    semantic_scholar_id: s2Paper.paperId,
    citation_count: s2Paper.citationCount || 0,
    semantic_scholar_url: s2Paper.url || null,
    updated_at: new Date().toISOString(),
  };

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
