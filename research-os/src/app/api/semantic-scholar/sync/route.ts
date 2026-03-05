import { NextResponse } from "next/server";
import {
  getPaperByArxivId,
  searchPaperByTitle,
} from "@/lib/semantic-scholar";
import { getServerSupabase } from "@/lib/supabase-server";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function POST(request: Request) {
  const supabase = getServerSupabase(request.headers.get("authorization"));
  const body = await request.json().catch(() => ({}));
  const batchSize = body.batchSize || 10;

  // Prioritize papers with arxiv_ids (fast direct lookup, no rate limit)
  const { data: withArxiv } = await supabase
    .from("research_papers")
    .select("id, title, arxiv_id, semantic_scholar_id, citation_count")
    .is("semantic_scholar_id", null)
    .not("arxiv_id", "is", null)
    .order("updated_at", { ascending: true })
    .limit(batchSize);

  // Fill remaining slots with papers that need title search
  const remaining = batchSize - (withArxiv?.length || 0);
  const { data: withoutArxiv } = remaining > 0
    ? await supabase
        .from("research_papers")
        .select("id, title, arxiv_id, semantic_scholar_id, citation_count")
        .is("semantic_scholar_id", null)
        .is("arxiv_id", null)
        .order("updated_at", { ascending: true })
        .limit(remaining)
    : { data: [] };

  const papers = [...(withArxiv || []), ...(withoutArxiv || [])];

  if (!papers.length) {
    return NextResponse.json({ message: "All papers synced!", synced: 0, notFound: 0 });
  }

  const results: { id: string; title: string; status: string; citationCount?: number; error?: string }[] = [];

  for (const paper of papers) {
    try {
      let s2Paper = null;

      if (paper.arxiv_id) {
        await delay(150);
        s2Paper = await getPaperByArxivId(paper.arxiv_id);
      }

      if (!s2Paper) {
        // Title search — needs longer delay to avoid 429
        await delay(1500);
        s2Paper = await searchPaperByTitle(paper.title);
      }

      if (!s2Paper) {
        // Mark updated_at so this paper goes to the back of the queue
        await supabase
          .from("research_papers")
          .update({ updated_at: new Date().toISOString() })
          .eq("id", paper.id);
        results.push({ id: paper.id, title: paper.title, status: "not_found" });
        continue;
      }

      const updates: Record<string, unknown> = {
        semantic_scholar_id: s2Paper.paperId,
        citation_count: s2Paper.citationCount || 0,
        semantic_scholar_url: s2Paper.url || null,
        updated_at: new Date().toISOString(),
      };

      if (paper.citation_count && s2Paper.citationCount) {
        updates.citation_velocity = s2Paper.citationCount - paper.citation_count;
      }

      if (s2Paper.abstract) {
        const { data: current } = await supabase
          .from("research_papers")
          .select("abstract")
          .eq("id", paper.id)
          .single();
        if (!current?.abstract) {
          updates.abstract = s2Paper.abstract;
        }
      }

      await supabase.from("research_papers").update(updates).eq("id", paper.id);

      // Update author data
      if (s2Paper.authors?.length) {
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

      results.push({
        id: paper.id,
        title: paper.title,
        status: "synced",
        citationCount: s2Paper.citationCount,
      });
    } catch (err) {
      results.push({
        id: paper.id,
        title: paper.title,
        status: "error",
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  }

  return NextResponse.json({
    synced: results.filter((r) => r.status === "synced").length,
    notFound: results.filter((r) => r.status === "not_found").length,
    errors: results.filter((r) => r.status === "error").length,
    total: papers.length,
    results,
  });
}
