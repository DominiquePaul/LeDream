import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import {
  getPaperByArxivId,
  searchPaperByTitle,
} from "@/lib/semantic-scholar";

function getSupabase() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function POST(request: Request) {
  const supabase = getSupabase();
  const body = await request.json().catch(() => ({}));
  const batchSize = body.batchSize || 10;

  // Fetch papers: unsynced first, or all if force
  const query = supabase
    .from("research_papers")
    .select("id, title, arxiv_id, semantic_scholar_id, citation_count")
    .order("updated_at", { ascending: true })
    .limit(batchSize);

  if (!body.force) {
    query.is("semantic_scholar_id", null);
  }

  const { data: papers, error } = await query;
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  if (!papers?.length) return NextResponse.json({ message: "All papers synced", synced: 0 });

  const results: { id: string; title: string; status: string; citationCount?: number; error?: string }[] = [];

  for (const paper of papers) {
    try {
      // Prefer direct arxiv lookup (no rate limit) over title search
      let s2Paper = null;
      if (paper.arxiv_id) {
        await delay(150);
        s2Paper = await getPaperByArxivId(paper.arxiv_id);
      }

      if (!s2Paper) {
        // Title search is rate-limited, add longer delay
        await delay(1000);
        s2Paper = await searchPaperByTitle(paper.title);
      }

      if (!s2Paper) {
        results.push({ id: paper.id, title: paper.title, status: "not_found" });
        continue;
      }

      // Update paper with S2 data
      const updates: Record<string, unknown> = {
        semantic_scholar_id: s2Paper.paperId,
        citation_count: s2Paper.citationCount || 0,
        semantic_scholar_url: s2Paper.url || null,
        updated_at: new Date().toISOString(),
      };

      // Citation velocity = change since last sync
      if (paper.citation_count && s2Paper.citationCount) {
        updates.citation_velocity = s2Paper.citationCount - paper.citation_count;
      }

      // Fill in abstract if missing
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
    results,
  });
}
