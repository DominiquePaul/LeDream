import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { getServerSupabase } from "@/lib/supabase-server";

/**
 * AI-enrichment: finds project URLs for papers.
 */
export async function POST(request: Request) {
  const supabase = getServerSupabase(request.headers.get("authorization"));

  const { data: papers } = await supabase
    .from("research_papers")
    .select("id, title, year, arxiv_id")
    .is("project_url", null)
    .order("updated_at", { ascending: true })
    .limit(10);

  if (!papers?.length) {
    return NextResponse.json({ message: "All papers enriched!", enriched: 0 });
  }

  const anthropic = new Anthropic();

  const paperList = papers
    .map((p) => `- "${p.title}" (${p.year || "?"})${p.arxiv_id ? ` [arXiv: ${p.arxiv_id}]` : ""}`)
    .join("\n");

  const msg = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 2000,
    messages: [
      {
        role: "user",
        content: `For each of these research papers, provide the official project page URL if one exists.
Project pages are official websites by the authors (e.g., danijar.com/dreamer, sites.google.com/..., github project pages with demos).
Do NOT include arxiv links, GitHub code-only repos, or general lab pages.

Papers:
${paperList}

Return a JSON array (no markdown):
[{"title": "exact title", "project_url": "https://... or null"}]

Only include entries where you're confident a real project page exists.`,
      },
    ],
  });

  const text = msg.content[0].type === "text" ? msg.content[0].text : "[]";
  let results: { title: string; project_url: string | null }[];
  try {
    results = JSON.parse(text.trim());
  } catch {
    return NextResponse.json({ error: "Failed to parse AI response", enriched: 0 });
  }

  let enriched = 0;
  for (const result of results) {
    if (!result.project_url) continue;
    const paper = papers.find((p) => p.title === result.title);
    if (paper) {
      await supabase
        .from("research_papers")
        .update({ project_url: result.project_url })
        .eq("id", paper.id);
      enriched++;
    }
  }

  return NextResponse.json({ enriched, total: papers.length });
}
