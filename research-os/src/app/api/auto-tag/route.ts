import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { getServerSupabase } from "@/lib/supabase-server";

export async function POST(request: Request) {
  const supabase = getServerSupabase(request.headers.get("authorization"));
  const body = await request.json().catch(() => ({}));
  const { paperIds } = body;

  // Fetch papers (optionally filtered by IDs)
  let query = supabase
    .from("research_papers")
    .select("id, title, one_liner, abstract, category");

  if (paperIds?.length) {
    query = query.in("id", paperIds);
  }

  const { data: papers } = await query;
  if (!papers?.length) {
    return NextResponse.json({ suggestions: [] });
  }

  // Fetch existing tags
  const { data: tags } = await supabase.from("research_tags").select("id, name, description");
  if (!tags?.length) {
    return NextResponse.json({ suggestions: [], message: "No tags exist yet" });
  }

  // Fetch existing assignments
  const { data: existingAssignments } = await supabase
    .from("research_paper_tags")
    .select("paper_id, tag_id");

  const assignedSet = new Set(
    (existingAssignments || []).map((a) => `${a.paper_id}:${a.tag_id}`)
  );

  const anthropic = new Anthropic();

  const paperList = papers
    .map((p) => `- ID: ${p.id} | Title: "${p.title}" | Category: ${p.category}${p.one_liner ? ` | Summary: ${p.one_liner}` : ""}`)
    .join("\n");

  const tagList = tags
    .map((t) => `- "${t.name}" (ID: ${t.id})${t.description ? `: ${t.description}` : ""}`)
    .join("\n");

  const msg = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 2000,
    messages: [
      {
        role: "user",
        content: `Given these research papers and available tags, suggest which tags should be assigned to which papers.

Papers:
${paperList}

Available Tags:
${tagList}

Return a JSON array of suggestions (no markdown):
[{"paper_id": "...", "tag_id": "...", "confidence": "high/medium/low", "reason": "brief reason"}]

Only suggest assignments where the match is meaningful. Skip papers that already fit no tags.`,
      },
    ],
  });

  const text = msg.content[0].type === "text" ? msg.content[0].text : "[]";
  let suggestions: { paper_id: string; tag_id: string; confidence: string; reason: string }[];
  try {
    suggestions = JSON.parse(text.trim());
  } catch {
    return NextResponse.json({ suggestions: [], error: "Failed to parse AI response" });
  }

  // Filter out already-assigned pairs
  const filtered = suggestions.filter(
    (s) => !assignedSet.has(`${s.paper_id}:${s.tag_id}`)
  );

  // Enrich with paper titles and tag names for display
  const paperMap = new Map(papers.map((p) => [p.id, p.title]));
  const tagMap = new Map(tags.map((t) => [t.id, t.name]));

  const enriched = filtered.map((s) => ({
    ...s,
    paper_title: paperMap.get(s.paper_id) || "Unknown",
    tag_name: tagMap.get(s.tag_id) || "Unknown",
  }));

  return NextResponse.json({ suggestions: enriched });
}
