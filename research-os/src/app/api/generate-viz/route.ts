import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { getServerSupabase } from "@/lib/supabase-server";

const SYSTEM_PROMPT =
  "You are an expert at creating clear, beautiful, interactive HTML visualizations for academic research papers. " +
  "Generate a SINGLE self-contained HTML document that can be embedded in an iframe. " +
  "Use inline CSS and vanilla JavaScript only (no external dependencies except CDN links to D3.js or Chart.js if needed). " +
  "The visualization should use a dark theme (background: #161822, text: #e2e4e9, accent: #638bd4). " +
  "Include interactive elements where appropriate (hover effects, toggles, sliders). " +
  "Structure the content with clear sections using h2/h3 headings. " +
  "Keep the HTML compact but well-formatted. " +
  "Do NOT include any markdown fences — return ONLY the raw HTML.";

export async function POST(request: Request) {
  const supabase = getServerSupabase(request.headers.get("authorization"));
  const { paperId, prompt } = await request.json();

  if (!paperId || !prompt) {
    return NextResponse.json({ error: "paperId and prompt required" }, { status: 400 });
  }

  const anthropic = new Anthropic();

  const msg = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 8000,
    system: SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  });

  const html = msg.content[0].type === "text" ? msg.content[0].text : "";

  // Save to DB
  await supabase
    .from("research_papers")
    .update({ visualization_html: html, updated_at: new Date().toISOString() })
    .eq("id", paperId);

  return NextResponse.json({ html });
}
