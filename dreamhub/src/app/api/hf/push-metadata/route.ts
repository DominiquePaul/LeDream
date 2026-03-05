import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const HF_API = "https://huggingface.co/api";

export async function POST(req: NextRequest) {
  // Verify auth
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
  const {
    data: { user },
  } = await supabase.auth.getUser(authHeader.split(" ")[1]);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { hfId, tags, metadata } = await req.json();
  if (!hfId) {
    return NextResponse.json({ error: "hfId required" }, { status: 400 });
  }

  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) {
    return NextResponse.json(
      { error: "HF_TOKEN not configured" },
      { status: 500 },
    );
  }

  // Build the README content with YAML front matter
  const yamlTags = (tags || []).map((t: string) => `  - ${t}`).join("\n");
  const meta = metadata || {};

  let readme = "---\n";
  if (yamlTags) {
    readme += `tags:\n${yamlTags}\n`;
  }
  readme += "---\n\n";
  readme += `# ${hfId.split("/")[1] || hfId}\n\n`;
  readme += `*Managed by [DreamHub](https://hub.dream-machines.eu)*\n\n`;

  if (meta.collectionConditions) {
    readme += `## Collection Conditions\n${meta.collectionConditions}\n\n`;
  }
  if (meta.teleoperatorInstructions) {
    readme += `## Teleoperator Instructions\n${meta.teleoperatorInstructions}\n\n`;
  }
  if (meta.knownIssues && meta.knownIssues.length > 0) {
    readme += `## Known Issues\n`;
    for (const issue of meta.knownIssues) {
      readme += `- ${issue}\n`;
    }
    readme += "\n";
  }
  if (meta.notes) {
    readme += `## Notes\n${meta.notes}\n\n`;
  }
  if (meta.estimatedHours) {
    readme += `**Estimated Hours:** ${meta.estimatedHours}\n`;
  }
  if (meta.episodeCount) {
    readme += `**Episode Count:** ${meta.episodeCount}\n`;
  }

  // Push README to HF via the API
  const encoder = new TextEncoder();
  const content = encoder.encode(readme);
  const base64Content = Buffer.from(content).toString("base64");

  const res = await fetch(
    `${HF_API}/datasets/${hfId}/commit/main`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${hfToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        summary: "Update metadata from DreamHub",
        files: [
          {
            path: "README.md",
            encoding: "base64",
            content: base64Content,
          },
        ],
      }),
    },
  );

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json(
      { error: `HF API error ${res.status}: ${text}` },
      { status: res.status },
    );
  }

  return NextResponse.json({ ok: true });
}
