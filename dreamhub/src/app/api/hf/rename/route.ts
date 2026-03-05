import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

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

  const { fromRepo, toRepo } = await req.json();
  if (!fromRepo || !toRepo) {
    return NextResponse.json(
      { error: "fromRepo and toRepo required" },
      { status: 400 },
    );
  }

  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) {
    return NextResponse.json(
      { error: "HF_TOKEN not configured on server" },
      { status: 500 },
    );
  }

  const res = await fetch("https://huggingface.co/api/repos/move", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${hfToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ fromRepo, toRepo, type: "dataset" }),
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json(
      { error: `HF API error ${res.status}: ${text}` },
      { status: res.status },
    );
  }

  return NextResponse.json({ ok: true });
}
