// Semantic Scholar API client
// Free API: 100 requests/sec, no auth required (but API key gets higher limits)

const S2_BASE = "https://api.semanticscholar.org/graph/v1";

interface S2Paper {
  paperId: string;
  externalIds?: { ArXiv?: string; DOI?: string };
  title: string;
  year?: number;
  abstract?: string;
  citationCount?: number;
  influentialCitationCount?: number;
  citations?: S2Citation[];
  references?: S2Citation[];
  authors?: S2Author[];
  url?: string;
}

interface S2Citation {
  paperId: string;
  title?: string;
  year?: number;
  externalIds?: { ArXiv?: string };
  authors?: { authorId: string; name: string }[];
  abstract?: string;
  citationCount?: number;
}

interface S2Author {
  authorId: string;
  name: string;
  affiliations?: string[];
  hIndex?: number;
  paperCount?: number;
  homepage?: string;
}

const PAPER_FIELDS = "paperId,externalIds,title,year,abstract,citationCount,influentialCitationCount,url,authors.authorId,authors.name,authors.affiliations,authors.hIndex,authors.paperCount,authors.homepage";

async function s2Fetch<T>(path: string): Promise<T | null> {
  const headers: Record<string, string> = {};
  const apiKey = process.env.S2_API_KEY;
  if (apiKey) headers["x-api-key"] = apiKey;

  const res = await fetch(`${S2_BASE}${path}`, { headers });
  if (!res.ok) {
    console.error(`S2 API error: ${res.status} for ${path}`);
    return null;
  }
  return res.json();
}

export async function searchPaperByTitle(title: string): Promise<S2Paper | null> {
  // Try exact title first
  const result = await s2Search(title);
  if (result) return result;

  // Try simplified title (remove common prefixes/suffixes, punctuation)
  const simplified = title
    .replace(/^(a |an |the )/i, "")
    .replace(/[:\-–—]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (simplified !== title) {
    const result2 = await s2Search(simplified);
    if (result2) return result2;
  }

  return null;
}

async function s2Search(query: string): Promise<S2Paper | null> {
  const res = await fetch(
    `${S2_BASE}/paper/search?query=${encodeURIComponent(query)}&limit=3&fields=${PAPER_FIELDS}`,
    {
      headers: process.env.S2_API_KEY ? { "x-api-key": process.env.S2_API_KEY } : {},
    }
  );
  if (!res.ok) return null;
  const data = await res.json();
  if (!data.data?.length) return null;

  // If only 1 result, return it
  if (data.data.length === 1) return data.data[0];

  // Pick the best match by title similarity
  const queryLower = query.toLowerCase();
  const scored = data.data.map((p: S2Paper) => ({
    paper: p,
    score: titleSimilarity(queryLower, p.title.toLowerCase()),
  }));
  scored.sort((a: { score: number }, b: { score: number }) => b.score - a.score);
  return scored[0].score > 0.3 ? scored[0].paper : null;
}

function titleSimilarity(a: string, b: string): number {
  const wordsA = new Set(a.split(/\s+/).filter((w) => w.length > 2));
  const wordsB = new Set(b.split(/\s+/).filter((w) => w.length > 2));
  if (wordsA.size === 0 || wordsB.size === 0) return 0;
  let overlap = 0;
  for (const w of wordsA) if (wordsB.has(w)) overlap++;
  return overlap / Math.max(wordsA.size, wordsB.size);
}

export async function getPaperByArxivId(arxivId: string): Promise<S2Paper | null> {
  return s2Fetch<S2Paper>(`/paper/ArXiv:${arxivId}?fields=${PAPER_FIELDS}`);
}

export async function getPaperByDOI(doi: string): Promise<S2Paper | null> {
  return s2Fetch<S2Paper>(`/paper/DOI:${doi}?fields=${PAPER_FIELDS}`);
}

export async function getPaperById(s2Id: string): Promise<S2Paper | null> {
  return s2Fetch<S2Paper>(`/paper/${s2Id}?fields=${PAPER_FIELDS}`);
}

export async function getPaperCitations(s2Id: string, limit = 100): Promise<S2Citation[]> {
  const res = await s2Fetch<{ data: S2Citation[] }>(
    `/paper/${s2Id}/citations?fields=paperId,title,year,externalIds,authors,abstract,citationCount&limit=${limit}`
  );
  return res?.data || [];
}

export async function getAuthorDetails(authorId: string): Promise<S2Author | null> {
  return s2Fetch<S2Author>(
    `/author/${authorId}?fields=authorId,name,affiliations,hIndex,paperCount,homepage`
  );
}

export type { S2Paper, S2Citation, S2Author };
