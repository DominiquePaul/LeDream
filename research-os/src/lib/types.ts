export interface Paper {
  id: string;
  slug: string | null;
  title: string;
  year: number | null;
  arxiv_id: string | null;
  abstract: string | null;
  one_liner: string | null;
  category: string;
  semantic_scholar_id: string | null;
  semantic_scholar_url: string | null;
  citation_count: number;
  citation_velocity: number;
  project_url: string | null;
  visualization_html: string | null;
  created_at: string;
  updated_at: string;
  // Joined data
  authors?: Author[];
  tags?: Tag[];
  notes?: Note[];
}

export interface Author {
  id: string;
  name: string;
  affiliation: string | null;
  affiliation_country: string | null;
  semantic_scholar_id: string | null;
  h_index: number | null;
  paper_count: number | null;
  homepage_url: string | null;
  created_at: string;
}

export interface Tag {
  id: string;
  name: string;
  color: string;
  description: string | null;
  created_at: string;
}

export interface Note {
  id: string;
  paper_id: string;
  granularity: "oneliner" | "rich";
  content: string;
  created_at: string;
  updated_at: string;
}

export interface PaperEdge {
  source_id: string;
  target_id: string;
}

export interface DiscoveryItem {
  id: string;
  paper_title: string;
  arxiv_id: string | null;
  authors: string | null;
  year: number | null;
  abstract: string | null;
  citing_paper_ids: string[];
  status: "unread" | "to-read" | "dismissed" | "added";
  discovered_at: string;
}

export interface CapsuleSynthesis {
  id: string;
  tag_id: string;
  synthesis_text: string;
  generated_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  tag_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface Citation {
  id: string;
  citing_paper_id: string;
  cited_paper_id: string;
  is_in_collection: boolean;
  created_at: string;
}

// Category display info
export const CATEGORIES: Record<string, { label: string; color: string }> = {
  rl: { label: "Reinforcement Learning", color: "#4a9b7f" },
  wm: { label: "World Models", color: "#6b73b5" },
  robotics: { label: "Robotics", color: "#c4884d" },
  technique: { label: "Techniques", color: "#638bd4" },
};
