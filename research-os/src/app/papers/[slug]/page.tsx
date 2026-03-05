"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MathJaxContext } from "better-react-mathjax";
import { useResearch } from "@/components/Shell";
import { fetchPaperBySlug } from "@/lib/supabase-db";
import { CATEGORIES } from "@/lib/types";
import { getVizComponent, hasVizComponent } from "@/lib/viz-registry";
import type { Paper } from "@/lib/types";

const mathjaxConfig = {
  loader: { load: ["[tex]/ams"] },
  tex: {
    packages: { "[+]": ["ams"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"]],
  },
};

function NoteEditor({ paper, onSave }: { paper: Paper; onSave: () => void }) {
  const { upsertNote } = useResearch();
  const richNote = paper.notes?.find((n) => n.granularity === "rich");
  const [content, setContent] = useState(richNote?.content || "");
  const [tab, setTab] = useState<"edit" | "preview">("edit");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await upsertNote(paper.id, content, "rich");
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      onSave();
    } catch (err) {
      console.error("Failed to save note:", err);
    } finally {
      setSaving(false);
    }
  }, [paper.id, content, upsertNote, onSave]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        handleSave();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleSave]);

  return (
    <div className="note-editor">
      <div className="note-editor__tabs">
        <button
          className={`note-editor__tab ${tab === "edit" ? "note-editor__tab--active" : ""}`}
          onClick={() => setTab("edit")}
        >
          Edit
        </button>
        <button
          className={`note-editor__tab ${tab === "preview" ? "note-editor__tab--active" : ""}`}
          onClick={() => setTab("preview")}
        >
          Preview
        </button>
      </div>
      {tab === "edit" ? (
        <textarea
          className="note-editor__textarea"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Write your notes in Markdown..."
        />
      ) : (
        <div className="note-editor__preview">
          {content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          ) : (
            <p style={{ color: "var(--text-tertiary)" }}>No notes yet. Switch to Edit to start writing.</p>
          )}
        </div>
      )}
      <div className="note-editor__actions">
        <button className="note-editor__save" onClick={handleSave} disabled={saving}>
          {saving ? "Saving..." : "Save"}
        </button>
        <span className="note-editor__status">{saved ? "Saved!" : "Ctrl+S to save"}</span>
      </div>
    </div>
  );
}

function TagAssigner({ paper, onUpdate }: { paper: Paper; onUpdate: () => void }) {
  const { tags, addTagToPaper, removeTagFromPaper } = useResearch();
  const paperTagIds = new Set(paper.tags?.map((t) => t.id) || []);

  const handleToggle = async (tagId: string) => {
    if (paperTagIds.has(tagId)) {
      await removeTagFromPaper(paper.id, tagId);
    } else {
      await addTagToPaper(paper.id, tagId);
    }
    onUpdate();
  };

  if (tags.length === 0) return null;

  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {tags.map((t) => (
        <button
          key={t.id}
          onClick={() => handleToggle(t.id)}
          style={{
            padding: "4px 12px",
            borderRadius: 6,
            border: `1px solid ${paperTagIds.has(t.id) ? t.color + "66" : "var(--border-default)"}`,
            background: paperTagIds.has(t.id) ? t.color + "22" : "transparent",
            color: paperTagIds.has(t.id) ? t.color : "var(--text-secondary)",
            fontSize: "0.75rem",
            cursor: "pointer",
            fontFamily: "var(--font-mono)",
          }}
        >
          {paperTagIds.has(t.id) ? "- " : "+ "}#{t.name}
        </button>
      ))}
    </div>
  );
}

function GenerateVizModal({
  paper,
  onClose,
  onGenerated,
}: {
  paper: Paper;
  onClose: () => void;
  onGenerated: (html: string) => void;
}) {
  const { session } = useResearch();
  const defaultPrompt = `Create an interactive visualization for this research paper:

Title: "${paper.title}"
${paper.year ? `Year: ${paper.year}` : ""}
${paper.one_liner ? `Summary: ${paper.one_liner}` : ""}
${paper.abstract ? `Abstract: ${paper.abstract}` : ""}

Include these sections:
1. **Big Ideas** - The core insights and innovations of this paper, explained clearly
2. **Key Contributions** - What this paper specifically contributes to the field
3. **Interactive Visualization** - An interactive diagram or visualization of the key concepts (use SVG, Canvas, or D3.js)
4. **Paper Lineage** - Which papers this builds on and which papers followed that built on it`;

  const [prompt, setPrompt] = useState(defaultPrompt);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      const res = await fetch("/api/generate-viz", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({ paperId: paper.id, prompt }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        onGenerated(data.html);
        onClose();
      }
    } catch {
      setError("Failed to generate visualization");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="viz-modal-overlay" onClick={onClose}>
      <div className="viz-modal" onClick={(e) => e.stopPropagation()}>
        <div className="viz-modal__header">
          <h2>Generate Visualization</h2>
          <button className="viz-modal__close" onClick={onClose}>x</button>
        </div>
        <p className="viz-modal__desc">
          Edit the prompt below, then click Generate. Claude will create an interactive HTML visualization.
        </p>
        <textarea
          className="viz-modal__prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={12}
        />
        {error && <p className="viz-modal__error">{error}</p>}
        <div className="viz-modal__actions">
          <button onClick={onClose} className="viz-modal__cancel">Cancel</button>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="viz-modal__generate"
          >
            {generating ? "Generating..." : "Generate Visualization"}
          </button>
        </div>
      </div>
    </div>
  );
}

function EmbeddedViz({ slug }: { slug: string }) {
  const VizComponent = getVizComponent(slug);
  if (!VizComponent) return null;

  return (
    <MathJaxContext
      version={3}
      config={mathjaxConfig}
      src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
    >
      <VizComponent />
    </MathJaxContext>
  );
}

function GeneratedViz({ html }: { html: string }) {
  return (
    <iframe
      srcDoc={html}
      className="paper-detail__generated-viz"
      sandbox="allow-scripts"
      title="Paper visualization"
    />
  );
}

export default function PaperDetailPage() {
  const params = useParams();
  const slug = params.slug as string;
  const [paper, setPaper] = useState<Paper | null>(null);
  const [loading, setLoading] = useState(true);
  const [showVizModal, setShowVizModal] = useState(false);
  const [vizTab, setVizTab] = useState<"viz" | "notes">("viz");
  const { edges, papers } = useResearch();

  const loadPaper = useCallback(async () => {
    const p = await fetchPaperBySlug(slug);
    setPaper(p);
    setLoading(false);
  }, [slug]);

  useEffect(() => {
    loadPaper();
  }, [loadPaper]);

  if (loading) {
    return <div className="paper-detail"><p style={{ color: "var(--text-secondary)" }}>Loading...</p></div>;
  }

  if (!paper) {
    return (
      <div className="paper-detail">
        <Link href="/" className="paper-detail__back">← Back to collection</Link>
        <h1>Paper not found</h1>
        <p style={{ color: "var(--text-secondary)" }}>No paper with slug &quot;{slug}&quot;</p>
      </div>
    );
  }

  const cat = CATEGORIES[paper.category];
  const authorStr = paper.authors?.map((a) => a.name).join(", ") || "";
  const hasExistingViz = paper.slug ? hasVizComponent(paper.slug) : false;
  const hasGeneratedViz = !!paper.visualization_html;
  const hasAnyViz = hasExistingViz || hasGeneratedViz;

  // Find related papers from lineage graph
  const parentIds = edges
    .filter((e) => e.source_id === paper.id)
    .map((e) => e.target_id);
  const childIds = edges
    .filter((e) => e.target_id === paper.id)
    .map((e) => e.source_id);
  const parentPapers = papers.filter((p) => parentIds.includes(p.id));
  const childPapers = papers.filter((p) => childIds.includes(p.id));

  return (
    <div className="paper-detail">
      <Link href="/" className="paper-detail__back">← Back to collection</Link>

      {/* Meta row */}
      <div className="paper-detail__meta">
        {cat && (
          <span
            className="paper-card__cat"
            style={{ background: cat.color + "18", color: cat.color, borderColor: cat.color + "33" }}
          >
            {cat.label}
          </span>
        )}
        {paper.year && (
          <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
            {paper.year}
          </span>
        )}
        {paper.arxiv_id && (
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: "0.8rem", color: "var(--accent-primary)" }}
          >
            arXiv
          </a>
        )}
        {paper.project_url && (
          <a
            href={paper.project_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: "0.8rem", color: "var(--accent-primary)" }}
          >
            Project Page
          </a>
        )}
        {paper.semantic_scholar_url && (
          <a
            href={paper.semantic_scholar_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}
          >
            Semantic Scholar
          </a>
        )}
      </div>

      <h1>{paper.title}</h1>
      {authorStr && <p className="paper-detail__authors">{authorStr}</p>}
      {paper.one_liner && <div className="paper-detail__oneliner">{paper.one_liner}</div>}

      {/* Citation info */}
      {paper.citation_count > 0 && (
        <div className="paper-detail__citations">
          <span className="paper-detail__citation-badge">
            {paper.citation_count} citations
            {paper.citation_velocity > 0 && (
              <span className="paper-detail__velocity">+{paper.citation_velocity} recent</span>
            )}
          </span>
        </div>
      )}

      {/* Tags */}
      <div className="paper-detail__section">
        <h2>Tags</h2>
        <TagAssigner paper={paper} onUpdate={loadPaper} />
      </div>

      {/* Visualization + Notes tabs */}
      <div className="paper-detail__content-tabs">
        <button
          className={`paper-detail__content-tab ${vizTab === "viz" ? "paper-detail__content-tab--active" : ""}`}
          onClick={() => setVizTab("viz")}
        >
          Visualization
        </button>
        <button
          className={`paper-detail__content-tab ${vizTab === "notes" ? "paper-detail__content-tab--active" : ""}`}
          onClick={() => setVizTab("notes")}
        >
          Notes
        </button>
      </div>

      {vizTab === "viz" ? (
        <div className="paper-detail__viz-section">
          {hasExistingViz && paper.slug ? (
            <EmbeddedViz slug={paper.slug} />
          ) : hasGeneratedViz ? (
            <GeneratedViz html={paper.visualization_html!} />
          ) : (
            <div className="paper-detail__no-viz">
              <p>No visualization yet for this paper.</p>
              <button
                className="paper-detail__generate-btn"
                onClick={() => setShowVizModal(true)}
              >
                Generate Visualization
              </button>
            </div>
          )}
          {/* Regenerate button even if viz exists (for generated ones) */}
          {hasGeneratedViz && !hasExistingViz && (
            <button
              className="paper-detail__regenerate-btn"
              onClick={() => setShowVizModal(true)}
            >
              Regenerate
            </button>
          )}
        </div>
      ) : (
        <NoteEditor paper={paper} onSave={loadPaper} />
      )}

      {/* Related Papers */}
      {(parentPapers.length > 0 || childPapers.length > 0) && (
        <div className="paper-detail__section">
          <h2>Related Papers</h2>
          <div className="lineage-grid">
            {parentPapers.length > 0 && (
              <div className="lineage-group">
                <h4>Builds on</h4>
                {parentPapers.map((p) => (
                  <div key={p.id} className="lineage-item" style={{ borderColor: CATEGORIES[p.category]?.color || "var(--border-default)" }}>
                    {p.slug ? (
                      <Link href={`/papers/${p.slug}`}>
                        <h5 style={{ color: "var(--accent-primary)" }}>{p.title}</h5>
                      </Link>
                    ) : (
                      <h5>{p.title}</h5>
                    )}
                    <p>{p.one_liner}</p>
                  </div>
                ))}
              </div>
            )}
            {childPapers.length > 0 && (
              <div className="lineage-group">
                <h4>Built upon by</h4>
                {childPapers.map((p) => (
                  <div key={p.id} className="lineage-item" style={{ borderColor: CATEGORIES[p.category]?.color || "var(--border-default)" }}>
                    {p.slug ? (
                      <Link href={`/papers/${p.slug}`}>
                        <h5 style={{ color: "var(--accent-primary)" }}>{p.title}</h5>
                      </Link>
                    ) : (
                      <h5>{p.title}</h5>
                    )}
                    <p>{p.one_liner}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Generate Viz Modal */}
      {showVizModal && (
        <GenerateVizModal
          paper={paper}
          onClose={() => setShowVizModal(false)}
          onGenerated={(html) => {
            setPaper((prev) => prev ? { ...prev, visualization_html: html } : prev);
          }}
        />
      )}
    </div>
  );
}
