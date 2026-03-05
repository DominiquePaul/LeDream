"use client";

import { useState, useMemo } from "react";
import { useResearch } from "@/components/Shell";

const DEFAULT_COLORS = [
  "#4a9b7f", "#6b73b5", "#c4884d", "#638bd4",
  "#9b6b9b", "#6b9b9b", "#b5736b", "#7fb54a",
  "#b5a84a", "#4a7fb5", "#b54a7f", "#4ab5a8",
];

interface TagSuggestion {
  paper_id: string;
  tag_id: string;
  paper_title: string;
  tag_name: string;
  confidence: string;
  reason: string;
}

export default function TagsPage() {
  const { tags, papers, createTag, deleteTag, addTagToPaper, session } = useResearch();
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState(DEFAULT_COLORS[0]);
  const [creating, setCreating] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState<TagSuggestion[]>([]);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const [applying, setApplying] = useState<Set<string>>(new Set());

  const tagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    papers.forEach((p) => {
      p.tags?.forEach((t) => {
        counts[t.id] = (counts[t.id] || 0) + 1;
      });
    });
    return counts;
  }, [papers]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await createTag(newName.trim(), newColor);
      setNewName("");
      setNewColor(DEFAULT_COLORS[Math.floor(Math.random() * DEFAULT_COLORS.length)]);
    } catch (err) {
      console.error("Failed to create tag:", err);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete tag "${name}"? This will remove it from all papers.`)) return;
    await deleteTag(id);
  };

  const handleAutoSuggest = async () => {
    setSuggesting(true);
    setSuggestError(null);
    setSuggestions([]);
    try {
      const res = await fetch("/api/auto-tag", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.error) {
        setSuggestError(data.error);
      } else if (data.suggestions?.length === 0) {
        setSuggestError(data.message || "No new suggestions found.");
      } else {
        setSuggestions(data.suggestions);
      }
    } catch {
      setSuggestError("Failed to get suggestions");
    } finally {
      setSuggesting(false);
    }
  };

  const handleApplySuggestion = async (suggestion: TagSuggestion) => {
    const key = `${suggestion.paper_id}:${suggestion.tag_id}`;
    setApplying((prev) => new Set(prev).add(key));
    try {
      await addTagToPaper(suggestion.paper_id, suggestion.tag_id);
      setSuggestions((prev) =>
        prev.filter((s) => !(s.paper_id === suggestion.paper_id && s.tag_id === suggestion.tag_id))
      );
    } catch (err) {
      console.error("Failed to apply suggestion:", err);
    } finally {
      setApplying((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  };

  const handleDismissSuggestion = (suggestion: TagSuggestion) => {
    setSuggestions((prev) =>
      prev.filter((s) => !(s.paper_id === suggestion.paper_id && s.tag_id === suggestion.tag_id))
    );
  };

  const handleApplyAll = async () => {
    for (const s of suggestions) {
      await handleApplySuggestion(s);
    }
  };

  return (
    <div className="tags-page">
      <h1>Manage Tags</h1>

      <div className="tags-page__create">
        <input
          className="tags-page__input"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New tag name..."
          onKeyDown={(e) => e.key === "Enter" && handleCreate()}
        />
        <input
          type="color"
          className="tags-page__color"
          value={newColor}
          onChange={(e) => setNewColor(e.target.value)}
        />
        <button
          className="tags-page__add-btn"
          onClick={handleCreate}
          disabled={creating || !newName.trim()}
        >
          {creating ? "..." : "Add Tag"}
        </button>
      </div>

      <div className="tags-page__list">
        {tags.length === 0 && (
          <p style={{ color: "var(--text-tertiary)", textAlign: "center", padding: 40 }}>
            No tags yet. Create one above.
          </p>
        )}
        {tags.map((t) => (
          <div key={t.id} className="tags-page__item">
            <div className="tags-page__item-color" style={{ background: t.color }} />
            <span className="tags-page__item-name">{t.name}</span>
            <span className="tags-page__item-count">{tagCounts[t.id] || 0} papers</span>
            <button
              className="tags-page__item-delete"
              onClick={() => handleDelete(t.id, t.name)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      {/* Auto-suggest section */}
      {tags.length > 0 && (
        <div className="tags-page__suggest">
          <h2>AI Tag Suggestions</h2>
          <p className="tags-page__suggest-desc">
            Let AI analyze your papers and suggest tag assignments based on content and categories.
          </p>
          <button
            className="tags-page__suggest-btn"
            onClick={handleAutoSuggest}
            disabled={suggesting}
          >
            {suggesting ? "Analyzing papers..." : "Get AI Suggestions"}
          </button>

          {suggestError && (
            <p className="tags-page__suggest-error">{suggestError}</p>
          )}

          {suggestions.length > 0 && (
            <div className="tags-page__suggestions">
              <div className="tags-page__suggestions-header">
                <span>{suggestions.length} suggestion{suggestions.length !== 1 ? "s" : ""}</span>
                <button className="tags-page__apply-all" onClick={handleApplyAll}>
                  Apply All
                </button>
              </div>
              {suggestions.map((s) => {
                const key = `${s.paper_id}:${s.tag_id}`;
                const tag = tags.find((t) => t.id === s.tag_id);
                return (
                  <div key={key} className="tags-page__suggestion">
                    <div className="tags-page__suggestion-info">
                      <span className="tags-page__suggestion-paper">{s.paper_title}</span>
                      <span className="tags-page__suggestion-arrow">→</span>
                      <span
                        className="tags-page__suggestion-tag"
                        style={{
                          background: (tag?.color || "#638bd4") + "22",
                          color: tag?.color || "#638bd4",
                          borderColor: (tag?.color || "#638bd4") + "44",
                        }}
                      >
                        #{s.tag_name}
                      </span>
                      <span className={`tags-page__suggestion-confidence tags-page__suggestion-confidence--${s.confidence}`}>
                        {s.confidence}
                      </span>
                    </div>
                    {s.reason && (
                      <p className="tags-page__suggestion-reason">{s.reason}</p>
                    )}
                    <div className="tags-page__suggestion-actions">
                      <button
                        className="tags-page__suggestion-apply"
                        onClick={() => handleApplySuggestion(s)}
                        disabled={applying.has(key)}
                      >
                        {applying.has(key) ? "..." : "Apply"}
                      </button>
                      <button
                        className="tags-page__suggestion-dismiss"
                        onClick={() => handleDismissSuggestion(s)}
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
