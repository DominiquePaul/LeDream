"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useResearch } from "@/components/Shell";
import { CATEGORIES } from "@/lib/types";
import type { Paper } from "@/lib/types";

function SyncButton({ onDone }: { onDone: () => void }) {
  const { session } = useResearch();
  const [syncing, setSyncing] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleSync = async () => {
    setSyncing(true);
    setResult(null);
    try {
      const res = await fetch("/api/semantic-scholar/sync", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({ batchSize: 10 }),
      });
      const data = await res.json();
      setResult(`Synced ${data.synced}, ${data.notFound || 0} not found`);
      onDone();
    } catch {
      setResult("Sync failed");
    } finally {
      setSyncing(false);
    }
  };

  const handleResolve = async () => {
    setResolving(true);
    setResult(null);
    try {
      const res = await fetch("/api/semantic-scholar/resolve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
      });
      const data = await res.json();
      if (data.message) {
        setResult(data.message);
      } else {
        const methods = data.results
          ?.filter((r: { status: string }) => r.status === "resolved")
          .map((r: { method: string }) => r.method);
        setResult(
          `AI resolved ${data.resolved}/${data.total} papers${methods?.length ? ` (${methods.join(", ")})` : ""}`
        );
      }
      onDone();
    } catch {
      setResult("Resolve failed");
    } finally {
      setResolving(false);
    }
  };

  const busy = syncing || resolving;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <button
        onClick={handleSync}
        disabled={busy}
        style={{
          padding: "6px 14px",
          background: "var(--bg-surface)",
          border: "1px solid var(--border-default)",
          borderRadius: 6,
          color: "var(--text-secondary)",
          fontSize: "0.75rem",
          cursor: busy ? "wait" : "pointer",
        }}
      >
        {syncing ? "Syncing..." : "Sync Citations"}
      </button>
      <button
        onClick={handleResolve}
        disabled={busy}
        style={{
          padding: "6px 14px",
          background: "var(--bg-surface)",
          border: "1px solid var(--border-default)",
          borderRadius: 6,
          color: "var(--accent-primary)",
          fontSize: "0.75rem",
          cursor: busy ? "wait" : "pointer",
        }}
      >
        {resolving ? "AI Resolving..." : "AI Resolve"}
      </button>
      {result && (
        <span style={{ fontSize: "0.7rem", color: "var(--text-tertiary)" }}>
          {result}
        </span>
      )}
    </div>
  );
}

function PaperCard({ paper }: { paper: Paper }) {
  const authorStr = paper.authors?.map((a) => a.name).join(", ") || "";
  const cat = CATEGORIES[paper.category];
  const href = paper.slug ? `/papers/${paper.slug}` : null;

  const card = (
    <div className="paper-card" data-category={paper.category}>
      <div className="paper-card__header">
        <span
          className="paper-card__cat"
          style={{ background: (cat?.color || "#638bd4") + "18", color: cat?.color || "#638bd4", borderColor: (cat?.color || "#638bd4") + "33" }}
        >
          {cat?.label || paper.category}
        </span>
        {paper.year && <span className="paper-card__year">{paper.year}</span>}
      </div>
      <h3 className="paper-card__title">{paper.title}</h3>
      {authorStr && <p className="paper-card__authors">{authorStr}</p>}
      {paper.one_liner && <p className="paper-card__oneliner">{paper.one_liner}</p>}
      {paper.tags && paper.tags.length > 0 && (
        <div className="paper-card__tags">
          {paper.tags.map((t) => (
            <span key={t.id} className="paper-card__tag" style={{ background: t.color + "22", color: t.color, borderColor: t.color + "44" }}>
              {t.name}
            </span>
          ))}
        </div>
      )}
      <div className="paper-card__footer">
        {paper.citation_count > 0 && (
          <span className="paper-card__citations">
            {paper.citation_count} citations
          </span>
        )}
        {paper.slug && (
          <span className="paper-card__viz-badge">Has visualization</span>
        )}
      </div>
    </div>
  );

  if (href) {
    return <Link href={href} style={{ textDecoration: "none", color: "inherit" }}>{card}</Link>;
  }
  return card;
}

function BulkTagBar({
  selected,
  onClear,
}: {
  selected: Set<string>;
  onClear: () => void;
}) {
  const { tags, addTagToPaper, removeTagFromPaper, papers, refresh } = useResearch();
  const [busy, setBusy] = useState(false);

  const selectedPapers = papers.filter((p) => selected.has(p.id));

  // For each tag, count how many selected papers have it
  const tagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    tags.forEach((t) => {
      counts[t.id] = selectedPapers.filter((p) =>
        p.tags?.some((pt) => pt.id === t.id)
      ).length;
    });
    return counts;
  }, [tags, selectedPapers]);

  const handleToggleTag = async (tagId: string) => {
    setBusy(true);
    const allHave = tagCounts[tagId] === selected.size;
    for (const paperId of selected) {
      if (allHave) {
        await removeTagFromPaper(paperId, tagId);
      } else {
        await addTagToPaper(paperId, tagId);
      }
    }
    await refresh();
    setBusy(false);
  };

  return (
    <div className="bulk-tag-bar">
      <div className="bulk-tag-bar__info">
        <span>{selected.size} paper{selected.size !== 1 ? "s" : ""} selected</span>
        <button onClick={onClear} className="bulk-tag-bar__clear">Deselect all</button>
      </div>
      <div className="bulk-tag-bar__tags">
        {tags.map((t) => {
          const count = tagCounts[t.id];
          const allHave = count === selected.size;
          const someHave = count > 0 && !allHave;
          return (
            <button
              key={t.id}
              onClick={() => handleToggleTag(t.id)}
              disabled={busy}
              className={`bulk-tag-bar__tag ${allHave ? "bulk-tag-bar__tag--all" : ""} ${someHave ? "bulk-tag-bar__tag--some" : ""}`}
              style={{
                borderColor: allHave ? t.color + "66" : someHave ? t.color + "44" : undefined,
                background: allHave ? t.color + "22" : undefined,
                color: allHave || someHave ? t.color : undefined,
              }}
            >
              {allHave ? "- " : "+ "}#{t.name}
              {someHave && <span className="bulk-tag-bar__tag-count">({count})</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TableView({
  papers,
  selected,
  onToggle,
  onToggleAll,
}: {
  papers: Paper[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onToggleAll: () => void;
}) {
  const allSelected = papers.length > 0 && papers.every((p) => selected.has(p.id));
  const someSelected = papers.some((p) => selected.has(p.id)) && !allSelected;

  return (
    <div className="table-view">
      <table className="table-view__table">
        <thead>
          <tr>
            <th className="table-view__th-check">
              <input
                type="checkbox"
                checked={allSelected}
                ref={(el) => { if (el) el.indeterminate = someSelected; }}
                onChange={onToggleAll}
                className="table-view__checkbox"
              />
            </th>
            <th>Title</th>
            <th>Category</th>
            <th>Year</th>
            <th>Authors</th>
            <th>Tags</th>
            <th>Citations</th>
          </tr>
        </thead>
        <tbody>
          {papers.map((p) => {
            const cat = CATEGORIES[p.category];
            return (
              <tr
                key={p.id}
                className={`table-view__row ${selected.has(p.id) ? "table-view__row--selected" : ""}`}
              >
                <td className="table-view__td-check">
                  <input
                    type="checkbox"
                    checked={selected.has(p.id)}
                    onChange={() => onToggle(p.id)}
                    className="table-view__checkbox"
                  />
                </td>
                <td>
                  {p.slug ? (
                    <Link href={`/papers/${p.slug}`} className="table-view__title-link">
                      {p.title}
                    </Link>
                  ) : (
                    <span className="table-view__title">{p.title}</span>
                  )}
                </td>
                <td>
                  <span
                    className="paper-card__cat"
                    style={{
                      background: (cat?.color || "#638bd4") + "18",
                      color: cat?.color || "#638bd4",
                      borderColor: (cat?.color || "#638bd4") + "33",
                    }}
                  >
                    {cat?.label || p.category}
                  </span>
                </td>
                <td className="table-view__year">{p.year || "—"}</td>
                <td className="table-view__authors">
                  {p.authors?.map((a) => a.name).join(", ") || "—"}
                </td>
                <td>
                  <div className="table-view__tags">
                    {(p.tags || []).map((t) => (
                      <span
                        key={t.id}
                        className="paper-card__tag"
                        style={{
                          background: t.color + "22",
                          color: t.color,
                          borderColor: t.color + "44",
                        }}
                      >
                        {t.name}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="table-view__citations">
                  {p.citation_count > 0 ? p.citation_count : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function CollectionPage() {
  const { papers, tags, loading, refresh } = useResearch();
  const searchParams = useSearchParams();
  const [search, setSearch] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(
    searchParams.get("category")
  );
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const filtered = useMemo(() => {
    let result = papers;

    if (selectedCategory) {
      result = result.filter((p) => p.category === selectedCategory);
    }

    if (selectedTag) {
      result = result.filter((p) => p.tags?.some((t) => t.id === selectedTag));
    }

    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.one_liner?.toLowerCase().includes(q) ||
          p.authors?.some((a) => a.name.toLowerCase().includes(q))
      );
    }

    return result;
  }, [papers, search, selectedCategory, selectedTag]);

  // Category stats
  const catCounts = useMemo(() => {
    return papers.reduce<Record<string, number>>((acc, p) => {
      acc[p.category] = (acc[p.category] || 0) + 1;
      return acc;
    }, {});
  }, [papers]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    const allFilteredSelected = filtered.every((p) => selected.has(p.id));
    if (allFilteredSelected) {
      setSelected((prev) => {
        const next = new Set(prev);
        filtered.forEach((p) => next.delete(p.id));
        return next;
      });
    } else {
      setSelected((prev) => {
        const next = new Set(prev);
        filtered.forEach((p) => next.add(p.id));
        return next;
      });
    }
  };

  if (loading) {
    return (
      <div className="collection">
        <div className="collection__loading">Loading papers...</div>
      </div>
    );
  }

  return (
    <div className="collection">
      {/* Header */}
      <div className="collection__header">
        <h1>Research Collection</h1>
        <p className="collection__subtitle">
          {papers.length} papers tracked across {Object.keys(catCounts).length} categories
        </p>
        <SyncButton onDone={refresh} />
      </div>

      {/* Stats */}
      <div className="collection__stats">
        {Object.entries(CATEGORIES).map(([key, cat]) => (
          <button
            key={key}
            className={`collection__stat ${selectedCategory === key ? "collection__stat--active" : ""}`}
            onClick={() => setSelectedCategory(selectedCategory === key ? null : key)}
            style={{ borderColor: selectedCategory === key ? cat.color : undefined }}
          >
            <span className="collection__stat-count" style={{ color: cat.color }}>
              {catCounts[key] || 0}
            </span>
            <span className="collection__stat-label">{cat.label}</span>
          </button>
        ))}
      </div>

      {/* Search + Filters */}
      <div className="collection__filters">
        <div className="collection__filter-row">
          <input
            type="text"
            placeholder="Search papers, authors..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="collection__search"
          />
          <div className="collection__view-toggle">
            <button
              className={`collection__view-btn ${viewMode === "grid" ? "collection__view-btn--active" : ""}`}
              onClick={() => setViewMode("grid")}
              title="Grid view"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <rect x="1" y="1" width="6" height="6" rx="1" />
                <rect x="9" y="1" width="6" height="6" rx="1" />
                <rect x="1" y="9" width="6" height="6" rx="1" />
                <rect x="9" y="9" width="6" height="6" rx="1" />
              </svg>
            </button>
            <button
              className={`collection__view-btn ${viewMode === "table" ? "collection__view-btn--active" : ""}`}
              onClick={() => setViewMode("table")}
              title="Table view"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <rect x="1" y="2" width="14" height="2" rx="0.5" />
                <rect x="1" y="7" width="14" height="2" rx="0.5" />
                <rect x="1" y="12" width="14" height="2" rx="0.5" />
              </svg>
            </button>
          </div>
        </div>
        {tags.length > 0 && (
          <div className="collection__tag-filters">
            {tags.map((t) => (
              <button
                key={t.id}
                className={`collection__tag-btn ${selectedTag === t.id ? "collection__tag-btn--active" : ""}`}
                onClick={() => setSelectedTag(selectedTag === t.id ? null : t.id)}
                style={{
                  background: selectedTag === t.id ? t.color + "22" : undefined,
                  color: selectedTag === t.id ? t.color : undefined,
                  borderColor: selectedTag === t.id ? t.color + "44" : undefined,
                }}
              >
                # {t.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Bulk tag bar */}
      {selected.size > 0 && (
        <BulkTagBar selected={selected} onClear={() => setSelected(new Set())} />
      )}

      {/* Results */}
      <div className="collection__count">
        {filtered.length} paper{filtered.length !== 1 ? "s" : ""}
        {(selectedCategory || selectedTag || search) && (
          <button
            className="collection__clear"
            onClick={() => { setSelectedCategory(null); setSelectedTag(null); setSearch(""); }}
          >
            Clear filters
          </button>
        )}
      </div>

      {viewMode === "grid" ? (
        <div className="collection__grid">
          {filtered.map((p) => (
            <PaperCard key={p.id} paper={p} />
          ))}
        </div>
      ) : (
        <TableView
          papers={filtered}
          selected={selected}
          onToggle={toggleSelect}
          onToggleAll={toggleSelectAll}
        />
      )}

      {filtered.length === 0 && (
        <div className="collection__empty">
          No papers match your filters.
        </div>
      )}
    </div>
  );
}
