"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import {
  ExternalLink,
  Search,
  AlertTriangle,
  Trash2,
  ChevronDown,
  ChevronRight,
  Pencil,
  Check,
  X,
  Tag as TagIcon,
  Eye,
  Type,
  Loader2,
  Upload,
  Plus,
  ClipboardList,
} from "lucide-react";

function fmtHours(h: number): string {
  if (h === 0) return "0m";
  const hours = Math.floor(h);
  const mins = Math.round((h - hours) * 60);
  if (hours === 0) return `${mins}m`;
  if (mins === 0) return `${hours}h`;
  return `${hours}h${mins}m`;
}

function visualizerUrl(hfId: string): string {
  return `https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F${encodeURIComponent(hfId)}%2Fepisode_0`;
}

export default function DatasetsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [search, setSearch] = useState("");
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [editingTags, setEditingTags] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameLoading, setRenameLoading] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [pushingHfId, setPushingHfId] = useState<string | null>(null);
  const [editMeta, setEditMeta] = useState({
    collectionConditions: "",
    teleoperatorInstructions: "",
    knownIssues: "",
    notes: "",
    estimatedHours: 0,
    episodeCount: 0,
  });

  const [showUnlabelled, setShowUnlabelled] = useState(false);
  const [showPlanForm, setShowPlanForm] = useState(false);
  const [planName, setPlanName] = useState("");
  const [planConditions, setPlanConditions] = useState("");
  const [planInstructions, setPlanInstructions] = useState("");
  const [planNotes, setPlanNotes] = useState("");
  const [planEpisodes, setPlanEpisodes] = useState(0);
  const [planHours, setPlanHours] = useState(0);
  const tagMap = new Map(data.tags.map((t) => [t.id, t]));

  const unlabelledCount = data.datasets.filter((d) => d.tags.length === 0).length;

  const filtered = data.datasets.filter((d) => {
    if (search && !d.name.toLowerCase().includes(search.toLowerCase()))
      return false;
    if (filterTag && !d.tags.includes(filterTag)) return false;
    if (showUnlabelled && d.tags.length > 0) return false;
    return true;
  });

  const startEdit = (d: (typeof data.datasets)[0]) => {
    setEditing(d.id);
    setEditMeta({
      collectionConditions: d.metadata.collectionConditions,
      teleoperatorInstructions: d.metadata.teleoperatorInstructions,
      knownIssues: d.metadata.knownIssues.join("\n"),
      notes: d.metadata.notes,
      estimatedHours: d.metadata.estimatedHours,
      episodeCount: d.metadata.episodeCount,
    });
  };

  const saveEdit = (id: string) => {
    ctx.updateDataset(id, {
      metadata: {
        ...editMeta,
        knownIssues: editMeta.knownIssues.split("\n").filter((s) => s.trim()),
      },
    });
    setEditing(null);
  };

  const toggleTag = (datasetId: string, tagId: string) => {
    const ds = data.datasets.find((d) => d.id === datasetId);
    if (!ds) return;
    const newTags = ds.tags.includes(tagId)
      ? ds.tags.filter((t) => t !== tagId)
      : [...ds.tags, tagId];
    ctx.updateDataset(datasetId, { tags: newTags });
  };

  const handleDeleteDataset = async (id: string, hfId: string, name: string) => {
    const expected = `permanently delete ${name}`;
    const input = prompt(`This will permanently delete "${name}" from DreamHub AND HuggingFace.\n\nType "${expected}" to confirm:`);
    if (input !== expected) {
      if (input !== null) alert("Text didn't match. Deletion cancelled.");
      return;
    }

    const token = await getAccessToken();
    if (!token) { alert("Not authenticated"); return; }

    setDeleteLoading(true);
    try {
      // Delete from HF if it exists there
      if (hfId) {
        const res = await fetch("/api/hf/delete", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ name: hfId }),
        });
        if (!res.ok) {
          const body = await res.json();
          throw new Error(body.error || `HF error ${res.status}`);
        }
      }
      // Delete from DreamHub
      ctx.deleteDataset(id);
    } catch (err) {
      alert(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDeleteLoading(false);
    }
  };

  const getAccessToken = async (): Promise<string | null> => {
    const { supabase } = await import("@/lib/supabase");
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token || null;
  };

  const handleRename = async (id: string, hfId: string) => {
    if (!renameValue.trim() || renameValue.trim() === hfId.split("/")[1]) {
      setRenamingId(null);
      return;
    }
    const owner = hfId.split("/")[0];
    const newName = renameValue.trim();
    const newHfId = `${owner}/${newName}`;

    if (!confirm(`Rename dataset on HuggingFace?\n\nFrom: ${hfId}\nTo: ${newHfId}\n\nThis changes the repo URL on HuggingFace. Existing links will break.`)) return;

    const token = await getAccessToken();
    if (!token) { alert("Not authenticated"); return; }

    setRenameLoading(true);
    try {
      const res = await fetch("/api/hf/rename", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ fromRepo: hfId, toRepo: newHfId }),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || `Error ${res.status}`);
      }
      ctx.updateDataset(id, {
        hfId: newHfId,
        name: newName,
        hfUrl: `https://huggingface.co/datasets/${newHfId}`,
      });
      setRenamingId(null);
    } catch (err) {
      alert(`Rename failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRenameLoading(false);
    }
  };

  const handlePushToHf = async (d: (typeof data.datasets)[0]) => {
    const token = await getAccessToken();
    if (!token) { alert("Not authenticated"); return; }

    setPushingHfId(d.id);
    try {
      const tagNames = d.tags.map(tid => tagMap.get(tid)?.name).filter(Boolean);
      const res = await fetch("/api/hf/push-metadata", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          hfId: d.hfId,
          tags: tagNames,
          metadata: d.metadata,
        }),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || `Error ${res.status}`);
      }
      alert("Metadata pushed to HuggingFace successfully.");
    } catch (err) {
      alert(`Push failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPushingHfId(null);
    }
  };

  const handlePlanCollection = () => {
    if (!planName.trim()) return;
    ctx.addDataset({
      name: planName.trim(),
      metadata: {
        collectionConditions: planConditions,
        teleoperatorInstructions: planInstructions,
        knownIssues: [],
        notes: planNotes,
        estimatedHours: planHours,
        episodeCount: planEpisodes,
      },
    });
    setPlanName("");
    setPlanConditions("");
    setPlanInstructions("");
    setPlanNotes("");
    setPlanEpisodes(0);
    setPlanHours(0);
    setShowPlanForm(false);
  };

  const plannedCount = data.datasets.filter((d) => d.status === "planned").length;

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-4 md:space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white">Datasets</h1>
          <p className="text-gray-400 text-sm mt-1">
            {data.datasets.filter(d => d.status === "synced").length} synced from HuggingFace
            {plannedCount > 0 && (
              <span className="text-cyan-400 ml-2">{plannedCount} planned</span>
            )}
          </p>
        </div>
        <button
          onClick={() => setShowPlanForm(!showPlanForm)}
          className="flex items-center gap-1.5 px-3 py-2 bg-cyan-900/30 text-cyan-400 rounded-lg hover:bg-cyan-900/50 text-sm transition-colors"
        >
          <Plus size={14} /> Plan Collection
        </button>
      </div>

      {/* Plan collection form */}
      {showPlanForm && (
        <div className="bg-cyan-950/30 border border-cyan-800/40 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <ClipboardList size={16} className="text-cyan-400" />
            <span className="text-sm font-medium text-cyan-300">Plan a Data Collection</span>
          </div>
          <p className="text-xs text-gray-400">
            Create a placeholder entry. When you upload to HF and sync, it will be matched by name.
          </p>
          <div>
            <label className="text-[10px] text-gray-500 uppercase">Dataset Name (must match future HF name)</label>
            <input
              value={planName}
              onChange={(e) => setPlanName(e.target.value)}
              placeholder="e.g., 500_chess_rook_moves"
              className="block w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              autoFocus
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Collection Conditions</label>
              <textarea
                value={planConditions}
                onChange={(e) => setPlanConditions(e.target.value)}
                rows={2}
                placeholder="How will variation be handled?"
                className="block w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              />
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Teleoperator Instructions</label>
              <textarea
                value={planInstructions}
                onChange={(e) => setPlanInstructions(e.target.value)}
                rows={2}
                placeholder="Instructions for the operator"
                className="block w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              />
            </div>
          </div>
          <div>
            <label className="text-[10px] text-gray-500 uppercase">Notes</label>
            <textarea
              value={planNotes}
              onChange={(e) => setPlanNotes(e.target.value)}
              rows={2}
              placeholder="Any additional notes for this collection"
              className="block w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
            />
          </div>
          <div className="flex gap-4">
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Target Episodes</label>
              <input
                type="number"
                value={planEpisodes}
                onChange={(e) => setPlanEpisodes(parseInt(e.target.value) || 0)}
                className="block w-24 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-white"
              />
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Estimated Hours</label>
              <input
                type="number"
                value={planHours}
                onChange={(e) => setPlanHours(parseFloat(e.target.value) || 0)}
                className="block w-24 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-white"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handlePlanCollection}
              disabled={!planName.trim()}
              className="flex items-center gap-1 px-3 py-1.5 bg-cyan-700 hover:bg-cyan-600 text-white text-xs rounded disabled:opacity-50"
            >
              <Check size={14} /> Create Plan
            </button>
            <button
              onClick={() => setShowPlanForm(false)}
              className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
            >
              <X size={14} /> Cancel
            </button>
          </div>
        </div>
      )}

      {/* Unlabelled notification */}
      {unlabelledCount > 0 && (
        <button
          onClick={() => { setShowUnlabelled(!showUnlabelled); setFilterTag(null); }}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm transition-colors w-full text-left ${
            showUnlabelled
              ? "bg-amber-900/30 border border-amber-700/50 text-amber-300"
              : "bg-amber-900/20 border border-amber-800/30 text-amber-400 hover:bg-amber-900/30"
          }`}
        >
          <AlertTriangle size={16} className="shrink-0" />
          <span>
            <strong>{unlabelledCount}</strong> dataset{unlabelledCount !== 1 ? "s" : ""} without tags
          </span>
          <span className="ml-auto text-xs text-amber-500">
            {showUnlabelled ? "Show all" : "Show unlabelled"}
          </span>
        </button>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:gap-4">
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search datasets..."
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-700"
          />
        </div>
        <select
          value={filterTag || ""}
          onChange={(e) => setFilterTag(e.target.value || null)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All tags</option>
          {data.tags.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>

      {/* Dataset list */}
      <div className="space-y-2">
        {filtered.map((d) => {
          const isExpanded = expandedId === d.id;
          const isEditing = editing === d.id;

          return (
            <div
              key={d.id}
              className={`rounded-lg border overflow-hidden ${
                d.status === "planned"
                  ? "bg-cyan-950/20 border-cyan-800/40"
                  : "bg-gray-900 border-gray-800"
              }`}
            >
              <div
                className="p-3 md:p-4 flex items-center gap-2 md:gap-3 cursor-pointer hover:bg-gray-800/30 transition-colors"
                onClick={() =>
                  setExpandedId(isExpanded ? null : d.id)
                }
              >
                {isExpanded ? (
                  <ChevronDown size={16} className="text-gray-500 shrink-0" />
                ) : (
                  <ChevronRight size={16} className="text-gray-500 shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-white font-medium truncate">
                      {d.name}
                    </span>
                    {d.status === "planned" && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-900/50 text-cyan-400 uppercase font-medium shrink-0">
                        Planned
                      </span>
                    )}
                    {d.metadata.knownIssues.length > 0 && (
                      <AlertTriangle
                        size={14}
                        className="text-amber-400 shrink-0"
                      />
                    )}
                  </div>
                  <div className="flex items-center gap-2 md:gap-3 mt-1 flex-wrap">
                    <span className="text-xs text-gray-500">
                      {d.metadata.episodeCount} eps
                    </span>
                    <span className="text-xs text-gray-500">
                      {fmtHours(d.metadata.estimatedHours)}
                    </span>
                    <span className="text-xs text-gray-500 hidden sm:inline">
                      {d.downloads} downloads
                    </span>
                    {d.tags.map((tid) => {
                      const tag = tagMap.get(tid);
                      return tag ? (
                        <span
                          key={tid}
                          className="text-[10px] px-1.5 py-0.5 rounded-full"
                          style={{
                            backgroundColor: tag.color + "22",
                            color: tag.color,
                          }}
                        >
                          {tag.name}
                        </span>
                      ) : null;
                    })}
                  </div>
                </div>
                <div className="flex items-center gap-1 md:gap-2" onClick={(e) => e.stopPropagation()}>
                  <a
                    href={visualizerUrl(d.hfId)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1.5 text-gray-500 hover:text-emerald-400"
                    title="Visualize dataset"
                  >
                    <Eye size={14} />
                  </a>
                  <a
                    href={d.hfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1.5 text-gray-500 hover:text-blue-400"
                    title="View on HuggingFace"
                  >
                    <ExternalLink size={14} />
                  </a>
                  <button
                    onClick={() => handleDeleteDataset(d.id, d.hfId, d.name)}
                    className="p-1.5 text-gray-500 hover:text-red-400"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="border-t border-gray-800 p-3 md:p-4 space-y-4">
                  {/* Links and actions */}
                  <div className="flex flex-wrap gap-2">
                    <a
                      href={visualizerUrl(d.hfId)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 bg-emerald-900/30 text-emerald-400 rounded-lg hover:bg-emerald-900/50 transition-colors"
                    >
                      <Eye size={12} /> Visualize
                    </a>
                    <a
                      href={d.hfUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 bg-blue-900/30 text-blue-400 rounded-lg hover:bg-blue-900/50 transition-colors"
                    >
                      <ExternalLink size={12} /> HuggingFace
                    </a>
                    <button
                      onClick={() => { setRenamingId(d.id); setRenameValue(d.hfId.split("/")[1] || d.name); }}
                      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
                    >
                      <Type size={12} /> Rename on HF
                    </button>
                    <button
                      onClick={() => handlePushToHf(d)}
                      disabled={pushingHfId === d.id}
                      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 bg-purple-900/30 text-purple-400 rounded-lg hover:bg-purple-900/50 transition-colors"
                    >
                      {pushingHfId === d.id ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />} Push to HF
                    </button>
                  </div>

                  {/* Rename form */}
                  {renamingId === d.id && (
                    <div className="flex items-center gap-2 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                      <span className="text-xs text-gray-400">{d.hfId.split("/")[0]}/</span>
                      <input
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-white font-mono"
                        onKeyDown={(e) => e.key === "Enter" && handleRename(d.id, d.hfId)}
                        autoFocus
                      />
                      <button
                        onClick={() => handleRename(d.id, d.hfId)}
                        disabled={renameLoading}
                        className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded disabled:opacity-50"
                      >
                        {renameLoading ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                      </button>
                      <button
                        onClick={() => setRenamingId(null)}
                        className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  )}

                  {isEditing ? (
                    <div className="space-y-3">
                      <Field
                        label="Collection Conditions"
                        value={editMeta.collectionConditions}
                        onChange={(v) =>
                          setEditMeta({ ...editMeta, collectionConditions: v })
                        }
                      />
                      <Field
                        label="Teleoperator Instructions"
                        value={editMeta.teleoperatorInstructions}
                        onChange={(v) =>
                          setEditMeta({
                            ...editMeta,
                            teleoperatorInstructions: v,
                          })
                        }
                      />
                      <Field
                        label="Known Issues (one per line)"
                        value={editMeta.knownIssues}
                        onChange={(v) =>
                          setEditMeta({ ...editMeta, knownIssues: v })
                        }
                        textarea
                      />
                      <Field
                        label="Notes"
                        value={editMeta.notes}
                        onChange={(v) =>
                          setEditMeta({ ...editMeta, notes: v })
                        }
                        textarea
                      />
                      <div className="flex gap-4">
                        <div>
                          <label className="text-[10px] text-gray-500 uppercase">
                            Estimated Hours
                          </label>
                          <input
                            type="number"
                            value={editMeta.estimatedHours}
                            onChange={(e) =>
                              setEditMeta({
                                ...editMeta,
                                estimatedHours: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="block w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-gray-500 uppercase">
                            Episode Count
                          </label>
                          <input
                            type="number"
                            value={editMeta.episodeCount}
                            onChange={(e) =>
                              setEditMeta({
                                ...editMeta,
                                episodeCount: parseInt(e.target.value) || 0,
                              })
                            }
                            className="block w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
                          />
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => saveEdit(d.id)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-green-700 hover:bg-green-600 text-white text-xs rounded"
                        >
                          <Check size={14} /> Save
                        </button>
                        <button
                          onClick={() => setEditing(null)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
                        >
                          <X size={14} /> Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex justify-end">
                        <button
                          onClick={() => startEdit(d)}
                          className="flex items-center gap-1 text-xs text-gray-400 hover:text-white"
                        >
                          <Pencil size={12} /> Edit Metadata
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <MetaField
                          label="Collection Conditions"
                          value={d.metadata.collectionConditions}
                        />
                        <MetaField
                          label="Teleoperator Instructions"
                          value={d.metadata.teleoperatorInstructions}
                        />
                      </div>
                      {d.metadata.knownIssues.length > 0 && (
                        <div>
                          <span className="text-[10px] text-gray-500 uppercase">
                            Known Issues
                          </span>
                          <div className="mt-1 space-y-1">
                            {d.metadata.knownIssues.map((issue, i) => (
                              <div
                                key={i}
                                className="flex items-center gap-2 text-xs text-amber-400"
                              >
                                <AlertTriangle size={12} />
                                {issue}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {d.metadata.notes && (
                        <MetaField label="Notes" value={d.metadata.notes} />
                      )}
                      {/* Tag assignment */}
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-[10px] text-gray-500 uppercase">Tags</span>
                          <button
                            onClick={() => setEditingTags(editingTags === d.id ? null : d.id)}
                            className="text-[10px] text-gray-500 hover:text-white flex items-center gap-1"
                          >
                            <TagIcon size={10} />
                            {editingTags === d.id ? "Done" : "Edit"}
                          </button>
                        </div>
                        {editingTags === d.id ? (
                          <div className="flex flex-wrap gap-1.5">
                            {data.tags.map((tag) => {
                              const isActive = d.tags.includes(tag.id);
                              return (
                                <button
                                  key={tag.id}
                                  onClick={() => toggleTag(d.id, tag.id)}
                                  className={`text-[10px] px-2 py-1 rounded-full border transition-colors ${
                                    isActive
                                      ? "border-transparent"
                                      : "border-gray-700 opacity-40 hover:opacity-70"
                                  }`}
                                  style={{
                                    backgroundColor: isActive ? tag.color + "33" : "transparent",
                                    color: tag.color,
                                  }}
                                >
                                  {isActive ? "\u2713 " : ""}{tag.name}
                                </button>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="flex flex-wrap gap-1.5">
                            {d.tags.length === 0 && (
                              <span className="text-[10px] text-gray-600 italic">No tags</span>
                            )}
                            {d.tags.map((tid) => {
                              const tag = tagMap.get(tid);
                              return tag ? (
                                <span
                                  key={tid}
                                  className="text-[10px] px-2 py-0.5 rounded-full"
                                  style={{ backgroundColor: tag.color + "22", color: tag.color }}
                                >
                                  {tag.name}
                                </span>
                              ) : null;
                            })}
                          </div>
                        )}
                      </div>

                      {/* Related Models */}
                      {(() => {
                        const related = data.models.filter(m =>
                          m.iterations.some(it => it.trainedOn.includes(d.id))
                        );
                        if (related.length === 0) return null;
                        return (
                          <div>
                            <span className="text-[10px] text-gray-500 uppercase">Related Models</span>
                            <div className="flex flex-wrap gap-1.5 mt-1">
                              {related.map(m => (
                                <a
                                  key={m.id}
                                  href={m.hfUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[11px] px-2 py-1 rounded-lg bg-emerald-900/20 text-emerald-400 hover:bg-emerald-900/40 transition-colors"
                                >
                                  {m.name} ({m.iterations.filter(it => it.trainedOn.includes(d.id)).length} checkpoint{m.iterations.filter(it => it.trainedOn.includes(d.id)).length !== 1 ? "s" : ""})
                                </a>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                      <div className="text-[10px] text-gray-600">
                        HF ID: {d.hfId} &middot; Last modified:{" "}
                        {new Date(d.lastModified).toLocaleDateString()}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetaField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[10px] text-gray-500 uppercase">{label}</span>
      <p className="text-sm text-gray-300 mt-0.5">
        {value || <span className="text-gray-600 italic">Not set</span>}
      </p>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  textarea,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  textarea?: boolean;
}) {
  return (
    <div>
      <label className="text-[10px] text-gray-500 uppercase">{label}</label>
      {textarea ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={3}
          className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
        />
      ) : (
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
        />
      )}
    </div>
  );
}
