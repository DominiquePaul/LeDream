"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import {
  ExternalLink,
  Search,
  Trash2,
  ChevronDown,
  ChevronRight,
  Pencil,
  Check,
  X,
  Tag as TagIcon,
  Database,
} from "lucide-react";

export default function ModelsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [search, setSearch] = useState("");
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingTags, setEditingTags] = useState<string | null>(null);
  const [editingIteration, setEditingIteration] = useState<string | null>(null);
  const [editIterNotes, setEditIterNotes] = useState("");
  const [editIterConfig, setEditIterConfig] = useState("");
  const [showConfig, setShowConfig] = useState<string | null>(null);
  const [editingDesc, setEditingDesc] = useState<string | null>(null);
  const [editDescText, setEditDescText] = useState("");

  const tagMap = new Map(data.tags.map((t) => [t.id, t]));
  const datasetMap = new Map(data.datasets.map((d) => [d.id, d]));

  const filtered = data.models.filter((m) => {
    if (search && !m.name.toLowerCase().includes(search.toLowerCase()))
      return false;
    if (filterTag && !m.tags.includes(filterTag)) return false;
    return true;
  });

  const handleDeleteModel = (id: string) => {
    if (!confirm("Remove this model from Research OS? (Does not delete from HuggingFace)")) return;
    ctx.deleteModel(id);
  };

  const toggleTag = (modelId: string, tagId: string) => {
    const m = data.models.find((m) => m.id === modelId);
    if (!m) return;
    const newTags = m.tags.includes(tagId)
      ? m.tags.filter((t) => t !== tagId)
      : [...m.tags, tagId];
    ctx.updateModel(modelId, { tags: newTags });
  };

  const startEditIteration = (modelId: string, iterationId: string) => {
    const m = data.models.find((m) => m.id === modelId);
    const it = m?.iterations.find((i) => i.id === iterationId);
    if (!it) return;
    setEditingIteration(iterationId);
    setEditIterNotes(it.notes);
    setEditIterConfig(JSON.stringify(it.config, null, 2));
  };

  const saveIteration = (modelId: string, iterationId: string) => {
    const m = data.models.find((m) => m.id === modelId);
    if (!m) return;
    let parsedConfig = {};
    try { parsedConfig = JSON.parse(editIterConfig); } catch { parsedConfig = {}; }
    const newIterations = m.iterations.map((it) =>
      it.id === iterationId
        ? { ...it, notes: editIterNotes, config: parsedConfig }
        : it
    );
    ctx.updateModel(modelId, { iterations: newIterations });
    setEditingIteration(null);
  };

  const startEditDesc = (m: (typeof data.models)[0]) => {
    setEditingDesc(m.id);
    setEditDescText(m.description);
  };

  const saveDesc = (id: string) => {
    ctx.updateModel(id, { description: editDescText });
    setEditingDesc(null);
  };

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-4 md:space-y-6">
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white">Models</h1>
        <p className="text-gray-400 text-sm mt-1">
          {data.models.length} model groups synced from HuggingFace
        </p>
      </div>

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
            placeholder="Search models..."
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

      {/* Model list */}
      <div className="space-y-2">
        {filtered.map((m) => {
          const isExpanded = expandedId === m.id;
          return (
            <div
              key={m.id}
              className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden"
            >
              <div
                className="p-4 flex items-center gap-3 cursor-pointer hover:bg-gray-800/30 transition-colors"
                onClick={() => setExpandedId(isExpanded ? null : m.id)}
              >
                {isExpanded ? (
                  <ChevronDown size={16} className="text-gray-500 shrink-0" />
                ) : (
                  <ChevronRight size={16} className="text-gray-500 shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-white font-medium truncate">
                      {m.name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
                      {m.policyType}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-gray-500">
                      {m.iterations.length} checkpoints
                    </span>
                    <span className="text-xs text-gray-500">
                      {m.downloads} downloads
                    </span>
                    {m.tags.map((tid) => {
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
                <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                  <a
                    href={m.hfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1.5 text-gray-500 hover:text-blue-400"
                  >
                    <ExternalLink size={14} />
                  </a>
                  <button
                    onClick={() => handleDeleteModel(m.id)}
                    className="p-1.5 text-gray-500 hover:text-red-400"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="border-t border-gray-800 p-4 space-y-4">
                  {/* Description */}
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] text-gray-500 uppercase">Description</span>
                      {editingDesc !== m.id && (
                        <button
                          onClick={() => startEditDesc(m)}
                          className="text-[10px] text-gray-500 hover:text-white flex items-center gap-1"
                        >
                          <Pencil size={10} /> Edit
                        </button>
                      )}
                    </div>
                    {editingDesc === m.id ? (
                      <div className="space-y-2">
                        <textarea
                          value={editDescText}
                          onChange={(e) => setEditDescText(e.target.value)}
                          rows={2}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white"
                          placeholder="Describe this model group..."
                        />
                        <div className="flex gap-2">
                          <button onClick={() => saveDesc(m.id)} className="text-green-400 hover:text-green-300">
                            <Check size={14} />
                          </button>
                          <button onClick={() => setEditingDesc(null)} className="text-gray-400 hover:text-gray-300">
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-gray-300">
                        {m.description || <span className="text-gray-600 italic">Not set</span>}
                      </p>
                    )}
                  </div>

                  {/* Tags */}
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[10px] text-gray-500 uppercase">Tags</span>
                      <button
                        onClick={() => setEditingTags(editingTags === m.id ? null : m.id)}
                        className="text-[10px] text-gray-500 hover:text-white flex items-center gap-1"
                      >
                        <TagIcon size={10} />
                        {editingTags === m.id ? "Done" : "Edit"}
                      </button>
                    </div>
                    {editingTags === m.id ? (
                      <div className="flex flex-wrap gap-1.5">
                        {data.tags.map((tag) => {
                          const isActive = m.tags.includes(tag.id);
                          return (
                            <button
                              key={tag.id}
                              onClick={() => toggleTag(m.id, tag.id)}
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
                        {m.tags.length === 0 && (
                          <span className="text-[10px] text-gray-600 italic">No tags</span>
                        )}
                        {m.tags.map((tid) => {
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

                  {/* Iterations */}
                  <div>
                    <h4 className="text-xs text-gray-500 uppercase mb-3">
                      Checkpoints / Iterations
                    </h4>
                    <div className="space-y-2">
                      {m.iterations.map((it) => {
                        const isEditingThis = editingIteration === it.id;
                        const hasConfig = Object.keys(it.config).length > 0;
                        const isShowingConfig = showConfig === it.id;
                        const trainedOnDatasets = it.trainedOn
                          .map((did) => datasetMap.get(did))
                          .filter(Boolean);

                        return (
                          <div
                            key={it.id}
                            className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/30"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2">
                                  <p className="text-sm text-white font-mono">
                                    {it.version}
                                  </p>
                                  <p className="text-[10px] text-gray-600">
                                    {new Date(it.createdAt).toLocaleDateString()}
                                  </p>
                                </div>

                                {isEditingThis ? (
                                  <div className="mt-2 space-y-2">
                                    <div>
                                      <label className="text-[10px] text-gray-500 uppercase">Notes</label>
                                      <textarea
                                        value={editIterNotes}
                                        onChange={(e) => setEditIterNotes(e.target.value)}
                                        rows={2}
                                        className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
                                        placeholder="Why was this config used? What changed?"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-[10px] text-gray-500 uppercase">Config (JSON)</label>
                                      <textarea
                                        value={editIterConfig}
                                        onChange={(e) => setEditIterConfig(e.target.value)}
                                        rows={4}
                                        className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs text-white font-mono mt-0.5"
                                        placeholder='{"learning_rate": 1e-5, "batch_size": 8}'
                                      />
                                    </div>
                                    <div className="flex gap-2">
                                      <button
                                        onClick={() => saveIteration(m.id, it.id)}
                                        className="flex items-center gap-1 px-2 py-1 bg-green-700 hover:bg-green-600 text-white text-xs rounded"
                                      >
                                        <Check size={12} /> Save
                                      </button>
                                      <button
                                        onClick={() => setEditingIteration(null)}
                                        className="flex items-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
                                      >
                                        <X size={12} /> Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  <>
                                    {it.notes && (
                                      <p className="text-xs text-gray-400 mt-1">{it.notes}</p>
                                    )}
                                    {hasConfig && (
                                      <button
                                        onClick={() => setShowConfig(isShowingConfig ? null : it.id)}
                                        className="text-[10px] text-blue-400 hover:text-blue-300 mt-1"
                                      >
                                        {isShowingConfig ? "Hide config" : "Show config"}
                                      </button>
                                    )}
                                    {isShowingConfig && hasConfig && (
                                      <pre className="mt-1 p-2 bg-gray-900 rounded text-[10px] text-gray-300 font-mono overflow-x-auto">
                                        {JSON.stringify(it.config, null, 2)}
                                      </pre>
                                    )}
                                    {trainedOnDatasets.length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-1.5">
                                        <Database size={10} className="text-gray-500 mt-0.5" />
                                        {trainedOnDatasets.map((ds) => ds && (
                                          <a
                                            key={ds.id}
                                            href={ds.hfUrl}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-[10px] px-1.5 py-0.5 bg-blue-900/30 text-blue-400 rounded hover:bg-blue-900/50"
                                          >
                                            {ds.name}
                                          </a>
                                        ))}
                                      </div>
                                    )}
                                  </>
                                )}
                              </div>
                              <div className="flex items-center gap-1.5 ml-2">
                                {!isEditingThis && (
                                  <button
                                    onClick={() => startEditIteration(m.id, it.id)}
                                    className="p-1 text-gray-500 hover:text-gray-300"
                                  >
                                    <Pencil size={12} />
                                  </button>
                                )}
                                <a
                                  href={`https://huggingface.co/${it.hfId}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-gray-500 hover:text-blue-400"
                                >
                                  <ExternalLink size={14} />
                                </a>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="text-[10px] text-gray-600">
                    HF ID: {m.hfId} &middot; Last modified:{" "}
                    {new Date(m.lastModified).toLocaleDateString()}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
