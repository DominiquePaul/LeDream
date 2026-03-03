"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import {
  ExternalLink,
  Search,
  Trash2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

export default function ModelsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [search, setSearch] = useState("");
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const tagMap = new Map(data.tags.map((t) => [t.id, t]));

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

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Models</h1>
        <p className="text-gray-400 text-sm mt-1">
          {data.models.length} model groups synced from HuggingFace
        </p>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
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
                <div className="border-t border-gray-800 p-4">
                  <h4 className="text-xs text-gray-500 uppercase mb-3">
                    Checkpoints / Iterations
                  </h4>
                  <div className="space-y-2">
                    {m.iterations.map((it) => (
                      <div
                        key={it.id}
                        className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-gray-700/30"
                      >
                        <div>
                          <p className="text-sm text-white font-mono">
                            {it.version}
                          </p>
                          {it.notes && (
                            <p className="text-xs text-gray-400 mt-0.5">
                              {it.notes}
                            </p>
                          )}
                          <p className="text-[10px] text-gray-600 mt-0.5">
                            {new Date(it.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                        <a
                          href={`https://huggingface.co/${it.hfId}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-gray-500 hover:text-blue-400"
                        >
                          <ExternalLink size={14} />
                        </a>
                      </div>
                    ))}
                  </div>
                  {m.description && (
                    <div className="mt-4">
                      <span className="text-[10px] text-gray-500 uppercase">
                        Description
                      </span>
                      <p className="text-sm text-gray-300 mt-0.5">
                        {m.description}
                      </p>
                    </div>
                  )}
                  <div className="text-[10px] text-gray-600 mt-3">
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
