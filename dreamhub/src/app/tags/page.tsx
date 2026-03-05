"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import { Plus, Trash2, Pencil, Check, X } from "lucide-react";

const CATEGORIES = ["task", "robot", "status", "custom"] as const;
const PRESET_COLORS = [
  "#EF4444", "#F97316", "#F59E0B", "#22C55E", "#10B981",
  "#14B8A6", "#3B82F6", "#6366F1", "#8B5CF6", "#A855F7",
  "#EC4899", "#64748B",
];

export default function TagsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState("#3B82F6");
  const [newCategory, setNewCategory] = useState<(typeof CATEGORIES)[number]>("custom");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editColor, setEditColor] = useState("");
  const [editCategory, setEditCategory] = useState<(typeof CATEGORIES)[number]>("custom");

  const handleCreateTag = () => {
    if (!newName.trim()) return;
    ctx.addTag({ name: newName, color: newColor, category: newCategory });
    setNewName("");
  };

  const handleDeleteTag = (id: string) => {
    if (!confirm("Delete this tag? It will be removed from all entities.")) return;
    ctx.deleteTag(id);
  };

  const startEdit = (tag: (typeof data.tags)[0]) => {
    setEditingId(tag.id);
    setEditName(tag.name);
    setEditColor(tag.color);
    setEditCategory(tag.category);
  };

  const saveEdit = () => {
    if (!editingId) return;
    ctx.updateTag(editingId, { name: editName, color: editColor, category: editCategory });
    setEditingId(null);
  };

  // Count usage
  const tagUsage = new Map<string, number>();
  for (const d of data.datasets) for (const t of d.tags) tagUsage.set(t, (tagUsage.get(t) || 0) + 1);
  for (const m of data.models) for (const t of m.tags) tagUsage.set(t, (tagUsage.get(t) || 0) + 1);
  for (const h of data.hypotheses) for (const t of h.tags) tagUsage.set(t, (tagUsage.get(t) || 0) + 1);

  const grouped = CATEGORIES.map((cat) => ({
    category: cat,
    tags: data.tags.filter((t) => t.category === cat),
  }));

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto space-y-6 md:space-y-8">
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white">Tags</h1>
        <p className="text-gray-400 text-sm mt-1">
          Manage tags for datasets, models, and experiments
        </p>
      </div>

      {/* Create new tag */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Create New Tag</h2>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[150px]">
            <label className="text-[10px] text-gray-500 uppercase">Name</label>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Tag name..."
              className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              onKeyDown={(e) => e.key === "Enter" && handleCreateTag()}
            />
          </div>
          <div>
            <label className="text-[10px] text-gray-500 uppercase">Category</label>
            <select
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value as typeof newCategory)}
              className="block bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-gray-500 uppercase">Color</label>
            <div className="flex flex-wrap gap-1 mt-0.5">
              {PRESET_COLORS.map((c) => (
                <button
                  key={c}
                  onClick={() => setNewColor(c)}
                  className={`w-6 h-6 rounded-full transition-all ${
                    newColor === c ? "ring-2 ring-white ring-offset-1 ring-offset-gray-900" : ""
                  }`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>
          <button
            onClick={handleCreateTag}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
          >
            <Plus size={14} />
            Add
          </button>
        </div>
      </div>

      {/* Tag list by category */}
      {grouped.map(({ category, tags }) => (
        <div key={category}>
          <h2 className="text-xs text-gray-500 uppercase font-semibold mb-2">
            {category} ({tags.length})
          </h2>
          <div className="space-y-1">
            {tags.map((tag) => {
              const isEditing = editingId === tag.id;
              const usage = tagUsage.get(tag.id) || 0;

              return (
                <div
                  key={tag.id}
                  className="flex items-center gap-3 p-3 bg-gray-900 rounded-lg border border-gray-800"
                >
                  {isEditing ? (
                    <>
                      <input
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
                      />
                      <select
                        value={editCategory}
                        onChange={(e) => setEditCategory(e.target.value as typeof editCategory)}
                        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                      >
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                      <div className="flex gap-1">
                        {PRESET_COLORS.map((c) => (
                          <button
                            key={c}
                            onClick={() => setEditColor(c)}
                            className={`w-4 h-4 rounded-full ${
                              editColor === c ? "ring-2 ring-white ring-offset-1 ring-offset-gray-900" : ""
                            }`}
                            style={{ backgroundColor: c }}
                          />
                        ))}
                      </div>
                      <button onClick={saveEdit} className="text-green-400 hover:text-green-300">
                        <Check size={16} />
                      </button>
                      <button onClick={() => setEditingId(null)} className="text-gray-400 hover:text-gray-300">
                        <X size={16} />
                      </button>
                    </>
                  ) : (
                    <>
                      <span
                        className="w-4 h-4 rounded-full shrink-0"
                        style={{ backgroundColor: tag.color }}
                      />
                      <span className="text-sm text-white flex-1">{tag.name}</span>
                      <span className="text-xs text-gray-500">{usage} uses</span>
                      <button
                        onClick={() => startEdit(tag)}
                        className="p-1 text-gray-500 hover:text-gray-300"
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        onClick={() => handleDeleteTag(tag.id)}
                        className="p-1 text-gray-500 hover:text-red-400"
                      >
                        <Trash2 size={14} />
                      </button>
                    </>
                  )}
                </div>
              );
            })}
            {tags.length === 0 && (
              <p className="text-xs text-gray-600 py-2">No tags in this category</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
