"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import {
  ChevronDown,
  ChevronRight,
  Plus,
  Trash2,
  ExternalLink,
  Pencil,
  X,
  Check,
  Link as LinkIcon,
  Eye,
} from "lucide-react";

function visualizerUrl(hfId: string): string {
  return `https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F${encodeURIComponent(hfId)}%2Fepisode_0`;
}

const STATUS_COLORS: Record<string, string> = {
  active: "bg-green-900/50 text-green-400",
  confirmed: "bg-blue-900/50 text-blue-400",
  rejected: "bg-red-900/50 text-red-400",
  exploring: "bg-yellow-900/50 text-yellow-400",
  planned: "bg-gray-800 text-gray-400",
  "in-progress": "bg-amber-900/50 text-amber-400",
  completed: "bg-emerald-900/50 text-emerald-400",
  failed: "bg-red-900/50 text-red-400",
};

export default function ExperimentsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [editingHyp, setEditingHyp] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [editingExp, setEditingExp] = useState<string | null>(null);
  const [editExpName, setEditExpName] = useState("");
  const [editExpNotes, setEditExpNotes] = useState("");
  const [editExpResults, setEditExpResults] = useState("");
  const [editExpStatus, setEditExpStatus] = useState("");
  const [editExpDatasets, setEditExpDatasets] = useState<string[]>([]);
  const [editExpModels, setEditExpModels] = useState<string[]>([]);
  const [showLinkPicker, setShowLinkPicker] = useState<"dataset" | "model" | null>(null);
  const [linkSearch, setLinkSearch] = useState("");

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleAddHypothesis = () => {
    ctx.addHypothesis({
      title: "New Hypothesis",
      description: "Describe your hypothesis here...",
      status: "exploring",
    });
  };

  const handleDeleteHypothesis = (id: string) => {
    if (!confirm("Delete this hypothesis and all its experiments?")) return;
    ctx.deleteHypothesis(id);
  };

  const startEditHypothesis = (h: (typeof data.hypotheses)[0]) => {
    setEditingHyp(h.id);
    setEditTitle(h.title);
    setEditDesc(h.description);
    setEditStatus(h.status);
  };

  const saveEditHypothesis = () => {
    if (!editingHyp) return;
    ctx.updateHypothesis(editingHyp, {
      title: editTitle,
      description: editDesc,
      status: editStatus as "active" | "confirmed" | "rejected" | "exploring",
    });
    setEditingHyp(null);
  };

  const handleAddExperiment = (hypothesisId: string) => {
    ctx.addExperiment({ name: "New Experiment", hypothesisId, status: "planned" });
  };

  const handleDeleteExperiment = (id: string) => {
    if (!confirm("Delete this experiment?")) return;
    ctx.deleteExperiment(id);
  };

  const startEditExperiment = (exp: (typeof data.experiments)[0]) => {
    setEditingExp(exp.id);
    setEditExpName(exp.name);
    setEditExpNotes(exp.notes);
    setEditExpResults(exp.results);
    setEditExpStatus(exp.status);
    setEditExpDatasets([...exp.datasetIds]);
    setEditExpModels([...exp.modelIds]);
    setShowLinkPicker(null);
    setLinkSearch("");
  };

  const saveEditExperiment = () => {
    if (!editingExp) return;
    ctx.updateExperiment(editingExp, {
      name: editExpName,
      notes: editExpNotes,
      results: editExpResults,
      status: editExpStatus as "planned" | "in-progress" | "completed" | "failed",
      datasetIds: editExpDatasets,
      modelIds: editExpModels,
    });
    setEditingExp(null);
    setShowLinkPicker(null);
  };

  const toggleDatasetLink = (did: string) => {
    setEditExpDatasets((prev) =>
      prev.includes(did) ? prev.filter((d) => d !== did) : [...prev, did]
    );
  };

  const toggleModelLink = (mid: string) => {
    setEditExpModels((prev) =>
      prev.includes(mid) ? prev.filter((m) => m !== mid) : [...prev, mid]
    );
  };

  const datasetMap = new Map(data.datasets.map((d) => [d.id, d]));
  const modelMap = new Map(data.models.map((m) => [m.id, m]));

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-4 md:space-y-6">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white">Experiments</h1>
          <p className="text-gray-400 text-sm mt-1">
            Hypothesis-centered view of all experiments
          </p>
        </div>
        <button
          onClick={handleAddHypothesis}
          className="flex items-center gap-2 px-3 md:px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors shrink-0"
        >
          <Plus size={16} />
          <span className="hidden sm:inline">New Hypothesis</span>
          <span className="sm:hidden">New</span>
        </button>
      </div>

      {data.hypotheses.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p>No hypotheses yet. Create one to get started.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {data.hypotheses.map((h) => {
            const isExpanded = expanded.has(h.id);
            const experiments = data.experiments.filter(
              (e) => e.hypothesisId === h.id
            );
            const isEditing = editingHyp === h.id;

            return (
              <div
                key={h.id}
                className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden"
              >
                <div className="p-4">
                  <div className="flex items-start gap-3">
                    <button
                      onClick={() => toggle(h.id)}
                      className="mt-1 text-gray-500 hover:text-gray-300"
                    >
                      {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </button>

                    <div className="flex-1 min-w-0">
                      {isEditing ? (
                        <div className="space-y-2">
                          <input
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
                          />
                          <textarea
                            value={editDesc}
                            onChange={(e) => setEditDesc(e.target.value)}
                            rows={2}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
                          />
                          <select
                            value={editStatus}
                            onChange={(e) => setEditStatus(e.target.value)}
                            className="bg-gray-800 border border-gray-700 rounded px-3 py-1 text-white text-xs"
                          >
                            <option value="exploring">Exploring</option>
                            <option value="active">Active</option>
                            <option value="confirmed">Confirmed</option>
                            <option value="rejected">Rejected</option>
                          </select>
                          <div className="flex gap-2">
                            <button onClick={saveEditHypothesis} className="text-green-400 hover:text-green-300">
                              <Check size={16} />
                            </button>
                            <button onClick={() => setEditingHyp(null)} className="text-gray-400 hover:text-gray-300">
                              <X size={16} />
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-center gap-2">
                            <h3 className="text-white font-medium text-sm">{h.title}</h3>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full ${STATUS_COLORS[h.status] || STATUS_COLORS.planned}`}>
                              {h.status}
                            </span>
                          </div>
                          {h.description && <p className="text-xs text-gray-400 mt-1">{h.description}</p>}
                          <p className="text-[10px] text-gray-600 mt-1">{experiments.length} experiments</p>
                        </>
                      )}
                    </div>

                    {!isEditing && (
                      <div className="flex gap-1">
                        <button onClick={() => startEditHypothesis(h)} className="p-1 text-gray-500 hover:text-gray-300">
                          <Pencil size={14} />
                        </button>
                        <button onClick={() => handleDeleteHypothesis(h.id)} className="p-1 text-gray-500 hover:text-red-400">
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-gray-800">
                    {experiments.length === 0 ? (
                      <p className="p-4 text-xs text-gray-500">No experiments yet.</p>
                    ) : (
                      <div className="divide-y divide-gray-800/50">
                        {experiments.map((exp) => {
                          const isEditingThisExp = editingExp === exp.id;

                          return (
                            <div key={exp.id} className="p-3 md:p-4 pl-6 md:pl-12">
                              {isEditingThisExp ? (
                                <div className="space-y-3">
                                  <div className="flex gap-3">
                                    <div className="flex-1">
                                      <label className="text-[10px] text-gray-500 uppercase">Name</label>
                                      <input
                                        value={editExpName}
                                        onChange={(e) => setEditExpName(e.target.value)}
                                        className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-[10px] text-gray-500 uppercase">Status</label>
                                      <select
                                        value={editExpStatus}
                                        onChange={(e) => setEditExpStatus(e.target.value)}
                                        className="block bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
                                      >
                                        <option value="planned">Planned</option>
                                        <option value="in-progress">In Progress</option>
                                        <option value="completed">Completed</option>
                                        <option value="failed">Failed</option>
                                      </select>
                                    </div>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-gray-500 uppercase">Notes</label>
                                    <textarea
                                      value={editExpNotes}
                                      onChange={(e) => setEditExpNotes(e.target.value)}
                                      rows={2}
                                      className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
                                      placeholder="Experiment notes..."
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-gray-500 uppercase">Results</label>
                                    <textarea
                                      value={editExpResults}
                                      onChange={(e) => setEditExpResults(e.target.value)}
                                      rows={2}
                                      className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
                                      placeholder="Experiment results..."
                                    />
                                  </div>

                                  {/* Dataset linker */}
                                  <div>
                                    <div className="flex items-center gap-2 mb-1">
                                      <label className="text-[10px] text-gray-500 uppercase">Linked Datasets</label>
                                      <button
                                        onClick={() => {
                                          setShowLinkPicker(showLinkPicker === "dataset" ? null : "dataset");
                                          setLinkSearch("");
                                        }}
                                        className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                      >
                                        <LinkIcon size={10} /> {showLinkPicker === "dataset" ? "Done" : "Add/Remove"}
                                      </button>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5 mb-1">
                                      {editExpDatasets.map((did) => {
                                        const ds = datasetMap.get(did);
                                        return ds ? (
                                          <span key={did} className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 bg-blue-900/30 text-blue-400 rounded-full">
                                            {ds.name}
                                            <button onClick={() => toggleDatasetLink(did)} className="hover:text-red-400">
                                              <X size={10} />
                                            </button>
                                          </span>
                                        ) : null;
                                      })}
                                      {editExpDatasets.length === 0 && <span className="text-[10px] text-gray-600 italic">None</span>}
                                    </div>
                                    {showLinkPicker === "dataset" && (
                                      <div className="bg-gray-800 border border-gray-700 rounded p-2 max-h-40 overflow-y-auto">
                                        <input
                                          value={linkSearch}
                                          onChange={(e) => setLinkSearch(e.target.value)}
                                          placeholder="Search datasets..."
                                          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white mb-2"
                                        />
                                        {data.datasets
                                          .filter((d) => !linkSearch || d.name.toLowerCase().includes(linkSearch.toLowerCase()))
                                          .slice(0, 20)
                                          .map((d) => (
                                            <button
                                              key={d.id}
                                              onClick={() => toggleDatasetLink(d.id)}
                                              className={`block w-full text-left px-2 py-1 text-xs rounded hover:bg-gray-700 ${
                                                editExpDatasets.includes(d.id) ? "text-blue-400" : "text-gray-300"
                                              }`}
                                            >
                                              {editExpDatasets.includes(d.id) ? "\u2713 " : ""}{d.name}
                                            </button>
                                          ))}
                                      </div>
                                    )}
                                  </div>

                                  {/* Model linker */}
                                  <div>
                                    <div className="flex items-center gap-2 mb-1">
                                      <label className="text-[10px] text-gray-500 uppercase">Linked Models</label>
                                      <button
                                        onClick={() => {
                                          setShowLinkPicker(showLinkPicker === "model" ? null : "model");
                                          setLinkSearch("");
                                        }}
                                        className="text-[10px] text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                                      >
                                        <LinkIcon size={10} /> {showLinkPicker === "model" ? "Done" : "Add/Remove"}
                                      </button>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5 mb-1">
                                      {editExpModels.map((mid) => {
                                        const m = modelMap.get(mid);
                                        return m ? (
                                          <span key={mid} className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 bg-emerald-900/30 text-emerald-400 rounded-full">
                                            {m.name}
                                            <button onClick={() => toggleModelLink(mid)} className="hover:text-red-400">
                                              <X size={10} />
                                            </button>
                                          </span>
                                        ) : null;
                                      })}
                                      {editExpModels.length === 0 && <span className="text-[10px] text-gray-600 italic">None</span>}
                                    </div>
                                    {showLinkPicker === "model" && (
                                      <div className="bg-gray-800 border border-gray-700 rounded p-2 max-h-40 overflow-y-auto">
                                        <input
                                          value={linkSearch}
                                          onChange={(e) => setLinkSearch(e.target.value)}
                                          placeholder="Search models..."
                                          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white mb-2"
                                        />
                                        {data.models
                                          .filter((m) => !linkSearch || m.name.toLowerCase().includes(linkSearch.toLowerCase()))
                                          .slice(0, 20)
                                          .map((m) => (
                                            <button
                                              key={m.id}
                                              onClick={() => toggleModelLink(m.id)}
                                              className={`block w-full text-left px-2 py-1 text-xs rounded hover:bg-gray-700 ${
                                                editExpModels.includes(m.id) ? "text-emerald-400" : "text-gray-300"
                                              }`}
                                            >
                                              {editExpModels.includes(m.id) ? "\u2713 " : ""}{m.name}
                                            </button>
                                          ))}
                                      </div>
                                    )}
                                  </div>

                                  <div className="flex gap-2">
                                    <button
                                      onClick={saveEditExperiment}
                                      className="flex items-center gap-1 px-3 py-1 bg-green-700 hover:bg-green-600 text-white text-xs rounded"
                                    >
                                      <Check size={14} /> Save
                                    </button>
                                    <button
                                      onClick={() => { setEditingExp(null); setShowLinkPicker(null); }}
                                      className="flex items-center gap-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
                                    >
                                      <X size={14} /> Cancel
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <div className="flex items-start justify-between">
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                      <span className="text-sm text-white font-medium">{exp.name}</span>
                                      <span className={`text-[10px] px-2 py-0.5 rounded-full ${STATUS_COLORS[exp.status] || STATUS_COLORS.planned}`}>
                                        {exp.status}
                                      </span>
                                    </div>
                                    {exp.notes && <p className="text-xs text-gray-400 mt-1">{exp.notes}</p>}
                                    {exp.results && <p className="text-xs text-emerald-400/70 mt-1">Result: {exp.results}</p>}
                                    {exp.datasetIds.length > 0 && (
                                      <div className="flex flex-wrap gap-1.5 mt-2">
                                        {exp.datasetIds.map((did) => {
                                          const ds = datasetMap.get(did);
                                          if (!ds) return null;
                                          return (
                                            <span key={did} className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 bg-blue-900/30 text-blue-400 rounded-full">
                                              <a href={visualizerUrl(ds.hfId)} target="_blank" rel="noopener noreferrer" className="hover:text-emerald-400" title="Visualize">
                                                <Eye size={10} />
                                              </a>
                                              <a href={ds.hfUrl} target="_blank" rel="noopener noreferrer" className="hover:text-blue-300">
                                                {ds.name}
                                              </a>
                                            </span>
                                          );
                                        })}
                                      </div>
                                    )}
                                    {exp.modelIds.length > 0 && (
                                      <div className="flex flex-wrap gap-1.5 mt-1">
                                        {exp.modelIds.map((mid) => {
                                          const m = modelMap.get(mid);
                                          if (!m) return null;
                                          return (
                                            <a key={mid} href={m.hfUrl} target="_blank" rel="noopener noreferrer"
                                              className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 bg-emerald-900/30 text-emerald-400 rounded-full hover:bg-emerald-900/50">
                                              <ExternalLink size={10} />{m.name}
                                            </a>
                                          );
                                        })}
                                      </div>
                                    )}
                                  </div>
                                  <div className="flex gap-1">
                                    <button onClick={() => startEditExperiment(exp)} className="p-1 text-gray-500 hover:text-gray-300">
                                      <Pencil size={14} />
                                    </button>
                                    <button onClick={() => handleDeleteExperiment(exp.id)} className="p-1 text-gray-500 hover:text-red-400">
                                      <Trash2 size={14} />
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="p-3 pl-6 md:pl-12">
                      <button onClick={() => handleAddExperiment(h.id)}
                        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300">
                        <Plus size={14} />Add Experiment
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
