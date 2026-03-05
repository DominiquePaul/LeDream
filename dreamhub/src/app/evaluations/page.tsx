"use client";

import { useState } from "react";
import { useData } from "@/components/Shell";
import type { Evaluation } from "@/lib/types";
import {
  Plus,
  Trash2,
  CheckCircle2,
  AlertCircle,
  XCircle,
  ExternalLink,
  ChevronDown,
  Eye,
} from "lucide-react";

const OUTCOME_CONFIG = {
  success: { label: "Success", color: "text-green-400", bg: "bg-green-900/30", icon: CheckCircle2 },
  partial: { label: "Partial", color: "text-amber-400", bg: "bg-amber-900/30", icon: AlertCircle },
  failure: { label: "Failure", color: "text-red-400", bg: "bg-red-900/30", icon: XCircle },
} as const;

export default function EvaluationsPage() {
  const ctx = useData();
  const { data } = ctx;
  const [showForm, setShowForm] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedIteration, setSelectedIteration] = useState("");
  const [selectedEvalDs, setSelectedEvalDs] = useState("");
  const [selectedExperiment, setSelectedExperiment] = useState("");
  const [outcome, setOutcome] = useState<Evaluation["outcome"]>("success");
  const [notes, setNotes] = useState("");
  const [filterModel, setFilterModel] = useState("");

  // Eval datasets are those tagged with "evaluation" or named eval_*
  const evalDatasets = data.datasets.filter(
    (d) => d.name.startsWith("eval_") || d.status === "synced",
  ).filter((d) => d.name.startsWith("eval_"));

  const model = data.models.find((m) => m.id === selectedModel);
  const iterations = model?.iterations || [];

  const handleSubmit = () => {
    if (!selectedModel) return;
    ctx.addEvaluation({
      modelId: selectedModel,
      iterationId: selectedIteration || null,
      evalDatasetId: selectedEvalDs || null,
      experimentId: selectedExperiment || null,
      outcome,
      notes,
    });
    setNotes("");
    setOutcome("success");
    setSelectedIteration("");
    setSelectedEvalDs("");
    setSelectedExperiment("");
  };

  // Group evaluations by model
  const evalsByModel = new Map<string, typeof data.evaluations>();
  for (const ev of data.evaluations) {
    if (!evalsByModel.has(ev.modelId)) evalsByModel.set(ev.modelId, []);
    evalsByModel.get(ev.modelId)!.push(ev);
  }

  // Filter
  const filteredEvals = filterModel
    ? data.evaluations.filter((ev) => ev.modelId === filterModel)
    : data.evaluations;

  // Stats
  const totalEvals = filteredEvals.length;
  const successCount = filteredEvals.filter((ev) => ev.outcome === "success").length;
  const partialCount = filteredEvals.filter((ev) => ev.outcome === "partial").length;
  const failureCount = filteredEvals.filter((ev) => ev.outcome === "failure").length;
  const successRate = totalEvals > 0 ? Math.round((successCount / totalEvals) * 100) : 0;

  // Model stats for summary cards
  const modelStats = data.models
    .filter((m) => evalsByModel.has(m.id))
    .map((m) => {
      const evals = evalsByModel.get(m.id)!;
      const s = evals.filter((ev) => ev.outcome === "success").length;
      return { model: m, total: evals.length, success: s, rate: Math.round((s / evals.length) * 100) };
    })
    .sort((a, b) => b.rate - a.rate);

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-4 md:space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white">Evaluations</h1>
          <p className="text-gray-400 text-sm mt-1">
            {totalEvals} rollout{totalEvals !== 1 ? "s" : ""} logged
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1.5 px-3 py-2 bg-green-900/30 text-green-400 rounded-lg hover:bg-green-900/50 text-sm transition-colors"
        >
          <Plus size={14} /> Log Rollout
        </button>
      </div>

      {/* Quick log form */}
      {showForm && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => { setSelectedModel(e.target.value); setSelectedIteration(""); }}
                className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              >
                <option value="">Select model...</option>
                {data.models.map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Checkpoint</label>
              <select
                value={selectedIteration}
                onChange={(e) => setSelectedIteration(e.target.value)}
                disabled={!selectedModel}
                className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5 disabled:opacity-50"
              >
                <option value="">Any / latest</option>
                {iterations.map((it) => (
                  <option key={it.id} value={it.id}>{it.version}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Eval Dataset (video)</label>
              <select
                value={selectedEvalDs}
                onChange={(e) => setSelectedEvalDs(e.target.value)}
                className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              >
                <option value="">None</option>
                {evalDatasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase">Experiment</label>
              <select
                value={selectedExperiment}
                onChange={(e) => setSelectedExperiment(e.target.value)}
                className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
              >
                <option value="">None</option>
                {data.experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>{exp.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="text-[10px] text-gray-500 uppercase mb-1 block">Outcome</label>
            <div className="flex gap-2">
              {(["success", "partial", "failure"] as const).map((o) => {
                const cfg = OUTCOME_CONFIG[o];
                const Icon = cfg.icon;
                return (
                  <button
                    key={o}
                    onClick={() => setOutcome(o)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors border ${
                      outcome === o
                        ? `${cfg.bg} ${cfg.color} border-current`
                        : "bg-gray-800 text-gray-400 border-gray-700 hover:bg-gray-700"
                    }`}
                  >
                    <Icon size={14} /> {cfg.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="text-[10px] text-gray-500 uppercase">Notes</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              placeholder="What happened during the rollout?"
              className="block w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-white mt-0.5"
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={!selectedModel}
            className="px-4 py-2 bg-green-700 hover:bg-green-600 text-white text-sm rounded-lg disabled:opacity-50 transition-colors"
          >
            Log Rollout
          </button>
        </div>
      )}

      {/* Stats bar */}
      {totalEvals > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Total Rollouts" value={totalEvals.toString()} />
          <StatCard label="Success Rate" value={`${successRate}%`} color="text-green-400" />
          <StatCard label="Partial" value={partialCount.toString()} color="text-amber-400" />
          <StatCard label="Failures" value={failureCount.toString()} color="text-red-400" />
        </div>
      )}

      {/* Model summary cards */}
      {modelStats.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-gray-400 mb-2">By Model</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {modelStats.map(({ model: m, total, success, rate }) => (
              <button
                key={m.id}
                onClick={() => setFilterModel(filterModel === m.id ? "" : m.id)}
                className={`text-left p-3 rounded-lg border transition-colors ${
                  filterModel === m.id
                    ? "bg-gray-800 border-blue-700"
                    : "bg-gray-900 border-gray-800 hover:bg-gray-800/50"
                }`}
              >
                <div className="text-sm text-white font-medium truncate">{m.name}</div>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-xs text-gray-500">{total} rollouts</span>
                  <span className="text-xs text-green-400">{rate}% success</span>
                  <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-green-500 rounded-full"
                      style={{ width: `${rate}%` }}
                    />
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Filter indicator */}
      {filterModel && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">
            Filtered by: <strong className="text-white">{data.models.find(m => m.id === filterModel)?.name}</strong>
          </span>
          <button
            onClick={() => setFilterModel("")}
            className="text-xs text-gray-500 hover:text-white underline"
          >
            Clear
          </button>
        </div>
      )}

      {/* Evaluation log */}
      <div className="space-y-2">
        {filteredEvals.length === 0 && (
          <div className="text-center py-12 text-gray-600">
            <p className="text-sm">No evaluations logged yet.</p>
            <p className="text-xs mt-1">Click &quot;Log Rollout&quot; to record your first evaluation.</p>
          </div>
        )}
        {[...filteredEvals]
          .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
          .map((ev) => {
            const m = data.models.find((m) => m.id === ev.modelId);
            const it = m?.iterations.find((i) => i.id === ev.iterationId);
            const evalDs = ev.evalDatasetId ? data.datasets.find((d) => d.id === ev.evalDatasetId) : null;
            const exp = ev.experimentId ? data.experiments.find((e) => e.id === ev.experimentId) : null;
            const cfg = OUTCOME_CONFIG[ev.outcome];
            const Icon = cfg.icon;

            return (
              <div key={ev.id} className="bg-gray-900 rounded-lg border border-gray-800 p-3 flex items-start gap-3">
                <div className={`mt-0.5 ${cfg.color}`}>
                  <Icon size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-white font-medium">
                      {m?.name || "Unknown model"}
                    </span>
                    {it && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-gray-800 text-gray-400 rounded">
                        {it.version}
                      </span>
                    )}
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${cfg.bg} ${cfg.color}`}>
                      {cfg.label}
                    </span>
                  </div>
                  {ev.notes && (
                    <p className="text-xs text-gray-400 mt-1">{ev.notes}</p>
                  )}
                  <div className="flex items-center gap-3 mt-1.5 flex-wrap">
                    {evalDs && (
                      <a
                        href={evalDs.hfUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1"
                      >
                        <Eye size={10} /> {evalDs.name}
                      </a>
                    )}
                    {exp && (
                      <span className="text-[10px] text-gray-500">
                        Exp: {exp.name}
                      </span>
                    )}
                    <span className="text-[10px] text-gray-600">
                      {new Date(ev.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => {
                    if (confirm("Delete this evaluation?")) ctx.deleteEvaluation(ev.id);
                  }}
                  className="p-1 text-gray-600 hover:text-red-400 transition-colors shrink-0"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={`text-xl font-bold mt-0.5 ${color || "text-white"}`}>{value}</div>
    </div>
  );
}
