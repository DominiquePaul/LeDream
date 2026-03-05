"use client";

import Link from "next/link";
import { useData } from "@/components/Shell";
import {
  Database,
  Box,
  FlaskConical,
  Lightbulb,
  Clock,
  ExternalLink,
  ArrowRight,
} from "lucide-react";

function fmtHours(h: number): string {
  if (h === 0) return "0m";
  const hours = Math.floor(h);
  const mins = Math.round((h - hours) * 60);
  if (hours === 0) return `${mins}m`;
  if (mins === 0) return `${hours}h`;
  return `${hours}h${mins}m`;
}

export default function Dashboard() {
  const { data } = useData();

  const totalHours = data.datasets.reduce(
    (sum, d) => sum + d.metadata.estimatedHours,
    0
  );
  const totalEpisodes = data.datasets.reduce(
    (sum, d) => sum + d.metadata.episodeCount,
    0
  );

  // Hours and episodes by tag
  const tagMap = new Map(data.tags.map((t) => [t.id, t]));
  const hoursByTag = new Map<string, number>();
  const episodesByTag = new Map<string, number>();
  for (const ds of data.datasets) {
    for (const tid of ds.tags) {
      const tag = tagMap.get(tid);
      if (tag) {
        hoursByTag.set(
          tag.name,
          (hoursByTag.get(tag.name) || 0) + ds.metadata.estimatedHours
        );
        episodesByTag.set(
          tag.name,
          (episodesByTag.get(tag.name) || 0) + ds.metadata.episodeCount
        );
      }
    }
  }

  // Checkpoints count
  const totalCheckpoints = data.models.reduce(
    (sum, m) => sum + m.iterations.length,
    0
  );

  const statCards = [
    {
      label: "Datasets",
      value: data.datasets.length,
      icon: Database,
      color: "text-blue-400",
      href: "/datasets",
    },
    {
      label: "Models",
      value: data.models.length,
      icon: Box,
      color: "text-emerald-400",
      href: "/models",
    },
    {
      label: "Checkpoints",
      value: totalCheckpoints,
      icon: Box,
      color: "text-teal-400",
      href: "/models",
    },
    {
      label: "Experiments",
      value: data.experiments.length,
      icon: FlaskConical,
      color: "text-amber-400",
      href: "/experiments",
    },
    {
      label: "Total Hours",
      value: fmtHours(totalHours),
      icon: Clock,
      color: "text-rose-400",
      href: "/datasets",
    },
    {
      label: "Total Episodes",
      value: totalEpisodes.toLocaleString(),
      icon: Database,
      color: "text-cyan-400",
      href: "/datasets",
    },
  ];

  // Recent datasets (last 5 modified)
  const recentDatasets = [...data.datasets]
    .sort((a, b) => b.lastModified.localeCompare(a.lastModified))
    .slice(0, 5);

  // Active hypotheses
  const activeHypotheses = data.hypotheses.filter(
    (h) => h.status === "active" || h.status === "exploring"
  );

  // Experiment status breakdown
  const expByStatus = new Map<string, number>();
  for (const e of data.experiments) {
    expByStatus.set(e.status, (expByStatus.get(e.status) || 0) + 1);
  }

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-6 md:space-y-8">
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400 text-sm mt-1">
          Overview of your robotics experiments
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 md:gap-4">
        {statCards.map((s) => (
          <Link
            key={s.label}
            href={s.href}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors group"
          >
            <div className="flex items-center gap-2 mb-2">
              <s.icon size={16} className={s.color} />
              <span className="text-xs text-gray-400">{s.label}</span>
            </div>
            <p className="text-2xl font-bold text-white">{s.value}</p>
          </Link>
        ))}
      </div>

      {/* Hours and Episodes by Tag */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">
            Data Hours by Tag
          </h2>
          {hoursByTag.size === 0 ? (
            <p className="text-sm text-gray-500">No data yet</p>
          ) : (
            <div className="space-y-3">
              {Array.from(hoursByTag.entries())
                .sort((a, b) => b[1] - a[1])
                .map(([name, hours]) => {
                  const tag = data.tags.find((t) => t.name === name);
                  const pct = totalHours > 0 ? (hours / totalHours) * 100 : 0;
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <span
                        className="inline-block w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: tag?.color || "#6B7280" }}
                      />
                      <span className="text-sm text-gray-300 w-32 truncate">
                        {name}
                      </span>
                      <div className="flex-1 bg-gray-800 rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: tag?.color || "#6B7280",
                          }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-16 text-right">
                        {fmtHours(hours)}
                      </span>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">
            Episodes by Tag
          </h2>
          {episodesByTag.size === 0 ? (
            <p className="text-sm text-gray-500">No data yet</p>
          ) : (
            <div className="space-y-3">
              {Array.from(episodesByTag.entries())
                .sort((a, b) => b[1] - a[1])
                .map(([name, episodes]) => {
                  const tag = data.tags.find((t) => t.name === name);
                  const pct = totalEpisodes > 0 ? (episodes / totalEpisodes) * 100 : 0;
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <span
                        className="inline-block w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: tag?.color || "#6B7280" }}
                      />
                      <span className="text-sm text-gray-300 w-32 truncate">
                        {name}
                      </span>
                      <div className="flex-1 bg-gray-800 rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: tag?.color || "#6B7280",
                          }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-20 text-right">
                        {episodes.toLocaleString()}
                      </span>
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Active Hypotheses */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-300">
              Active Hypotheses
            </h2>
            <Link href="/experiments" className="text-xs text-gray-500 hover:text-white flex items-center gap-1">
              View all <ArrowRight size={12} />
            </Link>
          </div>
          {activeHypotheses.length === 0 ? (
            <p className="text-gray-500 text-sm">No active hypotheses</p>
          ) : (
            <div className="space-y-3">
              {activeHypotheses.map((h) => (
                <Link
                  key={h.id}
                  href="/experiments"
                  className="block p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 hover:border-gray-600 transition-colors"
                >
                  <p className="text-sm text-white font-medium">{h.title}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        h.status === "active"
                          ? "bg-green-900/50 text-green-400"
                          : "bg-yellow-900/50 text-yellow-400"
                      }`}
                    >
                      {h.status}
                    </span>
                    <span className="text-xs text-gray-500">
                      {h.experimentIds.length} experiments
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Recent Datasets */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-300">
              Recently Updated Datasets
            </h2>
            <Link href="/datasets" className="text-xs text-gray-500 hover:text-white flex items-center gap-1">
              View all <ArrowRight size={12} />
            </Link>
          </div>
          <div className="space-y-2">
            {recentDatasets.map((d) => (
              <div
                key={d.id}
                className="flex items-center justify-between p-2 rounded hover:bg-gray-800/50 transition-colors"
              >
                <div>
                  <p className="text-sm text-white">{d.name}</p>
                  <p className="text-xs text-gray-500">
                    {d.metadata.episodeCount} episodes &middot;{" "}
                    {fmtHours(d.metadata.estimatedHours)}
                  </p>
                </div>
                <a
                  href={d.hfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-500 hover:text-blue-400"
                >
                  <ExternalLink size={14} />
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Experiment Status Breakdown */}
      {data.experiments.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">
            Experiment Status
          </h2>
          <div className="flex gap-4 flex-wrap">
            {[
              { status: "planned", color: "bg-gray-600" },
              { status: "in-progress", color: "bg-amber-500" },
              { status: "completed", color: "bg-emerald-500" },
              { status: "failed", color: "bg-red-500" },
            ].map(({ status, color }) => {
              const count = expByStatus.get(status) || 0;
              if (count === 0) return null;
              return (
                <div key={status} className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
                  <span className="text-sm text-gray-300">{status}</span>
                  <span className="text-sm text-white font-semibold">{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
