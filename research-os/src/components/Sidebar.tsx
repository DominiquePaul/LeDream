"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FlaskConical,
  Network,
  Database,
  Box,
  Tags,
  RefreshCw,
  AlertCircle,
  Download,
  Upload,
  Trash2,
} from "lucide-react";
import { readData, writeData, clearData } from "@/lib/storage";
import type { ResearchData } from "@/lib/types";

const nav = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/graph", label: "Graph", icon: Network },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/models", label: "Models", icon: Box },
  { href: "/tags", label: "Tags", icon: Tags },
];

export default function Sidebar({
  syncing,
  syncError,
  lastSynced,
  datasetCount,
  modelCount,
  onSync,
  onReset,
}: {
  syncing: boolean;
  syncError: string | null;
  lastSynced: string;
  datasetCount: number;
  modelCount: number;
  onSync: () => void;
  onReset: () => void;
}) {
  const pathname = usePathname();

  const handleExport = () => {
    const data = readData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research-os-backup-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const imported = JSON.parse(ev.target?.result as string) as ResearchData;
          if (!imported.datasets || !imported.models || !imported.tags) {
            alert("Invalid Research OS data file.");
            return;
          }
          writeData(imported);
          window.location.reload();
        } catch {
          alert("Failed to parse JSON file.");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const handleReset = () => {
    if (!confirm("Reset ALL data? This will clear everything and re-sync from HuggingFace. This cannot be undone.")) return;
    onReset();
  };

  return (
    <aside className="w-56 bg-gray-950 text-gray-300 flex flex-col border-r border-gray-800 shrink-0">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-lg font-bold text-white tracking-tight">
          Research OS
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">Robotics Experiment Hub</p>
      </div>

      <nav className="flex-1 py-2">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-4 py-2 text-sm transition-colors ${
                active
                  ? "bg-gray-800 text-white font-medium"
                  : "hover:bg-gray-900 hover:text-white"
              }`}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-800 space-y-2">
        <button
          onClick={onSync}
          disabled={syncing}
          className="flex items-center gap-2 text-xs text-gray-400 hover:text-white transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={syncing ? "animate-spin" : ""} />
          {syncing ? "Syncing HF..." : "Sync with HuggingFace"}
        </button>
        {syncError && (
          <div className="flex items-start gap-1.5 text-[10px] text-red-400">
            <AlertCircle size={12} className="mt-0.5 shrink-0" />
            <span>Sync failed: {syncError}</span>
          </div>
        )}
        {lastSynced && (
          <div className="text-[10px] text-gray-600">
            <p>Last: {new Date(lastSynced).toLocaleString()}</p>
            <p>{datasetCount} datasets, {modelCount} models</p>
          </div>
        )}

        <div className="pt-2 border-t border-gray-800/50 space-y-1">
          <button
            onClick={handleExport}
            className="flex items-center gap-2 text-[11px] text-gray-500 hover:text-white transition-colors w-full"
          >
            <Download size={12} /> Export Data
          </button>
          <button
            onClick={handleImport}
            className="flex items-center gap-2 text-[11px] text-gray-500 hover:text-white transition-colors w-full"
          >
            <Upload size={12} /> Import Data
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 text-[11px] text-gray-500 hover:text-red-400 transition-colors w-full"
          >
            <Trash2 size={12} /> Reset All Data
          </button>
        </div>
      </div>
    </aside>
  );
}
