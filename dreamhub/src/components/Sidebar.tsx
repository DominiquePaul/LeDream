"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  LayoutDashboard,
  FlaskConical,
  Network,
  Database,
  Box,
  ClipboardCheck,
  Tags,
  RefreshCw,
  AlertCircle,
  Download,
  Upload,
  Trash2,
  Menu,
  X,
  LogOut,
} from "lucide-react";
import { supabase } from "@/lib/supabase";
import type { ResearchData } from "@/lib/types";

const nav = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/graph", label: "Graph", icon: Network },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/models", label: "Models", icon: Box },
  { href: "/evaluations", label: "Evaluations", icon: ClipboardCheck },
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
  onImport,
  data,
  userEmail,
}: {
  syncing: boolean;
  syncError: string | null;
  lastSynced: string;
  datasetCount: number;
  modelCount: number;
  onSync: () => void;
  onReset: () => void;
  onImport: (d: ResearchData) => Promise<void>;
  data: ResearchData;
  userEmail: string;
}) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dreamhub-backup-${new Date().toISOString().slice(0, 10)}.json`;
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
      reader.onload = async (ev) => {
        try {
          const imported = JSON.parse(
            ev.target?.result as string,
          ) as ResearchData;
          if (!imported.datasets || !imported.models || !imported.tags) {
            alert("Invalid DreamHub data file.");
            return;
          }
          await onImport(imported);
        } catch {
          alert("Failed to parse JSON file.");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const handleReset = () => {
    if (
      !confirm(
        "Reset ALL data? This will clear everything and re-sync from HuggingFace. This cannot be undone.",
      )
    )
      return;
    onReset();
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
  };

  const sidebarContent = (
    <>
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight font-[var(--font-dm-mono)]">
            DreamHub
          </h1>
          <p className="text-[10px] text-gray-500 mt-0.5 tracking-wide uppercase">
            Dream Machines
          </p>
        </div>
        <button
          onClick={() => setMobileOpen(false)}
          className="md:hidden p-1 text-gray-400 hover:text-white"
        >
          <X size={20} />
        </button>
      </div>

      <nav className="flex-1 py-2">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              onClick={() => setMobileOpen(false)}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
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
            <p>
              {datasetCount} datasets, {modelCount} models
            </p>
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

        <div className="pt-2 border-t border-gray-800/50">
          <div className="text-[10px] text-gray-600 truncate mb-1">
            {userEmail}
          </div>
          <button
            onClick={handleSignOut}
            className="flex items-center gap-2 text-[11px] text-gray-500 hover:text-white transition-colors w-full"
          >
            <LogOut size={12} /> Sign Out
          </button>
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile top bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-40 bg-gray-950 border-b border-gray-800 px-4 py-3 flex items-center justify-between">
        <button
          onClick={() => setMobileOpen(true)}
          className="p-1 text-gray-400 hover:text-white"
        >
          <Menu size={22} />
        </button>
        <h1 className="text-sm font-bold text-white font-[var(--font-dm-mono)]">
          DreamHub
        </h1>
        <button
          onClick={onSync}
          disabled={syncing}
          className="p-1 text-gray-400 hover:text-white disabled:opacity-50"
        >
          <RefreshCw size={16} className={syncing ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-50 bg-black/60"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile slide-out sidebar */}
      <aside
        className={`md:hidden fixed top-0 left-0 bottom-0 z-50 w-64 bg-gray-950 text-gray-300 flex flex-col border-r border-gray-800 transform transition-transform duration-200 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {sidebarContent}
      </aside>

      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-56 bg-gray-950 text-gray-300 flex-col border-r border-gray-800 shrink-0">
        {sidebarContent}
      </aside>
    </>
  );
}
