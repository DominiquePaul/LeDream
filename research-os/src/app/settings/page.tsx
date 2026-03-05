"use client";

import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/supabase";
import { useResearch } from "@/components/Shell";

interface SyncLog {
  id: string;
  action: string;
  summary: string;
  details: string;
  created_at: string;
}

export default function SettingsPage() {
  const { session, papers } = useResearch();
  const [logs, setLogs] = useState<SyncLog[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [expandedLog, setExpandedLog] = useState<string | null>(null);

  const unsyncedCount = papers.filter((p) => !p.semantic_scholar_id).length;
  const syncedCount = papers.filter((p) => p.semantic_scholar_id).length;

  const loadLogs = useCallback(async () => {
    const { data } = await supabase
      .from("research_sync_log")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(20);
    setLogs((data || []) as SyncLog[]);
  }, []);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  const triggerSync = async () => {
    setSyncing(true);
    setResult(null);
    try {
      // Run standard sync
      const syncRes = await fetch("/api/semantic-scholar/sync", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({ batchSize: 15 }),
      });
      const syncData = await syncRes.json();

      // Then run AI resolve
      const resolveRes = await fetch("/api/semantic-scholar/resolve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session?.access_token}`,
        },
      });
      const resolveData = await resolveRes.json();

      setResult(
        `Sync: ${syncData.synced} synced, ${syncData.notFound || 0} not found. ` +
        `AI Resolve: ${resolveData.resolved || 0} resolved, ${resolveData.notFound || 0} not found.`
      );
      await loadLogs();
    } catch {
      setResult("Sync failed");
    } finally {
      setSyncing(false);
    }
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const parseSummary = (summary: string) => {
    try {
      return JSON.parse(summary);
    } catch {
      return null;
    }
  };

  return (
    <div className="settings-page">
      <h1>Settings</h1>

      {/* Sync Status */}
      <section className="settings-page__section">
        <h2>Semantic Scholar Sync</h2>
        <p className="settings-page__description">
          Papers are automatically synced with Semantic Scholar every 6 hours via cron job.
          This fetches citation counts, abstracts, and author metadata.
        </p>

        <div className="settings-page__stats">
          <div className="settings-page__stat">
            <span className="settings-page__stat-value">{syncedCount}</span>
            <span className="settings-page__stat-label">Synced</span>
          </div>
          <div className="settings-page__stat">
            <span className="settings-page__stat-value settings-page__stat-value--warn">
              {unsyncedCount}
            </span>
            <span className="settings-page__stat-label">Unsynced</span>
          </div>
          <div className="settings-page__stat">
            <span className="settings-page__stat-value">{papers.length}</span>
            <span className="settings-page__stat-label">Total</span>
          </div>
        </div>

        <div className="settings-page__actions">
          <button
            className="settings-page__sync-btn"
            onClick={triggerSync}
            disabled={syncing}
          >
            {syncing ? "Running sync + AI resolve..." : "Run Manual Sync"}
          </button>
          {result && (
            <p className="settings-page__result">{result}</p>
          )}
        </div>
      </section>

      {/* Sync History */}
      <section className="settings-page__section">
        <h2>Sync History</h2>
        {logs.length === 0 ? (
          <p className="settings-page__empty">No sync logs yet. Syncs are logged automatically.</p>
        ) : (
          <div className="settings-page__logs">
            {logs.map((log) => {
              const summary = parseSummary(log.summary);
              const details = log.details ? JSON.parse(log.details) : [];
              const isExpanded = expandedLog === log.id;
              return (
                <div key={log.id} className="settings-page__log">
                  <div
                    className="settings-page__log-header"
                    onClick={() => setExpandedLog(isExpanded ? null : log.id)}
                  >
                    <span className="settings-page__log-date">
                      {formatDate(log.created_at)}
                    </span>
                    <span className="settings-page__log-action">{log.action}</span>
                    {summary && (
                      <span className="settings-page__log-summary">
                        {summary.synced > 0 && `${summary.synced} synced`}
                        {summary.resolved > 0 && ` ${summary.resolved} resolved`}
                        {summary.refreshed > 0 && ` ${summary.refreshed} refreshed`}
                        {summary.notFound > 0 && ` ${summary.notFound} not found`}
                        {summary.errors > 0 && ` ${summary.errors} errors`}
                      </span>
                    )}
                    <span className="settings-page__log-expand">
                      {isExpanded ? "▾" : "▸"}
                    </span>
                  </div>
                  {isExpanded && details.length > 0 && (
                    <div className="settings-page__log-details">
                      {details.map((line: string, i: number) => (
                        <div key={i} className="settings-page__log-line">
                          {line}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
