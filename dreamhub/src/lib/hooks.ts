"use client";

import { useState, useEffect, useCallback } from "react";
import { v4 as uuid } from "uuid";
import type { ResearchData, Dataset, Evaluation, Hypothesis, Experiment, Tag } from "./types";
import { syncFromHuggingFace } from "./hf-sync";
import { seedData } from "./seed";
import {
  fetchAllData,
  saveSyncResult,
  dbUpsertTag,
  dbDeleteTag,
  dbUpsertDataset,
  dbDeleteDataset,
  dbUpsertModel,
  dbDeleteModel,
  dbUpsertHypothesis,
  dbDeleteHypothesis,
  dbUpsertExperiment,
  dbDeleteExperiment,
  dbUpsertEvaluation,
  dbDeleteEvaluation,
  dbResetAll,
} from "./supabase-db";

const EMPTY: ResearchData = {
  hypotheses: [],
  experiments: [],
  datasets: [],
  models: [],
  evaluations: [],
  tags: [],
  lastSynced: "",
};

export function useResearchData() {
  const [data, setData] = useState<ResearchData>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);

  // Load from Supabase on mount
  useEffect(() => {
    fetchAllData()
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load data from Supabase:", err);
        setLoading(false);
      });
  }, []);

  const sync = useCallback(async () => {
    setSyncing(true);
    setSyncError(null);
    try {
      const current = await fetchAllData();
      console.log(
        "[DreamHub] Starting HF sync, current datasets:",
        current.datasets.length,
      );
      const synced = await syncFromHuggingFace(current);
      const seeded = seedData(synced);
      console.log(
        "[DreamHub] HF sync done, datasets:",
        seeded.datasets.length,
        "models:",
        seeded.models.length,
      );
      await saveSyncResult(seeded);
      setData(seeded);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("Sync failed:", msg, err);
      setSyncError(msg);
    } finally {
      setSyncing(false);
    }
  }, []);

  // Auto-sync on first load if never synced
  useEffect(() => {
    if (!loading && data.lastSynced === "") {
      sync();
    }
  }, [loading, data.lastSynced, sync]);

  // --- CRUD operations ---

  const addHypothesis = useCallback(
    (partial: Partial<Hypothesis>) => {
      const now = new Date().toISOString();
      const h: Hypothesis = {
        id: uuid(),
        title: partial.title || "New Hypothesis",
        description: partial.description || "",
        status: partial.status || "exploring",
        experimentIds: [],
        tags: partial.tags || [],
        createdAt: now,
        updatedAt: now,
      };
      setData((prev) => ({ ...prev, hypotheses: [...prev.hypotheses, h] }));
      dbUpsertHypothesis(h).catch((err) =>
        console.error("Failed to save hypothesis:", err),
      );
      return h;
    },
    [],
  );

  const updateHypothesis = useCallback(
    (id: string, updates: Partial<Hypothesis>) => {
      setData((prev) => {
        const next = {
          ...prev,
          hypotheses: prev.hypotheses.map((h) =>
            h.id === id
              ? { ...h, ...updates, updatedAt: new Date().toISOString() }
              : h,
          ),
        };
        const updated = next.hypotheses.find((h) => h.id === id);
        if (updated)
          dbUpsertHypothesis(updated).catch((err) =>
            console.error("Failed to update hypothesis:", err),
          );
        return next;
      });
    },
    [],
  );

  const deleteHypothesis = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      hypotheses: prev.hypotheses.filter((h) => h.id !== id),
      experiments: prev.experiments.filter((e) => e.hypothesisId !== id),
    }));
    dbDeleteHypothesis(id).catch((err) =>
      console.error("Failed to delete hypothesis:", err),
    );
  }, []);

  const addExperiment = useCallback(
    (partial: Partial<Experiment> & { hypothesisId: string }) => {
      const now = new Date().toISOString();
      const e: Experiment = {
        id: uuid(),
        name: partial.name || "New Experiment",
        hypothesisId: partial.hypothesisId,
        datasetIds: partial.datasetIds || [],
        modelIds: partial.modelIds || [],
        status: partial.status || "planned",
        notes: partial.notes || "",
        results: partial.results || "",
        createdAt: now,
        updatedAt: now,
      };
      setData((prev) => ({
        ...prev,
        experiments: [...prev.experiments, e],
        hypotheses: prev.hypotheses.map((h) =>
          h.id === e.hypothesisId
            ? { ...h, experimentIds: [...h.experimentIds, e.id] }
            : h,
        ),
      }));
      dbUpsertExperiment(e).catch((err) =>
        console.error("Failed to save experiment:", err),
      );
      return e;
    },
    [],
  );

  const updateExperiment = useCallback(
    (id: string, updates: Partial<Experiment>) => {
      setData((prev) => {
        const next = {
          ...prev,
          experiments: prev.experiments.map((e) =>
            e.id === id
              ? { ...e, ...updates, updatedAt: new Date().toISOString() }
              : e,
          ),
        };
        const updated = next.experiments.find((e) => e.id === id);
        if (updated)
          dbUpsertExperiment(updated).catch((err) =>
            console.error("Failed to update experiment:", err),
          );
        return next;
      });
    },
    [],
  );

  const deleteExperiment = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      experiments: prev.experiments.filter((e) => e.id !== id),
      hypotheses: prev.hypotheses.map((h) => ({
        ...h,
        experimentIds: h.experimentIds.filter((eid) => eid !== id),
      })),
    }));
    dbDeleteExperiment(id).catch((err) =>
      console.error("Failed to delete experiment:", err),
    );
  }, []);

  const addDataset = useCallback(
    (partial: Partial<Dataset>) => {
      const now = new Date().toISOString();
      const d: Dataset = {
        id: uuid(),
        hfId: "",
        name: partial.name || "New Collection",
        description: partial.description || "",
        tags: partial.tags || [],
        metadata: partial.metadata || {
          collectionConditions: "",
          teleoperatorInstructions: "",
          knownIssues: [],
          notes: "",
          estimatedHours: 0,
          episodeCount: 0,
        },
        status: "planned",
        hfUrl: "",
        downloads: 0,
        lastModified: now,
        createdAt: now,
      };
      setData((prev) => ({ ...prev, datasets: [...prev.datasets, d] }));
      dbUpsertDataset(d).catch((err) =>
        console.error("Failed to save dataset:", err),
      );
      return d;
    },
    [],
  );

  const updateDataset = useCallback(
    (id: string, updates: Record<string, unknown>) => {
      setData((prev) => {
        const next = {
          ...prev,
          datasets: prev.datasets.map((d) =>
            d.id === id ? { ...d, ...updates } : d,
          ),
        };
        const updated = next.datasets.find((d) => d.id === id);
        if (updated)
          dbUpsertDataset(updated).catch((err) =>
            console.error("Failed to update dataset:", err),
          );
        return next;
      });
    },
    [],
  );

  const deleteDataset = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      datasets: prev.datasets.filter((d) => d.id !== id),
      experiments: prev.experiments.map((e) => ({
        ...e,
        datasetIds: e.datasetIds.filter((did) => did !== id),
      })),
    }));
    dbDeleteDataset(id).catch((err) =>
      console.error("Failed to delete dataset:", err),
    );
  }, []);

  const updateModel = useCallback(
    (id: string, updates: Record<string, unknown>) => {
      setData((prev) => {
        const next = {
          ...prev,
          models: prev.models.map((m) =>
            m.id === id ? { ...m, ...updates } : m,
          ),
        };
        const updated = next.models.find((m) => m.id === id);
        if (updated)
          dbUpsertModel(updated).catch((err) =>
            console.error("Failed to update model:", err),
          );
        return next;
      });
    },
    [],
  );

  const deleteModel = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      models: prev.models.filter((m) => m.id !== id),
      experiments: prev.experiments.map((e) => ({
        ...e,
        modelIds: e.modelIds.filter((mid) => mid !== id),
      })),
    }));
    dbDeleteModel(id).catch((err) =>
      console.error("Failed to delete model:", err),
    );
  }, []);

  const addEvaluation = useCallback(
    (partial: Partial<Evaluation> & { modelId: string }) => {
      const ev: Evaluation = {
        id: uuid(),
        experimentId: partial.experimentId || null,
        modelId: partial.modelId,
        iterationId: partial.iterationId || null,
        evalDatasetId: partial.evalDatasetId || null,
        outcome: partial.outcome || "success",
        notes: partial.notes || "",
        createdAt: new Date().toISOString(),
      };
      setData((prev) => ({ ...prev, evaluations: [...prev.evaluations, ev] }));
      dbUpsertEvaluation(ev).catch((err) =>
        console.error("Failed to save evaluation:", err),
      );
      return ev;
    },
    [],
  );

  const updateEvaluation = useCallback(
    (id: string, updates: Partial<Evaluation>) => {
      setData((prev) => {
        const next = {
          ...prev,
          evaluations: prev.evaluations.map((ev) =>
            ev.id === id ? { ...ev, ...updates } : ev,
          ),
        };
        const updated = next.evaluations.find((ev) => ev.id === id);
        if (updated)
          dbUpsertEvaluation(updated).catch((err) =>
            console.error("Failed to update evaluation:", err),
          );
        return next;
      });
    },
    [],
  );

  const deleteEvaluation = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      evaluations: prev.evaluations.filter((ev) => ev.id !== id),
    }));
    dbDeleteEvaluation(id).catch((err) =>
      console.error("Failed to delete evaluation:", err),
    );
  }, []);

  const addTag = useCallback((partial: Partial<Tag>) => {
    const t: Tag = {
      id: uuid(),
      name: partial.name || "New Tag",
      color: partial.color || "#6B7280",
      category: partial.category || "custom",
    };
    setData((prev) => ({ ...prev, tags: [...prev.tags, t] }));
    dbUpsertTag(t).catch((err) =>
      console.error("Failed to save tag:", err),
    );
    return t;
  }, []);

  const updateTag = useCallback((id: string, updates: Partial<Tag>) => {
    setData((prev) => {
      const next = {
        ...prev,
        tags: prev.tags.map((t) => (t.id === id ? { ...t, ...updates } : t)),
      };
      const updated = next.tags.find((t) => t.id === id);
      if (updated)
        dbUpsertTag(updated).catch((err) =>
          console.error("Failed to update tag:", err),
        );
      return next;
    });
  }, []);

  const deleteTag = useCallback((id: string) => {
    setData((prev) => ({
      ...prev,
      tags: prev.tags.filter((t) => t.id !== id),
      datasets: prev.datasets.map((d) => ({
        ...d,
        tags: d.tags.filter((t) => t !== id),
      })),
      models: prev.models.map((m) => ({
        ...m,
        tags: m.tags.filter((t) => t !== id),
      })),
      hypotheses: prev.hypotheses.map((h) => ({
        ...h,
        tags: h.tags.filter((t) => t !== id),
      })),
    }));
    dbDeleteTag(id).catch((err) =>
      console.error("Failed to delete tag:", err),
    );
  }, []);

  const resetAll = useCallback(async () => {
    setData(EMPTY);
    await dbResetAll();
  }, []);

  const importData = useCallback(async (imported: ResearchData) => {
    await dbResetAll();
    await saveSyncResult(imported);
    setData(imported);
  }, []);

  return {
    data,
    loading,
    syncing,
    syncError,
    sync,
    addHypothesis,
    updateHypothesis,
    deleteHypothesis,
    addExperiment,
    updateExperiment,
    deleteExperiment,
    addDataset,
    updateDataset,
    deleteDataset,
    updateModel,
    deleteModel,
    addEvaluation,
    updateEvaluation,
    deleteEvaluation,
    addTag,
    updateTag,
    deleteTag,
    resetAll,
    importData,
  };
}
