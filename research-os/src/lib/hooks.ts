"use client";

import { useState, useEffect, useCallback } from "react";
import { v4 as uuid } from "uuid";
import type { ResearchData, Hypothesis, Experiment, Tag } from "./types";
import { readData, writeData } from "./storage";
import { syncFromHuggingFace } from "./hf-sync";
import { seedData } from "./seed";

const EMPTY: ResearchData = {
  hypotheses: [],
  experiments: [],
  datasets: [],
  models: [],
  tags: [],
  lastSynced: "",
};

export function useResearchData() {
  const [data, setData] = useState<ResearchData>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = readData();
    setData(stored);
    setLoading(false);
  }, []);

  // Save to localStorage whenever data changes (skip initial empty state)
  const save = useCallback((newData: ResearchData) => {
    setData(newData);
    writeData(newData);
  }, []);

  const sync = useCallback(async () => {
    setSyncing(true);
    try {
      const current = readData();
      let synced = await syncFromHuggingFace(current);
      synced = seedData(synced);
      save(synced);
    } catch (err) {
      console.error("Sync failed:", err);
    } finally {
      setSyncing(false);
    }
  }, [save]);

  // Auto-sync on first load if never synced
  useEffect(() => {
    if (!loading && data.lastSynced === "") {
      sync();
    }
  }, [loading, data.lastSynced, sync]);

  // --- CRUD operations ---

  const addHypothesis = useCallback((partial: Partial<Hypothesis>) => {
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
    const next = { ...readData() };
    next.hypotheses = [...next.hypotheses, h];
    save(next);
    return h;
  }, [save]);

  const updateHypothesis = useCallback((id: string, updates: Partial<Hypothesis>) => {
    const next = { ...readData() };
    next.hypotheses = next.hypotheses.map(h =>
      h.id === id ? { ...h, ...updates, updatedAt: new Date().toISOString() } : h
    );
    save(next);
  }, [save]);

  const deleteHypothesis = useCallback((id: string) => {
    const next = { ...readData() };
    next.hypotheses = next.hypotheses.filter(h => h.id !== id);
    next.experiments = next.experiments.filter(e => e.hypothesisId !== id);
    save(next);
  }, [save]);

  const addExperiment = useCallback((partial: Partial<Experiment> & { hypothesisId: string }) => {
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
    const next = { ...readData() };
    next.experiments = [...next.experiments, e];
    next.hypotheses = next.hypotheses.map(h =>
      h.id === e.hypothesisId ? { ...h, experimentIds: [...h.experimentIds, e.id] } : h
    );
    save(next);
    return e;
  }, [save]);

  const updateExperiment = useCallback((id: string, updates: Partial<Experiment>) => {
    const next = { ...readData() };
    next.experiments = next.experiments.map(e =>
      e.id === id ? { ...e, ...updates, updatedAt: new Date().toISOString() } : e
    );
    save(next);
  }, [save]);

  const deleteExperiment = useCallback((id: string) => {
    const next = { ...readData() };
    next.experiments = next.experiments.filter(e => e.id !== id);
    next.hypotheses = next.hypotheses.map(h => ({
      ...h, experimentIds: h.experimentIds.filter(eid => eid !== id)
    }));
    save(next);
  }, [save]);

  const updateDataset = useCallback((id: string, updates: Record<string, unknown>) => {
    const next = { ...readData() };
    next.datasets = next.datasets.map(d =>
      d.id === id ? { ...d, ...updates } : d
    );
    save(next);
  }, [save]);

  const deleteDataset = useCallback((id: string) => {
    const next = { ...readData() };
    next.datasets = next.datasets.filter(d => d.id !== id);
    next.experiments = next.experiments.map(e => ({
      ...e, datasetIds: e.datasetIds.filter(did => did !== id)
    }));
    save(next);
  }, [save]);

  const updateModel = useCallback((id: string, updates: Record<string, unknown>) => {
    const next = { ...readData() };
    next.models = next.models.map(m =>
      m.id === id ? { ...m, ...updates } : m
    );
    save(next);
  }, [save]);

  const deleteModel = useCallback((id: string) => {
    const next = { ...readData() };
    next.models = next.models.filter(m => m.id !== id);
    next.experiments = next.experiments.map(e => ({
      ...e, modelIds: e.modelIds.filter(mid => mid !== id)
    }));
    save(next);
  }, [save]);

  const addTag = useCallback((partial: Partial<Tag>) => {
    const t: Tag = {
      id: uuid(),
      name: partial.name || "New Tag",
      color: partial.color || "#6B7280",
      category: partial.category || "custom",
    };
    const next = { ...readData() };
    next.tags = [...next.tags, t];
    save(next);
    return t;
  }, [save]);

  const updateTag = useCallback((id: string, updates: Partial<Tag>) => {
    const next = { ...readData() };
    next.tags = next.tags.map(t => t.id === id ? { ...t, ...updates } : t);
    save(next);
  }, [save]);

  const deleteTag = useCallback((id: string) => {
    const next = { ...readData() };
    next.tags = next.tags.filter(t => t.id !== id);
    next.datasets = next.datasets.map(d => ({ ...d, tags: d.tags.filter(t => t !== id) }));
    next.models = next.models.map(m => ({ ...m, tags: m.tags.filter(t => t !== id) }));
    next.hypotheses = next.hypotheses.map(h => ({ ...h, tags: h.tags.filter(t => t !== id) }));
    save(next);
  }, [save]);

  const resetAll = useCallback(() => {
    save(EMPTY);
  }, [save]);

  return {
    data,
    loading,
    syncing,
    sync,
    addHypothesis,
    updateHypothesis,
    deleteHypothesis,
    addExperiment,
    updateExperiment,
    deleteExperiment,
    updateDataset,
    deleteDataset,
    updateModel,
    deleteModel,
    addTag,
    updateTag,
    deleteTag,
    resetAll,
  };
}
