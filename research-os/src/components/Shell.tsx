"use client";

import { useResearchData } from "@/lib/hooks";
import Sidebar from "./Sidebar";
import type { ResearchData, Hypothesis, Experiment, Tag } from "@/lib/types";
import { createContext, useContext } from "react";

interface DataCtx {
  data: ResearchData;
  loading: boolean;
  syncing: boolean;
  syncError: string | null;
  sync: () => Promise<void>;
  addHypothesis: (p: Partial<Hypothesis>) => Hypothesis;
  updateHypothesis: (id: string, u: Partial<Hypothesis>) => void;
  deleteHypothesis: (id: string) => void;
  addExperiment: (p: Partial<Experiment> & { hypothesisId: string }) => Experiment;
  updateExperiment: (id: string, u: Partial<Experiment>) => void;
  deleteExperiment: (id: string) => void;
  updateDataset: (id: string, u: Record<string, unknown>) => void;
  deleteDataset: (id: string) => void;
  updateModel: (id: string, u: Record<string, unknown>) => void;
  deleteModel: (id: string) => void;
  addTag: (p: Partial<Tag>) => Tag;
  updateTag: (id: string, u: Partial<Tag>) => void;
  deleteTag: (id: string) => void;
  resetAll: () => void;
}

const noop = () => {};
const noopAsync = async () => {};
const noopReturn = (() => ({} as never));

const DataContext = createContext<DataCtx>({
  data: { hypotheses: [], experiments: [], datasets: [], models: [], tags: [], lastSynced: "" },
  loading: true,
  syncing: false,
  syncError: null,
  sync: noopAsync,
  addHypothesis: noopReturn,
  updateHypothesis: noop,
  deleteHypothesis: noop,
  addExperiment: noopReturn,
  updateExperiment: noop,
  deleteExperiment: noop,
  updateDataset: noop,
  deleteDataset: noop,
  updateModel: noop,
  deleteModel: noop,
  addTag: noopReturn,
  updateTag: noop,
  deleteTag: noop,
  resetAll: noop,
});

export const useData = () => useContext(DataContext);

export default function Shell({ children }: { children: React.ReactNode }) {
  const ctx = useResearchData();

  return (
    <DataContext.Provider value={ctx}>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        <Sidebar
          syncing={ctx.syncing}
          syncError={ctx.syncError}
          lastSynced={ctx.data.lastSynced}
          datasetCount={ctx.data.datasets.length}
          modelCount={ctx.data.models.length}
          onSync={ctx.sync}
        />
        <main className="flex-1 overflow-auto">
          {ctx.loading && ctx.data.lastSynced === "" ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-gray-700 border-t-blue-500 rounded-full animate-spin mx-auto" />
                <p className="text-gray-400 mt-4">Loading &amp; syncing with HuggingFace...</p>
              </div>
            </div>
          ) : (
            children
          )}
        </main>
      </div>
    </DataContext.Provider>
  );
}
