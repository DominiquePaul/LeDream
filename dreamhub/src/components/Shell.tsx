"use client";

import { useResearchData } from "@/lib/hooks";
import { supabase } from "@/lib/supabase";
import Sidebar from "./Sidebar";
import type { ResearchData, Dataset, Evaluation, Hypothesis, Experiment, Tag } from "@/lib/types";
import { createContext, useContext, useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";

interface DataCtx {
  data: ResearchData;
  loading: boolean;
  syncing: boolean;
  syncError: string | null;
  sync: () => Promise<void>;
  addHypothesis: (p: Partial<Hypothesis>) => Hypothesis;
  updateHypothesis: (id: string, u: Partial<Hypothesis>) => void;
  deleteHypothesis: (id: string) => void;
  addExperiment: (
    p: Partial<Experiment> & { hypothesisId: string },
  ) => Experiment;
  updateExperiment: (id: string, u: Partial<Experiment>) => void;
  deleteExperiment: (id: string) => void;
  addDataset: (p: Partial<Dataset>) => Dataset;
  updateDataset: (id: string, u: Record<string, unknown>) => void;
  deleteDataset: (id: string) => void;
  updateModel: (id: string, u: Record<string, unknown>) => void;
  deleteModel: (id: string) => void;
  addEvaluation: (p: Partial<Evaluation> & { modelId: string }) => Evaluation;
  updateEvaluation: (id: string, u: Partial<Evaluation>) => void;
  deleteEvaluation: (id: string) => void;
  addTag: (p: Partial<Tag>) => Tag;
  updateTag: (id: string, u: Partial<Tag>) => void;
  deleteTag: (id: string) => void;
  resetAll: () => void;
  importData: (d: ResearchData) => Promise<void>;
  session: Session | null;
}

const noop = () => {};
const noopAsync = async () => {};
const noopReturn = () => ({}) as never;

const DataContext = createContext<DataCtx>({
  data: {
    hypotheses: [],
    experiments: [],
    datasets: [],
    models: [],
    evaluations: [],
    tags: [],
    lastSynced: "",
  },
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
  addDataset: noopReturn,
  updateDataset: noop,
  deleteDataset: noop,
  updateModel: noop,
  deleteModel: noop,
  addEvaluation: noopReturn,
  updateEvaluation: noop,
  deleteEvaluation: noop,
  addTag: noopReturn,
  updateTag: noop,
  deleteTag: noop,
  resetAll: noop,
  importData: noopAsync,
  session: null,
});

export const useData = () => useContext(DataContext);

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSignUp, setIsSignUp] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const { error: authError } = isSignUp
      ? await supabase.auth.signUp({ email, password })
      : await supabase.auth.signInWithPassword({ email, password });

    if (authError) {
      setError(authError.message);
      setLoading(false);
    }
    // On success, onAuthStateChange in Shell will pick up the session
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-950">
      <div className="w-full max-w-sm mx-4">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white font-[var(--font-dm-mono)]">
            DreamHub
          </h1>
          <p className="text-xs text-gray-500 mt-1 tracking-wide uppercase">
            Dream Machines
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4"
        >
          <h2 className="text-lg font-semibold text-white">
            {isSignUp ? "Create Account" : "Sign In"}
          </h2>

          {error && (
            <div className="text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg p-3">
              {error}
            </div>
          )}

          <div>
            <label className="text-xs text-gray-400 uppercase">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white mt-1 focus:outline-none focus:border-blue-500"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 uppercase">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white mt-1 focus:outline-none focus:border-blue-500"
              placeholder="Min. 6 characters"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
          >
            {loading ? "..." : isSignUp ? "Create Account" : "Sign In"}
          </button>

          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError(null);
            }}
            className="w-full text-xs text-gray-500 hover:text-white transition-colors"
          >
            {isSignUp
              ? "Already have an account? Sign in"
              : "First time? Create account"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function Shell({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      setSession(s);
      setAuthLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s);
    });

    return () => subscription.unsubscribe();
  }, []);

  // Show loading while checking auth
  if (authLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-950">
        <div className="w-8 h-8 border-2 border-gray-700 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  // Show login if not authenticated
  if (!session) {
    return <LoginForm />;
  }

  return <AuthenticatedShell session={session}>{children}</AuthenticatedShell>;
}

function AuthenticatedShell({
  children,
  session,
}: {
  children: React.ReactNode;
  session: Session;
}) {
  const ctx = useResearchData();

  return (
    <DataContext.Provider value={{ ...ctx, session }}>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        <Sidebar
          syncing={ctx.syncing}
          syncError={ctx.syncError}
          lastSynced={ctx.data.lastSynced}
          datasetCount={ctx.data.datasets.length}
          modelCount={ctx.data.models.length}
          onSync={ctx.sync}
          onReset={ctx.resetAll}
          onImport={ctx.importData}
          data={ctx.data}
          userEmail={session.user.email || ""}
        />
        <main className="flex-1 overflow-auto pt-14 md:pt-0">
          {ctx.loading && ctx.data.lastSynced === "" ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-gray-700 border-t-blue-500 rounded-full animate-spin mx-auto" />
                <p className="text-gray-400 mt-4">
                  Loading &amp; syncing with HuggingFace...
                </p>
                <p className="text-gray-600 text-xs mt-1 font-[var(--font-dm-mono)]">
                  DreamHub
                </p>
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
