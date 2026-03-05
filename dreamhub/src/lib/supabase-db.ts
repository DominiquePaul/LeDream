import { supabase } from "./supabase";
import type {
  ResearchData,
  Tag,
  Dataset,
  Model,
  ModelIteration,
  Hypothesis,
  Experiment,
  Evaluation,
} from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groupJunction(
  rows: Record<string, string>[] | null,
  keyField: string,
  valueField: string,
): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const r of rows || []) {
    const k = r[keyField];
    if (!map.has(k)) map.set(k, []);
    map.get(k)!.push(r[valueField]);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Fetch all data
// ---------------------------------------------------------------------------

export async function fetchAllData(): Promise<ResearchData> {
  const [
    { data: tags },
    { data: datasets },
    { data: models },
    { data: iterations },
    { data: hypotheses },
    { data: experiments },
    { data: evaluations },
    { data: datasetTags },
    { data: modelTags },
    { data: hypothesisTags },
    { data: expDatasets },
    { data: expModels },
    { data: iterDatasets },
    { data: settings },
  ] = await Promise.all([
    supabase.from("tags").select("*"),
    supabase.from("datasets").select("*"),
    supabase.from("models").select("*"),
    supabase.from("model_iterations").select("*"),
    supabase.from("hypotheses").select("*"),
    supabase.from("experiments").select("*"),
    supabase.from("evaluations").select("*"),
    supabase.from("dataset_tags").select("*"),
    supabase.from("model_tags").select("*"),
    supabase.from("hypothesis_tags").select("*"),
    supabase.from("experiment_datasets").select("*"),
    supabase.from("experiment_models").select("*"),
    supabase.from("iteration_datasets").select("*"),
    supabase.from("app_settings").select("*").eq("key", "lastSynced"),
  ]);

  const dsTagMap = groupJunction(datasetTags, "dataset_id", "tag_id");
  const mdTagMap = groupJunction(modelTags, "model_id", "tag_id");
  const hyTagMap = groupJunction(hypothesisTags, "hypothesis_id", "tag_id");
  const expDsMap = groupJunction(expDatasets, "experiment_id", "dataset_id");
  const expMdMap = groupJunction(expModels, "experiment_id", "model_id");
  const iterDsMap = groupJunction(iterDatasets, "iteration_id", "dataset_id");

  // Group iterations by model_id
  const iterByModel = new Map<string, typeof iterations>();
  for (const it of iterations || []) {
    if (!iterByModel.has(it.model_id)) iterByModel.set(it.model_id, []);
    iterByModel.get(it.model_id)!.push(it);
  }

  // Group experiment IDs by hypothesis_id
  const expByHyp = new Map<string, string[]>();
  for (const e of experiments || []) {
    if (!expByHyp.has(e.hypothesis_id)) expByHyp.set(e.hypothesis_id, []);
    expByHyp.get(e.hypothesis_id)!.push(e.id);
  }

  return {
    tags: (tags || []).map((t) => ({
      id: t.id,
      name: t.name,
      color: t.color,
      category: t.category as Tag["category"],
    })),
    datasets: (datasets || []).map((d) => ({
      id: d.id,
      hfId: d.hf_id,
      name: d.name,
      description: d.description,
      tags: dsTagMap.get(d.id) || [],
      metadata: {
        collectionConditions: d.collection_conditions,
        teleoperatorInstructions: d.teleoperator_instructions,
        knownIssues: d.known_issues || [],
        notes: d.notes,
        estimatedHours: d.estimated_hours,
        episodeCount: d.episode_count,
      },
      status: (d.status || "synced") as Dataset["status"],
      hfUrl: d.hf_url,
      downloads: d.downloads,
      lastModified: d.last_modified,
      createdAt: d.created_at,
    })),
    models: (models || []).map((m) => ({
      id: m.id,
      hfId: m.hf_id,
      name: m.name,
      description: m.description,
      tags: mdTagMap.get(m.id) || [],
      policyType: m.policy_type,
      iterations: (iterByModel.get(m.id) || [])
        .map((it) => ({
          id: it.id,
          version: it.version,
          config: it.config || {},
          notes: it.notes,
          trainedOn: iterDsMap.get(it.id) || [],
          hfId: it.hf_id,
          hfUrl: it.hf_url,
          createdAt: it.created_at,
        }))
        .sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
      hfUrl: m.hf_url,
      downloads: m.downloads,
      lastModified: m.last_modified,
      createdAt: m.created_at,
    })),
    hypotheses: (hypotheses || []).map((h) => ({
      id: h.id,
      title: h.title,
      description: h.description,
      status: h.status as Hypothesis["status"],
      experimentIds: expByHyp.get(h.id) || [],
      tags: hyTagMap.get(h.id) || [],
      createdAt: h.created_at,
      updatedAt: h.updated_at,
    })),
    experiments: (experiments || []).map((e) => ({
      id: e.id,
      name: e.name,
      hypothesisId: e.hypothesis_id,
      datasetIds: expDsMap.get(e.id) || [],
      modelIds: expMdMap.get(e.id) || [],
      status: e.status as Experiment["status"],
      notes: e.notes,
      results: e.results,
      createdAt: e.created_at,
      updatedAt: e.updated_at,
    })),
    evaluations: (evaluations || []).map((ev) => ({
      id: ev.id,
      experimentId: ev.experiment_id,
      modelId: ev.model_id,
      iterationId: ev.iteration_id,
      evalDatasetId: ev.eval_dataset_id,
      outcome: ev.outcome as Evaluation["outcome"],
      notes: ev.notes,
      createdAt: ev.created_at,
    })),
    lastSynced: settings?.[0]?.value || "",
  };
}

// ---------------------------------------------------------------------------
// Bulk save after HF sync
// ---------------------------------------------------------------------------

export async function saveSyncResult(data: ResearchData): Promise<void> {
  // 1. Upsert tags
  if (data.tags.length > 0) {
    const { error } = await supabase.from("tags").upsert(
      data.tags.map((t) => ({
        id: t.id,
        name: t.name,
        color: t.color,
        category: t.category,
      })),
      { onConflict: "id" },
    );
    if (error) console.error("Tags upsert error:", error);
  }

  // 2. Upsert datasets + junction
  for (const d of data.datasets) {
    await dbUpsertDataset(d);
  }

  // 3. Upsert models + iterations + junctions
  for (const m of data.models) {
    await dbUpsertModel(m);
  }

  // 4. Upsert hypotheses + tags
  for (const h of data.hypotheses) {
    await dbUpsertHypothesis(h);
  }

  // 5. Upsert experiments + junctions
  for (const e of data.experiments) {
    await dbUpsertExperiment(e);
  }

  // 6. Update lastSynced
  await dbSetLastSynced(data.lastSynced);
}

// ---------------------------------------------------------------------------
// Tag CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertTag(tag: Tag): Promise<void> {
  const { error } = await supabase.from("tags").upsert({
    id: tag.id,
    name: tag.name,
    color: tag.color,
    category: tag.category,
  });
  if (error) throw error;
}

export async function dbDeleteTag(id: string): Promise<void> {
  // CASCADE handles junction tables
  const { error } = await supabase.from("tags").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Dataset CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertDataset(dataset: Dataset): Promise<void> {
  const { error } = await supabase.from("datasets").upsert({
    id: dataset.id,
    hf_id: dataset.hfId,
    name: dataset.name,
    description: dataset.description,
    collection_conditions: dataset.metadata.collectionConditions,
    teleoperator_instructions: dataset.metadata.teleoperatorInstructions,
    known_issues: dataset.metadata.knownIssues,
    notes: dataset.metadata.notes,
    estimated_hours: dataset.metadata.estimatedHours,
    episode_count: dataset.metadata.episodeCount,
    status: dataset.status || "synced",
    hf_url: dataset.hfUrl,
    downloads: dataset.downloads,
    last_modified: dataset.lastModified,
    created_at: dataset.createdAt,
  });
  if (error) throw error;

  // Resync tags
  await supabase.from("dataset_tags").delete().eq("dataset_id", dataset.id);
  if (dataset.tags.length > 0) {
    const { error: tagError } = await supabase.from("dataset_tags").insert(
      dataset.tags.map((tid) => ({ dataset_id: dataset.id, tag_id: tid })),
    );
    if (tagError) console.error("Dataset tags insert error:", tagError);
  }
}

export async function dbDeleteDataset(id: string): Promise<void> {
  // CASCADE handles junction tables
  const { error } = await supabase.from("datasets").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Model CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertModel(model: Model): Promise<void> {
  const { error } = await supabase.from("models").upsert({
    id: model.id,
    hf_id: model.hfId,
    name: model.name,
    description: model.description,
    policy_type: model.policyType,
    hf_url: model.hfUrl,
    downloads: model.downloads,
    last_modified: model.lastModified,
    created_at: model.createdAt,
  });
  if (error) throw error;

  // Resync tags
  await supabase.from("model_tags").delete().eq("model_id", model.id);
  if (model.tags.length > 0) {
    const { error: tagError } = await supabase.from("model_tags").insert(
      model.tags.map((tid) => ({ model_id: model.id, tag_id: tid })),
    );
    if (tagError) console.error("Model tags insert error:", tagError);
  }

  // Upsert iterations
  for (const it of model.iterations) {
    const { error: itError } = await supabase.from("model_iterations").upsert({
      id: it.id,
      model_id: model.id,
      version: it.version,
      config: it.config,
      notes: it.notes,
      hf_id: it.hfId,
      hf_url: it.hfUrl,
      created_at: it.createdAt,
    });
    if (itError) console.error("Iteration upsert error:", itError);

    // Resync trainedOn
    await supabase
      .from("iteration_datasets")
      .delete()
      .eq("iteration_id", it.id);
    if (it.trainedOn.length > 0) {
      await supabase.from("iteration_datasets").insert(
        it.trainedOn.map((did) => ({ iteration_id: it.id, dataset_id: did })),
      );
    }
  }
}

export async function dbDeleteModel(id: string): Promise<void> {
  // CASCADE handles iterations and junction tables
  const { error } = await supabase.from("models").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Hypothesis CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertHypothesis(hypothesis: Hypothesis): Promise<void> {
  const { error } = await supabase.from("hypotheses").upsert({
    id: hypothesis.id,
    title: hypothesis.title,
    description: hypothesis.description,
    status: hypothesis.status,
    created_at: hypothesis.createdAt,
    updated_at: hypothesis.updatedAt,
  });
  if (error) throw error;

  // Resync tags
  await supabase
    .from("hypothesis_tags")
    .delete()
    .eq("hypothesis_id", hypothesis.id);
  if (hypothesis.tags.length > 0) {
    await supabase.from("hypothesis_tags").insert(
      hypothesis.tags.map((tid) => ({
        hypothesis_id: hypothesis.id,
        tag_id: tid,
      })),
    );
  }
}

export async function dbDeleteHypothesis(id: string): Promise<void> {
  // CASCADE handles experiments and junction tables
  const { error } = await supabase.from("hypotheses").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Experiment CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertExperiment(experiment: Experiment): Promise<void> {
  const { error } = await supabase.from("experiments").upsert({
    id: experiment.id,
    name: experiment.name,
    hypothesis_id: experiment.hypothesisId,
    status: experiment.status,
    notes: experiment.notes,
    results: experiment.results,
    created_at: experiment.createdAt,
    updated_at: experiment.updatedAt,
  });
  if (error) throw error;

  // Resync dataset links
  await supabase
    .from("experiment_datasets")
    .delete()
    .eq("experiment_id", experiment.id);
  if (experiment.datasetIds.length > 0) {
    await supabase.from("experiment_datasets").insert(
      experiment.datasetIds.map((did) => ({
        experiment_id: experiment.id,
        dataset_id: did,
      })),
    );
  }

  // Resync model links
  await supabase
    .from("experiment_models")
    .delete()
    .eq("experiment_id", experiment.id);
  if (experiment.modelIds.length > 0) {
    await supabase.from("experiment_models").insert(
      experiment.modelIds.map((mid) => ({
        experiment_id: experiment.id,
        model_id: mid,
      })),
    );
  }
}

export async function dbDeleteExperiment(id: string): Promise<void> {
  // CASCADE handles junction tables
  const { error } = await supabase.from("experiments").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Evaluation CRUD
// ---------------------------------------------------------------------------

export async function dbUpsertEvaluation(evaluation: Evaluation): Promise<void> {
  const { error } = await supabase.from("evaluations").upsert({
    id: evaluation.id,
    experiment_id: evaluation.experimentId,
    model_id: evaluation.modelId,
    iteration_id: evaluation.iterationId,
    eval_dataset_id: evaluation.evalDatasetId,
    outcome: evaluation.outcome,
    notes: evaluation.notes,
    created_at: evaluation.createdAt,
  });
  if (error) throw error;
}

export async function dbDeleteEvaluation(id: string): Promise<void> {
  const { error } = await supabase.from("evaluations").delete().eq("id", id);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function dbSetLastSynced(ts: string): Promise<void> {
  const { error } = await supabase.from("app_settings").upsert({
    key: "lastSynced",
    value: ts,
  });
  if (error) console.error("Failed to set lastSynced:", error);
}

// ---------------------------------------------------------------------------
// Reset all data
// ---------------------------------------------------------------------------

export async function dbResetAll(): Promise<void> {
  // Delete in order that respects dependencies (CASCADE handles most, but be safe)
  await supabase.from("evaluations").delete().neq("id", "");
  await supabase.from("experiment_datasets").delete().neq("experiment_id", "");
  await supabase.from("experiment_models").delete().neq("experiment_id", "");
  await supabase.from("iteration_datasets").delete().neq("iteration_id", "");
  await supabase.from("dataset_tags").delete().neq("dataset_id", "");
  await supabase.from("model_tags").delete().neq("model_id", "");
  await supabase.from("hypothesis_tags").delete().neq("hypothesis_id", "");
  await supabase.from("model_iterations").delete().neq("id", "");
  await supabase.from("experiments").delete().neq("id", "");
  await supabase.from("hypotheses").delete().neq("id", "");
  await supabase.from("datasets").delete().neq("id", "");
  await supabase.from("models").delete().neq("id", "");
  await supabase.from("tags").delete().neq("id", "");
  await supabase.from("app_settings").delete().eq("key", "lastSynced");
}
