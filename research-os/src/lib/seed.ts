import { v4 as uuid } from "uuid";
import type { ResearchData, Hypothesis, Experiment } from "./types";

/**
 * Pre-populate hypotheses and experiments based on known HF data.
 * Only seeds if no hypotheses exist yet (won't overwrite user changes).
 */
export function seedData(data: ResearchData): ResearchData {
  if (data.hypotheses.length > 0) return data;

  const now = new Date().toISOString();

  // Helper to find dataset IDs by name pattern
  const findDatasets = (pattern: RegExp) =>
    data.datasets.filter(d => pattern.test(d.name)).map(d => d.id);
  const findModels = (pattern: RegExp) =>
    data.models.filter(m => pattern.test(m.name)).map(m => m.id);

  const hypotheses: Hypothesis[] = [];
  const experiments: Experiment[] = [];

  // Hypothesis 1: Chess robot can learn piece movement from demonstrations
  const h1Id = uuid();
  const e1Id = uuid();
  const e2Id = uuid();
  const e3Id = uuid();
  hypotheses.push({
    id: h1Id,
    title: "A robot arm can learn chess piece movement from teleoperated demonstrations",
    description: "Testing whether imitation learning (ACT, SmolVLA, Diffusion) can teach a SO-100 arm to reliably move chess pieces. Progressive scaling from 100 to 1500 demonstrations.",
    status: "confirmed",
    experimentIds: [e1Id, e2Id, e3Id],
    tags: [],
    createdAt: "2025-05-01T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e1Id,
    name: "100 rook movements - ACT baseline",
    hypothesisId: h1Id,
    datasetIds: findDatasets(/100_rook/),
    modelIds: findModels(/100_rooks/),
    status: "completed",
    notes: "Initial test with 100 rook movement episodes. ACT policy at various checkpoints (20k-100k steps).",
    results: "Model learned basic movement but lacked precision. Needed more data.",
    createdAt: "2025-07-01T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e2Id,
    name: "500 chess moves - ACT vs SmolVLA",
    hypothesisId: h1Id,
    datasetIds: findDatasets(/500_chess/),
    modelIds: findModels(/500_chess/),
    status: "completed",
    notes: "Scaled to 500 episodes. Compared ACT and SmolVLA architectures at 20k-100k steps.",
    results: "ACT outperformed SmolVLA at this scale. Both showed improvement over 100-episode baseline.",
    createdAt: "2025-07-05T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e3Id,
    name: "1500 chess moves - ACT vs SmolVLA vs Diffusion",
    hypothesisId: h1Id,
    datasetIds: findDatasets(/1500_chess/),
    modelIds: findModels(/1500_chess/),
    status: "completed",
    notes: "Largest chess dataset. Three architectures compared across many checkpoints up to 1M steps.",
    results: "ACT at 240k-300k steps showed best performance. Diffusion competitive but slower inference.",
    createdAt: "2025-07-10T00:00:00Z",
    updatedAt: now,
  });

  // Hypothesis 2: ACT can learn PCB placement
  const h2Id = uuid();
  const e4Id = uuid();
  const e5Id = uuid();
  hypotheses.push({
    id: h2Id,
    title: "ACT policy can learn precise PCB component placement",
    description: "Testing whether the ACT architecture that worked for chess can transfer to PCB placement tasks requiring higher precision.",
    status: "active",
    experimentIds: [e4Id, e5Id],
    tags: [],
    createdAt: "2026-01-15T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e4Id,
    name: "PCB placement v1 - ACT baseline",
    hypothesisId: h2Id,
    datasetIds: findDatasets(/pcb_placement_v1/),
    modelIds: findModels(/pcb_placement_v1/),
    status: "completed",
    notes: "Initial PCB placement experiment with ACT policy. Checkpoints at 1k-3k steps.",
    results: "Promising initial results but needs more data and longer training.",
    createdAt: "2026-02-01T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e5Id,
    name: "PCB 1st item - ACT 48 action chunks, long training",
    hypothesisId: h2Id,
    datasetIds: findDatasets(/pcb_placement_1/),
    modelIds: findModels(/pcb_placement_1st_item/),
    status: "in-progress",
    notes: "Extended training with 48 action chunks. Comparing with and without VAE. Training up to 50k steps.",
    results: "",
    createdAt: "2026-02-15T00:00:00Z",
    updatedAt: now,
  });

  // Hypothesis 3: YOLO for chess piece detection
  const h3Id = uuid();
  const e6Id = uuid();
  hypotheses.push({
    id: h3Id,
    title: "YOLO can reliably detect chess pieces and board corners for state estimation",
    description: "Computer vision pipeline for chess: detect pieces, segment board, find corners for coordinate mapping.",
    status: "confirmed",
    experimentIds: [e6Id],
    tags: [],
    createdAt: "2025-06-01T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e6Id,
    name: "Chess piece detection - merged dataset YOLO training",
    hypothesisId: h3Id,
    datasetIds: findDatasets(/chess.*(pieces|board|segmentation)/),
    modelIds: findModels(/chess.*(piece|board|corner)/),
    status: "completed",
    notes: "Trained YOLO models on merged Roboflow + custom datasets for piece detection and board segmentation.",
    results: "chess-piece-detector-merged-v2 achieved good accuracy. Board segmentation and corner detection working.",
    createdAt: "2025-06-15T00:00:00Z",
    updatedAt: now,
  });

  // Hypothesis 4: Grasping from hackathon
  const h4Id = uuid();
  const e7Id = uuid();
  hypotheses.push({
    id: h4Id,
    title: "Merged multi-object grasping datasets improve generalization",
    description: "Zurich hackathon: testing whether merging datasets of different objects (cola, tape, mate) improves grasping policy.",
    status: "exploring",
    experimentIds: [e7Id],
    tags: [],
    createdAt: "2025-05-15T00:00:00Z",
    updatedAt: now,
  });

  experiments.push({
    id: e7Id,
    name: "Zurich grasping - merged dataset progression",
    hypothesisId: h4Id,
    datasetIds: findDatasets(/merged.*(cola|grab)/),
    modelIds: findModels(/zrh_grasping/),
    status: "completed",
    notes: "Iterative merging of grasping datasets. Multiple versions of merged data and policy checkpoints.",
    results: "Progressive improvement with more diverse data. v7 merged dataset with 80k checkpoint showed best results.",
    createdAt: "2025-05-20T00:00:00Z",
    updatedAt: now,
  });

  data.hypotheses = hypotheses;
  data.experiments = experiments;
  return data;
}
