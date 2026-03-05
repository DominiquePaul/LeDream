export interface Tag {
  id: string;
  name: string;
  color: string; // hex color
  category: "task" | "robot" | "status" | "custom";
}

export interface DatasetMetadata {
  collectionConditions: string; // how variation was handled
  teleoperatorInstructions: string; // instructions given to teleoperator
  knownIssues: string[]; // e.g., "missing IR filter"
  notes: string;
  estimatedHours: number;
  episodeCount: number;
}

export interface Dataset {
  id: string;
  hfId: string; // e.g., "dopaul/1500_chess_moves" — empty for planned
  name: string;
  description: string;
  tags: string[]; // tag IDs
  metadata: DatasetMetadata;
  status: "planned" | "synced"; // planned = not yet on HF
  hfUrl: string;
  downloads: number;
  lastModified: string;
  createdAt: string;
}

export interface ModelIteration {
  id: string;
  version: string; // e.g., "act_100k"
  config: Record<string, unknown>; // hyperparameters
  notes: string; // why this config was chosen/updated
  trainedOn: string[]; // dataset IDs
  hfId: string; // e.g., "dopaul/1500_chess_moves_act_100k"
  hfUrl: string;
  createdAt: string;
}

export interface Model {
  id: string;
  hfId: string;
  name: string;
  description: string;
  tags: string[]; // tag IDs
  policyType: string; // ACT, SmolVLA, Diffusion, YOLO
  iterations: ModelIteration[];
  hfUrl: string;
  downloads: number;
  lastModified: string;
  createdAt: string;
}

export interface Experiment {
  id: string;
  name: string;
  hypothesisId: string;
  datasetIds: string[];
  modelIds: string[];
  status: "planned" | "in-progress" | "completed" | "failed";
  notes: string;
  results: string;
  createdAt: string;
  updatedAt: string;
}

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  status: "active" | "confirmed" | "rejected" | "exploring";
  experimentIds: string[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

export interface Evaluation {
  id: string;
  experimentId: string | null;
  modelId: string;
  iterationId: string | null;
  evalDatasetId: string | null; // eval_ dataset with video
  outcome: "success" | "partial" | "failure";
  notes: string;
  createdAt: string;
}

export interface ResearchData {
  hypotheses: Hypothesis[];
  experiments: Experiment[];
  datasets: Dataset[];
  models: Model[];
  evaluations: Evaluation[];
  tags: Tag[];
  lastSynced: string;
}
