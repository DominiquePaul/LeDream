import { v4 as uuid } from "uuid";
import type { Dataset, Model, ModelIteration, ResearchData, Tag } from "./types";

const HF_API = "https://huggingface.co/api";
const HF_USER = "dopaul";

interface HFDataset {
  id: string;
  lastModified: string;
  downloads: number;
  tags?: string[];
  description?: string;
}

interface HFModel {
  id: string;
  lastModified?: string;
  createdAt?: string;
  downloads: number;
  tags?: string[];
  pipeline_tag?: string;
  library_name?: string;
  description?: string;
}

function inferDatasetMeta(hfId: string, tags: string[]): {
  estimatedHours: number;
  episodeCount: number;
  taskType: string;
  robotForm: string;
} {
  const name = hfId.split("/")[1] || "";
  let estimatedHours = 0;
  let episodeCount = 0;
  let taskType = "unknown";
  let robotForm = "unknown";

  if (name.includes("1500")) { episodeCount = 1500; estimatedHours = 12; }
  else if (name.includes("500")) { episodeCount = 500; estimatedHours = 4; }
  else if (name.includes("100_rook") || name.includes("100x")) { episodeCount = 100; estimatedHours = 1; }
  else if (name.startsWith("game_")) { episodeCount = 1; estimatedHours = 0.5; }
  else if (name.includes("eval_")) { episodeCount = 10; estimatedHours = 0.5; }
  else { episodeCount = 50; estimatedHours = 0.5; }

  if (name.includes("chess") || name.includes("pawn") || name.includes("rook") || name.includes("game_")) {
    taskType = "chess";
  } else if (name.includes("pcb")) {
    taskType = "pcb-placement";
  } else if (name.includes("grasp") || name.includes("grab") || name.includes("cola")) {
    taskType = "grasping";
  } else if (["house", "bike", "cat", "spider", "waterbottle", "microphone", "lamp", "boat", "chair"].some(o => name === o)) {
    taskType = "object-interaction";
  }

  if (tags.includes("so100") || name.includes("so-100")) {
    robotForm = "SO-100";
  } else if (tags.includes("phospho-dk") || tags.includes("phosphobot")) {
    robotForm = "Phosphobot";
  }

  return { estimatedHours, episodeCount, taskType, robotForm };
}

function inferPolicyType(hfId: string, tags: string[]): string {
  const name = hfId.split("/")[1] || "";
  if (name.includes("act") || name.includes("act_")) return "ACT";
  if (name.includes("smolvla")) return "SmolVLA";
  if (name.includes("diffusion")) return "Diffusion";
  if (tags.includes("ultralytics") || name.includes("detector") || name.includes("segmentation") || name.includes("detection")) return "YOLO";
  return "unknown";
}

function getModelDate(m: HFModel): string {
  return m.lastModified || m.createdAt || new Date().toISOString();
}

function groupModelIterations(models: HFModel[]): Map<string, HFModel[]> {
  const groups = new Map<string, HFModel[]>();
  for (const model of models) {
    const name = model.id.split("/")[1] || "";
    const baseName = name.replace(/_\d+k?$/, "").replace(/-\d+$/, "");
    if (!groups.has(baseName)) groups.set(baseName, []);
    groups.get(baseName)!.push(model);
  }
  return groups;
}

export async function syncFromHuggingFace(existing: ResearchData): Promise<ResearchData> {
  console.log("[HF Sync] Fetching from", `${HF_API}/datasets?author=${HF_USER}&limit=200`);

  const [datasetsRes, modelsRes] = await Promise.all([
    fetch(`${HF_API}/datasets?author=${HF_USER}&limit=200`),
    fetch(`${HF_API}/models?author=${HF_USER}&limit=200`),
  ]);

  if (!datasetsRes.ok || !modelsRes.ok) {
    const dsText = !datasetsRes.ok ? await datasetsRes.text().catch(() => "") : "";
    const mdText = !modelsRes.ok ? await modelsRes.text().catch(() => "") : "";
    throw new Error(
      `HF API error: datasets=${datasetsRes.status} ${dsText}, models=${modelsRes.status} ${mdText}`
    );
  }

  const hfDatasets: HFDataset[] = await datasetsRes.json();
  const hfModels: HFModel[] = await modelsRes.json();
  console.log("[HF Sync] Got", hfDatasets.length, "datasets,", hfModels.length, "models");

  const data: ResearchData = {
    ...existing,
    datasets: [...existing.datasets],
    models: [...existing.models],
    tags: [...existing.tags],
  };

  // Ensure default tags exist
  const defaultTags: Array<{ name: string; color: string; category: Tag["category"] }> = [
    { name: "chess", color: "#8B5CF6", category: "task" },
    { name: "pcb-placement", color: "#10B981", category: "task" },
    { name: "grasping", color: "#F59E0B", category: "task" },
    { name: "object-interaction", color: "#6366F1", category: "task" },
    { name: "SO-100", color: "#3B82F6", category: "robot" },
    { name: "Phosphobot", color: "#EC4899", category: "robot" },
    { name: "ACT", color: "#14B8A6", category: "custom" },
    { name: "SmolVLA", color: "#F97316", category: "custom" },
    { name: "Diffusion", color: "#A855F7", category: "custom" },
    { name: "YOLO", color: "#EF4444", category: "custom" },
    { name: "evaluation", color: "#64748B", category: "status" },
    { name: "training-data", color: "#22C55E", category: "status" },
  ];

  for (const dt of defaultTags) {
    if (!data.tags.find(t => t.name === dt.name)) {
      data.tags.push({ id: uuid(), name: dt.name, color: dt.color, category: dt.category });
    }
  }

  const findTagId = (name: string) => data.tags.find(t => t.name === name)?.id;

  // Sync datasets
  for (const hfDs of hfDatasets) {
    const existingDs = data.datasets.find(d => d.hfId === hfDs.id);
    const inferred = inferDatasetMeta(hfDs.id, hfDs.tags || []);
    const tagIds: string[] = [];
    if (inferred.taskType !== "unknown") {
      const tid = findTagId(inferred.taskType);
      if (tid) tagIds.push(tid);
    }
    if (inferred.robotForm !== "unknown") {
      const tid = findTagId(inferred.robotForm);
      if (tid) tagIds.push(tid);
    }
    const dsName = hfDs.id.split("/")[1] || hfDs.id;
    if (dsName.includes("eval_")) {
      const tid = findTagId("evaluation");
      if (tid) tagIds.push(tid);
    } else {
      const tid = findTagId("training-data");
      if (tid) tagIds.push(tid);
    }

    if (existingDs) {
      existingDs.downloads = hfDs.downloads || 0;
      existingDs.lastModified = hfDs.lastModified || existingDs.lastModified;
    } else {
      const modified = hfDs.lastModified || new Date().toISOString();
      data.datasets.push({
        id: uuid(),
        hfId: hfDs.id,
        name: dsName,
        description: hfDs.description || "",
        tags: tagIds,
        metadata: {
          collectionConditions: "",
          teleoperatorInstructions: "",
          knownIssues: [],
          notes: "",
          estimatedHours: inferred.estimatedHours,
          episodeCount: inferred.episodeCount,
        },
        hfUrl: `https://huggingface.co/datasets/${hfDs.id}`,
        downloads: hfDs.downloads || 0,
        lastModified: modified,
        createdAt: modified,
      });
    }
  }

  // Sync models
  const modelGroups = groupModelIterations(hfModels);

  for (const [baseName, groupModels] of modelGroups) {
    const existingModel = data.models.find(m => {
      const mBase = (m.hfId.split("/")[1] || "").replace(/_\d+k?$/, "").replace(/-\d+$/, "");
      return mBase === baseName;
    });

    const policyType = inferPolicyType(groupModels[0].id, groupModels[0].tags || []);
    const tagIds: string[] = [];
    if (policyType !== "unknown") {
      const tid = findTagId(policyType);
      if (tid) tagIds.push(tid);
    }
    if (baseName.includes("chess") || baseName.includes("rook") || baseName.includes("pawn")) {
      const tid = findTagId("chess");
      if (tid) tagIds.push(tid);
    } else if (baseName.includes("pcb")) {
      const tid = findTagId("pcb-placement");
      if (tid) tagIds.push(tid);
    } else if (baseName.includes("grasp") || baseName.includes("zrh")) {
      const tid = findTagId("grasping");
      if (tid) tagIds.push(tid);
    }

    const iterations: ModelIteration[] = groupModels.map(m => ({
      id: uuid(),
      version: m.id.split("/")[1] || "",
      config: {},
      notes: "",
      trainedOn: [],
      hfId: m.id,
      hfUrl: `https://huggingface.co/${m.id}`,
      createdAt: getModelDate(m),
    }));

    const totalDownloads = groupModels.reduce((sum, m) => sum + (m.downloads || 0), 0);

    if (existingModel) {
      existingModel.downloads = totalDownloads;
      for (const it of iterations) {
        if (!existingModel.iterations.find(ei => ei.hfId === it.hfId)) {
          existingModel.iterations.push(it);
        }
      }
    } else {
      const dates = groupModels.map(m => getModelDate(m)).sort();
      data.models.push({
        id: uuid(),
        hfId: groupModels[0].id,
        name: baseName,
        description: "",
        tags: tagIds,
        policyType,
        iterations: iterations.sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
        hfUrl: `https://huggingface.co/${groupModels[0].id}`,
        downloads: totalDownloads,
        lastModified: dates[dates.length - 1],
        createdAt: dates[0],
      });
    }
  }

  data.lastSynced = new Date().toISOString();
  return data;
}
