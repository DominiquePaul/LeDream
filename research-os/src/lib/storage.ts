import type { ResearchData } from "./types";

const STORAGE_KEY = "research-os-data";

const EMPTY_DATA: ResearchData = {
  hypotheses: [],
  experiments: [],
  datasets: [],
  models: [],
  tags: [],
  lastSynced: "",
};

export function readData(): ResearchData {
  if (typeof window === "undefined") return { ...EMPTY_DATA };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...EMPTY_DATA };
    return JSON.parse(raw) as ResearchData;
  } catch {
    return { ...EMPTY_DATA };
  }
}

export function writeData(data: ResearchData): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function clearData(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(STORAGE_KEY);
}
