import dynamic from "next/dynamic";
import type { ComponentType } from "react";

// Registry of existing interactive visualization components
// Each maps a paper slug to its method page component
const registry: Record<string, ComponentType> = {
  bcq: dynamic(() => import("@/app/methods/bcq/page")),
  bear: dynamic(() => import("@/app/methods/bear/page")),
  cql: dynamic(() => import("@/app/methods/cql/page")),
  ddpg: dynamic(() => import("@/app/methods/ddpg/page")),
  dqn: dynamic(() => import("@/app/methods/dqn/page")),
  dreamer: dynamic(() => import("@/app/methods/dreamer/page")),
  dreamerv2: dynamic(() => import("@/app/methods/dreamerv2/page")),
  dreamerv3: dynamic(() => import("@/app/methods/dreamerv3/page")),
  fast: dynamic(() => import("@/app/methods/fast/page")),
  iql: dynamic(() => import("@/app/methods/iql/page")),
  layernorm: dynamic(() => import("@/app/methods/layernorm/page")),
  planet: dynamic(() => import("@/app/methods/planet/page")),
  ppo: dynamic(() => import("@/app/methods/ppo/page")),
  rlhf: dynamic(() => import("@/app/methods/rlhf/page")),
  rlpd: dynamic(() => import("@/app/methods/rlpd/page")),
  "robot-controllers": dynamic(() => import("@/app/methods/robot-controllers/page")),
  sac: dynamic(() => import("@/app/methods/sac/page")),
  vjepa: dynamic(() => import("@/app/methods/vjepa/page")),
  vjepa2: dynamic(() => import("@/app/methods/vjepa2/page")),
  "world-models": dynamic(() => import("@/app/methods/world-models/page")),
};

export function getVizComponent(slug: string): ComponentType | null {
  return registry[slug] || null;
}

export function hasVizComponent(slug: string): boolean {
  return slug in registry;
}
