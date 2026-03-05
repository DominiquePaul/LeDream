"use client";

import { useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  MarkerType,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useData } from "@/components/Shell";

const NODE_COLORS: Record<string, string> = {
  hypothesis: "#8B5CF6",
  experiment: "#F59E0B",
  dataset: "#3B82F6",
  model: "#10B981",
};

export default function GraphPage() {
  const { data } = useData();
  const router = useRouter();

  const { initialNodes, initialEdges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    let hypY = 0;

    for (const h of data.hypotheses) {
      nodes.push({
        id: `h-${h.id}`,
        position: { x: 0, y: hypY },
        data: { label: h.title.slice(0, 60) + (h.title.length > 60 ? "..." : "") },
        style: {
          background: "#1E1B4B",
          color: "#C4B5FD",
          border: "1px solid #8B5CF6",
          borderRadius: 8,
          padding: "8px 12px",
          fontSize: 11,
          maxWidth: 250,
          cursor: "pointer",
        },
      });

      const experiments = data.experiments.filter(
        (e) => e.hypothesisId === h.id
      );
      let expY = hypY;

      for (const exp of experiments) {
        nodes.push({
          id: `e-${exp.id}`,
          position: { x: 350, y: expY },
          data: { label: exp.name },
          style: {
            background: "#1C1917",
            color: "#FCD34D",
            border: "1px solid #F59E0B",
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 11,
            maxWidth: 200,
            cursor: "pointer",
          },
        });

        edges.push({
          id: `h-${h.id}-e-${exp.id}`,
          source: `h-${h.id}`,
          target: `e-${exp.id}`,
          markerEnd: { type: MarkerType.ArrowClosed, color: "#F59E0B" },
          style: { stroke: "#F59E0B44" },
        });

        let dsY = expY;
        for (const did of exp.datasetIds) {
          const nodeId = `d-${did}`;
          if (!nodes.find((n) => n.id === nodeId)) {
            const ds = data.datasets.find((d) => d.id === did);
            if (ds) {
              nodes.push({
                id: nodeId,
                position: { x: 700, y: dsY },
                data: { label: ds.name },
                style: {
                  background: "#172554",
                  color: "#93C5FD",
                  border: "1px solid #3B82F6",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 11,
                  maxWidth: 180,
                  cursor: "pointer",
                },
              });
              dsY += 70;
            }
          }
          edges.push({
            id: `e-${exp.id}-d-${did}`,
            source: `e-${exp.id}`,
            target: nodeId,
            markerEnd: { type: MarkerType.ArrowClosed, color: "#3B82F6" },
            style: { stroke: "#3B82F644" },
          });
        }

        let mdY = dsY;
        for (const mid of exp.modelIds) {
          const nodeId = `m-${mid}`;
          if (!nodes.find((n) => n.id === nodeId)) {
            const model = data.models.find((m) => m.id === mid);
            if (model) {
              nodes.push({
                id: nodeId,
                position: { x: 1000, y: mdY },
                data: { label: model.name },
                style: {
                  background: "#052E16",
                  color: "#6EE7B7",
                  border: "1px solid #10B981",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 11,
                  maxWidth: 180,
                  cursor: "pointer",
                },
              });
              mdY += 70;
            }
          }
          edges.push({
            id: `e-${exp.id}-m-${mid}`,
            source: `e-${exp.id}`,
            target: nodeId,
            markerEnd: { type: MarkerType.ArrowClosed, color: "#10B981" },
            style: { stroke: "#10B98144" },
          });
        }

        expY = Math.max(expY + 100, dsY, mdY);
      }

      hypY = Math.max(hypY + 150, expY + 50);
    }

    return { initialNodes: nodes, initialEdges: edges };
  }, [data]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const id = node.id;
    if (id.startsWith("h-") || id.startsWith("e-")) {
      router.push("/experiments");
    } else if (id.startsWith("d-")) {
      router.push("/datasets");
    } else if (id.startsWith("m-")) {
      router.push("/models");
    }
  }, [router]);

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 md:p-6 pb-0">
        <h1 className="text-xl md:text-2xl font-bold text-white">Relational Graph</h1>
        <p className="text-gray-400 text-sm mt-1">
          Visual map of hypotheses, experiments, datasets, and models. Click a node to navigate.
        </p>
        <div className="flex flex-wrap gap-3 md:gap-4 mt-3">
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-gray-400 capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 m-3 md:m-6 rounded-lg border border-gray-800 overflow-hidden">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          className="bg-gray-950"
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1F2937" gap={20} />
          <Controls className="!bg-gray-900 !border-gray-800 !rounded-lg [&>button]:!bg-gray-900 [&>button]:!border-gray-700 [&>button]:!text-gray-400 [&>button:hover]:!bg-gray-800" />
          <MiniMap
            nodeColor={(node) => {
              const id = node.id;
              if (id.startsWith("h-")) return NODE_COLORS.hypothesis;
              if (id.startsWith("e-")) return NODE_COLORS.experiment;
              if (id.startsWith("d-")) return NODE_COLORS.dataset;
              if (id.startsWith("m-")) return NODE_COLORS.model;
              return "#6B7280";
            }}
            className="!bg-gray-900 !border-gray-800"
          />
        </ReactFlow>
      </div>
    </div>
  );
}
