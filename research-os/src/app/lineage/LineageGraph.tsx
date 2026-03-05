'use client';

import { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const dagre = require('@dagrejs/dagre');
import { papers, edges, type PaperNode } from '@/lib/lineage-data';
import s from './lineage.module.css';

// ── Layout constants ────────────────────────────────────────────────────────
const NODE_W = 120;
const NODE_H = 40;
const MARGIN = 40;

// ── Color helpers ───────────────────────────────────────────────────────────

function nodeFill(p: PaperNode): string {
  if (!p.slug) return 'transparent';
  switch (p.category) {
    case 'rl':
      return 'rgba(74, 155, 127, 0.18)';
    case 'wm':
      return 'rgba(107, 115, 181, 0.18)';
    case 'robotics':
      return 'rgba(196, 136, 77, 0.18)';
    default:
      return 'rgba(99, 139, 212, 0.12)';
  }
}

function nodeStroke(p: PaperNode): string {
  if (!p.slug) return 'var(--text-tertiary)';
  switch (p.category) {
    case 'rl':
      return 'var(--cat-rl)';
    case 'wm':
      return 'var(--cat-wm)';
    case 'robotics':
      return 'var(--cat-robotics)';
    default:
      return 'var(--accent-primary)';
  }
}

function labelFill(p: PaperNode): string {
  if (!p.slug) return 'var(--text-tertiary)';
  return 'var(--text-primary)';
}

// ── Dagre layout ────────────────────────────────────────────────────────────

interface LayoutNode {
  paper: PaperNode;
  x: number;
  y: number;
}

interface LayoutEdge {
  source: string;
  target: string;
  points: { x: number; y: number }[];
}

function computeLayout() {
  const allPapers = Object.values(papers);

  // ── Phase 1: Run dagre to get a good initial X ordering ──
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: 'TB',
    nodesep: 30,
    ranksep: 60,
    marginx: MARGIN,
    marginy: MARGIN,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const p of allPapers) {
    g.setNode(p.id, { width: NODE_W, height: NODE_H });
  }
  for (const e of edges) {
    g.setEdge(e.target, e.source);
  }
  dagre.layout(g);

  const dagreX: Record<string, number> = {};
  for (const id of g.nodes()) {
    const n = g.node(id);
    if (n) dagreX[id] = n.x;
  }

  // ── Phase 2: Assign rows strictly by year ──
  const sortedYears = [
    ...new Set(
      allPapers.filter((p) => p.year != null).map((p) => p.year as number),
    ),
  ].sort((a, b) => a - b);

  const yearToRow = new Map<number, number>();
  sortedYears.forEach((y, i) => yearToRow.set(y, i));

  const rowOf: Record<string, number> = {};
  for (const p of allPapers) {
    if (p.year != null) {
      rowOf[p.id] = yearToRow.get(p.year)!;
    }
  }

  // Null-year nodes: place one row after their latest dependency
  for (const p of allPapers) {
    if (p.year == null) {
      let maxDepRow = -1;
      for (const e of edges) {
        if (e.source === p.id && rowOf[e.target] != null) {
          maxDepRow = Math.max(maxDepRow, rowOf[e.target]);
        }
      }
      rowOf[p.id] = maxDepRow >= 0 ? maxDepRow + 1 : 0;
    }
  }

  // Group by row
  const rowGroups = new Map<number, string[]>();
  for (const p of allPapers) {
    const r = rowOf[p.id];
    const arr = rowGroups.get(r) ?? [];
    arr.push(p.id);
    rowGroups.set(r, arr);
  }

  // Seed each row's order with dagre X
  for (const [, ids] of rowGroups) {
    ids.sort((a, b) => (dagreX[a] ?? 0) - (dagreX[b] ?? 0));
  }

  // ── Phase 3: Barycenter crossing reduction ──
  const upNbrs = new Map<string, string[]>();
  const downNbrs = new Map<string, string[]>();
  for (const p of allPapers) {
    upNbrs.set(p.id, []);
    downNbrs.set(p.id, []);
  }
  for (const e of edges) {
    const rS = rowOf[e.source];
    const rT = rowOf[e.target];
    if (rS > rT) {
      upNbrs.get(e.source)!.push(e.target);
      downNbrs.get(e.target)!.push(e.source);
    } else if (rS < rT) {
      upNbrs.get(e.target)!.push(e.source);
      downNbrs.get(e.source)!.push(e.target);
    }
  }

  const sortedRowKeys = [...rowGroups.keys()].sort((a, b) => a - b);

  for (let iter = 0; iter < 6; iter++) {
    // Top-down
    for (const r of sortedRowKeys) {
      const ids = rowGroups.get(r)!;
      const scored = ids.map((id, origIdx) => {
        const ups = upNbrs.get(id) ?? [];
        if (ups.length === 0) return { id, score: origIdx, origIdx };
        let sum = 0;
        for (const uid of ups) {
          const uRow = rowGroups.get(rowOf[uid]);
          if (uRow) sum += uRow.indexOf(uid);
        }
        return { id, score: sum / ups.length, origIdx };
      });
      scored.sort((a, b) => a.score - b.score || a.origIdx - b.origIdx);
      rowGroups.set(
        r,
        scored.map((x) => x.id),
      );
    }
    // Bottom-up
    for (let i = sortedRowKeys.length - 1; i >= 0; i--) {
      const r = sortedRowKeys[i];
      const ids = rowGroups.get(r)!;
      const scored = ids.map((id, origIdx) => {
        const downs = downNbrs.get(id) ?? [];
        if (downs.length === 0) return { id, score: origIdx, origIdx };
        let sum = 0;
        for (const did of downs) {
          const dRow = rowGroups.get(rowOf[did]);
          if (dRow) sum += dRow.indexOf(did);
        }
        return { id, score: sum / downs.length, origIdx };
      });
      scored.sort((a, b) => a.score - b.score || a.origIdx - b.origIdx);
      rowGroups.set(
        r,
        scored.map((x) => x.id),
      );
    }
  }

  // ── Phase 4: Assign final positions ──
  const RANK_SEP = 100;
  const NODE_GAP = 24;
  const CATEGORY_GAP = 48; // extra gap between different category groups

  // Category sort order for clustering within rows
  const catOrder: Record<string, number> = {
    rl: 0,
    wm: 1,
    robotics: 2,
    technique: 3,
  };

  // Sort each row by category (secondary to barycenter ordering within category)
  for (const [, ids] of rowGroups) {
    ids.sort((a, b) => {
      const catA = catOrder[papers[a]?.category ?? 'technique'] ?? 3;
      const catB = catOrder[papers[b]?.category ?? 'technique'] ?? 3;
      if (catA !== catB) return catA - catB;
      return 0; // preserve barycenter order within same category
    });
  }

  // Compute row widths accounting for category gaps
  function rowWidth(ids: string[]): number {
    if (ids.length === 0) return 0;
    let w = ids.length * NODE_W;
    for (let i = 1; i < ids.length; i++) {
      const prevCat = papers[ids[i - 1]]?.category;
      const currCat = papers[ids[i]]?.category;
      w += prevCat !== currCat ? CATEGORY_GAP : NODE_GAP;
    }
    return w;
  }

  let maxRowW = 0;
  for (const [, ids] of rowGroups) {
    maxRowW = Math.max(maxRowW, rowWidth(ids));
  }

  const centerX = MARGIN + maxRowW / 2;
  const totalRows = Math.max(...rowGroups.keys()) + 1;

  const posMap: Record<string, { x: number; y: number }> = {};
  for (const [rowIdx, ids] of rowGroups) {
    const rw = rowWidth(ids);
    let curX = centerX - rw / 2 + NODE_W / 2;
    const y = MARGIN + NODE_H / 2 + rowIdx * RANK_SEP;
    for (let i = 0; i < ids.length; i++) {
      posMap[ids[i]] = { x: curX, y };
      if (i < ids.length - 1) {
        const currCat = papers[ids[i]]?.category;
        const nextCat = papers[ids[i + 1]]?.category;
        curX += NODE_W + (currCat !== nextCat ? CATEGORY_GAP : NODE_GAP);
      }
    }
  }

  // Build node list
  const nodes: LayoutNode[] = [];
  for (const [id, pos] of Object.entries(posMap)) {
    if (papers[id]) nodes.push({ paper: papers[id], x: pos.x, y: pos.y });
  }

  // Build edges
  const layoutEdges: LayoutEdge[] = [];
  for (const e of edges) {
    const src = posMap[e.target]; // older (top)
    const tgt = posMap[e.source]; // newer (bottom)
    if (!src || !tgt) continue;

    if (Math.abs(src.y - tgt.y) < 1) {
      // Same-row edge: arc above the row
      const midX = (src.x + tgt.x) / 2;
      const arcY = src.y - NODE_H - 6;
      layoutEdges.push({
        source: e.target,
        target: e.source,
        points: [
          { x: src.x, y: src.y - NODE_H / 2 },
          { x: midX, y: arcY },
          { x: tgt.x, y: tgt.y - NODE_H / 2 },
        ],
      });
    } else {
      // Normal edge: straight line from bottom of src to top of tgt
      layoutEdges.push({
        source: e.target,
        target: e.source,
        points: [
          { x: src.x, y: src.y + NODE_H / 2 },
          { x: tgt.x, y: tgt.y - NODE_H / 2 },
        ],
      });
    }
  }

  const width = maxRowW + MARGIN * 2;
  const height = totalRows * RANK_SEP + MARGIN * 2;

  return { nodes, edges: layoutEdges, width, height };
}

// ── Path builder ────────────────────────────────────────────────────────────

function buildPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return '';
  const [start, ...rest] = points;
  let d = `M${start.x},${start.y}`;
  if (rest.length === 1) {
    d += `L${rest[0].x},${rest[0].y}`;
  } else {
    // Use quadratic curves through intermediate points
    for (let i = 0; i < rest.length - 1; i++) {
      const curr = rest[i];
      const next = rest[i + 1];
      const midX = (curr.x + next.x) / 2;
      const midY = (curr.y + next.y) / 2;
      d += `Q${curr.x},${curr.y} ${midX},${midY}`;
    }
    const last = rest[rest.length - 1];
    d += `L${last.x},${last.y}`;
  }
  return d;
}

// ── Adjacency helper ────────────────────────────────────────────────────────

function buildAdjacency() {
  const adj = new Map<string, Set<string>>();
  for (const id of Object.keys(papers)) {
    adj.set(id, new Set());
  }
  for (const e of edges) {
    adj.get(e.source)?.add(e.target);
    adj.get(e.target)?.add(e.source);
  }
  return adj;
}

// ── Component ───────────────────────────────────────────────────────────────

export function LineageGraph() {
  const router = useRouter();
  const layout = useMemo(computeLayout, []);
  const adjacency = useMemo(buildAdjacency, []);
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const [hovered, setHovered] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    paper: PaperNode;
    x: number;
    y: number;
  } | null>(null);

  // Zoom / pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, px: 0, py: 0 });

  // Fit graph to wrapper on mount
  useEffect(() => {
    if (wrapperRef.current) {
      const wrapperW = wrapperRef.current.clientWidth;
      const fitZoom = Math.min(1, wrapperW / layout.width);
      setZoom(fitZoom);
    }
  }, [layout.width]);

  const neighbors = useMemo(() => {
    if (!hovered) return null;
    const set = adjacency.get(hovered) ?? new Set<string>();
    return new Set([hovered, ...set]);
  }, [hovered, adjacency]);

  const highlightedEdges = useMemo(() => {
    if (!hovered) return null;
    const set = new Set<string>();
    for (const e of edges) {
      if (e.source === hovered || e.target === hovered) {
        set.add(`${e.target}->${e.source}`);
      }
    }
    return set;
  }, [hovered]);

  const handleNodeEnter = useCallback(
    (paper: PaperNode, e: React.MouseEvent) => {
      setHovered(paper.id);
      const rect = wrapperRef.current?.getBoundingClientRect();
      if (rect) {
        setTooltip({
          paper,
          x: e.clientX - rect.left + 12,
          y: e.clientY - rect.top - 8,
        });
      }
    },
    [],
  );

  const handleNodeMove = useCallback(
    (paper: PaperNode, e: React.MouseEvent) => {
      const rect = wrapperRef.current?.getBoundingClientRect();
      if (rect) {
        setTooltip({
          paper,
          x: e.clientX - rect.left + 12,
          y: e.clientY - rect.top - 8,
        });
      }
    },
    [],
  );

  const handleNodeLeave = useCallback(() => {
    setHovered(null);
    setTooltip(null);
  }, []);

  const handleNodeClick = useCallback(
    (paper: PaperNode) => {
      if (paper.slug) {
        router.push(`/methods/${paper.slug}`);
      }
    },
    [router],
  );

  // Zoom handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.min(3, Math.max(0.2, z - e.deltaY * 0.001)));
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      // Only start pan if clicking on the SVG background, not on a node
      const target = e.target as SVGElement;
      if (target.closest(`.${s.nodeGroup}`)) return;
      setIsPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
    },
    [pan],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning) return;
      setPan({
        x: panStart.current.px + (e.clientX - panStart.current.x),
        y: panStart.current.py + (e.clientY - panStart.current.y),
      });
    },
    [isPanning],
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const zoomIn = useCallback(() => setZoom((z) => Math.min(3, z + 0.2)), []);
  const zoomOut = useCallback(
    () => setZoom((z) => Math.max(0.2, z - 0.2)),
    [],
  );
  const resetView = useCallback(() => {
    if (wrapperRef.current) {
      const wrapperW = wrapperRef.current.clientWidth;
      const fitZoom = Math.min(1, wrapperW / layout.width);
      setZoom(fitZoom);
    }
    setPan({ x: 0, y: 0 });
  }, [layout.width]);

  return (
    <div className={s.graphWrapper} ref={wrapperRef}>
      <svg
        ref={svgRef}
        className={s.svgCanvas}
        width="100%"
        height="100%"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L10,5 L0,10 Z" className={s.arrowMarker} />
          </marker>
          <marker
            id="arrow-hl"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L10,5 L0,10 Z" fill="var(--accent-primary)" />
          </marker>
        </defs>

        <g
          transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}
        >
          {/* Edges */}
          {layout.edges.map((edge) => {
            const edgeKey = `${edge.source}->${edge.target}`;
            const isHl = highlightedEdges?.has(edgeKey);
            const isDimmed = neighbors && !isHl;
            return (
              <g
                key={edgeKey}
                className={[
                  isDimmed ? s.dimmed : '',
                  isHl ? s.highlighted : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                <path
                  d={buildPath(edge.points)}
                  className={s.edgePath}
                  markerEnd={isHl ? 'url(#arrow-hl)' : 'url(#arrow)'}
                />
              </g>
            );
          })}

          {/* Nodes */}
          {layout.nodes.map(({ paper: p, x, y }) => {
            const isOnSite = p.slug !== null;
            const isDimmed = neighbors && !neighbors.has(p.id);
            return (
              <g
                key={p.id}
                className={[
                  s.nodeGroup,
                  isOnSite ? s.onSite : s.offSite,
                  isDimmed ? s.dimmed : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                transform={`translate(${x - NODE_W / 2}, ${y - NODE_H / 2})`}
                onMouseEnter={(e) => handleNodeEnter(p, e)}
                onMouseMove={(e) => handleNodeMove(p, e)}
                onMouseLeave={handleNodeLeave}
                onClick={() => handleNodeClick(p)}
              >
                <rect
                  className={s.nodeRect}
                  width={NODE_W}
                  height={NODE_H}
                  fill={nodeFill(p)}
                  stroke={nodeStroke(p)}
                />
                <text
                  className={s.nodeLabel}
                  x={NODE_W / 2}
                  y={p.year ? NODE_H / 2 - 5 : NODE_H / 2}
                  fill={labelFill(p)}
                >
                  {p.title}
                </text>
                {p.year && (
                  <text
                    className={s.nodeYear}
                    x={NODE_W / 2}
                    y={NODE_H / 2 + 9}
                  >
                    {p.year}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className={s.tooltip}
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className={s.tooltipTitle}>{tooltip.paper.title}</div>
          <div className={s.tooltipMeta}>
            {tooltip.paper.authors}
            {tooltip.paper.authors && tooltip.paper.year ? ', ' : ''}
            {tooltip.paper.year ?? ''}
          </div>
          {tooltip.paper.oneLiner && (
            <div className={s.tooltipOneLiner}>{tooltip.paper.oneLiner}</div>
          )}
          {tooltip.paper.slug && (
            <div className={s.tooltipHint}>Click to view page →</div>
          )}
        </div>
      )}

      {/* Zoom controls */}
      <div className={s.controls}>
        <button className={s.controlBtn} onClick={zoomIn} aria-label="Zoom in">
          +
        </button>
        <button
          className={s.controlBtn}
          onClick={zoomOut}
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          className={s.controlBtn}
          onClick={resetView}
          aria-label="Reset view"
          style={{ fontSize: '12px' }}
        >
          ⟲
        </button>
      </div>

      {/* Legend */}
      <div className={s.legend}>
        <div className={s.legendItem}>
          <span
            className={s.legendSwatch}
            style={{
              background: 'rgba(74, 155, 127, 0.18)',
              borderColor: 'var(--cat-rl)',
            }}
          />
          Standard RL
        </div>
        <div className={s.legendItem}>
          <span
            className={s.legendSwatch}
            style={{
              background: 'rgba(107, 115, 181, 0.18)',
              borderColor: 'var(--cat-wm)',
            }}
          />
          World Models
        </div>
        <div className={s.legendItem}>
          <span
            className={s.legendSwatch}
            style={{
              background: 'rgba(99, 139, 212, 0.12)',
              borderColor: 'var(--accent-primary)',
            }}
          />
          Techniques
        </div>
        <div className={s.legendItem}>
          <span
            className={`${s.legendSwatch} ${s.legendSwatchDashed}`}
            style={{ borderColor: 'var(--text-tertiary)' }}
          />
          Off-site reference
        </div>
        <div className={s.legendItem}>
          <svg className={s.legendArrow} viewBox="0 0 24 12">
            <line
              x1="0"
              y1="6"
              x2="18"
              y2="6"
              stroke="var(--text-tertiary)"
              strokeWidth="1.5"
            />
            <polygon points="18,2 24,6 18,10" fill="var(--text-tertiary)" />
          </svg>
          Builds on
        </div>
      </div>
    </div>
  );
}
