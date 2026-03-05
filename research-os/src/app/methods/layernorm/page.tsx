'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './layernorm.module.css';

const coreIdeas = [
  {
    title: 'Hypersphere Projection Bounds Activations',
    desc: 'LayerNorm projects activations onto a hypersphere of radius sqrt(d), discarding magnitude while preserving direction. This prevents out-of-distribution inputs from producing extreme outputs.',
    detail: 'By centering (subtract mean) and normalizing (divide by standard deviation) each activation vector, LayerNorm constrains all hidden representations to lie on a hypersphere. The learnable parameters gamma and beta allow the network to undo the normalization selectively, but the default bounded behavior prevents catastrophic extrapolation on unseen states.',
  },
  {
    title: 'Bounded and Unbiased Extrapolation for RL',
    desc: 'Unlike conservative penalties (CQL) that bound Q-values but introduce systematic bias, LayerNorm bounds representations without any bias \u2014 and at negligible computational cost.',
    detail: 'Without normalization, Q-values can explode exponentially on out-of-distribution states. Conservative methods like CQL add penalties that bound the Q-values but systematically underestimate them, biasing exploration. LayerNorm achieves the same bounding effect naturally through its geometric projection, with zero bias and no auxiliary losses \u2014 just a mean and variance computation per layer.',
  },
];

const keyFeatures = [
  { title: 'Bounded Representations', desc: 'Magnitude capped at every layer \u2014 no exponential blow-up on OOD inputs' },
  { title: 'Zero Bias', desc: 'No systematic over- or under-estimation, unlike conservative Q-penalties' },
  { title: 'Negligible Cost', desc: 'Just mean and variance per layer \u2014 no extra sampling, no auxiliary losses' },
];

/* ================================================================
   HELPERS
   ================================================================ */

function layerNorm(h: number[]): number[] {
  const d = h.length;
  const eps = 1e-5;
  const mu = h.reduce((a, b) => a + b, 0) / d;
  const variance = h.reduce((a, b) => a + (b - mu) ** 2, 0) / d;
  const sigma = Math.sqrt(variance + eps);
  return h.map((v) => (v - mu) / sigma);
}

function gaussPdf(x: number, mu: number, sigma: number): number {
  return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
}

const COLORS = {
  green: '#10b981',
  blue: '#3b82f6',
  purple: '#a855f7',
  red: '#ef4444',
  orange: '#f59e0b',
  cyan: '#00d9ff',
};

/* ================================================================
   COMPONENT
   ================================================================ */

export default function LayerNormPage() {
  const [oodScale, setOodScale] = useState(3.0);
  const [showLN, setShowLN] = useState(false);

  /* ---- canvas refs ---- */
  const sphereCanvasRef = useRef<HTMLCanvasElement>(null);
  const distCanvasRef = useRef<HTMLCanvasElement>(null);
  const extrapolCanvasRef = useRef<HTMLCanvasElement>(null);

  const { width: sw, height: sh } = useCanvasResize(sphereCanvasRef);
  const { width: dw, height: dh } = useCanvasResize(distCanvasRef);
  const { width: ew, height: eh } = useCanvasResize(extrapolCanvasRef);

  /* ================================================================
     DRAW: Hypersphere / Geometry Visualization (MAIN VISUAL)
     ================================================================ */
  const drawSphere = useCallback(() => {
    const canvas = sphereCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (sw === 0 || sh === 0) return;

    ctx.clearRect(0, 0, sw, sh);

    const cx = sw / 2;
    const cy = sh / 2 + 5;
    const radius = Math.min(sw, sh) * 0.34;
    const scale = radius / 1.5;

    // ---- Subtle radial gradient background ----
    const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.6);
    bgGrad.addColorStop(0, 'rgba(16,185,129,0.04)');
    bgGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, sw, sh);

    // ---- Draw the hypersphere circle (always behind points) ----
    // Subtle inner fill to hint at the surface
    if (showLN) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(16,185,129,0.04)';
      ctx.fill();
    }
    // Dashed circle outline — always subtle so it reads as background
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = showLN ? 'rgba(16,185,129,0.25)' : 'rgba(16,185,129,0.15)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = showLN ? 'rgba(16,185,129,0.5)' : 'rgba(16,185,129,0.3)';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('||h|| = √d', cx, cy - radius - 14);
    ctx.font = '10px sans-serif';
    ctx.fillStyle = 'rgba(16,185,129,0.3)';
    ctx.fillText('hypersphere', cx, cy - radius - 2);

    // ---- Axes ----
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - radius - 35, cy);
    ctx.lineTo(cx + radius + 35, cy);
    ctx.moveTo(cx, cy - radius - 35);
    ctx.lineTo(cx, cy + radius + 35);
    ctx.stroke();

    ctx.fillStyle = '#444';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('feature 1', cx + radius + 25, cy + 14);
    ctx.fillText('feature 2', cx + 10, cy - radius - 22);

    // ---- Generate points ----
    // We work in 3D (d=3) so LayerNorm has enough degrees of freedom,
    // then project down to 2D for display. In 2D, mean-centering kills
    // one DOF leaving only 2 possible outputs — a known artifact.
    const seed = 42;
    const rng = (i: number) => {
      const x = Math.sin(seed + i * 127.1 + 311.7) * 43758.5453;
      return x - Math.floor(x);
    };

    type Pt3 = [number, number, number];

    // In-distribution 3D points
    const inDistPoints3: Pt3[] = [];
    for (let i = 0; i < 15; i++) {
      const theta = rng(i) * 2 * Math.PI;
      const phi = rng(i + 30) * Math.PI;
      const mag = 0.5 + rng(i + 50) * 0.6;
      inDistPoints3.push([
        Math.sin(phi) * Math.cos(theta) * mag,
        Math.sin(phi) * Math.sin(theta) * mag,
        Math.cos(phi) * mag,
      ]);
    }

    // OOD 3D points
    const oodPoints3: Pt3[] = [];
    for (let i = 0; i < 8; i++) {
      const theta = rng(i + 100) * 2 * Math.PI;
      const phi = rng(i + 130) * Math.PI;
      const mag = oodScale * (0.7 + rng(i + 150) * 0.5);
      oodPoints3.push([
        Math.sin(phi) * Math.cos(theta) * mag,
        Math.sin(phi) * Math.sin(theta) * mag,
        Math.cos(phi) * mag,
      ]);
    }

    // Apply LayerNorm in 3D (d=3), then project first two dims for display
    function applyLN3(p: Pt3): [number, number] {
      const d = 3;
      const mu = (p[0] + p[1] + p[2]) / d;
      const c: Pt3 = [p[0] - mu, p[1] - mu, p[2] - mu];
      const sigma = Math.sqrt((c[0] ** 2 + c[1] ** 2 + c[2] ** 2) / d + 1e-5);
      // return first 2 components for 2D display
      return [c[0] / sigma, c[1] / sigma];
    }

    // Project 3D → 2D for display (take first 2 coords)
    function proj2(p: Pt3): [number, number] {
      return [p[0], p[1]];
    }

    const inDistPoints = inDistPoints3.map(proj2);
    const oodPoints = oodPoints3.map(proj2);

    // ---- Draw all points with proper layering ----
    // When LN is on, we draw in layers so trails are behind everything,
    // faded originals are in the middle, and green projected dots are on top.

    interface PointData {
      p2d: [number, number];
      p3d: Pt3;
      color: string;
      isOOD: boolean;
    }

    const allPoints: PointData[] = [
      ...inDistPoints.map((p, i) => ({ p2d: p, p3d: inDistPoints3[i], color: COLORS.blue, isOOD: false })),
      ...oodPoints.map((p, i) => ({ p2d: p, p3d: oodPoints3[i], color: COLORS.red, isOOD: true })),
    ];

    if (showLN) {
      // Layer 1: Dashed projection trails (behind everything)
      for (const pt of allPoints) {
        const px = cx + pt.p2d[0] * scale;
        const py = cy - pt.p2d[1] * scale;
        const ln = applyLN3(pt.p3d);
        const lx = cx + ln[0] * scale;
        const ly = cy - ln[1] * scale;

        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = pt.isOOD ? 'rgba(239,68,68,0.25)' : 'rgba(59,130,246,0.25)';
        ctx.lineWidth = 1;
        ctx.moveTo(px, py);
        ctx.lineTo(lx, ly);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Layer 2: Faded original points
      for (const pt of allPoints) {
        const px = cx + pt.p2d[0] * scale;
        const py = cy - pt.p2d[1] * scale;

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fillStyle = pt.isOOD ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)';
        ctx.fill();
      }

      // Layer 3: Green projected points on top
      for (const pt of allPoints) {
        const ln = applyLN3(pt.p3d);
        const lx = cx + ln[0] * scale;
        const ly = cy - ln[1] * scale;

        ctx.beginPath();
        ctx.arc(lx, ly, 6, 0, 2 * Math.PI);
        ctx.fillStyle = COLORS.green;
        ctx.fill();
        ctx.strokeStyle = 'rgba(16,185,129,0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    } else {
      // No LN — just draw original points
      for (const pt of allPoints) {
        const px = cx + pt.p2d[0] * scale;
        const py = cy - pt.p2d[1] * scale;

        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // ---- Legend ----
    const ly = sh - 22;
    ctx.font = '10px sans-serif';

    ctx.beginPath(); ctx.arc(20, ly, 4, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.blue; ctx.fill();
    ctx.fillStyle = '#999'; ctx.textAlign = 'left';
    ctx.fillText('In-distribution', 30, ly + 3);

    ctx.beginPath(); ctx.arc(140, ly, 4, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.red; ctx.fill();
    ctx.fillStyle = '#999';
    ctx.fillText('Out-of-distribution', 150, ly + 3);

    if (showLN) {
      ctx.beginPath(); ctx.arc(290, ly, 4, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS.green; ctx.fill();
      ctx.fillStyle = '#999';
      ctx.fillText('After LayerNorm', 300, ly + 3);
    }
  }, [sw, sh, oodScale, showLN]);

  useEffect(() => { drawSphere(); }, [drawSphere]);

  /* ================================================================
     DRAW: Activation Distribution Before/After
     ================================================================ */
  const drawDistribution = useCallback(() => {
    const canvas = distCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (dw === 0 || dh === 0) return;

    ctx.clearRect(0, 0, dw, dh);

    const pad = { left: 45, right: 15, top: 25, bottom: 35 };
    const pw = dw - pad.left - pad.right;
    const ph = dh - pad.top - pad.bottom;

    const xMin = -10, xMax = 10;
    const yMin = 0, yMax = 0.55;

    function toX(v: number) { return pad.left + ((v - xMin) / (xMax - xMin)) * pw; }
    function toY(v: number) { return dh - pad.bottom - ((v - yMin) / (yMax - yMin)) * ph; }

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, dh - pad.bottom);
    ctx.lineTo(dw - pad.right, dh - pad.bottom);
    ctx.stroke();

    // X labels
    ctx.fillStyle = '#555';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    for (let x = -8; x <= 8; x += 4) {
      ctx.fillText(x.toString(), toX(x), dh - pad.bottom + 14);
    }
    ctx.fillText('Activation value', dw / 2, dh - 5);

    const numPts = 300;

    // Before LN distribution
    function beforePdf(x: number): number {
      const inDist = gaussPdf(x, 1, 1.5);
      const ood = gaussPdf(x, oodScale * 1.5, oodScale * 0.8) * 0.4;
      return inDist * 0.7 + ood;
    }

    // After LN distribution
    function afterPdf(x: number): number {
      const inDist = gaussPdf(x, 0, 1.0);
      const ood = gaussPdf(x, 0.3, 1.1) * 0.4;
      return inDist * 0.7 + ood;
    }

    // "Before" fill + stroke
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(0));
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      ctx.lineTo(toX(x), toY(Math.min(beforePdf(x), yMax)));
    }
    ctx.lineTo(toX(xMax), toY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,68,68,0.1)';
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const cx2 = toX(x);
      const cy2 = toY(Math.min(beforePdf(x), yMax));
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    }
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 2;
    ctx.stroke();

    // "After" fill + stroke
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(0));
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      ctx.lineTo(toX(x), toY(Math.min(afterPdf(x), yMax)));
    }
    ctx.lineTo(toX(xMax), toY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(16,185,129,0.1)';
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const cx2 = toX(x);
      const cy2 = toY(Math.min(afterPdf(x), yMax));
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    }
    ctx.strokeStyle = COLORS.green;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Labels
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = COLORS.red;
    ctx.textAlign = 'left';
    ctx.fillText('Before LN', toX(4), toY(0.38));
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.fillText('OOD tail →', toX(4), toY(0.33));

    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = COLORS.green;
    ctx.textAlign = 'right';
    ctx.fillText('After LN', toX(-2), toY(0.48));
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(16,185,129,0.5)';
    ctx.fillText('bounded', toX(-2), toY(0.43));
  }, [dw, dh, oodScale]);

  useEffect(() => { drawDistribution(); }, [drawDistribution]);

  /* ================================================================
     DRAW: Extrapolation Error Comparison
     ================================================================ */
  const drawExtrapolation = useCallback(() => {
    const canvas = extrapolCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (ew === 0 || eh === 0) return;

    ctx.clearRect(0, 0, ew, eh);

    const pad = { left: 55, right: 15, top: 25, bottom: 40 };
    const pw = ew - pad.left - pad.right;
    const ph = eh - pad.top - pad.bottom;

    const xMin = 0, xMax = 5;
    const yMin = 0, yMax = 12;

    function toX(v: number) { return pad.left + ((v - xMin) / (xMax - xMin)) * pw; }
    function toY(v: number) { return eh - pad.bottom - ((v - yMin) / (yMax - yMin)) * ph; }

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, eh - pad.bottom);
    ctx.lineTo(ew - pad.right, eh - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = '#555';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Distance from training data', ew / 2, eh - 6);

    ctx.save();
    ctx.translate(12, eh / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('|Q-value error|', 0, 0);
    ctx.restore();

    // Training region highlight
    ctx.fillStyle = 'rgba(59,130,246,0.06)';
    ctx.fillRect(pad.left, pad.top, toX(1) - pad.left, ph);
    ctx.fillStyle = 'rgba(59,130,246,0.35)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training', (pad.left + toX(1)) / 2, pad.top + 14);

    const numPts = 200;

    // No normalization: exponential blow-up
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const y = 0.3 * Math.exp(0.7 * x);
      i === 0 ? ctx.moveTo(toX(x), toY(Math.min(y, yMax))) : ctx.lineTo(toX(x), toY(Math.min(y, yMax)));
    }
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // CQL-style conservative: bounded but biased
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const y = 0.5 + 0.8 * Math.log(1 + x) + 0.15 * Math.sin(x * 2.5);
      i === 0 ? ctx.moveTo(toX(x), toY(Math.min(Math.max(y, 0), yMax))) : ctx.lineTo(toX(x), toY(Math.min(Math.max(y, 0), yMax)));
    }
    ctx.strokeStyle = COLORS.orange;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // LayerNorm: bounded + unbiased
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const y = 0.2 + 0.5 * Math.sqrt(x);
      i === 0 ? ctx.moveTo(toX(x), toY(Math.min(y, yMax))) : ctx.lineTo(toX(x), toY(Math.min(y, yMax)));
    }
    ctx.strokeStyle = COLORS.green;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Labels
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';

    ctx.fillStyle = COLORS.red;
    ctx.fillText('No normalization', toX(3), toY(10.5));
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.fillText('exponential blow-up', toX(3), toY(9.5));

    ctx.font = 'bold 10px sans-serif';
    ctx.fillStyle = COLORS.orange;
    ctx.fillText('Conservative (CQL)', toX(3), toY(3));
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(245,158,11,0.5)';
    ctx.fillText('bounded but biased', toX(3), toY(2.2));

    ctx.font = 'bold 10px sans-serif';
    ctx.fillStyle = COLORS.green;
    ctx.fillText('LayerNorm', toX(3), toY(1.5));
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(16,185,129,0.5)';
    ctx.fillText('bounded + unbiased ✓', toX(3), toY(0.7));
  }, [ew, eh]);

  useEffect(() => { drawExtrapolation(); }, [drawExtrapolation]);

  /* ---- live numerical example ---- */
  const exampleH = [2.0, -1.0, 0.5, 3.5 * (1 + (oodScale - 1) * 0.5)];
  const exampleLN = layerNorm(exampleH);
  const magBefore = Math.sqrt(exampleH.reduce((a, b) => a + b * b, 0));
  const magAfter = Math.sqrt(exampleLN.reduce((a, b) => a + b * b, 0));

  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className="method-page">
      <h1 className={styles.title}>LayerNorm in RL</h1>
      <p className={styles.subtitle}>
        How Layer Normalization Tames Extrapolation &mdash; Ba, Kiros &amp; Hinton &middot; 2016
      </p>
      <div className={styles.linkRow}>
        <a href="https://arxiv.org/abs/1607.06450" target="_blank" rel="noopener noreferrer" className={styles.paperLink}>
          &rarr; arxiv.org/abs/1607.06450
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== The One Formula ==================== */}
      <div className={styles.cardMb}>
        <div className={styles.formula}>
          <MathJax>{'\\( \\text{LayerNorm}(\\mathbf{h}) = \\vec{\\gamma} \\odot \\frac{\\mathbf{h} - \\mu}{\\sigma} + \\vec{\\beta} \\qquad \\text{center} \\to \\text{normalize} \\to \\text{rescale} \\)'}</MathJax>
        </div>
        <div className={styles.annotationCallout}>
          <strong>The one-line intuition:</strong> LayerNorm projects activations onto a hypersphere — discarding <em>how big</em> a signal is while keeping <em>which direction</em> it points. This prevents out-of-distribution inputs from producing extreme outputs.
          The learnable parameters γ and β exist so the network can <em>undo</em> the normalization if it needs to — γ rescales each feature and β re-shifts it, letting the network learn which dimensions should be large or offset while still keeping the default bounded.
        </div>
      </div>

      {/* ==================== Main Visual: Hypersphere ==================== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          The Geometric View
        </div>
        <div className={styles.toggleBtns}>
          <button
            className={!showLN ? styles.toggleBtnActive : styles.toggleBtn}
            onClick={() => setShowLN(false)}
          >
            Raw Activations
          </button>
          <button
            className={showLN ? styles.toggleBtnActive : styles.toggleBtn}
            onClick={() => setShowLN(true)}
          >
            With LayerNorm
          </button>
        </div>
        <div className={styles.canvasContainer}>
          <div className={styles.canvasInner}>
            <canvas ref={sphereCanvasRef} className={styles.vizCanvasTall} />
          </div>
        </div>
        <div className={styles.sliderContainer}>
          <label>OOD magnitude:</label>
          <input
            type="range"
            className={styles.slider}
            min={1}
            max={6}
            step={0.1}
            value={oodScale}
            onChange={(e) => setOodScale(parseFloat(e.target.value))}
          />
          <span className={styles.sliderValue}>{oodScale.toFixed(1)}×</span>
        </div>
        <p className={styles.vizNote}>
          {showLN
            ? 'Toggle LayerNorm: all points snap to the green circle. Direction preserved, magnitude gone.'
            : 'Red OOD points can be arbitrarily far away. Drag the slider to increase their distance.'}
        </p>
      </div>

      {/* ==================== Live Numerical Example + Distribution ==================== */}
      <div className={styles.mainGrid}>
        {/* Live example */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Live Example
          </div>
          <div className={styles.metricsRow}>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: COLORS.red }}>{magBefore.toFixed(1)}</div>
              <div className={styles.metricLabel}>||h|| before</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#888' }}>&rarr;</div>
              <div className={styles.metricLabel}>LN</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: COLORS.green }}>{magAfter.toFixed(1)}</div>
              <div className={styles.metricLabel}>||h|| after</div>
            </div>
          </div>
          <div style={{ marginTop: 12, fontFamily: 'monospace', fontSize: '0.82rem', color: '#aaa', lineHeight: 1.8 }}>
            <div style={{ color: COLORS.red }}>
              h = [{exampleH.map(v => v.toFixed(1)).join(', ')}]
            </div>
            <div style={{ color: '#666' }}>
              μ = {(exampleH.reduce((a, b) => a + b, 0) / exampleH.length).toFixed(2)},&nbsp;
              σ = {Math.sqrt(exampleH.reduce((a, b) => a + (b - exampleH.reduce((c, d) => c + d, 0) / exampleH.length) ** 2, 0) / exampleH.length + 1e-5).toFixed(2)}
            </div>
            <div style={{ color: COLORS.green }}>
              LN(h) = [{exampleLN.map(v => v.toFixed(2)).join(', ')}]
            </div>
          </div>
          <div className={styles.sliderContainer}>
            <label>OOD severity:</label>
            <input
              type="range"
              className={styles.slider}
              min={1}
              max={6}
              step={0.1}
              value={oodScale}
              onChange={(e) => setOodScale(parseFloat(e.target.value))}
            />
            <span className={styles.sliderValue}>{oodScale.toFixed(1)}×</span>
          </div>
          <p className={styles.vizNote}>
            Drag the slider: ||h|| changes wildly, but ||LN(h)|| stays ≈ {Math.sqrt(exampleH.length).toFixed(1)}
          </p>
        </div>

        {/* Distribution Before/After */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Activation Distribution
          </div>
          <div className={styles.canvasContainer}>
            <div className={styles.canvasInner}>
              <canvas ref={distCanvasRef} className={styles.vizCanvas} />
            </div>
          </div>
          <p className={styles.vizNote}>
            <span style={{ color: COLORS.red }}>Before LN:</span> OOD inputs create a long tail.&nbsp;
            <span style={{ color: COLORS.green }}>After LN:</span> everything collapses to μ=0, σ=1.
          </p>
        </div>
      </div>

      {/* ==================== Extrapolation Error + Network Diagram ==================== */}
      <div className={styles.mainGrid}>
        {/* Extrapolation Error */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Extrapolation Error
          </div>
          <div className={styles.canvasContainer}>
            <div className={styles.canvasInner}>
              <canvas ref={extrapolCanvasRef} className={styles.vizCanvas} />
            </div>
          </div>
          <div className={styles.annotationCallout}>
            <strong>Why this matters for RL:</strong> Without normalization, Q-values explode exponentially on unseen states.
            Conservative penalties (CQL) bound it but bias exploration.
            LayerNorm bounds it <em>and</em> stays unbiased.
          </div>
        </div>

        {/* Network Architecture */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Where It Goes in the Network
          </div>
          <svg className={styles.pipelineSvg} viewBox="0 0 440 200">
            <defs>
              <marker id="lnArr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#888" />
              </marker>
              <marker id="lnArrG" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#10b981" />
              </marker>
            </defs>

            {/* Input */}
            <rect x="5" y="70" width="60" height="40" rx="8" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5" />
            <text x="35" y="93" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">State</text>

            <path d="M 65 90 L 85 90" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#lnArr)" />

            {/* Linear */}
            <rect x="88" y="70" width="60" height="40" rx="8" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5" />
            <text x="118" y="93" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold">Linear</text>

            <path d="M 148 90 L 165 90" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#lnArr)" />

            {/* LayerNorm (highlighted) */}
            <rect x="168" y="60" width="70" height="60" rx="10" fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="2.5" />
            <text x="203" y="85" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">LN</text>
            <text x="203" y="100" textAnchor="middle" fill="rgba(16,185,129,0.5)" fontSize="7">γ⊙(h−μ)/σ+β</text>

            <path d="M 238 90 L 255 90" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#lnArrG)" />

            {/* ReLU */}
            <rect x="258" y="75" width="45" height="30" rx="6" fill="rgba(245,158,11,0.12)" stroke="#f59e0b" strokeWidth="1" />
            <text x="280" y="93" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">ReLU</text>

            <path d="M 303 90 L 320 90" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#lnArr)" />

            {/* Output */}
            <rect x="323" y="70" width="80" height="40" rx="8" fill="rgba(0,217,255,0.15)" stroke="#00d9ff" strokeWidth="1.5" />
            <text x="363" y="87" textAnchor="middle" fill="#00d9ff" fontSize="10" fontWeight="bold">Q(s,a)</text>
            <text x="363" y="100" textAnchor="middle" fill="#888" fontSize="8">or π(a|s)</text>

            {/* Annotation */}
            <path d="M 203 58 L 203 35" stroke="rgba(16,185,129,0.4)" strokeWidth="1" strokeDasharray="3" fill="none" />
            <text x="203" y="28" textAnchor="middle" fill="rgba(16,185,129,0.7)" fontSize="9" fontWeight="bold">||h|| capped</text>

            {/* Repeated pattern */}
            <rect x="95" y="140" width="250" height="24" rx="6" fill="rgba(16,185,129,0.04)" stroke="rgba(16,185,129,0.15)" strokeWidth="1" strokeDasharray="4" />
            <text x="220" y="155" textAnchor="middle" fill="rgba(16,185,129,0.5)" fontSize="8">Linear → LN → ReLU (repeated per layer)</text>
          </svg>
        </div>
      </div>

      {/* ==================== Paper Lineage ==================== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Paper Lineage
        </div>
        <div className={styles.lineageGrid}>
          <div>
            <h4 className={styles.lineageSectionTitle} style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className={styles.lineageList}>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>Batch Normalization</div>
                <div className={styles.lineageItemDesc}>Ioffe &amp; Szegedy, 2015 &mdash; Normalizes across batch. Breaks in RL (non-stationary data).</div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>Weight Normalization</div>
                <div className={styles.lineageItemDesc}>Salimans &amp; Kingma, 2016 &mdash; Decouples weight magnitude from direction.</div>
              </div>
            </div>
          </div>
          <div>
            <h4 className={styles.lineageSectionTitle} style={{ color: '#10b981' }}>Applied In (RL)</h4>
            <div className={styles.lineageList}>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>RLPD / Cal-QL / SERL</div>
                <div className={styles.lineageItemDesc}>LayerNorm in critics enables stable offline-to-online fine-tuning and real-world robot learning.</div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>TD7 / CrossQ / BRO</div>
                <div className={styles.lineageItemDesc}>LayerNorm as a core ingredient for state-of-the-art continuous control.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
