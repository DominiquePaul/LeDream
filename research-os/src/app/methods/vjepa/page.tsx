'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './vjepa.module.css';

/* ================================================================
   Data
   ================================================================ */

const coreIdeas = [
  {
    title: 'Feature Prediction, Not Pixels',
    desc: 'Instead of reconstructing raw pixels like MAE/VideoMAE, predict abstract feature representations. Eliminates irrelevant low-level details and focuses on semantic content.',
    detail: 'The key insight is that predicting in representation space rather than pixel space eliminates the need to model irrelevant details. A pixel-level prediction model must reconstruct exact textures, lighting, and backgrounds \u2014 wasting capacity on unpredictable low-level details. V-JEPA instead predicts abstract features, allowing it to focus on semantic concepts like objects, motion, and spatial relationships. This leads to more versatile representations that transfer well to downstream tasks, while training significantly faster than pixel-prediction methods like VideoMAE.',
  },
  {
    title: 'Joint-Embedding Predictive Architecture',
    desc: 'Two encoders: x-encoder (context, ViT) processes visible tokens, y-encoder (EMA copy, stop-gradient) produces targets from masked regions. Predictor maps context features + mask tokens to predicted targets. L1 loss.',
    detail: 'V-JEPA uses a Joint-Embedding Predictive Architecture. The x-encoder processes only the visible (unmasked) tokens \u2014 about 10% of the video. The y-encoder is an exponential moving average (EMA) copy of the x-encoder that processes the full video to produce target features. A stop-gradient on the y-encoder prevents representation collapse (where everything maps to the same point). The predictor is a narrow transformer (12 layers, dim 384) that takes x-encoder outputs and learnable mask tokens with positional embeddings, and outputs predicted features at masked positions. Training uses L1 loss between predictions and (stopped gradient) targets.',
  },
  {
    title: 'Multi-Block Masking for Video',
    desc: 'Mask ~90% of video tokens using short-range (8 blocks at 15% each) + long-range (2 blocks at 70% each) spatial-temporal masks. Forces model to predict across space and time.',
    detail: 'V-JEPA uses a specific masking strategy optimized for video. Short-range masks: union of 8 randomly sampled blocks covering 15% of each frame, with aspect ratios in [0.75, 1.5]. Long-range masks: union of 2 blocks covering 70% of each frame. Both types span the full temporal dimension. The result is about 90% of tokens masked. The multi-block strategy forces the model to make predictions after removing large contiguous spatial-temporal regions, which encourages learning of high-level scene understanding rather than local pixel interpolation.',
  },
];

const keyFeatures = [
  { title: 'No Pixel Reconstruction', desc: 'Predicts features not pixels \u2014 more semantic, more efficient' },
  { title: '90% Masking Efficiency', desc: 'Processes only 10% of tokens, massive computational savings' },
  { title: 'Frozen Backbone', desc: 'Representations so good they work with frozen backbone + attentive probe' },
];

const archComponentInfo: Record<string, { title: string; desc: string }> = {
  video: {
    title: 'Video Clip Input',
    desc: 'Input is a video clip of T frames at resolution H\u00d7W. The video is patchified into non-overlapping 16\u00d716 pixel patches spanning 2 consecutive frames (tubelet embedding). This creates a grid of T/2 \u00d7 H/16 \u00d7 W/16 tokens, each representing a small spatial-temporal region of the video.',
  },
  mask: {
    title: 'Binary Mask (~90%)',
    desc: 'A binary mask determines which tokens are visible (context for x-encoder) and which are masked (targets from y-encoder). The mask is generated using the multi-block strategy: union of 8 short-range blocks (15% each) and 2 long-range blocks (70% each), all spanning the full temporal dimension. About 90% of all tokens are masked.',
  },
  x_encoder: {
    title: 'x-encoder (Context Encoder)',
    desc: 'Standard Vision Transformer (ViT-L/16 or ViT-H/16) that processes only the visible (unmasked) tokens \u2014 approximately 10% of the video. This is the main encoder whose weights are learned through gradient descent. Processing only 10% of tokens makes training very efficient compared to methods that process all tokens.',
  },
  y_encoder: {
    title: 'y-encoder (EMA Target Encoder)',
    desc: 'An exponential moving average (EMA) copy of the x-encoder that processes the full video including masked regions to produce target features. The stop-gradient prevents gradients from flowing through the y-encoder, which is critical for avoiding representation collapse. The EMA momentum \u03c4 increases from 0.996 to 1.0 over training.',
  },
  remove_unmasked: {
    title: 'Remove Unmasked Tokens',
    desc: 'After the y-encoder processes all tokens, the features at unmasked (visible) positions are discarded, keeping only the features at masked positions [M\u00d7d]. These serve as the prediction targets \u2014 the predictor must output features matching these target representations.',
  },
  mask_tokens: {
    title: 'Learnable Mask Tokens',
    desc: 'Learnable embeddings that serve as placeholders for masked positions. Combined with positional embeddings encoding the spatial-temporal location of each masked token. These are concatenated with the x-encoder outputs and fed to the predictor, allowing it to know where predictions should be made.',
  },
  predictor: {
    title: 'Predictor Network',
    desc: 'A narrow Vision Transformer (12 layers, dimension 384) that takes x-encoder outputs concatenated with learnable mask tokens (with positional embeddings) and outputs predicted features at masked positions. The predictor is intentionally narrow to prevent it from simply copying input features, forcing the x-encoder to learn rich representations.',
  },
  loss: {
    title: 'L1 Loss in Feature Space',
    desc: 'The training objective is the L1 (absolute) distance between the predictor\u2019s output at masked positions and the stop-gradient target features from the y-encoder. L1 loss is preferred over L2 because it is less sensitive to outliers and produces sharper feature predictions. The stop-gradient on targets is essential to prevent collapse.',
  },
};

const benchmarkData: Record<
  string,
  { methods: Record<string, { score: number; color: string }>; maxVal: number }
> = {
  k400: {
    methods: {
      'V-JEPA (H/16)': { score: 82.0, color: '#00d9ff' },
      DINOv2: { score: 83.4, color: '#a855f7' },
      VideoMAE: { score: 79.8, color: '#f59e0b' },
      OmniMAE: { score: 78.3, color: '#10b981' },
      OpenCLIP: { score: 81.0, color: '#888' },
    },
    maxVal: 90,
  },
  ssv2: {
    methods: {
      'V-JEPA (H/16)': { score: 71.4, color: '#00d9ff' },
      DINOv2: { score: 50.6, color: '#a855f7' },
      VideoMAE: { score: 66.6, color: '#f59e0b' },
      OmniMAE: { score: 63.5, color: '#10b981' },
      OpenCLIP: { score: 52.1, color: '#888' },
    },
    maxVal: 80,
  },
  in1k: {
    methods: {
      'V-JEPA (H/16)': { score: 75.9, color: '#00d9ff' },
      DINOv2: { score: 86.2, color: '#a855f7' },
      VideoMAE: { score: 72.7, color: '#f59e0b' },
      OmniMAE: { score: 73.8, color: '#10b981' },
      OpenCLIP: { score: 84.4, color: '#888' },
    },
    maxVal: 95,
  },
};

/* ================================================================
   Masking helpers
   ================================================================ */

type MaskGrid = number[][][]; // [frame][row][col] = 0 | 1 | 2

function generateMasks(
  gridW: number,
  gridH: number,
  numFrames: number,
  ratio: number,
  mode: string,
): MaskGrid {
  const masks: MaskGrid = [];
  for (let f = 0; f < numFrames; f++) {
    masks.push([]);
    for (let r = 0; r < gridH; r++) {
      masks[f].push(new Array(gridW).fill(0));
    }
  }

  const totalTokens = gridW * gridH * numFrames;
  const targetMasked = Math.floor(totalTokens * (ratio / 100));

  function applyBlock(
    m: MaskGrid,
    blockW: number,
    blockH: number,
    maskType: number,
  ) {
    const startR = Math.floor(Math.random() * (gridH - blockH + 1));
    const startC = Math.floor(Math.random() * (gridW - blockW + 1));
    for (let f = 0; f < numFrames; f++) {
      for (let r = startR; r < startR + blockH && r < gridH; r++) {
        for (let c = startC; c < startC + blockW && c < gridW; c++) {
          if (m[f][r][c] === 0) {
            m[f][r][c] = maskType;
          }
        }
      }
    }
  }

  if (mode === 'short' || mode === 'combined') {
    const blockArea = Math.floor(gridW * gridH * 0.15);
    for (let b = 0; b < 8; b++) {
      const aspect = 0.75 + Math.random() * 0.75;
      let bH = Math.max(1, Math.round(Math.sqrt(blockArea / aspect)));
      let bW = Math.max(1, Math.round(blockArea / bH));
      bH = Math.min(bH, gridH);
      bW = Math.min(bW, gridW);
      applyBlock(masks, bW, bH, 1);
    }
  }

  if (mode === 'long' || mode === 'combined') {
    const blockArea = Math.floor(gridW * gridH * 0.7);
    for (let b = 0; b < 2; b++) {
      const aspect = 0.75 + Math.random() * 0.75;
      let bH = Math.max(1, Math.round(Math.sqrt(blockArea / aspect)));
      let bW = Math.max(1, Math.round(blockArea / bH));
      bH = Math.min(bH, gridH);
      bW = Math.min(bW, gridW);
      applyBlock(masks, bW, bH, 2);
    }
  }

  // Count currently masked
  let maskedCount = 0;
  for (let f = 0; f < numFrames; f++) {
    for (let r = 0; r < gridH; r++) {
      for (let c = 0; c < gridW; c++) {
        if (masks[f][r][c] > 0) maskedCount++;
      }
    }
  }

  // Fill up to target ratio
  if (maskedCount < targetMasked) {
    const unmaskTokens: [number, number, number][] = [];
    for (let f = 0; f < numFrames; f++) {
      for (let r = 0; r < gridH; r++) {
        for (let c = 0; c < gridW; c++) {
          if (masks[f][r][c] === 0) unmaskTokens.push([f, r, c]);
        }
      }
    }
    for (let i = unmaskTokens.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [unmaskTokens[i], unmaskTokens[j]] = [unmaskTokens[j], unmaskTokens[i]];
    }
    const toMask = Math.min(targetMasked - maskedCount, unmaskTokens.length);
    for (let i = 0; i < toMask; i++) {
      const [f, r, c] = unmaskTokens[i];
      masks[f][r][c] = mode === 'long' ? 2 : 1;
    }
  }

  return masks;
}

/* ================================================================
   Component
   ================================================================ */

export default function VJEPAPage() {
  /* ---------- State ---------- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);
  const [activeModelItem, setActiveModelItem] = useState<string | null>(null);
  const [activeLossItem, setActiveLossItem] = useState<string | null>(null);
  const [maskMode, setMaskMode] = useState('combined');
  const [maskRatio, setMaskRatio] = useState(90);
  const [taskKey, setTaskKey] = useState('k400');
  const [maskState, setMaskState] = useState<MaskGrid | null>(null);

  // Metrics derived from drawing
  const [metricMaskRatio, setMetricMaskRatio] = useState('90%');
  const [metricVisible, setMetricVisible] = useState('10%');
  const [metricSpeedup, setMetricSpeedup] = useState('~10x');

  /* ---------- Refs ---------- */
  const canvasRef = useRef<HTMLCanvasElement>(null);

  /* ---------- Canvas resize ---------- */
  const canvasSize = useCanvasResize(canvasRef);

  /* ---------- Generate masks on mode/ratio change ---------- */
  useEffect(() => {
    setMaskState(generateMasks(14, 14, 8, maskRatio, maskMode));
  }, [maskMode, maskRatio]);

  /* ---------- Draw masking visualization ---------- */
  const drawMaskingViz = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !maskState) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvasSize.width;
    const h = canvasSize.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const numFrames = 8;
    const gridW = 14;
    const gridH = 14;
    const padding = 8;
    const frameGap = 6;
    const totalGaps = (numFrames - 1) * frameGap;
    const availableW = w - padding * 2 - totalGaps;
    const frameW = availableW / numFrames;
    const cellSize = Math.min(
      frameW / gridW,
      (h - padding * 2 - 30) / gridH,
    );
    const actualFrameW = cellSize * gridW;
    const actualFrameH = cellSize * gridH;
    const startY = padding + 20;

    let totalTokens = 0;
    let maskedTokens = 0;

    for (let f = 0; f < numFrames; f++) {
      const frameX = padding + f * (actualFrameW + frameGap);

      // Frame label
      ctx.fillStyle = '#888';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`t=${f + 1}`, frameX + actualFrameW / 2, startY - 6);

      for (let r = 0; r < gridH; r++) {
        for (let c = 0; c < gridW; c++) {
          const x = frameX + c * cellSize;
          const y = startY + r * cellSize;
          const val = maskState[f][r][c];
          totalTokens++;

          if (val === 0) {
            ctx.fillStyle = 'rgba(0, 217, 255, 0.6)';
          } else if (val === 1) {
            ctx.fillStyle = 'rgba(100, 100, 100, 0.4)';
            maskedTokens++;
          } else {
            ctx.fillStyle = 'rgba(60, 60, 60, 0.5)';
            maskedTokens++;
          }

          ctx.fillRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);

          ctx.strokeStyle = 'rgba(255,255,255,0.05)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
        }
      }

      // Frame border
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 1;
      ctx.strokeRect(frameX, startY, actualFrameW, actualFrameH);
    }

    // Legend
    const legendY = startY + actualFrameH + 12;
    const legendItems = [
      { color: 'rgba(0, 217, 255, 0.6)', label: 'Visible (context x)' },
      { color: 'rgba(100, 100, 100, 0.4)', label: 'Short-range masked' },
      { color: 'rgba(60, 60, 60, 0.5)', label: 'Long-range masked' },
    ];

    let legendX = w / 2 - 180;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    legendItems.forEach((item) => {
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, legendY - 8, 12, 12);
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.strokeRect(legendX, legendY - 8, 12, 12);
      ctx.fillStyle = '#888';
      ctx.fillText(item.label, legendX + 16, legendY + 2);
      legendX += 130;
    });

    // Update metrics
    const actualRatioNum = ((maskedTokens / totalTokens) * 100).toFixed(0);
    const visiblePct = (100 - (maskedTokens / totalTokens) * 100).toFixed(0);
    const speedup = (
      totalTokens / Math.max(1, totalTokens - maskedTokens)
    ).toFixed(0);

    setMetricMaskRatio(`${actualRatioNum}%`);
    setMetricVisible(`${visiblePct}%`);
    setMetricSpeedup(`~${speedup}x`);
  }, [canvasSize, maskState]);

  useEffect(() => {
    drawMaskingViz();
  }, [drawMaskingViz]);

  /* ---------- Benchmark rendering ---------- */
  const currentBenchmark = benchmarkData[taskKey];

  /* ---------- Render ---------- */
  return (
    <div className="method-page">
      <div className={styles.container}>
        <h1>V-JEPA</h1>
        <p className={styles.subtitle}>
          Revisiting Feature Prediction for Learning Visual Representations
          from Video &mdash; Bardes et al., 2024
        </p>
        <div className={styles.githubLink}>
          <a
            href="https://github.com/facebookresearch/jepa"
            target="_blank"
            rel="noopener noreferrer"
          >
            <span style={{ marginRight: 5 }}>&rarr;</span>{' '}
            github.com/facebookresearch/jepa
          </a>
        </div>

        <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

        {/* ===================== Model Components + Architecture ===================== */}
        <div className={styles.mainGrid}>
          {/* Model Components */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Model Components
            </div>
            <div className={styles.modelEquations}>
              {[
                {
                  key: 'x-encoder',
                  color: '#00d9ff',
                  title: 'x-encoder (Context)',
                  desc: 'Standard ViT (L/16, H/16, or H/16_384). Processes only visible ~10% of tokens. Very efficient \u2014 the key to training speed.',
                },
                {
                  key: 'y-encoder',
                  color: '#a855f7',
                  title: 'y-encoder (EMA Target)',
                  desc: 'Exponential moving average of x-encoder weights. Processes full video including masked regions. Stop-gradient prevents collapse. \u03c4: 0.996\u21921.0',
                },
                {
                  key: 'predictor',
                  color: '#10b981',
                  title: 'Predictor',
                  desc: 'Narrow ViT (12 layers, dim 384). Takes x-encoder outputs + learnable mask tokens with positional embeddings. Outputs predicted features at masked positions.',
                },
                {
                  key: 'masking',
                  color: '#f59e0b',
                  title: 'Multi-Block Masking',
                  desc: 'Short-range (8 blocks, 15%) + long-range (2 blocks, 70%). ~90% masking ratio. Patches are 16\u00d716 pixels spanning 2 frames.',
                },
                {
                  key: 'probing',
                  color: '#ef4444',
                  title: 'Attentive Probing',
                  desc: 'Evaluation protocol. Freeze backbone, add cross-attention layer with learnable query token, then linear classifier. Better than linear probing for unnormalized features.',
                },
              ].map((item) => (
                <div
                  key={item.key}
                  className={`${styles.modelItem} ${activeModelItem === item.key ? styles.modelItemActive : ''}`}
                  onClick={() => setActiveModelItem(item.key)}
                >
                  <h4>
                    <span
                      className={styles.modelDot}
                      style={{ background: item.color }}
                    />
                    {item.title}
                  </h4>
                  <p className={styles.modelItemDesc}>{item.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* V-JEPA Architecture SVG */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              V-JEPA Architecture
            </div>
            <svg className={styles.pipelineSvg} viewBox="0 0 500 420">
              {/* Arrow markers */}
              <defs>
                <marker
                  id="arrowCyan"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff" />
                </marker>
                <marker
                  id="arrowPurple"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
                </marker>
                <marker
                  id="arrowGreen"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
                </marker>
                <marker
                  id="arrowOrange"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
                </marker>
                <marker
                  id="arrowRed"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
                </marker>
                <marker
                  id="arrowGray"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#888" />
                </marker>
              </defs>

              {/* Video Clip */}
              <rect
                x="160"
                y="5"
                width="180"
                height="35"
                rx="8"
                fill="rgba(245,158,11,0.2)"
                stroke="#f59e0b"
                strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'video' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('video')}
              />
              <text
                x="250"
                y="20"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                {'Video Clip (T\u00d7H\u00d7W)'}
              </text>
              <text
                x="250"
                y="33"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                {'Patchified \u2192 tokens'}
              </text>

              {/* Mask */}
              <rect
                x="190"
                y="50"
                width="120"
                height="22"
                rx="6"
                fill="rgba(245,158,11,0.15)"
                stroke="#f59e0b"
                strokeWidth="1"
                strokeDasharray="4"
                className={`${styles.pipelineStage} ${activeComponent === 'mask' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('mask')}
              />
              <text
                x="250"
                y="64"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="8"
                pointerEvents="none"
              >
                Binary Mask (~90%)
              </text>

              {/* Arrow from video to mask */}
              <path
                d="M 250 40 L 250 48"
                stroke="#f59e0b"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowOrange)"
              />

              {/* Split arrows */}
              <path
                d="M 210 72 L 110 95"
                stroke="#00d9ff"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowCyan)"
              />
              <path
                d="M 290 72 L 390 95"
                stroke="#a855f7"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowPurple)"
              />

              {/* Labels for split */}
              <text
                x="140"
                y="82"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="7"
              >
                Visible (~10%)
              </text>
              <text
                x="360"
                y="82"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="7"
              >
                All tokens
              </text>

              {/* x-encoder */}
              <rect
                x="40"
                y="100"
                width="140"
                height="55"
                rx="10"
                fill="rgba(0,217,255,0.2)"
                stroke="#00d9ff"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'x_encoder' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('x_encoder')}
              />
              <text
                x="110"
                y="120"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                {'x-encoder (E\u03b8)'}
              </text>
              <text
                x="110"
                y="135"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                ViT-L/16 or ViT-H/16
              </text>
              <text
                x="110"
                y="147"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="7"
                pointerEvents="none"
              >
                {'Context features [N\u00d7d]'}
              </text>

              {/* y-encoder (EMA, dashed border) */}
              <rect
                x="320"
                y="100"
                width="140"
                height="55"
                rx="10"
                fill="rgba(168,85,247,0.15)"
                stroke="#a855f7"
                strokeWidth="2"
                strokeDasharray="6 3"
                className={`${styles.pipelineStage} ${activeComponent === 'y_encoder' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('y_encoder')}
              />
              <text
                x="390"
                y="120"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                {'y-encoder (\u0112\u03b8)'}
              </text>
              <text
                x="390"
                y="135"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                EMA copy, stop-gradient
              </text>
              <text
                x="390"
                y="147"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="7"
                pointerEvents="none"
              >
                {'Target features [L\u00d7d]'}
              </text>

              {/* EMA update arrow (dashed) */}
              <path
                d="M 180 115 Q 250 90 320 115"
                stroke="#a855f7"
                strokeWidth="1.5"
                fill="none"
                strokeDasharray="5 3"
                className={styles.flowArrow}
              />
              <text
                x="250"
                y="98"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="7"
                fontWeight="bold"
              >
                EMA update
              </text>

              {/* Remove unmasked tokens */}
              <rect
                x="340"
                y="170"
                width="120"
                height="22"
                rx="6"
                fill="rgba(168,85,247,0.1)"
                stroke="#a855f7"
                strokeWidth="1"
                className={`${styles.pipelineStage} ${activeComponent === 'remove_unmasked' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('remove_unmasked')}
              />
              <text
                x="400"
                y="184"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="7"
                pointerEvents="none"
              >
                {'Remove unmasked \u2192 [M\u00d7d]'}
              </text>
              <path
                d="M 390 155 L 400 168"
                stroke="#a855f7"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowPurple)"
              />

              {/* Mask tokens + x-encoder output merge */}
              <rect
                x="70"
                y="180"
                width="80"
                height="22"
                rx="6"
                fill="rgba(16,185,129,0.15)"
                stroke="#10b981"
                strokeWidth="1"
                className={`${styles.pipelineStage} ${activeComponent === 'mask_tokens' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('mask_tokens')}
              />
              <text
                x="110"
                y="194"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
                pointerEvents="none"
              >
                Mask tokens + pos emb
              </text>

              {/* Arrow from x-encoder down */}
              <path
                d="M 110 155 L 110 178"
                stroke="#00d9ff"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowCyan)"
              />

              {/* Predictor */}
              <rect
                x="40"
                y="220"
                width="200"
                height="55"
                rx="10"
                fill="rgba(16,185,129,0.2)"
                stroke="#10b981"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'predictor' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('predictor')}
              />
              <text
                x="140"
                y="240"
                textAnchor="middle"
                fill="#10b981"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                {'Predictor (P\u03c6)'}
              </text>
              <text
                x="140"
                y="255"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                Narrow ViT: 12 layers, dim 384
              </text>
              <text
                x="140"
                y="267"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
                pointerEvents="none"
              >
                {'Predicted features [M\u00d7d]'}
              </text>

              {/* Arrows into predictor */}
              <path
                d="M 110 202 L 120 218"
                stroke="#10b981"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowGreen)"
              />

              {/* L1 Loss */}
              <rect
                x="160"
                y="305"
                width="180"
                height="40"
                rx="10"
                fill="rgba(239,68,68,0.15)"
                stroke="#ef4444"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'loss' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent('loss')}
              />
              <text
                x="250"
                y="322"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                L1 Loss
              </text>
              <text
                x="250"
                y="337"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                {'||predicted - sg(target)||\u2081'}
              </text>

              {/* Arrow from predictor to loss */}
              <path
                d="M 180 275 L 220 303"
                stroke="#10b981"
                strokeWidth="2"
                fill="none"
                markerEnd="url(#arrowGreen)"
                className={styles.flowArrow}
              />
              <text
                x="180"
                y="295"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
              >
                predicted
              </text>

              {/* Arrow from target to loss with stop-gradient */}
              <path
                d="M 400 192 L 400 290 L 310 310"
                stroke="#a855f7"
                strokeWidth="2"
                fill="none"
                markerEnd="url(#arrowPurple)"
              />
              <text
                x="410"
                y="250"
                textAnchor="start"
                fill="#a855f7"
                fontSize="7"
              >
                target
              </text>

              {/* Stop-gradient symbol */}
              <circle
                cx="380"
                cy="290"
                r="10"
                fill="rgba(239,68,68,0.3)"
                stroke="#ef4444"
                strokeWidth="1.5"
              />
              <text
                x="380"
                y="294"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="11"
                fontWeight="bold"
              >
                {'\u2298'}
              </text>
              <text x="395" y="290" textAnchor="start" fill="#888" fontSize="7">
                sg
              </text>

              {/* Summary bar */}
              <rect
                x="20"
                y="365"
                width="460"
                height="45"
                rx="10"
                fill="rgba(0,217,255,0.08)"
                stroke="#00d9ff"
                strokeWidth="1"
              />
              <text
                x="250"
                y="382"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="8"
                fontWeight="bold"
              >
                Key: Predict features at masked positions, NOT pixels
              </text>
              <text
                x="250"
                y="397"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
              >
                x-encoder sees ~10% tokens | y-encoder (EMA) sees 100% |
                Predictor bridges the gap | L1 loss in feature space
              </text>
            </svg>

            {activeComponent && archComponentInfo[activeComponent] && (
              <div className={styles.infoPanel}>
                <h4>{archComponentInfo[activeComponent].title}</h4>
                <p>{archComponentInfo[activeComponent].desc}</p>
              </div>
            )}
          </div>
        </div>

        {/* ===================== Canvas + Benchmark ===================== */}
        <div className={styles.mainGrid}>
          {/* Interactive Canvas: Video Token Masking */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Video Token Masking
            </div>
            <div className={styles.toggleBtns}>
              {[
                { mode: 'short', label: 'Short-range masks' },
                { mode: 'long', label: 'Long-range masks' },
                { mode: 'combined', label: 'Combined (V-JEPA)' },
              ].map((item) => (
                <button
                  key={item.mode}
                  className={`${styles.toggleBtn} ${maskMode === item.mode ? styles.toggleBtnActive : ''}`}
                  onClick={() => setMaskMode(item.mode)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <div className={styles.sliderContainer}>
              <label>Masking ratio:</label>
              <input
                type="range"
                className={styles.slider}
                min={50}
                max={95}
                step={1}
                value={maskRatio}
                onChange={(e) => setMaskRatio(parseInt(e.target.value))}
              />
              <span className={styles.sliderValue}>{maskRatio}%</span>
            </div>
            <div className={styles.canvasContainer}>
              <canvas ref={canvasRef} className={styles.maskingCanvas} />
            </div>
            <div className={styles.metricsRow}>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: '#888' }}>
                  {metricMaskRatio}
                </div>
                <div className={styles.metricLabel}>Masking Ratio</div>
              </div>
              <div className={styles.metric}>
                <div
                  className={styles.metricValue}
                  style={{ color: '#00d9ff' }}
                >
                  {metricVisible}
                </div>
                <div className={styles.metricLabel}>Visible Tokens</div>
              </div>
              <div className={styles.metric}>
                <div
                  className={styles.metricValue}
                  style={{ color: '#10b981' }}
                >
                  {metricSpeedup}
                </div>
                <div className={styles.metricLabel}>Processing Speedup</div>
              </div>
            </div>
          </div>

          {/* Benchmark Results */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Benchmark Results (Frozen Evaluation)
            </div>
            <div className={styles.comparisonContainer}>
              <select
                className={styles.taskSelect}
                value={taskKey}
                onChange={(e) => setTaskKey(e.target.value)}
              >
                <option value="k400">Kinetics-400 (K400)</option>
                <option value="ssv2">Something-Something v2 (SSv2)</option>
                <option value="in1k">ImageNet-1K (IN1K)</option>
              </select>
              <div>
                {Object.entries(currentBenchmark.methods).map(
                  ([method, info]) => {
                    const pct =
                      (info.score / currentBenchmark.maxVal) * 100;
                    const isVJEPA = method.includes('V-JEPA');
                    return (
                      <div key={method} className={styles.comparisonRow}>
                        <span
                          className={styles.comparisonLabel}
                          style={{
                            color: info.color,
                            fontWeight: isVJEPA ? 700 : 400,
                          }}
                        >
                          {method}
                        </span>
                        <div className={styles.comparisonBarTrack}>
                          <div
                            className={styles.comparisonBarFill}
                            style={{
                              width: `${pct}%`,
                              background: info.color,
                              opacity: isVJEPA ? 1 : 0.7,
                            }}
                          />
                        </div>
                        <span
                          className={styles.comparisonValue}
                          style={{ color: info.color }}
                        >
                          {info.score}
                        </span>
                      </div>
                    );
                  },
                )}
              </div>
              <p className={styles.benchmarkNote}>
                All results with frozen backbone + attentive probing. V-JEPA
                achieves strong performance across video and image tasks
                without pixel reconstruction.
              </p>
            </div>
          </div>
        </div>

        {/* ===================== Training Objectives ===================== */}
        <div className={`${styles.card} ${styles.cardMb}`}>
          <div className={styles.cardTitle}>
            Training Objectives
          </div>
          <div className={styles.modelEquations}>
            {/* V-JEPA Loss */}
            <div
              className={`${styles.modelItem} ${activeLossItem === 'vjepa' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLossItem('vjepa')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#10b981' }}
                />
                V-JEPA Loss
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '\\[\\mathcal{L} = \\|P_\\phi(E_\\theta(x), \\Delta_y) - \\text{sg}(\\bar{E}_{\\bar\\theta}(y))\\|_1\\]'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                L1 loss between predicted features at masked positions and
                stopped-gradient target features from the EMA encoder.
                &Delta;<sub>y</sub> denotes the learnable mask tokens with
                positional embeddings.
              </p>
            </div>

            {/* EMA Update */}
            <div
              className={`${styles.modelItem} ${activeLossItem === 'ema' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLossItem('ema')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#a855f7' }}
                />
                EMA Update
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '\\[\\bar{\\theta} \\leftarrow \\tau\\bar{\\theta} + (1-\\tau)\\theta, \\quad \\tau: 0.996 \\to 1.0\\]'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                Exponential moving average update for the y-encoder weights.
                &tau; linearly increases from 0.996 to 1.0 over training,
                making the target encoder increasingly stable.
              </p>
            </div>

            {/* Multi-Block Masking */}
            <div
              className={`${styles.modelItem} ${activeLossItem === 'masking' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLossItem('masking')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#f59e0b' }}
                />
                Multi-Block Masking
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '\\[M = M_{\\text{short}} \\cup M_{\\text{long}}, \\quad \\frac{|M|}{|T|} \\approx 0.9\\]'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                Union of short-range (8 blocks &times; 15%) and long-range (2
                blocks &times; 70%) masks. Total masking ratio ~90%. All masks
                span the full temporal dimension.
              </p>
            </div>
          </div>
        </div>

        {/* ===================== Paper Lineage ===================== */}
        <div className={`${styles.card} ${styles.cardMb}`}>
          <div className={styles.cardTitle}>
            Paper Lineage
          </div>
          <div className={styles.lineageGrid}>
            <div>
              <h4
                className={styles.lineageSectionTitle}
                style={{ color: '#f59e0b' }}
              >
                Builds On
              </h4>
              <div className={styles.lineageItems}>
                {[
                  {
                    name: 'I-JEPA',
                    desc: 'Assran et al., 2023 \u2014 Image-based JEPA. V-JEPA extends the approach to video with temporal masking strategies.',
                  },
                  {
                    name: 'BYOL',
                    desc: 'Grill et al., 2020 \u2014 Bootstrap Your Own Latent. Pioneered EMA target encoder + stop-gradient for self-supervised learning.',
                  },
                  {
                    name: 'MAE',
                    desc: 'He et al., 2022 \u2014 Masked Autoencoders. V-JEPA replaces pixel reconstruction with feature prediction.',
                  },
                  {
                    name: 'ViT',
                    desc: 'Dosovitskiy et al., 2021 \u2014 Vision Transformer. Core backbone architecture used in both encoders and predictor.',
                  },
                  {
                    name: 'BEiT',
                    desc: 'Bao et al., 2022 \u2014 Bidirectional Encoder representations from Image Transformers. Masking strategy inspiration.',
                  },
                ].map((item) => (
                  <div
                    key={item.name}
                    className={styles.lineageItem}
                    style={{ borderLeftColor: '#f59e0b' }}
                  >
                    <h5>{item.name}</h5>
                    <p>{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4
                className={styles.lineageSectionTitle}
                style={{ color: '#10b981' }}
              >
                Built Upon By
              </h4>
              <div className={styles.lineageItems}>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#10b981' }}
                >
                  <h5>V-JEPA 2</h5>
                  <p>
                    Assran et al., 2025 &mdash; Extends V-JEPA with improved
                    training, action-conditioned world model, and planning
                    capabilities.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
