'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './vjepa2.module.css';

const coreIdeas = [
  {
    title: 'Scaling Self-Supervised Video Pretraining',
    desc: 'Scale along ALL axes simultaneously: data (2M\u219222M videos), model (300M\u21921B params), training (90K\u2192252K iters), and progressive resolution (256px\u2192384px, 16\u219264 frames). Each contributes +1\u20131.5 points.',
    detail: 'V-JEPA 2 demonstrates that scaling self-supervised video pretraining improves all downstream capabilities. Four key ingredients: (1) Data scaling: increasing from VideoMix2M to VideoMix22M gives +1.0 point. (2) Model scaling: going from ViT-L (300M params) to ViT-g (1B params) gives +1.5 points. (3) Longer training: extending from 90K to 252K iterations gives +0.8 points. (4) Progressive resolution: training at 256px then scaling to 384px and from 16 to 64 frames gives +1.0 point. Total improvement: +4.0 points, reaching 88.2% average accuracy across 6 understanding tasks.',
  },
  {
    title: 'Action-Conditioned World Model (V-JEPA 2-AC)',
    desc: 'Freeze the video encoder, train a 300M-param autoregressive transformer predictor on top using only 62 hours of robot data. Block-causal attention predicts next-frame representations conditioned on actions.',
    detail: 'V-JEPA 2-AC adds an action-conditioned predictor on top of the frozen V-JEPA 2 backbone. The predictor is a 300M-parameter transformer (24 layers, 16 heads, 1024 hidden dim) with block-causal attention. It processes interleaved sequences of (frame features, actions, end-effector states) and predicts next-frame representations autoregressively. Trained with teacher-forcing loss + rollout loss on only 62 hours of robot manipulation data from the Droid dataset (7-DOF Franka Panda arm).',
  },
  {
    title: 'Planning via Energy Minimization',
    desc: 'Plan robot actions by minimizing L1 distance between predicted future state and goal state in representation space. No reward function needed \u2014 just provide a goal image.',
    detail: 'V-JEPA 2-AC enables robot planning without any reward function. Given a current observation and a goal image, both are encoded into feature maps z_current and z_goal. The system optimizes an action sequence a_{1:H} to minimize E(a) = ||P(a_{1:H}, z_0) - z_goal||_1, where P is the autoregressive predictor. Only the first action is executed, then replanning occurs.',
  },
];

const keyFeatures = [
  { title: '1M Hours of Video', desc: 'Pretrained on unprecedented scale of internet video data (22M videos from 5 datasets)' },
  { title: '62 Hours to Robot', desc: 'Only 62 hours of robot data needed for the action-conditioned world model on top of frozen backbone' },
  { title: 'Goal-Conditioned Planning', desc: 'Plans by minimizing distance to goal in feature space \u2014 no reward function or task-specific training' },
];

/* ================================================================
   Data
   ================================================================ */

const ideaInfo: Record<number, { title: string; desc: string }> = {
  1: {
    title: 'Scaling Self-Supervised Video Pretraining',
    desc: 'V-JEPA 2 demonstrates that scaling self-supervised video pretraining improves all downstream capabilities. Four key ingredients: (1) Data scaling: increasing from VideoMix2M (2M videos) to VideoMix22M (22M videos from SSv2, Kinetics, HowTo100M, YT-Temporal-1B, ImageNet) gives +1.0 point. (2) Model scaling: going from ViT-L (300M params) to ViT-g (1B params) gives +1.5 points. (3) Longer training: extending from 90K to 252K iterations with warmup-constant-decay schedule gives +0.8 points. (4) Progressive resolution: training at 256px then scaling to 384px and from 16 to 64 frames gives +1.0 point. Total improvement: +4.0 points, reaching 88.2% average accuracy across 6 understanding tasks.',
  },
  2: {
    title: 'Action-Conditioned World Model (V-JEPA 2-AC)',
    desc: "V-JEPA 2-AC adds an action-conditioned predictor on top of the frozen V-JEPA 2 backbone. The predictor is a 300M-parameter transformer (24 layers, 16 heads, 1024 hidden dim) with block-causal attention. It processes interleaved sequences of (frame features, actions, end-effector states) and predicts next-frame representations autoregressively. Trained with teacher-forcing loss + rollout loss on only 62 hours of robot manipulation data from the Droid dataset (7-DOF Franka Panda arm). Uses 3D-RoPE for spatiotemporal position encoding. The frozen backbone means all visual understanding comes from internet video pretraining.",
  },
  3: {
    title: 'Planning via Energy Minimization',
    desc: 'V-JEPA 2-AC enables robot planning without any reward function. Given a current observation and a goal image, both are encoded by the frozen video encoder into feature maps z_current and z_goal. The system optimizes an action sequence a_{1:H} to minimize E(a) = ||P(a_{1:H}, z_0) - z_goal||_1, where P is the autoregressive predictor. This is done via gradient-based optimization (MPPI or similar). Only the first action is executed, then replanning occurs. This enables zero-shot manipulation in new environments without any task-specific training or reward engineering.',
  },
};

const componentInfo: Record<string, { title: string; desc: string }> = {
  video: {
    title: 'Video Dataset (VideoMix22M)',
    desc: "V-JEPA 2 is pretrained on VideoMix22M, a collection of 22 million videos from 5 datasets: Something-Something v2 (SSv2), Kinetics-710, HowTo100M, YT-Temporal-1B, and ImageNet. This represents a massive scale-up from the original V-JEPA's 2M videos, contributing +1.0 point improvement in downstream accuracy.",
  },
  mask: {
    title: 'Masking Strategy',
    desc: 'Following the JEPA paradigm, a large portion (~90%) of the video patches are masked. The x-encoder only sees the unmasked patches (context), while the predictor must predict the features of the masked regions. This forces the model to learn meaningful spatiotemporal representations.',
  },
  xencoder: {
    title: 'x-encoder (ViT-g, 1B params)',
    desc: 'The online encoder processes only the visible (unmasked) video patches. It is a ViT-g with 1 billion parameters using 3D-RoPE for position encoding. During pretraining, it is updated by gradient descent. After pretraining, this encoder is frozen and serves as the visual backbone for all downstream tasks including the action-conditioned world model.',
  },
  pretrain_pred: {
    title: 'Pretraining Predictor',
    desc: 'During V-JEPA pretraining, the predictor takes the encoded visible patches and positional information about the masked regions, then predicts the features of those masked patches. The targets come from the EMA y-encoder with stop-gradient. This predictor is discarded after pretraining.',
  },
  yencoder: {
    title: 'y-encoder (EMA Target)',
    desc: 'The target encoder is an exponential moving average (EMA) of the x-encoder. It processes the full (unmasked) video to produce target features. Stop-gradient prevents the targets from collapsing. This asymmetric design (online encoder sees partial input, target encoder sees full input) is key to the JEPA self-supervised paradigm.',
  },
  pretrain_loss: {
    title: 'Pretraining L1 Loss',
    desc: "The pretraining objective minimizes the L1 distance between the predictor's output and the stop-gradient target features. Training runs for 252K iterations with a warmup-constant-decay learning rate schedule. Progressive resolution: starts at 256px with 16 frames, scales to 384px with 64 frames.",
  },
  frozen_enc: {
    title: 'Frozen Encoder (Pretrained ViT-g)',
    desc: 'After pretraining on 22M internet videos, the ViT-g encoder is completely frozen. All visual understanding capability comes from this pretrained backbone. For V-JEPA 2-AC, only the autoregressive predictor on top is trained, using just 62 hours of robot data. This transfer learning approach is key: internet video provides general visual understanding, robot data provides action-conditioned dynamics.',
  },
  ac_predictor: {
    title: 'Autoregressive Predictor (V-JEPA 2-AC)',
    desc: 'A 300M-parameter transformer with 24 layers, 16 attention heads, and 1024 hidden dimension. Uses block-causal attention to process interleaved sequences of (frame features z_t, actions a_t, end-effector states s_t). Predicts next-frame features autoregressively. Trained with teacher-forcing + rollout loss on 62 hours of Droid dataset (7-DOF Franka Panda arm). Uses 3D-RoPE for spatiotemporal position encoding.',
  },
  energy_plan: {
    title: 'Energy-Based Planning',
    desc: 'Given current observation and goal image, both encoded by the frozen backbone into z_current and z_goal, the planner optimizes: a*_{1:H} = argmin ||P_AC(z_0, s_0, a_{1:H}) - z_goal||_1. This is solved via gradient-based optimization (e.g., MPPI). Only the first action a*_1 is executed, then replanning occurs with the new observation. No reward function or task-specific training needed.',
  },
};

interface BenchEntry {
  info: string;
  methods: Record<string, number>;
  maxVal: number;
  colors: Record<string, string>;
}

const benchData: Record<string, BenchEntry> = {
  scaling: {
    info: 'Cumulative improvement from scaling each axis (avg accuracy on 6 tasks)',
    methods: {
      Baseline: 84.2,
      '+ Data (22M)': 85.2,
      '+ Model (ViT-g)': 86.7,
      '+ Training (252K)': 87.5,
      '+ Resolution (384px)': 88.2,
    },
    maxVal: 92,
    colors: {
      Baseline: '#888',
      '+ Data (22M)': '#3b82f6',
      '+ Model (ViT-g)': '#8b7ec8',
      '+ Training (252K)': '#c4884d',
      '+ Resolution (384px)': '#10b981',
    },
  },
  comparison: {
    info: 'Model comparison on average video understanding accuracy (%)',
    methods: {
      'V-JEPA 2 (ViT-g)': 88.2,
      DINOv2: 75.8,
      Hiera: 74.7,
      'V-JEPA (H/16)': 71.5,
      VideoMAE: 71.4,
    },
    maxVal: 92,
    colors: {
      'V-JEPA 2 (ViT-g)': '#638bd4',
      DINOv2: '#8b7ec8',
      Hiera: '#c4884d',
      'V-JEPA (H/16)': '#3b82f6',
      VideoMAE: '#888',
    },
  },
};

/* ================================================================
   Planning helpers
   ================================================================ */

interface Point {
  x: number;
  y: number;
}

function generateTrajectories(horizon: number, maxIter: number): Point[][] {
  const startX = 60,
    startY = 200;
  const goalX = 500,
    goalY = 80;
  const trajectories: Point[][] = [];

  for (let iter = 0; iter <= maxIter; iter++) {
    const t = iter / maxIter;
    const points: Point[] = [];

    for (let i = 0; i <= horizon; i++) {
      const frac = i / horizon;
      const randOffsetX =
        (1 - t) * Math.sin(frac * Math.PI * 2.5) * 80 * (1 - frac * 0.3);
      const randOffsetY = (1 - t) * Math.cos(frac * Math.PI * 1.8) * 60;
      const optX = startX + (goalX - startX) * frac;
      const optY =
        startY +
        (goalY - startY) * frac -
        Math.sin(frac * Math.PI) * 40 * (1 - t * 0.5);
      points.push({ x: optX + randOffsetX, y: optY + randOffsetY });
    }
    trajectories.push(points);
  }
  return trajectories;
}

function getEnergy(iter: number, maxIter: number): number {
  const t = iter / maxIter;
  return 12.4 * Math.exp(-3.5 * t) + 0.3;
}

function getEnergyHistory(currentIter: number, maxIter: number): number[] {
  const history: number[] = [];
  for (let i = 1; i <= currentIter; i++) {
    history.push(getEnergy(i, maxIter));
  }
  return history;
}

function drawStar(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  outerR: number,
  numPoints: number,
  color: string,
) {
  const innerR = outerR * 0.4;
  ctx.beginPath();
  for (let i = 0; i < numPoints * 2; i++) {
    const r = i % 2 === 0 ? outerR : innerR;
    const angle = (i * Math.PI) / numPoints - Math.PI / 2;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

/* ================================================================
   Component
   ================================================================ */

export default function VJEPA2Page() {
  // ---- Architecture state ----
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  // ---- Model item state ----
  const [activeModelItem, setActiveModelItem] = useState<string | null>(null);

  // ---- Training Objectives state ----
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  // ---- Planning state ----
  const [iteration, setIteration] = useState(1);
  const [horizon, setHorizon] = useState(8);
  const planningCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trajectoriesRef = useRef<Point[][] | null>(null);
  const cachedHorizonRef = useRef<number | null>(null);
  const canvasSize = useCanvasResize(planningCanvasRef);

  // ---- Benchmark state ----
  const [activeBench, setActiveBench] = useState<string>('scaling');

  // ---- Computed metrics ----
  const maxIter = 30;
  const energy = getEnergy(iteration, maxIter);

  // Ensure trajectories are cached for current horizon
  if (cachedHorizonRef.current !== horizon) {
    trajectoriesRef.current = generateTrajectories(horizon, maxIter);
    cachedHorizonRef.current = horizon;
  }

  // Compute trajectory length for display
  const computeTrajLen = useCallback(() => {
    if (!trajectoriesRef.current) return 0;
    const currentPts = trajectoriesRef.current[iteration];
    if (!currentPts) return 0;
    const w = canvasSize.width;
    const h = canvasSize.height;
    const scaleX = w / 600;
    const scaleY = h / 300;
    let trajLen = 0;
    for (let i = 1; i < currentPts.length; i++) {
      const dx = (currentPts[i].x - currentPts[i - 1].x) * scaleX;
      const dy = (currentPts[i].y - currentPts[i - 1].y) * scaleY;
      trajLen += Math.sqrt(dx * dx + dy * dy);
    }
    return trajLen / 50;
  }, [iteration, canvasSize.width, canvasSize.height]);

  const trajLen = computeTrajLen();

  // ---- Canvas drawing ----
  useEffect(() => {
    const canvas = planningCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvasSize.width;
    const h = canvasSize.height;
    if (w === 0 || h === 0) return;

    // Regenerate if needed
    if (cachedHorizonRef.current !== horizon) {
      trajectoriesRef.current = generateTrajectories(horizon, maxIter);
      cachedHorizonRef.current = horizon;
    }

    const trajectories = trajectoriesRef.current;
    if (!trajectories) return;

    ctx.clearRect(0, 0, w, h);

    const scaleX = w / 600;
    const scaleY = h / 300;
    const startX = 60 * scaleX;
    const startY = 200 * scaleY;
    const goalX = 500 * scaleX;
    const goalY = 80 * scaleY;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Representation Space', w / 2, h - 8);

    // Previous trajectories (faded)
    if (iteration > 1) {
      for (
        let prevIter = Math.max(1, iteration - 5);
        prevIter < iteration;
        prevIter++
      ) {
        const pts = trajectories[prevIter];
        const alpha = 0.05 + 0.05 * (prevIter - (iteration - 5));
        ctx.beginPath();
        ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
        ctx.lineWidth = 1;
        for (let i = 0; i < pts.length; i++) {
          const px = pts[i].x * scaleX;
          const py = pts[i].y * scaleY;
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    }

    // Current trajectory
    const currentPts = trajectories[iteration];
    ctx.beginPath();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.5;
    for (let i = 0; i < currentPts.length; i++) {
      const px = currentPts[i].x * scaleX;
      const py = currentPts[i].y * scaleY;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Intermediate dots
    for (let i = 1; i < currentPts.length - 1; i++) {
      const px = currentPts[i].x * scaleX;
      const py = currentPts[i].y * scaleY;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(245, 158, 11, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Start (blue dot)
    ctx.beginPath();
    ctx.arc(startX, startY, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('z_0', startX, startY + 25);
    ctx.fillStyle = '#3b82f6';
    ctx.font = '9px sans-serif';
    ctx.fillText('Current', startX, startY + 37);

    // Goal (green star)
    drawStar(ctx, goalX, goalY, 12, 5, '#10b981');
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('z_goal', goalX, goalY + 25);
    ctx.fillStyle = '#10b981';
    ctx.font = '9px sans-serif';
    ctx.fillText('Goal', goalX, goalY + 37);

    // Predicted end state
    const endPt = currentPts[currentPts.length - 1];
    const endX = endPt.x * scaleX;
    const endY = endPt.y * scaleY;
    ctx.beginPath();
    ctx.arc(endX, endY, 7, 0, 2 * Math.PI);
    ctx.fillStyle = '#f59e0b';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dashed line from predicted to goal
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(endX, endY);
    ctx.lineTo(goalX, goalY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Energy label
    const currentEnergy = getEnergy(iteration, maxIter);
    const midLineX = (endX + goalX) / 2;
    const midLineY = (endY + goalY) / 2;
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('E=' + currentEnergy.toFixed(1), midLineX, midLineY - 8);

    // Mini energy chart
    const chartW = 100,
      chartH = 50;
    const chartX = w - chartW - 15,
      chartY2 = 10;

    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(chartX, chartY2, chartW, chartH);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(chartX, chartY2, chartW, chartH);

    ctx.fillStyle = '#888';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Energy vs Iteration', chartX + chartW / 2, chartY2 + 10);

    const energyHistory = getEnergyHistory(iteration, maxIter);
    if (energyHistory.length > 0) {
      const maxE = 13;
      ctx.beginPath();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < energyHistory.length; i++) {
        const cx = chartX + 5 + (i / (maxIter - 1)) * (chartW - 10);
        const cy =
          chartY2 +
          chartH -
          5 -
          (energyHistory[i] / maxE) * (chartH - 18);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      const lastI = energyHistory.length - 1;
      const dotCx = chartX + 5 + (lastI / (maxIter - 1)) * (chartW - 10);
      const dotCy =
        chartY2 +
        chartH -
        5 -
        (energyHistory[lastI] / maxE) * (chartH - 18);
      ctx.beginPath();
      ctx.arc(dotCx, dotCy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
    }
  }, [iteration, horizon, canvasSize]);

  // ---- Benchmark rendering ----
  const bench = benchData[activeBench];
  const benchEntries = Object.entries(bench.methods);

  function isTopEntry(benchKey: string, method: string, idx: number): boolean {
    if (idx === 0) return true;
    if (benchKey === 'scaling' && method === '+ Resolution (384px)') return true;
    if (benchKey === 'comparison' && method === 'V-JEPA 2 (ViT-g)') return true;
    return false;
  }

  /* ================================================================
     Render
     ================================================================ */

  return (
    <div className="method-page">
      <div className={styles.container}>
        <h1>V-JEPA 2</h1>
        <p className={styles.subtitle}>
          Self-Supervised Video Models Enable Understanding, Prediction and
          Planning &mdash; Assran et al., 2025
        </p>
        <div className={styles.githubLink}>
          <a
            href="https://github.com/facebookresearch/vjepa2"
            target="_blank"
            rel="noopener noreferrer"
          >
            <span style={{ marginRight: 5 }}>&rarr;</span>
            github.com/facebookresearch/vjepa2
          </a>
        </div>

        <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

        <div className={styles.mainGrid}>
          {/* ==================== Model Components ==================== */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Model Components
            </div>
            <div className={styles.modelEquations}>
              {/* Backbone */}
              <div
                className={`${styles.modelItem} ${activeModelItem === 'backbone' ? styles.modelItemActive : ''}`}
                onClick={() => setActiveModelItem(activeModelItem === 'backbone' ? null : 'backbone')}
              >
                <h4>
                  <span
                    className={styles.modelDot}
                    style={{ background: '#638bd4' }}
                  />
                  ViT-g Backbone (1B params)
                </h4>
                <p className={styles.modelItemDesc}>
                  Vision Transformer with 1B parameters trained on 22M videos.
                  Uses 3D-RoPE instead of absolute position embeddings.
                  Progressive training: 256px&#8594;384px, 16&#8594;64 frames.
                </p>
              </div>

              {/* Predictor */}
              <div
                className={`${styles.modelItem} ${activeModelItem === 'predictor' ? styles.modelItemActive : ''}`}
                onClick={() => setActiveModelItem(activeModelItem === 'predictor' ? null : 'predictor')}
              >
                <h4>
                  <span
                    className={styles.modelDot}
                    style={{ background: '#c4884d' }}
                  />
                  Autoregressive Predictor (V-JEPA 2-AC)
                </h4>
                <p className={styles.modelItemDesc}>
                  300M params, 24 transformer layers, 16 heads, 1024 dim.
                  Block-causal attention. Takes (features, actions, poses)
                  &#8594; predicts next features.
                </p>
              </div>

              {/* TF Loss */}
              <div
                className={`${styles.modelItem} ${activeModelItem === 'tfloss' ? styles.modelItemActive : ''}`}
                onClick={() => setActiveModelItem(activeModelItem === 'tfloss' ? null : 'tfloss')}
              >
                <h4>
                  <span
                    className={styles.modelDot}
                    style={{ background: '#10b981' }}
                  />
                  Teacher-Forcing Loss
                </h4>
                <div className={styles.formula}>
                  <MathJax inline>
                    {'\\( \\mathcal{L}_{\\text{TF}} = \\frac{1}{T}\\sum_{k=1}^{T}\\|\\hat{z}_k - z_k\\|_1 \\)'}
                  </MathJax>
                </div>
                <p className={styles.modelItemDesc}>
                  L1 between predicted and actual backbone features at each
                  timestep
                </p>
              </div>

              {/* Rollout Loss */}
              <div
                className={`${styles.modelItem} ${activeModelItem === 'rollout' ? styles.modelItemActive : ''}`}
                onClick={() => setActiveModelItem(activeModelItem === 'rollout' ? null : 'rollout')}
              >
                <h4>
                  <span
                    className={styles.modelDot}
                    style={{ background: '#8b7ec8' }}
                  />
                  Rollout Loss
                </h4>
                <div className={styles.formula}>
                  <MathJax inline>
                    {'\\( \\mathcal{L}_{\\text{rollout}} = \\|P(a_{1:T}; z_0) - z_{T+1}\\|_1 \\)'}
                  </MathJax>
                </div>
                <p className={styles.modelItemDesc}>
                  L1 error over multi-step autoregressive rollouts. Helps reduce
                  compounding errors.
                </p>
              </div>

              {/* Energy */}
              <div
                className={`${styles.modelItem} ${activeModelItem === 'energy' ? styles.modelItemActive : ''}`}
                onClick={() => setActiveModelItem(activeModelItem === 'energy' ? null : 'energy')}
              >
                <h4>
                  <span
                    className={styles.modelDot}
                    style={{ background: '#ef4444' }}
                  />
                  Energy-Based Planning
                </h4>
                <div className={styles.formula}>
                  <MathJax inline>
                    {'\\( E(a_{1:H}) = \\|P_{\\text{AC}}(z_0, s_0, a_{1:H}) - z_{\\text{goal}}\\|_1 \\)'}
                  </MathJax>
                </div>
                <p className={styles.modelItemDesc}>
                  Minimize over action sequences. No reward function needed.
                </p>
              </div>
            </div>
          </div>

          {/* ==================== Architecture SVG ==================== */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              V-JEPA 2 Architecture
            </div>
            <svg className={styles.pipelineSvg} viewBox="0 0 500 470">
              {/* Arrow marker defs */}
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
                  id="arrowBlue"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
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
                  id="arrowGray"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#888" />
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
              </defs>

              {/* Part 1: Pretraining */}
              <text
                x="250"
                y="15"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="10"
                fontWeight="bold"
              >
                V-JEPA 2 Pretraining
              </text>

              {/* Video */}
              <rect
                x="15"
                y="30"
                width="70"
                height="40"
                rx="6"
                fill="#3b82f6"
                className={`${styles.pipelineStage} ${activeComponent === 'video' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'video' ? null : 'video')}
              />
              <text
                x="50"
                y="48"
                textAnchor="middle"
                fill="white"
                fontSize="8"
                fontWeight="bold"
                pointerEvents="none"
              >
                Video
              </text>
              <text
                x="50"
                y="59"
                textAnchor="middle"
                fill="white"
                fontSize="7"
                pointerEvents="none"
              >
                22M videos
              </text>

              {/* Mask */}
              <rect
                x="100"
                y="30"
                width="50"
                height="40"
                rx="6"
                fill="rgba(239,68,68,0.3)"
                stroke="#ef4444"
                strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'mask' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'mask' ? null : 'mask')}
              />
              <text
                x="125"
                y="48"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="8"
                fontWeight="bold"
                pointerEvents="none"
              >
                Mask
              </text>
              <text
                x="125"
                y="59"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                90%
              </text>

              {/* x-encoder */}
              <rect
                x="165"
                y="25"
                width="90"
                height="50"
                rx="8"
                fill="rgba(0,217,255,0.2)"
                stroke="#00d9ff"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'xencoder' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'xencoder' ? null : 'xencoder')}
              />
              <text
                x="210"
                y="45"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                x-encoder
              </text>
              <text
                x="210"
                y="58"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                ViT-g (1B)
              </text>

              {/* Predictor */}
              <rect
                x="270"
                y="25"
                width="75"
                height="50"
                rx="8"
                fill="rgba(168,85,247,0.2)"
                stroke="#a855f7"
                strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'pretrain_pred' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'pretrain_pred' ? null : 'pretrain_pred')}
              />
              <text
                x="307"
                y="45"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                Predictor
              </text>
              <text
                x="307"
                y="58"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                features
              </text>

              {/* y-encoder (EMA) */}
              <rect
                x="165"
                y="90"
                width="90"
                height="40"
                rx="8"
                fill="rgba(245,158,11,0.2)"
                stroke="#f59e0b"
                strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'yencoder' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'yencoder' ? null : 'yencoder')}
              />
              <text
                x="210"
                y="107"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                y-encoder (EMA)
              </text>
              <text
                x="210"
                y="120"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                stop-grad targets
              </text>

              {/* L1 Loss */}
              <rect
                x="370"
                y="30"
                width="60"
                height="38"
                rx="8"
                fill="rgba(16,185,129,0.2)"
                stroke="#10b981"
                strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'pretrain_loss' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'pretrain_loss' ? null : 'pretrain_loss')}
              />
              <text
                x="400"
                y="48"
                textAnchor="middle"
                fill="#10b981"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                L1 Loss
              </text>
              <text
                x="400"
                y="58"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                252K iters
              </text>

              {/* Arrows for pretraining */}
              <path
                d="M 85 50 L 98 50"
                stroke="#3b82f6"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowBlue)"
              />
              <path
                d="M 150 50 L 163 50"
                stroke="#ef4444"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowRed)"
              />
              <path
                d="M 255 50 L 268 50"
                stroke="#a855f7"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowPurple)"
              />
              <path
                d="M 345 50 L 368 50"
                stroke="#10b981"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowGreen)"
                className={styles.flowArrow}
              />
              <path
                d="M 50 70 Q 50 110 163 110"
                stroke="#f59e0b"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowOrange)"
              />
              <path
                d="M 255 110 Q 340 110 390 70"
                stroke="#f59e0b"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowOrange)"
                strokeDasharray="4"
              />
              <text x="330" y="100" fill="#888" fontSize="7">
                stop-grad
              </text>

              {/* Scale labels */}
              <rect
                x="440"
                y="25"
                width="50"
                height="50"
                rx="6"
                fill="rgba(0,217,255,0.1)"
                stroke="#00d9ff"
                strokeWidth="1"
                strokeDasharray="3"
              />
              <text
                x="465"
                y="40"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="7"
              >
                1B params
              </text>
              <text
                x="465"
                y="52"
                textAnchor="middle"
                fill="#888"
                fontSize="6"
              >
                22M videos
              </text>
              <text
                x="465"
                y="64"
                textAnchor="middle"
                fill="#888"
                fontSize="6"
              >
                384px/64f
              </text>

              {/* Separator */}
              <line
                x1="15"
                y1="145"
                x2="485"
                y2="145"
                stroke="rgba(255,255,255,0.15)"
                strokeWidth="1"
                strokeDasharray="6"
              />

              {/* Part 2: V-JEPA 2-AC */}
              <text
                x="250"
                y="165"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="10"
                fontWeight="bold"
              >
                V-JEPA 2-AC (Action-Conditioned World Model)
              </text>

              {/* Frame 0 */}
              <rect
                x="15"
                y="180"
                width="55"
                height="35"
                rx="6"
                fill="rgba(59,130,246,0.3)"
                stroke="#3b82f6"
                strokeWidth="1.5"
              />
              <text
                x="42"
                y="195"
                textAnchor="middle"
                fill="#3b82f6"
                fontSize="7"
                fontWeight="bold"
              >
                Frame x_0
              </text>
              <text
                x="42"
                y="206"
                textAnchor="middle"
                fill="#888"
                fontSize="6"
              >
                current
              </text>

              {/* Frame 1 */}
              <rect
                x="80"
                y="180"
                width="55"
                height="35"
                rx="6"
                fill="rgba(59,130,246,0.3)"
                stroke="#3b82f6"
                strokeWidth="1.5"
              />
              <text
                x="107"
                y="195"
                textAnchor="middle"
                fill="#3b82f6"
                fontSize="7"
                fontWeight="bold"
              >
                Frame x_1
              </text>

              {/* Frame dots */}
              <text x="152" y="200" fill="#888" fontSize="12">
                ...
              </text>

              {/* Frozen Encoder */}
              <rect
                x="15"
                y="230"
                width="155"
                height="40"
                rx="8"
                fill="rgba(0,217,255,0.15)"
                stroke="#00d9ff"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'frozen_enc' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'frozen_enc' ? null : 'frozen_enc')}
              />
              <text
                x="92"
                y="248"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="9"
                fontWeight="bold"
                pointerEvents="none"
              >
                {'Frozen Encoder \u2744\uFE0F'}
              </text>
              <text
                x="92"
                y="261"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                ViT-g (1B) &mdash; pretrained
              </text>

              {/* Arrows frames to encoder */}
              <path
                d="M 42 215 L 42 228"
                stroke="#3b82f6"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowBlue)"
              />
              <path
                d="M 107 215 L 107 228"
                stroke="#3b82f6"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowBlue)"
              />

              {/* Features z_0, z_1 */}
              <rect
                x="15"
                y="285"
                width="55"
                height="28"
                rx="6"
                fill="rgba(0,217,255,0.2)"
                stroke="#00d9ff"
                strokeWidth="1"
              />
              <text
                x="42"
                y="303"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="8"
                fontWeight="bold"
              >
                z_0
              </text>

              <rect
                x="80"
                y="285"
                width="55"
                height="28"
                rx="6"
                fill="rgba(0,217,255,0.2)"
                stroke="#00d9ff"
                strokeWidth="1"
              />
              <text
                x="107"
                y="303"
                textAnchor="middle"
                fill="#00d9ff"
                fontSize="8"
                fontWeight="bold"
              >
                z_1
              </text>

              {/* Arrows encoder to features */}
              <path
                d="M 50 270 L 42 283"
                stroke="#00d9ff"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowCyan)"
              />
              <path
                d="M 100 270 L 107 283"
                stroke="#00d9ff"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowCyan)"
              />

              {/* Actions */}
              <rect
                x="15"
                y="330"
                width="55"
                height="25"
                rx="6"
                fill="rgba(16,185,129,0.2)"
                stroke="#10b981"
                strokeWidth="1"
              />
              <text
                x="42"
                y="346"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
                fontWeight="bold"
              >
                a_0, s_0
              </text>

              <rect
                x="80"
                y="330"
                width="55"
                height="25"
                rx="6"
                fill="rgba(16,185,129,0.2)"
                stroke="#10b981"
                strokeWidth="1"
              />
              <text
                x="107"
                y="346"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
                fontWeight="bold"
              >
                a_1, s_1
              </text>

              <text x="152" y="345" fill="#888" fontSize="12">
                ...
              </text>

              {/* Autoregressive Predictor */}
              <rect
                x="185"
                y="280"
                width="170"
                height="85"
                rx="10"
                fill="rgba(245,158,11,0.15)"
                stroke="#f59e0b"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'ac_predictor' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'ac_predictor' ? null : 'ac_predictor')}
              />
              <text
                x="270"
                y="305"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                Autoregressive
              </text>
              <text
                x="270"
                y="318"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="10"
                fontWeight="bold"
                pointerEvents="none"
              >
                Predictor
              </text>
              <text
                x="270"
                y="335"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                300M params &middot; 24 layers
              </text>
              <text
                x="270"
                y="347"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                Block-causal &middot; 3D-RoPE
              </text>
              <text
                x="270"
                y="359"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
                pointerEvents="none"
              >
                62 hours robot data
              </text>

              {/* Arrows: features & actions to predictor */}
              <path
                d="M 135 300 L 183 300"
                stroke="#00d9ff"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowCyan)"
                className={styles.flowArrow}
              />
              <path
                d="M 135 342 L 183 342"
                stroke="#10b981"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowGreen)"
                className={styles.flowArrow}
              />

              {/* Predicted output */}
              <rect
                x="375"
                y="295"
                width="60"
                height="35"
                rx="8"
                fill="rgba(245,158,11,0.3)"
                stroke="#f59e0b"
                strokeWidth="2"
              />
              <text
                x="405"
                y="312"
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="9"
                fontWeight="bold"
              >
                {'\u0302z_{t+1}'}
              </text>
              <text
                x="405"
                y="324"
                textAnchor="middle"
                fill="#888"
                fontSize="7"
              >
                predicted
              </text>

              <path
                d="M 355 315 L 373 315"
                stroke="#f59e0b"
                strokeWidth="2"
                fill="none"
                markerEnd="url(#arrowOrange)"
                className={styles.flowArrow}
              />

              {/* Goal image path */}
              <rect
                x="375"
                y="180"
                width="60"
                height="35"
                rx="6"
                fill="rgba(16,185,129,0.3)"
                stroke="#10b981"
                strokeWidth="1.5"
              />
              <text
                x="405"
                y="195"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
                fontWeight="bold"
              >
                Goal
              </text>
              <text
                x="405"
                y="206"
                textAnchor="middle"
                fill="#10b981"
                fontSize="7"
              >
                Image
              </text>

              <rect
                x="375"
                y="230"
                width="60"
                height="28"
                rx="6"
                fill="rgba(16,185,129,0.15)"
                stroke="#10b981"
                strokeWidth="1"
              />
              <text
                x="405"
                y="248"
                textAnchor="middle"
                fill="#10b981"
                fontSize="8"
                fontWeight="bold"
              >
                z_goal
              </text>

              <path
                d="M 405 215 L 405 228"
                stroke="#10b981"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowGreen)"
              />
              <text x="445" y="225" fill="#888" fontSize="6">
                Frozen
              </text>
              <text x="445" y="233" fill="#888" fontSize="6">
                {`Enc \u2744\uFE0F`}
              </text>

              {/* Energy */}
              <rect
                x="448"
                y="270"
                width="45"
                height="55"
                rx="8"
                fill="rgba(239,68,68,0.2)"
                stroke="#ef4444"
                strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'energy_plan' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'energy_plan' ? null : 'energy_plan')}
              />
              <text
                x="470"
                y="290"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="8"
                fontWeight="bold"
                pointerEvents="none"
              >
                Energy
              </text>
              <text
                x="470"
                y="303"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="7"
                pointerEvents="none"
              >
                {`||\u0302z - z_g||`}
              </text>
              <text
                x="470"
                y="315"
                textAnchor="middle"
                fill="#888"
                fontSize="6"
                pointerEvents="none"
              >
                minimize
              </text>

              <path
                d="M 405 258 L 460 268"
                stroke="#ef4444"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowRed)"
              />
              <path
                d="M 435 312 L 450 305"
                stroke="#ef4444"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowRed)"
              />

              {/* Planning loop arrow */}
              <path
                d="M 470 325 Q 470 370 270 370 Q 70 370 42 357"
                stroke="#ef4444"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#arrowRed)"
                strokeDasharray="6"
              />
              <text
                x="270"
                y="385"
                textAnchor="middle"
                fill="#ef4444"
                fontSize="7"
                fontWeight="bold"
              >
                Gradient-based action optimization (replanning)
              </text>

              {/* Training Info */}
              <rect
                x="15"
                y="400"
                width="478"
                height="28"
                rx="8"
                fill="rgba(168,85,247,0.1)"
                stroke="#a855f7"
                strokeWidth="1"
              />
              <text
                x="250"
                y="418"
                textAnchor="middle"
                fill="#a855f7"
                fontSize="9"
                fontWeight="bold"
              >
                {
                  'L_total = L_TF + \u03BB\u00B7L_rollout  |  Frozen backbone \u2014 only predictor trained  |  Goal-conditioned planning via energy min'
                }
              </text>
            </svg>

            {activeComponent && componentInfo[activeComponent] && (
              <div className={styles.infoPanel}>
                <h4>{componentInfo[activeComponent].title}</h4>
                <p>{componentInfo[activeComponent].desc}</p>
              </div>
            )}
          </div>
        </div>

        {/* ==================== Planning + Benchmarks ==================== */}
        <div className={styles.mainGrid}>
          {/* Planning Canvas */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Planning via Energy Minimization
            </div>
            <div className={styles.sliderContainer}>
              <label>Iteration:</label>
              <input
                type="range"
                className={styles.slider}
                min={1}
                max={30}
                step={1}
                value={iteration}
                onChange={(e) => setIteration(parseInt(e.target.value))}
              />
              <span className={styles.sliderValue}>{iteration}</span>
            </div>
            <div className={styles.sliderContainer}>
              <label>Horizon H:</label>
              <input
                type="range"
                className={styles.slider}
                min={4}
                max={20}
                step={1}
                value={horizon}
                onChange={(e) => {
                  cachedHorizonRef.current = null;
                  setHorizon(parseInt(e.target.value));
                }}
              />
              <span className={styles.sliderValue}>{horizon}</span>
            </div>
            <div className={styles.canvasContainer}>
              <canvas ref={planningCanvasRef} className={styles.planningCanvas} />
            </div>
            <div className={styles.metricsRow}>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: '#ef4444' }}>
                  {energy.toFixed(1)}
                </div>
                <div className={styles.metricLabel}>Energy</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: 'var(--accent-primary, #638bd4)' }}>
                  {iteration}
                </div>
                <div className={styles.metricLabel}>Iteration</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: 'var(--accent-secondary, #8b7ec8)' }}>
                  {trajLen.toFixed(1)}
                </div>
                <div className={styles.metricLabel}>Trajectory Length</div>
              </div>
            </div>
          </div>

          {/* Benchmark Results */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Benchmark Results
            </div>
            <div className={styles.comparisonContainer}>
              <div className={styles.toggleBtns}>
                <button
                  className={`${styles.toggleBtn} ${activeBench === 'scaling' ? styles.toggleBtnActive : ''}`}
                  onClick={() => setActiveBench('scaling')}
                >
                  Scaling Ingredients
                </button>
                <button
                  className={`${styles.toggleBtn} ${activeBench === 'comparison' ? styles.toggleBtnActive : ''}`}
                  onClick={() => setActiveBench('comparison')}
                >
                  Model Comparison
                </button>
              </div>
              <div className={styles.benchInfo}>{bench.info}</div>
              <div>
                {benchEntries.map(([method, score], idx) => {
                  const pct = (Math.max(score, 0) / bench.maxVal) * 100;
                  const color = bench.colors[method] || '#888';
                  const top = isTopEntry(activeBench, method, idx);
                  return (
                    <div className={styles.comparisonRow} key={method}>
                      <span
                        className={styles.comparisonLabel}
                        style={{
                          color,
                          fontWeight: top ? 700 : 400,
                        }}
                      >
                        {method}
                      </span>
                      <div className={styles.comparisonBarTrack}>
                        <div
                          className={styles.comparisonBarFill}
                          style={{
                            width: `${pct}%`,
                            background: color,
                            opacity: top ? 1 : 0.7,
                          }}
                        />
                      </div>
                      <span
                        className={styles.comparisonValue}
                        style={{ color }}
                      >
                        {score}%
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className={styles.benchNote}>
                Average accuracy across 6 video understanding benchmarks.
                Scaling all axes yields +4.0 points total improvement.
              </p>
            </div>
          </div>
        </div>

        {/* ==================== Training Objectives ==================== */}
        <div className={`${styles.card} ${styles.cardMb}`}>
          <div className={styles.cardTitle}>
            Training Objectives
          </div>
          <div className={styles.trainingObjectives}>
            {/* Pretrain */}
            <div
              className={`${styles.modelItem} ${activeLoss === 'pretrain' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLoss(activeLoss === 'pretrain' ? null : 'pretrain')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#638bd4' }}
                />
                Pretraining (V-JEPA Objective)
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '$$\\mathcal{L}_{\\text{pretrain}} = \\|P_\\phi(\\Delta_y, E_\\theta(x)) - \\text{sg}(\\bar{E}_{\\bar\\theta}(y))\\|_1$$'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                Predict masked video features from visible context. The target
                encoder is an EMA of the online encoder with stop-gradient.
              </p>
            </div>

            {/* Teacher-Forcing */}
            <div
              className={`${styles.modelItem} ${activeLoss === 'tf' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLoss(activeLoss === 'tf' ? null : 'tf')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#10b981' }}
                />
                Teacher-Forcing Loss
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '$$\\mathcal{L}_{\\text{TF}}(\\phi) = \\frac{1}{T}\\sum_{k=1}^{T}\\left\\|P_\\phi\\big((a_t, s_t, E(x_l))_{l\\leq k}\\big) - E(x_{k+1})\\right\\|_1$$'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                At each timestep, predict the next frame&apos;s features given
                all preceding frames, actions, and end-effector states.
              </p>
            </div>

            {/* Rollout */}
            <div
              className={`${styles.modelItem} ${activeLoss === 'rollout' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLoss(activeLoss === 'rollout' ? null : 'rollout')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#8b7ec8' }}
                />
                Rollout Loss
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '$$\\mathcal{L}_{\\text{rollout}}(\\phi) = \\|P_\\phi(a_{1:T}; s_1, z_1) - z_{T+1}\\|_1$$'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                Multi-step autoregressive prediction loss. Reduces compounding
                errors during long-horizon rollouts.
              </p>
            </div>

            {/* Planning Energy */}
            <div
              className={`${styles.modelItem} ${activeLoss === 'planning' ? styles.modelItemActive : ''}`}
              onClick={() => setActiveLoss(activeLoss === 'planning' ? null : 'planning')}
            >
              <h4>
                <span
                  className={styles.modelDot}
                  style={{ background: '#ef4444' }}
                />
                Planning Energy
              </h4>
              <div className={styles.formula}>
                <MathJax>
                  {
                    '$$a_{1:H}^* = \\arg\\min_{a_{1:H}} \\|P_{\\text{AC}}(z_0, s_0, a_{1:H}) - z_{\\text{goal}}\\|_1$$'
                  }
                </MathJax>
              </div>
              <p className={styles.modelItemDesc}>
                Find the action sequence that minimizes the distance between
                predicted future and goal in representation space. Only the
                first action is executed, then replanning occurs.
              </p>
            </div>
          </div>
        </div>

        {/* ==================== Paper Lineage ==================== */}
        <div className={`${styles.card} ${styles.cardMb}`}>
          <div className={styles.cardTitle}>
            Paper Lineage
          </div>
          <div className={styles.lineageGrid}>
            <div>
              <h4
                className={styles.lineageSectionTitle}
                style={{ color: '#c4884d' }}
              >
                Builds On
              </h4>
              <div className={styles.lineageItems}>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#c4884d' }}
                >
                  <h5>V-JEPA</h5>
                  <p>
                    Bardes et al., 2024 &mdash; Original self-supervised video
                    prediction in latent space. V-JEPA 2 scales the data, model,
                    and training far beyond the original.
                  </p>
                </div>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#c4884d' }}
                >
                  <h5>Scaling ViT</h5>
                  <p>
                    Zhai et al., 2022 &mdash; Foundational work on scaling
                    Vision Transformers. V-JEPA 2 uses ViT-g (1B params) with
                    3D-RoPE.
                  </p>
                </div>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#c4884d' }}
                >
                  <h5>Droid Dataset</h5>
                  <p>
                    Khazatsky et al., 2024 &mdash; 62 hours of robot
                    manipulation data (7-DOF Franka Panda) used to train V-JEPA
                    2-AC.
                  </p>
                </div>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#c4884d' }}
                >
                  <h5>I-JEPA</h5>
                  <p>
                    Assran et al., 2023 &mdash; Image-level JEPA that inspired
                    the video extension. Predicts image features from context.
                  </p>
                </div>
                <div
                  className={styles.lineageItem}
                  style={{ borderLeftColor: '#c4884d' }}
                >
                  <h5>RoPE</h5>
                  <p>
                    Su et al., 2024 &mdash; Rotary Position Embedding extended
                    to 3D (spatial + temporal) for video transformers.
                  </p>
                </div>
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
                  <h5>Frontier Paper</h5>
                  <p>
                    V-JEPA 2 is a very recent contribution (2025). Follow-up
                    work building on this foundation is expected in robotics,
                    video understanding, and world models.
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
