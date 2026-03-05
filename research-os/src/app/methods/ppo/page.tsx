'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import styles from './ppo.module.css';

import type { CoreIdea, KeyFeature } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';

const coreIdeas: CoreIdea[] = [
  { title: 'Clipped Surrogate Objective', desc: 'Bound policy updates with simple ratio clipping instead of complex trust region constraints, preventing destructively large policy changes' },
  { title: 'Multi-Epoch Mini-Batch Updates', desc: 'Reuse collected on-policy data for K epochs of mini-batch SGD, balancing sample efficiency with on-policy stability' },
  { title: 'Generalized Advantage Estimation', desc: 'Use \u03BB-weighted TD residuals to balance bias and variance in advantage estimates, controlling the effective lookahead horizon' },
];

const keyFeatures: KeyFeature[] = [
  { title: 'Clipped Surrogate', desc: 'Simple clipping replaces TRPO\u2019s complex trust region constraint, bounding policy updates without second-order optimization' },
  { title: 'On-Policy + Multi-Epoch', desc: 'Collects fresh data each iteration but reuses it for K epochs of mini-batch SGD, balancing stability and sample efficiency' },
  { title: 'Universal Applicability', desc: 'Works for discrete and continuous actions, single and multi-agent settings, games to robotics to LLM alignment' },
];

/* ------------------------------------------------------------------ */
/*  Static Data                                                        */
/* ------------------------------------------------------------------ */

const componentInfo: Record<string, { title: string; desc: string }> = {
  env: {
    title: 'Environment',
    desc: 'The agent collects trajectories by running the current policy in the environment. PPO is on-policy: data is collected fresh each iteration and discarded after the update.',
  },
  rolloutBuffer: {
    title: 'Rollout Buffer',
    desc: 'Stores on-policy trajectories (s, a, r, log\u03C0, V(s)) from the current policy. Unlike a replay buffer, this data is used once for K epochs of mini-batch updates and then discarded.',
  },
  actor: {
    title: 'Actor (Policy \u03C0_\u03B8)',
    desc: 'Outputs action probabilities (discrete) or Gaussian parameters (continuous). Shared or separate network from the critic. Updated via the clipped surrogate objective to prevent large policy changes.',
  },
  critic: {
    title: 'Critic V_\u03C6',
    desc: 'State value function estimating V(s). Provides baseline for advantage estimation. Can share parameters with the actor (shared trunk) or be a separate network. Updated via MSE on returns.',
  },
  gae: {
    title: 'GAE (Generalized Advantage Estimation)',
    desc: 'Computes advantage \u00C2_t = \u03A3_l (\u03B3\u03BB)^l \u03B4_{t+l} where \u03B4_t = r_t + \u03B3V(s_{t+1}) - V(s_t). Interpolates between high-bias (TD, \u03BB=0) and high-variance (MC, \u03BB=1) estimates.',
  },
  clipping: {
    title: 'Clipping Mechanism',
    desc: 'The core PPO innovation. Clips the probability ratio r(\u03B8) = \u03C0_\u03B8(a|s)/\u03C0_old(a|s) to [1-\u03B5, 1+\u03B5], preventing destructive large updates. This creates a trust region without expensive second-order optimization.',
  },
};

const stepDetailsExpanded: Record<number, string> = {
  1: 'Run the current policy \u03C0_old in the environment for T timesteps (or N parallel environments for T/N steps each). Store (s_t, a_t, r_t, log \u03C0_old(a_t|s_t), V(s_t)) in the rollout buffer.',
  2: 'Compute TD residuals \u03B4_t = r_t + \u03B3V(s_{t+1}) - V(s_t), then compute GAE advantages: \u00C2_t = \u03A3_{l=0}^{T-t} (\u03B3\u03BB)^l \u03B4_{t+l}. Returns are \u0052_t = \u00C2_t + V(s_t).',
  3: 'For K epochs, shuffle the rollout buffer into mini-batches. Compute ratio r_t(\u03B8) = \u03C0_\u03B8(a_t|s_t) / \u03C0_old(a_t|s_t). Apply clipped surrogate: L = min(r\u00B7\u00C2, clip(r, 1\u00B1\u03B5)\u00B7\u00C2).',
  4: 'Update critic by minimizing MSE between V_\u03C6(s) and computed returns \u0052_t. Optionally clip the value function update as well for stability.',
  5: 'Optionally add entropy bonus H(\u03C0) to the actor loss to encourage exploration. The total loss combines actor loss, critic loss, and entropy: L = L_clip - c_1\u00B7L_VF + c_2\u00B7H.',
  6: 'Discard the rollout buffer and repeat. The old policy \u03C0_old is now the updated policy \u03C0_\u03B8. Each iteration collects new data with the latest policy (on-policy).',
};

interface StepData {
  number: number;
  title: string;
  defaultDesc: string;
}

const steps: StepData[] = [
  { number: 1, title: 'Collect Rollouts', defaultDesc: 'Run \u03C0_old in environment, store (s, a, r, log\u03C0, V)' },
  { number: 2, title: 'Compute Advantages (GAE)', defaultDesc: '\u00C2_t via TD residuals and exponential weighting (\u03B3\u03BB)' },
  { number: 3, title: 'Update Actor (Clipped Surrogate)', defaultDesc: 'K epochs of mini-batch updates with ratio clipping' },
  { number: 4, title: 'Update Critic', defaultDesc: 'MSE between V(s) and computed returns' },
  { number: 5, title: 'Entropy Bonus', defaultDesc: 'Optional exploration incentive H(\u03C0)' },
  { number: 6, title: 'Discard Buffer & Repeat', defaultDesc: 'New rollouts with updated policy (on-policy)' },
];

/* ------------------------------------------------------------------ */
/*  Clipping visualization helper                                      */
/* ------------------------------------------------------------------ */

function drawClipping(
  canvas: HTMLCanvasElement,
  epsilon: number,
  advantage: number,
  width: number,
  height: number,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx || width === 0 || height === 0) return;

  ctx.clearRect(0, 0, width, height);

  const padding = { left: 50, right: 20, top: 25, bottom: 35 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Axes
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#888';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('r(\u03B8) = \u03C0_\u03B8 / \u03C0_old', width / 2, height - 8);

  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('L_clip', 0, 0);
  ctx.restore();

  const xMin = 0;
  const xMax = 2.5;
  const yMin = -2;
  const yMax = 2;

  function toCanvasX(x: number) {
    return padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  }
  function toCanvasY(y: number) {
    return height - padding.bottom - ((y - yMin) / (yMax - yMin)) * plotHeight;
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.fillStyle = '#555';
  ctx.font = '10px sans-serif';

  for (let x = 0; x <= 2.5; x += 0.5) {
    const cx = toCanvasX(x);
    ctx.beginPath();
    ctx.moveTo(cx, padding.top);
    ctx.lineTo(cx, height - padding.bottom);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(x.toFixed(1), cx, height - padding.bottom + 12);
  }

  for (let y = -2; y <= 2; y += 1) {
    const cy = toCanvasY(y);
    ctx.beginPath();
    ctx.moveTo(padding.left, cy);
    ctx.lineTo(width - padding.right, cy);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(y.toString(), padding.left - 8, cy + 4);
  }

  // Zero line
  const zeroY = toCanvasY(0);
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);
  ctx.lineTo(width - padding.right, zeroY);
  ctx.stroke();

  // Clipping boundaries
  const clipLow = 1 - epsilon;
  const clipHigh = 1 + epsilon;

  ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(toCanvasX(clipLow), padding.top);
  ctx.lineTo(toCanvasX(clipLow), height - padding.bottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toCanvasX(clipHigh), padding.top);
  ctx.lineTo(toCanvasX(clipHigh), height - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels for clip boundaries
  ctx.fillStyle = '#ef4444';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`1-\u03B5=${clipLow.toFixed(2)}`, toCanvasX(clipLow), padding.top - 5);
  ctx.fillText(`1+\u03B5=${clipHigh.toFixed(2)}`, toCanvasX(clipHigh), padding.top - 5);

  // Unclipped objective: r * A (straight line through origin with slope A)
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  for (let i = 0; i <= plotWidth; i++) {
    const r = xMin + (i / plotWidth) * (xMax - xMin);
    const L = r * advantage;
    const cy = toCanvasY(Math.max(yMin, Math.min(yMax, L)));
    if (i === 0) ctx.moveTo(toCanvasX(r), cy);
    else ctx.lineTo(toCanvasX(r), cy);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Clipped objective: min(r*A, clip(r, 1-e, 1+e)*A)
  ctx.beginPath();
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 3;
  for (let i = 0; i <= plotWidth; i++) {
    const r = xMin + (i / plotWidth) * (xMax - xMin);
    const unclipped = r * advantage;
    const clippedR = Math.max(clipLow, Math.min(clipHigh, r));
    const clipped = clippedR * advantage;
    const L = Math.min(unclipped, clipped);
    const cy = toCanvasY(Math.max(yMin, Math.min(yMax, L)));
    if (i === 0) ctx.moveTo(toCanvasX(r), cy);
    else ctx.lineTo(toCanvasX(r), cy);
  }
  ctx.stroke();

  // Clip region fill
  ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
  ctx.fillRect(toCanvasX(clipLow), padding.top, toCanvasX(clipHigh) - toCanvasX(clipLow), plotHeight);

  // r=1 marker
  ctx.strokeStyle = 'rgba(0, 217, 255, 0.5)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(toCanvasX(1), padding.top);
  ctx.lineTo(toCanvasX(1), height - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#00d9ff';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('r=1', toCanvasX(1), height - padding.bottom + 24);

  // Legend
  ctx.font = '10px sans-serif';

  ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(width - 150, padding.top + 10);
  ctx.lineTo(width - 125, padding.top + 10);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#a855f7';
  ctx.textAlign = 'left';
  ctx.fillText('r \u00B7 \u00C2 (unclipped)', width - 120, padding.top + 14);

  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(width - 150, padding.top + 28);
  ctx.lineTo(width - 125, padding.top + 28);
  ctx.stroke();
  ctx.fillStyle = '#10b981';
  ctx.fillText('L_clip (PPO)', width - 120, padding.top + 32);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function PPOPage() {
  /* ---- Architecture interaction ---- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  /* ---- Algorithm steps ---- */
  const [activeStep, setActiveStep] = useState<number | null>(null);

  /* ---- Loss items ---- */
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  /* ---- Clipping visualization ---- */
  const [epsilon, setEpsilon] = useState(0.2);
  const [advantage, setAdvantage] = useState(1.0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = useCanvasResize(canvasRef);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawClipping(canvas, epsilon, advantage, canvasSize.width, canvasSize.height);
  }, [epsilon, advantage, canvasSize]);

  /* ---- GAE lambda visualization ---- */
  const [lambda, setLambda] = useState(0.95);
  const gaeCanvasRef = useRef<HTMLCanvasElement>(null);
  const gaeCanvasSize = useCanvasResize(gaeCanvasRef);

  const drawGAE = useCallback(() => {
    const canvas = gaeCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = gaeCanvasSize.width;
    const height = gaeCanvasSize.height;
    if (width === 0 || height === 0) return;

    ctx.clearRect(0, 0, width, height);

    const padding = { left: 50, right: 20, top: 20, bottom: 35 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Lookahead steps (l)', width / 2, height - 8);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Weight (\u03B3\u03BB)^l', 0, 0);
    ctx.restore();

    const nSteps = 20;
    const gamma = 0.99;
    const barWidth = plotWidth / (nSteps + 1);

    for (let l = 0; l < nSteps; l++) {
      const weight = Math.pow(gamma * lambda, l);
      const barHeight = weight * plotHeight * 0.9;
      const x = padding.left + (l + 0.5) * barWidth;
      const y = height - padding.bottom - barHeight;

      const hue = 168 - l * 6;
      ctx.fillStyle = `hsl(${Math.max(0, hue)}, 70%, 50%)`;
      ctx.fillRect(x, y, barWidth * 0.7, barHeight);

      if (l < 10 || l % 5 === 0) {
        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(l.toString(), x + barWidth * 0.35, height - padding.bottom + 12);
      }
    }

    // Effective horizon label
    const effectiveHorizon = 1 / (1 - gamma * lambda);
    ctx.fillStyle = '#00d9ff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`Eff. horizon \u2248 ${effectiveHorizon.toFixed(1)} steps`, width - padding.right, padding.top + 15);
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.fillText(`\u03BB = ${lambda.toFixed(2)}, \u03B3 = ${gamma.toFixed(2)}`, width - padding.right, padding.top + 30);
  }, [gaeCanvasSize, lambda]);

  useEffect(() => {
    drawGAE();
  }, [drawGAE]);

  /* ---- Render ---- */
  return (
    <div className="method-page">
      <h1>Proximal Policy Optimization (PPO)</h1>
      <p className="subtitle">
        Clipped Surrogate Policy Gradient &mdash; Schulman et al., 2017
      </p>
      <div className={styles.githubLink}>
        <a
          href="https://arxiv.org/abs/1707.06347"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span style={{ marginRight: 5 }}>&rarr;</span> arxiv.org/abs/1707.06347
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ============ Architecture Diagram ============ */}
      <section>
        <h2>Network Architecture</h2>
        <div className="diagram-frame">
          <svg className={styles.architectureSvg} viewBox="0 0 500 420">
            {/* Defs (arrow markers) */}
            <defs>
              <marker id="ppoArrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
              </marker>
              <marker id="ppoArrowPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
              </marker>
              <marker id="ppoArrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
              </marker>
              <marker id="ppoArrowBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
              </marker>
              <marker id="ppoArrowCyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff" />
              </marker>
              <marker id="ppoArrowRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
              </marker>
            </defs>

            {/* Environment */}
            <rect
              x="175" y="10" width="150" height="55" rx="10"
              fill="#10b981"
              className={`${styles.networkBox}${activeComponent === 'env' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('env')}
            />
            <text x="250" y="35" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold" pointerEvents="none">
              Environment
            </text>
            <text x="250" y="52" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              s_t, r_t, done
            </text>

            {/* Rollout Buffer */}
            <rect
              x="350" y="100" width="130" height="65" rx="10"
              fill="#f59e0b"
              className={`${styles.networkBox}${activeComponent === 'rolloutBuffer' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('rolloutBuffer')}
            />
            <text x="415" y="125" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Rollout Buffer
            </text>
            <text x="415" y="143" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              (s, a, r, log{'\u03C0'}, V)
            </text>

            {/* Actor (Policy) */}
            <rect
              x="20" y="180" width="140" height="75" rx="10"
              fill="#a855f7"
              className={`${styles.networkBox}${activeComponent === 'actor' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('actor')}
            />
            <text x="90" y="205" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Actor {'\u03C0'}_{'{\u03B8}'}
            </text>
            <text x="90" y="225" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              Policy Network
            </text>
            <text x="90" y="242" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              {'\u03C0'}(a|s)
            </text>

            {/* Critic */}
            <rect
              x="200" y="180" width="140" height="75" rx="10"
              fill="#3b82f6"
              className={`${styles.networkBox}${activeComponent === 'critic' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('critic')}
            />
            <text x="270" y="205" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Critic V_{'{\u03C6}'}
            </text>
            <text x="270" y="225" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              Value Network
            </text>
            <text x="270" y="242" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              V(s)
            </text>

            {/* GAE box */}
            <rect
              x="180" y="300" width="140" height="55" rx="10"
              fill="rgba(0, 217, 255, 0.15)" stroke="#00d9ff" strokeWidth="2"
              className={`${styles.networkBox}${activeComponent === 'gae' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('gae')}
            />
            <text x="250" y="323" textAnchor="middle" fill="#00d9ff" fontSize="11" fontWeight="bold" pointerEvents="none">
              GAE
            </text>
            <text x="250" y="343" textAnchor="middle" fill="#00d9ff" fontSize="9" pointerEvents="none">
              {'\u00C2'}_t = {'\u03A3'} ({'\u03B3\u03BB'})^l {'\u03B4'}_{'{t+l}'}
            </text>

            {/* Clipping box */}
            <rect
              x="20" y="300" width="130" height="55" rx="10"
              fill="rgba(239, 68, 68, 0.15)" stroke="#ef4444" strokeWidth="2"
              className={`${styles.networkBox}${activeComponent === 'clipping' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('clipping')}
            />
            <text x="85" y="323" textAnchor="middle" fill="#ef4444" fontSize="10" fontWeight="bold" pointerEvents="none">
              Clipped Surrogate
            </text>
            <text x="85" y="343" textAnchor="middle" fill="#ef4444" fontSize="9" pointerEvents="none">
              clip(r, 1{'\u00B1\u03B5'}) {'\u00B7'} {'\u00C2'}
            </text>

            {/* ---- Flow Arrows ---- */}

            {/* Actor -> Environment (action) */}
            <path
              d="M 90 180 Q 90 100 175 37"
              stroke="#a855f7" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowPurple)"
              className={styles.flowArrow}
            />
            <text x="95" y="120" fill="#a855f7" fontSize="9">action a</text>

            {/* Environment -> Rollout Buffer */}
            <path
              d="M 325 37 Q 430 37 415 95"
              stroke="#10b981" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowGreen)"
              className={styles.flowArrow}
            />
            <text x="390" y="60" fill="#10b981" fontSize="9">trajectory</text>

            {/* Rollout Buffer -> Actor (mini-batches) */}
            <path
              d="M 350 132 Q 220 140 160 185"
              stroke="#f59e0b" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowOrange)"
            />

            {/* Rollout Buffer -> Critic (mini-batches) */}
            <path
              d="M 350 145 L 340 195"
              stroke="#f59e0b" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowOrange)"
            />
            <text x="360" y="178" fill="#f59e0b" fontSize="8">mini-batches</text>

            {/* Critic -> GAE */}
            <path
              d="M 270 255 L 260 295"
              stroke="#3b82f6" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowBlue)"
            />
            <text x="280" y="280" fill="#3b82f6" fontSize="8">V(s)</text>

            {/* GAE -> Clipping */}
            <path
              d="M 180 327 L 155 327"
              stroke="#00d9ff" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowCyan)"
            />
            <text x="165" y="320" fill="#00d9ff" fontSize="8">{'\u00C2'}_t</text>

            {/* Clipping -> Actor (gradient) */}
            <path
              d="M 85 300 L 85 260"
              stroke="#ef4444" strokeWidth="2" fill="none"
              markerEnd="url(#ppoArrowRed)"
              strokeDasharray="4"
            />
            <text x="55" y="283" fill="#ef4444" fontSize="8">{'\u2207\u03B8'}</text>

            {/* Environment -> Critic (states for V estimation) */}
            <path
              d="M 250 65 L 265 175"
              stroke="#10b981" strokeWidth="1.5" fill="none"
              markerEnd="url(#ppoArrowGreen)"
              strokeDasharray="4"
            />

            {/* Shared trunk indicator */}
            <rect x="80" y="265" width="230" height="20" rx="6" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.15)" strokeDasharray="4" />
            <text x="195" y="279" textAnchor="middle" fill="#666" fontSize="8" pointerEvents="none">
              optionally shared trunk
            </text>

            {/* Parallel envs indicator */}
            <rect x="335" y="15" width="60" height="20" rx="6" fill="rgba(16,185,129,0.2)" />
            <text x="365" y="29" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">
              N envs
            </text>

            {/* On-policy label */}
            <rect x="370" y="380" width="120" height="25" rx="6" fill="rgba(239,68,68,0.15)" />
            <text x="430" y="397" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="bold" pointerEvents="none">
              On-Policy: discard {'\u2192'} recollect
            </text>
          </svg>
        </div>

        {/* Component Info Panel */}
        {activeComponent && componentInfo[activeComponent] && (
          <div className={styles.infoPanel}>
            <h4>{componentInfo[activeComponent].title}</h4>
            <p>{componentInfo[activeComponent].desc}</p>
          </div>
        )}
      </section>

      {/* ============ Algorithm Steps ============ */}
      <section>
        <h2>Algorithm Steps</h2>
        <p className={styles.stepsHint}>
          One PPO iteration. Repeat until convergence. Data is collected and discarded each iteration (on-policy).
        </p>
        <ol className="algo-steps">
          {steps.map((s) => (
            <li
              key={s.number}
              className={activeStep === s.number ? 'active' : ''}
              onClick={() => setActiveStep(activeStep === s.number ? null : s.number)}
              style={{ cursor: 'pointer' }}
            >
              <strong>{s.title}</strong>
              <br />
              {activeStep === s.number
                ? stepDetailsExpanded[s.number]
                : s.defaultDesc}
            </li>
          ))}
        </ol>
      </section>

      {/* ============ Loss Functions ============ */}
      <section>
        <h2>Loss Functions</h2>
        <div className={styles.lossEquations}>
          {/* Clipped Surrogate */}
          <div
            className={`${styles.lossItem}${activeLoss === 'clip' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'clip' ? null : 'clip')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#10b981' }} />
              Clipped Surrogate Objective
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( L^{\\text{CLIP}}(\\theta) = \\mathbb{E}_t \\left[ \\min \\left( r_t(\\theta) \\hat{A}_t,\\; \\text{clip}(r_t(\\theta),\\, 1-\\varepsilon,\\, 1+\\varepsilon) \\hat{A}_t \\right) \\right] \\)'}
              </MathJax>
            </div>
            <p className={styles.lossHint}>
              where r_t({'\u03B8'}) = {'\u03C0_\u03B8'}(a_t|s_t) / {'\u03C0_{old}'}(a_t|s_t) is the probability ratio
            </p>
          </div>

          {/* Value Loss */}
          <div
            className={`${styles.lossItem}${activeLoss === 'value' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'value' ? null : 'value')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#3b82f6' }} />
              Value Function Loss
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( L^{\\text{VF}}(\\phi) = \\mathbb{E}_t \\left[ \\left( V_\\phi(s_t) - \\hat{R}_t \\right)^2 \\right] \\)'}
              </MathJax>
            </div>
          </div>

          {/* Combined PPO Loss */}
          <div
            className={`${styles.lossItem}${activeLoss === 'combined' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'combined' ? null : 'combined')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#a855f7' }} />
              Combined Objective
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( L(\\theta, \\phi) = L^{\\text{CLIP}} - c_1 \\cdot L^{\\text{VF}} + c_2 \\cdot H[\\pi_\\theta] \\)'}
              </MathJax>
            </div>
            <p className={styles.lossHint}>
              c_1 = 0.5 (value coef), c_2 = 0.01 (entropy coef) are typical defaults
            </p>
          </div>
        </div>

        {/* Notation Guide */}
        <div className={styles.notationPanel}>
          <h4>Notation Guide</h4>
          <p>
            <strong style={{ color: '#10b981' }}>r_t({'\u03B8'})</strong> &mdash;
            importance sampling ratio:{' '}
            <MathJax inline>{'\\(\\pi_\\theta(a_t|s_t) / \\pi_{\\text{old}}(a_t|s_t)\\)'}</MathJax>
          </p>
          <p>
            <strong style={{ color: '#00d9ff' }}>{'\u00C2'}_t</strong> &mdash;
            GAE advantage estimate, computed from TD residuals weighted by{' '}
            <MathJax inline>{'\\((\\gamma\\lambda)^l\\)'}</MathJax>
          </p>
          <p style={{ fontSize: '0.85rem', color: '#888' }}>
            The clipping ensures |r({'\u03B8'}) - 1| {'\u2264 \u03B5'}, which bounds
            the KL divergence between old and new policies without computing it explicitly.
          </p>
        </div>
      </section>

      {/* ============ Clipping Visualization ============ */}
      <section>
        <h2>Clipping Visualization</h2>
        <div className={styles.entropyViz}>
          <div className={styles.entropySliderContainer}>
            <label>{'\u03B5'} (clip range):</label>
            <input
              type="range"
              className={styles.entropySlider}
              min="0.05"
              max="0.5"
              step="0.01"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            />
            <span className={styles.entropyValue}>{epsilon.toFixed(2)}</span>
          </div>
          <div className={styles.entropySliderContainer}>
            <label>{'\u00C2'} (advantage):</label>
            <input
              type="range"
              className={styles.entropySlider}
              min="-2"
              max="2"
              step="0.1"
              value={advantage}
              onChange={(e) => setAdvantage(parseFloat(e.target.value))}
            />
            <span className={styles.entropyValue} style={{ color: advantage >= 0 ? '#10b981' : '#ef4444' }}>
              {advantage >= 0 ? '+' : ''}{advantage.toFixed(1)}
            </span>
          </div>
          <canvas ref={canvasRef} className={styles.distributionCanvas} />
          <p className={styles.entropyHint}>
            {advantage >= 0
              ? 'Positive advantage: clipping caps how much the policy can increase the probability of good actions.'
              : 'Negative advantage: clipping caps how much the policy can decrease the probability of bad actions.'}
            <br />
            The flat regions prevent destructively large policy updates.
          </p>
        </div>
      </section>

      {/* ============ GAE Visualization ============ */}
      <section>
        <h2>GAE: Bias-Variance Tradeoff</h2>
        <div className={styles.entropyViz}>
          <div className={styles.entropySliderContainer}>
            <label>{'\u03BB'} (GAE lambda):</label>
            <input
              type="range"
              className={styles.entropySlider}
              min="0"
              max="1"
              step="0.01"
              value={lambda}
              onChange={(e) => setLambda(parseFloat(e.target.value))}
            />
            <span className={styles.entropyValue}>{lambda.toFixed(2)}</span>
          </div>
          <canvas ref={gaeCanvasRef} className={styles.distributionCanvas} />
          <p className={styles.entropyHint}>
            {lambda < 0.3
              ? '\u03BB \u2248 0: TD-like. Low variance but high bias \u2014 only looks one step ahead.'
              : lambda < 0.7
              ? '\u03BB \u2248 0.5: Moderate tradeoff between bias and variance.'
              : lambda < 0.95
              ? '\u03BB \u2248 0.9: Standard setting. Good balance for most environments.'
              : '\u03BB \u2265 0.95: Near Monte Carlo. Low bias but high variance \u2014 uses many future steps.'}
          </p>
        </div>
      </section>

      {/* ============ Paper Lineage ============ */}
      <section>
        <h2>Paper Lineage</h2>
        <div className={styles.lineageGrid}>
          {/* Builds On */}
          <div>
            <h4 className={styles.lineageSectionTitle} style={{ color: '#f59e0b' }}>
              Builds On
            </h4>
            <div className={styles.lineageItems}>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>REINFORCE</h5>
                <p>
                  Williams, 1992 &mdash; Classic policy gradient algorithm; high variance
                  Monte Carlo estimates that PPO stabilizes via clipping and baselines
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>A3C / A2C</h5>
                <p>
                  Mnih et al., 2016 &mdash; Parallel actor-critic; PPO inherits the
                  actor-critic framework and parallel environment design
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>TRPO</h5>
                <p>
                  Schulman et al., 2015 &mdash; Trust Region Policy Optimization; PPO&apos;s
                  direct predecessor. PPO replaces TRPO&apos;s expensive conjugate gradient
                  constraint with simple clipping
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>GAE</h5>
                <p>
                  Schulman et al., 2016 &mdash; Generalized Advantage Estimation; the
                  {'\u03BB'}-weighted advantage computation used in PPO
                </p>
              </div>
            </div>
          </div>

          {/* Built Upon By */}
          <div>
            <h4 className={styles.lineageSectionTitle} style={{ color: '#10b981' }}>
              Built Upon By
            </h4>
            <div className={styles.lineageItems}>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>RLHF / InstructGPT</h5>
                <p>
                  Ouyang et al., 2022 &mdash; Uses PPO as the RL optimizer for
                  aligning LLMs with human preferences via reward models
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>OpenAI Five / Dota</h5>
                <p>
                  Berner et al., 2019 &mdash; Scaled PPO to play Dota 2 at
                  superhuman level with massive parallelization
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>GRPO</h5>
                <p>
                  Shao et al., 2024 &mdash; Group Relative Policy Optimization;
                  drops the critic and uses group-level baselines for LLM fine-tuning
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>Dreamer (Actor)</h5>
                <p>
                  Hafner et al., 2020 &mdash; Uses PPO-style policy gradients with
                  backprop through learned dynamics for model-based RL
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
