'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './planet.module.css';

/* ===== Data ===== */

const coreIdeas = [
  {
    title: 'Recurrent State Space Model (RSSM)',
    desc: 'Combine deterministic recurrent path h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) with stochastic latent s_t ~ p(s_t|h_t). Both paths essential \u2014 deterministic for memory, stochastic for uncertainty.',
    detail: 'PlaNet introduces the Recurrent State Space Model (RSSM), which splits the latent state into a deterministic component h_t (GRU hidden state) and a stochastic component s_t (diagonal Gaussian). The deterministic path allows reliable memory over many steps, while the stochastic path captures environmental randomness. Experiments show BOTH paths are necessary \u2014 purely deterministic (GRU) or purely stochastic (SSM) models fail. The RSSM became the foundation for all subsequent Dreamer models.',
  },
  {
    title: 'Online Planning with CEM',
    desc: 'No actor/critic network. At each step, use Cross-Entropy Method to search for best action sequence: sample J=1000 candidates, evaluate by model rollout, refit to top K=100, repeat I=10 iterations. Execute only first action.',
    detail: 'Instead of learning a policy network, PlaNet plans online at each timestep using the Cross-Entropy Method (CEM). It samples J=1000 random action sequences of length H=12, rolls them out through the learned RSSM, sums predicted rewards, and fits a Gaussian to the top K=100 sequences. After I=10 iterations, it executes the mean of the first action. This replanning after each observation avoids compounding model errors. The downside is computational cost \u2014 Dreamer later replaced CEM with a learned actor.',
  },
  {
    title: 'Latent Overshooting',
    desc: 'Standard variational bound only trains 1-step transitions. Latent overshooting trains multi-step predictions (d=1..D steps ahead) in latent space, improving long-term accuracy.',
    detail: 'The standard variational bound (Eq. 3) only trains the transition model for 1-step predictions: p(s_t|s_{t-1}) is compared to q(s_t|o_\u2264t) via KL. But for planning over H steps, we need accurate multi-step predictions. Latent overshooting (Eq. 7) adds losses for d-step predictions p(s_t|s_{t-d}) against the same posterior, with weighting factors \u03b2_d. This encourages consistency between 1-step and multi-step model rollouts, improving planning performance.',
  },
];

const keyFeatures = [
  { title: 'No Policy Network', desc: 'Pure model-based planning via CEM at inference time \u2014 no actor or critic networks needed' },
  { title: '200x Data Efficient', desc: 'Matches D4PG using 200\u00d7 fewer environment episodes by leveraging the learned world model' },
  { title: 'RSSM Foundation', desc: 'Introduced the Recurrent State Space Model architecture that powers all subsequent Dreamer models' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  h_prev: {
    title: 'Deterministic State h_{t-1}',
    desc: 'The GRU hidden state from the previous timestep. This deterministic path carries forward information across many steps without the lossy bottleneck of stochastic sampling. Together with the stochastic state s_{t-1}, it forms the full model state used for the next prediction.',
  },
  gru1: {
    title: 'GRU Recurrent Unit',
    desc: 'A Gated Recurrent Unit that takes the previous hidden state h_{t-1}, previous stochastic state s_{t-1}, and action a_{t-1} to produce the next deterministic state h_t. The GRU provides reliable long-term memory that the stochastic path alone cannot achieve. The GRU has 200 hidden units in the standard configuration.',
  },
  gru2: {
    title: 'GRU Recurrent Unit (next step)',
    desc: 'Same GRU architecture applied at the next timestep, taking h_t, s_t, and a_t as inputs. The unrolled structure shows how information flows through time via the deterministic path.',
  },
  h_t: {
    title: 'Deterministic State h_t',
    desc: 'Current deterministic hidden state. Combined with the stochastic state s_t to form the full latent representation used for observation decoding, reward prediction, and as input to the next timestep. The deterministic path is what distinguishes the RSSM from a purely stochastic state space model.',
  },
  h_next: {
    title: 'Deterministic State h_{t+1}',
    desc: 'The deterministic hidden state at the next timestep, produced by the GRU from h_t, s_t, and a_t. During planning, the model rolls forward through these states without needing any observations.',
  },
  s_prev: {
    title: 'Stochastic State s_{t-1}',
    desc: 'The stochastic latent variable from the previous timestep, sampled from a diagonal Gaussian. It captures environmental uncertainty that the deterministic path cannot represent. The stochastic state has 30 dimensions with learned mean and variance.',
  },
  s_t: {
    title: 'Stochastic State s_t',
    desc: 'Current stochastic latent state. During training, it is sampled from the posterior q(s_t|h_t, o_t) which sees the observation. During planning, it is sampled from the prior p(s_t|h_t) which predicts without the observation. The KL loss encourages these distributions to match.',
  },
  s_next: {
    title: 'Stochastic State s_{t+1}',
    desc: 'The stochastic state at the next timestep. During CEM planning, this is sampled from the prior p(s_{t+1}|h_{t+1}) and used to predict the reward r_{t+1} for trajectory evaluation.',
  },
  encoder: {
    title: 'Posterior Encoder (CNN + MLP)',
    desc: 'The encoder network that computes the posterior distribution q(s_t|h_{t-1}, a_{t-1}, o_t). It processes the 64\u00d764 image observation through a CNN, concatenates with the deterministic state, and outputs the mean and variance of a diagonal Gaussian. During training, the posterior provides a better estimate than the prior by conditioning on the actual observation.',
  },
  obs: {
    title: 'Observation o_t (64\u00d764 image)',
    desc: 'The raw image observation from the environment, downscaled to 64\u00d764\u00d73 pixels. During training, observations are fed to the encoder to compute the posterior. During planning (CEM), no observations are available \u2014 the model uses only the prior to predict future states.',
  },
  decoder: {
    title: 'Observation Decoder (Deconvolution)',
    desc: 'The decoder network p(o_t|h_t, s_t) that reconstructs images from the combined deterministic and stochastic state. Uses transposed convolutions to upsample from the latent vector to a 64\u00d764\u00d73 image. The reconstruction loss provides learning signal for the encoder and dynamics.',
  },
  rewardmodel: {
    title: 'Reward Model (MLP)',
    desc: 'The reward predictor p(r_t|h_t, s_t) is a small MLP that predicts the scalar reward from the latent state. During CEM planning, the reward model evaluates candidate action sequences by summing predicted rewards along imagined trajectories. It is trained jointly with the rest of the model.',
  },
  cem: {
    title: 'Cross-Entropy Method (CEM) Planning',
    desc: 'At each timestep, PlaNet plans by: (1) initializing a Gaussian distribution over action sequences of length H=12, (2) sampling J=1000 candidate sequences, (3) rolling each through the RSSM to predict rewards, (4) selecting the top K=100 sequences by total reward, (5) refitting the Gaussian to these elites, (6) repeating for I=10 iterations. Only the first action of the best sequence is executed, then the agent replans from the next observation. This model-predictive control approach avoids compounding errors.',
  },
};

const benchmarkData: Record<string, Record<string, number>> = {
  cartpole: { PlaNet: 821, 'D4PG (100k ep)': 746, 'A3C (100k ep)': 494, 'CEM+true dyn.': 856 },
  reacher: { PlaNet: 821, 'D4PG (100k ep)': 601, 'A3C (100k ep)': 310, 'CEM+true dyn.': 893 },
  cheetah: { PlaNet: 832, 'D4PG (100k ep)': 480, 'A3C (100k ep)': 262, 'CEM+true dyn.': 872 },
  finger: { PlaNet: 662, 'D4PG (100k ep)': 700, 'A3C (100k ep)': 264, 'CEM+true dyn.': 899 },
  cup: { PlaNet: 930, 'D4PG (100k ep)': 723, 'A3C (100k ep)': 382, 'CEM+true dyn.': 966 },
  walker: { PlaNet: 951, 'D4PG (100k ep)': 890, 'A3C (100k ep)': 287, 'CEM+true dyn.': 969 },
};

const methodColors: Record<string, string> = {
  PlaNet: '#3b82f6',
  'D4PG (100k ep)': '#f59e0b',
  'A3C (100k ep)': '#888',
  'CEM+true dyn.': '#10b981',
};

/* ===== Helpers ===== */

function rewardAt(x: number, y: number): number {
  const r1 = Math.sin(x * 2.5) * Math.cos(y * 2.0) * 0.4;
  const r2 = Math.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.08) * 0.8;
  const r3 = Math.exp(-((x - 0.5) ** 2 + (y - 0.8) ** 2) / 0.12) * 0.5;
  const r4 = -Math.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.05) * 0.3;
  return r1 + r2 + r3 + r4;
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/* ===== Component ===== */

export default function PlaNetPage() {
  // --- Architecture SVG component info ---
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  // --- Model items state ---
  const [activeModelItem, setActiveModelItem] = useState<string | null>(null);

  // --- Training objectives state ---
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  // --- CEM sliders ---
  const [cemIteration, setCemIteration] = useState(1);
  const [cemHorizon, setCemHorizon] = useState(12);

  // --- Benchmark task select ---
  const [selectedTask, setSelectedTask] = useState('cartpole');

  // --- Canvas ref ---
  const cemCanvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = useCanvasResize(cemCanvasRef);

  // --- CEM drawing ---
  const drawCEMViz = useCallback(() => {
    const canvas = cemCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvasSize.width;
    const h = canvasSize.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const iteration = cemIteration;
    const horizon = cemHorizon;
    const pad = 10;
    const pw = w - 2 * pad;
    const ph = h - 2 * pad;

    // Draw reward heatmap
    const resolution = 4;
    for (let px = 0; px < pw; px += resolution) {
      for (let py = 0; py < ph; py += resolution) {
        const nx = px / pw;
        const ny = py / ph;
        const reward = rewardAt(nx, ny);
        const normalized = (reward + 0.5) / 1.5;
        const clamped = Math.max(0, Math.min(1, normalized));
        const r = Math.floor(clamped * 40);
        const g = Math.floor(clamped * 120 + 20);
        const b = Math.floor((1 - clamped) * 80 + 40);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(pad + px, pad + py, resolution, resolution);
      }
    }

    // Agent position
    const agentX = 0.1;
    const agentY = 0.5;
    const agentPx = pad + agentX * pw;
    const agentPy = pad + agentY * ph;

    // CEM parameters
    const numCandidates = 50;
    const spreadFactor = Math.max(0.05, 1.0 / Math.pow(iteration, 0.7));
    const targetX = 0.7;
    const targetY = 0.3;
    const convergence = 1 - spreadFactor;
    const meanDirX = agentX + convergence * (targetX - agentX);
    const meanDirY = agentY + convergence * (targetY - agentY);

    let bestReward = -Infinity;
    let bestTrajectory: { x: number; y: number }[] | null = null;
    const trajectories: { points: { x: number; y: number }[]; totalReward: number }[] = [];

    for (let c = 0; c < numCandidates; c++) {
      const rngTraj = seededRandom(c * 137 + iteration * 17);
      const points = [{ x: agentX, y: agentY }];
      let totalReward = 0;

      for (let step = 0; step < horizon; step++) {
        const t = (step + 1) / horizon;
        const baseX = agentX + t * (meanDirX - agentX) + (rngTraj() - 0.5) * spreadFactor * 0.8;
        const baseY = agentY + t * (meanDirY - agentY) + (rngTraj() - 0.5) * spreadFactor * 0.8;
        const cx = Math.max(0, Math.min(1, baseX));
        const cy = Math.max(0, Math.min(1, baseY));
        points.push({ x: cx, y: cy });
        totalReward += rewardAt(cx, cy);
      }

      trajectories.push({ points, totalReward });
      if (totalReward > bestReward) {
        bestReward = totalReward;
        bestTrajectory = points;
      }
    }

    // Sort by reward for coloring
    trajectories.sort((a, b) => a.totalReward - b.totalReward);

    // Draw all trajectories
    trajectories.forEach((traj, idx) => {
      if (traj.points === bestTrajectory) return;
      const quality = idx / trajectories.length;
      const alpha = 0.15 + quality * 0.3;
      const r = Math.floor(100 + quality * 155);
      const g = Math.floor(100 + (1 - quality) * 100);
      const b = Math.floor(200 - quality * 100);

      ctx.beginPath();
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.lineWidth = 1;
      traj.points.forEach((p, i) => {
        const ppx = pad + p.x * pw;
        const ppy = pad + p.y * ph;
        i === 0 ? ctx.moveTo(ppx, ppy) : ctx.lineTo(ppx, ppy);
      });
      ctx.stroke();
    });

    // Draw best trajectory bold
    if (bestTrajectory) {
      ctx.beginPath();
      ctx.strokeStyle = '#00d9ff';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00d9ff';
      ctx.shadowBlur = 8;
      bestTrajectory.forEach((p, i) => {
        const ppx = pad + p.x * pw;
        const ppy = pad + p.y * ph;
        i === 0 ? ctx.moveTo(ppx, ppy) : ctx.lineTo(ppx, ppy);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      const lastP = bestTrajectory[bestTrajectory.length - 1];
      ctx.beginPath();
      ctx.arc(pad + lastP.x * pw, pad + lastP.y * ph, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#00d9ff';
      ctx.fill();
    }

    // Draw agent dot
    ctx.beginPath();
    ctx.arc(agentPx, agentPy, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw target peak indicator
    const peakPx = pad + targetX * pw;
    const peakPy = pad + targetY * ph;
    ctx.beginPath();
    ctx.arc(peakPx, peakPy, 6, 0, 2 * Math.PI);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('peak', peakPx + 10, peakPy + 4);

    // Legend
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(w - 130, pad + 2, 120, 55);
    ctx.font = '10px sans-serif';

    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(w - 118, pad + 15, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('Agent', w - 110, pad + 19);

    ctx.strokeStyle = '#00d9ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w - 122, pad + 32);
    ctx.lineTo(w - 112, pad + 32);
    ctx.stroke();
    ctx.fillStyle = '#00d9ff';
    ctx.fillText('Best traj.', w - 110, pad + 36);

    ctx.fillStyle = 'rgba(200,200,200,0.5)';
    ctx.fillRect(w - 122, pad + 45, 10, 2);
    ctx.fillStyle = '#aaa';
    ctx.fillText('Candidates', w - 110, pad + 50);

    // Store bestReward/spreadFactor for metrics
    setBestReward(bestReward);
    setCandidateStd(spreadFactor);
  }, [canvasSize, cemIteration, cemHorizon]);

  // CEM metrics displayed below canvas
  const [bestReward, setBestReward] = useState(0);
  const [candidateStd, setCandidateStd] = useState(1);

  useEffect(() => {
    drawCEMViz();
  }, [drawCEMViz]);

  // --- Benchmark rendering ---
  const benchData = benchmarkData[selectedTask];
  const maxBenchVal = Math.max(...Object.values(benchData), 1);

  return (
    <div className="method-page">
      <h1 className={styles.title}>PlaNet</h1>
      <p className={styles.subtitle}>
        Learning Latent Dynamics for Planning from Pixels &mdash; Hafner et al., ICML 2019
      </p>
      <div className={styles.linkRow}>
        <a
          href="https://danijar.com/planet"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.paperLink}
        >
          <span className={styles.linkIcon}>&rarr;</span> danijar.com/planet
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ===== Model Components + Architecture ===== */}
      <div className={styles.mainGrid}>
        {/* Model Components (RSSM) */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Model Components (RSSM)
          </div>
          <div className={styles.modelEquations}>
            {/* Deterministic Path */}
            <div
              className={activeModelItem === 'deterministic' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'deterministic' ? null : 'deterministic')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#3b82f6' }} />
                Deterministic Path (GRU)
              </h4>
              <div className={styles.formula}>
                <MathJax inline>{'\\( h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemDesc}>Provides memory for long-term dependencies via a GRU recurrent unit</p>
            </div>
            {/* Stochastic Prior */}
            <div
              className={activeModelItem === 'prior' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'prior' ? null : 'prior')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#10b981' }} />
                Stochastic Prior
              </h4>
              <div className={styles.formula}>
                <MathJax inline>{'\\( s_t \\sim p(s_t \\mid h_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemDesc}>Diagonal Gaussian. Captures environment uncertainty in the latent state</p>
            </div>
            {/* Posterior (Encoder) */}
            <div
              className={activeModelItem === 'posterior' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'posterior' ? null : 'posterior')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#f59e0b' }} />
                Posterior (Encoder)
              </h4>
              <div className={styles.formula}>
                <MathJax inline>{'\\( q(s_t \\mid h_{t-1}, a_{t-1}, o_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemDesc}>CNN encoder + MLP. Conditions on actual observation for better latent estimate</p>
            </div>
            {/* Observation Model */}
            <div
              className={activeModelItem === 'observation' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'observation' ? null : 'observation')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#a855f7' }} />
                Observation Model
              </h4>
              <div className={styles.formula}>
                <MathJax inline>{'\\( p(o_t \\mid h_t, s_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemDesc}>Deconvolutional decoder. Reconstructs 64&times;64 images from the latent state</p>
            </div>
            {/* Reward Model */}
            <div
              className={activeModelItem === 'reward' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'reward' ? null : 'reward')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#ef4444' }} />
                Reward Model
              </h4>
              <div className={styles.formula}>
                <MathJax inline>{'\\( p(r_t \\mid h_t, s_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemDesc}>MLP. Predicts scalar reward from latent state for planning evaluation</p>
            </div>
          </div>
        </div>

        {/* Architecture SVG */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            RSSM Architecture (Unrolled)
          </div>
          <svg className={styles.pipelineSvg} viewBox="0 0 500 420">
            <defs>
              <marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
              </marker>
              <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
              </marker>
              <marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
              </marker>
              <marker id="arrowPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
              </marker>
              <marker id="arrowRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
              </marker>
              <marker id="arrowGray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#888" />
              </marker>
              <marker id="arrowCyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff" />
              </marker>
            </defs>

            {/* Title labels */}
            <text x="250" y="15" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">Deterministic Path (GRU)</text>

            {/* Actions above GRU boxes */}
            <text x="145" y="35" textAnchor="middle" fill="#888" fontSize="9">{'a_{t-1}'}</text>
            <text x="305" y="35" textAnchor="middle" fill="#888" fontSize="9">{'a_t'}</text>
            <path d="M 145 38 L 145 48" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGray)" />
            <path d="M 305 38 L 305 48" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGray)" />

            {/* Deterministic states h */}
            <rect
              x="30" y="55" width="60" height="35" rx="8"
              fill="rgba(59,130,246,0.3)" stroke="#3b82f6" strokeWidth="2"
              className={activeComponent === 'h_prev' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('h_prev')}
            />
            <text x="60" y="77" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'h_{t-1}'}</text>

            {/* GRU box 1 */}
            <rect
              x="120" y="50" width="50" height="45" rx="6" fill="#3b82f6"
              className={activeComponent === 'gru1' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('gru1')}
            />
            <text x="145" y="72" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>GRU</text>
            <text x="145" y="84" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="7" style={{ pointerEvents: 'none' }}>{'f(\u00b7)'}</text>

            {/* h_t */}
            <rect
              x="200" y="55" width="60" height="35" rx="8"
              fill="rgba(59,130,246,0.3)" stroke="#3b82f6" strokeWidth="2"
              className={activeComponent === 'h_t' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('h_t')}
            />
            <text x="230" y="77" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'h_t'}</text>

            {/* GRU box 2 */}
            <rect
              x="280" y="50" width="50" height="45" rx="6" fill="#3b82f6"
              className={activeComponent === 'gru2' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('gru2')}
            />
            <text x="305" y="72" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>GRU</text>
            <text x="305" y="84" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="7" style={{ pointerEvents: 'none' }}>{'f(\u00b7)'}</text>

            {/* h_{t+1} */}
            <rect
              x="360" y="55" width="60" height="35" rx="8"
              fill="rgba(59,130,246,0.3)" stroke="#3b82f6" strokeWidth="2"
              className={activeComponent === 'h_next' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('h_next')}
            />
            <text x="390" y="77" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'h_{t+1}'}</text>

            {/* Arrows: deterministic path */}
            <path d="M 90 72 L 118 72" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" className={styles.flowArrow} />
            <path d="M 170 72 L 198 72" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" className={styles.flowArrow} />
            <path d="M 260 72 L 278 72" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" className={styles.flowArrow} />
            <path d="M 330 72 L 358 72" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" className={styles.flowArrow} />

            {/* Stochastic states s (circles below h) */}
            <circle
              cx="60" cy="140" r="18"
              fill="rgba(16,185,129,0.3)" stroke="#10b981" strokeWidth="2"
              className={activeComponent === 's_prev' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('s_prev')}
            />
            <text x="60" y="144" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'s_{t-1}'}</text>

            <circle
              cx="230" cy="140" r="18"
              fill="rgba(16,185,129,0.3)" stroke="#10b981" strokeWidth="2"
              className={activeComponent === 's_t' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('s_t')}
            />
            <text x="230" y="144" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'s_t'}</text>

            <circle
              cx="390" cy="140" r="18"
              fill="rgba(16,185,129,0.3)" stroke="#10b981" strokeWidth="2"
              className={activeComponent === 's_next' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('s_next')}
            />
            <text x="390" y="144" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'s_{t+1}'}</text>

            {/* Arrows: h to s (prior) */}
            <path d="M 60 90 L 60 120" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGreen)" />
            <path d="M 230 90 L 230 120" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGreen)" />
            <path d="M 390 90 L 390 120" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGreen)" />
            <text x="250" y="112" fill="#10b981" fontSize="7">{'p(s_t|h_t)'}</text>

            {/* s feeds back into next GRU */}
            <path d="M 78 138 Q 100 130 120 85" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="4" />
            <path d="M 248 138 Q 270 130 280 85" stroke="#10b981" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="4" />

            {/* Observation o_t and encoder (posterior) */}
            <rect
              x="175" y="185" width="55" height="30" rx="6"
              fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="1.5"
              className={activeComponent === 'encoder' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('encoder')}
            />
            <text x="202" y="200" textAnchor="middle" fill="#f59e0b" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Encoder</text>
            <text x="202" y="210" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>CNN+MLP</text>

            <rect
              x="175" y="228" width="55" height="25" rx="6" fill="#f59e0b"
              className={activeComponent === 'obs' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('obs')}
            />
            <text x="202" y="244" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'o_t'}</text>

            {/* Arrow o_t to encoder */}
            <path d="M 202 228 L 202 217" stroke="#f59e0b" strokeWidth="1.5" fill="none" markerEnd="url(#arrowOrange)" />
            {/* Arrow encoder to s_t (posterior, dashed) */}
            <path d="M 210 185 Q 222 168 228 160" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" strokeDasharray="5 3" />
            <text x="235" y="175" fill="#f59e0b" fontSize="7">{'q(s_t|h,a,o)'}</text>

            {/* Decoder arrows from (h_t, s_t) to reconstructed o_t */}
            <rect
              x="280" y="185" width="55" height="30" rx="6"
              fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5"
              className={activeComponent === 'decoder' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('decoder')}
            />
            <text x="307" y="200" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Decoder</text>
            <text x="307" y="210" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>DeConv</text>

            <rect x="280" y="228" width="55" height="25" rx="6" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5" />
            <text x="307" y="244" textAnchor="middle" fill="#a855f7" fontSize="8" style={{ pointerEvents: 'none' }}>{'o_t (recon)'}</text>

            {/* Arrow (h_t,s_t) to decoder */}
            <path d="M 248 140 Q 268 165 290 185" stroke="#a855f7" strokeWidth="1.5" fill="none" markerEnd="url(#arrowPurple)" />
            {/* Arrow decoder to reconstructed */}
            <path d="M 307 215 L 307 226" stroke="#a855f7" strokeWidth="1.5" fill="none" markerEnd="url(#arrowPurple)" />

            {/* Reward model */}
            <rect
              x="370" y="185" width="55" height="30" rx="6"
              fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5"
              className={activeComponent === 'rewardmodel' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('rewardmodel')}
            />
            <text x="397" y="200" textAnchor="middle" fill="#ef4444" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Reward</text>
            <text x="397" y="210" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>MLP</text>

            <rect x="377" y="228" width="40" height="25" rx="6" fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5" />
            <text x="397" y="244" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'r_t'}</text>

            {/* Arrow (h_t,s_t) to reward model */}
            <path d="M 248 145 Q 330 175 375 185" stroke="#ef4444" strokeWidth="1.5" fill="none" markerEnd="url(#arrowRed)" />
            {/* Arrow reward model to r_t */}
            <path d="M 397 215 L 397 226" stroke="#ef4444" strokeWidth="1.5" fill="none" markerEnd="url(#arrowRed)" />

            {/* CEM Planning Box at bottom */}
            <rect
              x="50" y="280" width="400" height="55" rx="10"
              fill="rgba(0,217,255,0.1)" stroke="#00d9ff" strokeWidth="2"
              className={activeComponent === 'cem' ? styles.pipelineStageActive : styles.pipelineStage}
              onClick={() => setActiveComponent('cem')}
            />
            <text x="250" y="298" textAnchor="middle" fill="#00d9ff" fontSize="11" fontWeight="bold" style={{ pointerEvents: 'none' }}>Cross-Entropy Method (CEM) Planning</text>

            {/* CEM sub-steps */}
            <rect x="70" y="308" width="80" height="20" rx="5" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1" />
            <text x="110" y="322" textAnchor="middle" fill="#3b82f6" fontSize="8">Sample J=1000</text>

            <rect x="165" y="308" width="75" height="20" rx="5" fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="1" />
            <text x="202" y="322" textAnchor="middle" fill="#10b981" fontSize="8">Evaluate</text>

            <rect x="255" y="308" width="80" height="20" rx="5" fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="1" />
            <text x="295" y="322" textAnchor="middle" fill="#f59e0b" fontSize="8">Refit K=100</text>

            <rect x="350" y="308" width="80" height="20" rx="5" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1" />
            <text x="390" y="322" textAnchor="middle" fill="#a855f7" fontSize="8">Repeat I=10</text>

            {/* Arrows between CEM steps */}
            <path d="M 150 318 L 163 318" stroke="#888" strokeWidth="1" fill="none" markerEnd="url(#arrowGray)" />
            <path d="M 240 318 L 253 318" stroke="#888" strokeWidth="1" fill="none" markerEnd="url(#arrowGray)" />
            <path d="M 335 318 L 348 318" stroke="#888" strokeWidth="1" fill="none" markerEnd="url(#arrowGray)" />

            {/* Arrow from model to CEM */}
            <path d="M 250 253 L 250 278" stroke="#00d9ff" strokeWidth="2" fill="none" markerEnd="url(#arrowCyan)" />
            <text x="265" y="268" fill="#00d9ff" fontSize="7">rollout</text>

            {/* Execute first action label */}
            <text x="250" y="355" textAnchor="middle" fill="#888" fontSize="9">{'Execute first action a*_1, observe o_{t+1}, replan'}</text>

            {/* Labels for stochastic zone */}
            <text x="450" y="140" textAnchor="end" fill="#10b981" fontSize="8" fontStyle="italic">stochastic</text>
            <text x="450" y="72" textAnchor="end" fill="#3b82f6" fontSize="8" fontStyle="italic">deterministic</text>
          </svg>

          {activeComponent && componentInfo[activeComponent] && (
            <div className={styles.infoPanel}>
              <h4>{componentInfo[activeComponent].title}</h4>
              <p>{componentInfo[activeComponent].desc}</p>
            </div>
          )}
        </div>
      </div>

      {/* ===== CEM Planning + Benchmark ===== */}
      <div className={styles.mainGrid}>
        {/* CEM Planning Interactive */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            CEM Planning Visualization
          </div>
          <div className={styles.sliderContainer}>
            <label>CEM Iteration:</label>
            <input
              type="range"
              className={styles.slider}
              min={1}
              max={10}
              step={1}
              value={cemIteration}
              onChange={(e) => setCemIteration(parseInt(e.target.value))}
            />
            <span className={styles.sliderValue}>{cemIteration}</span>
          </div>
          <div className={styles.sliderContainer}>
            <label>Planning Horizon:</label>
            <input
              type="range"
              className={styles.slider}
              min={4}
              max={20}
              step={1}
              value={cemHorizon}
              onChange={(e) => setCemHorizon(parseInt(e.target.value))}
            />
            <span className={styles.sliderValue}>{cemHorizon}</span>
          </div>
          <div className={styles.canvasContainer}>
            <canvas ref={cemCanvasRef} className={styles.cemCanvas} />
          </div>
          <div className={styles.metricsRow}>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#10b981' }}>{cemIteration}/10</div>
              <div className={styles.metricLabel}>Iteration</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#f59e0b' }}>{bestReward.toFixed(2)}</div>
              <div className={styles.metricLabel}>Best Reward</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#a855f7' }}>{candidateStd.toFixed(2)}</div>
              <div className={styles.metricLabel}>Candidate Std</div>
            </div>
          </div>
        </div>

        {/* Benchmark Results */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            DeepMind Control Suite Results
          </div>
          <div className={styles.comparisonContainer}>
            <p className={styles.comparisonNote}>Performance at 500 episodes (2000 for D4PG/A3C at 100k episodes):</p>
            <div>
              {Object.entries(benchData).map(([method, score]) => {
                const pct = Math.max(score, 0) / maxBenchVal * 100;
                const color = methodColors[method];
                const isPlaNet = method === 'PlaNet';
                return (
                  <div className={styles.comparisonRow} key={method}>
                    <span
                      className={styles.comparisonLabel}
                      style={{ color, fontWeight: isPlaNet ? 700 : 400 }}
                    >
                      {method}
                    </span>
                    <div className={styles.comparisonBarTrack}>
                      <div
                        className={styles.comparisonBarFill}
                        style={{ width: `${pct}%`, background: color, opacity: isPlaNet ? 1 : 0.7 }}
                      />
                    </div>
                    <span className={styles.comparisonValue} style={{ color }}>{score}</span>
                  </div>
                );
              })}
            </div>
            <div className={styles.taskSelectContainer}>
              <label className={styles.taskSelectLabel}>Task:</label>
              <select
                className={styles.taskSelect}
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
              >
                <option value="cartpole">Cartpole Swingup</option>
                <option value="reacher">Reacher Easy</option>
                <option value="cheetah">Cheetah Run</option>
                <option value="finger">Finger Spin</option>
                <option value="cup">Cup Catch</option>
                <option value="walker">Walker Walk</option>
              </select>
            </div>
            <p className={styles.comparisonFooter}>
              PlaNet matches D4PG (100k episodes) using only 500 episodes &mdash; 200&times; more data efficient.
            </p>
          </div>
        </div>
      </div>

      {/* ===== Training Objectives ===== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Training Objectives
        </div>
        <div className={styles.modelEquations}>
          {/* ELBO */}
          <div
            className={activeLoss === 'elbo' ? styles.modelItemActive : styles.modelItem}
            onClick={() => setActiveLoss(activeLoss === 'elbo' ? null : 'elbo')}
          >
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#3b82f6' }} />
              Variational Bound (ELBO)
            </h4>
            <div className={styles.formula}>
              <MathJax>{
                '$$\\ln p(o_{1:T}|a_{1:T}) \\geq \\sum_{t=1}^T \\left[\\mathbb{E}_{q(s_t|o_{\\leq t},a_{<t})} \\ln p(o_t|s_t,h_t) - \\text{KL}[q(s_t|o_{\\leq t},a_{<t}) \\| p(s_t|h_t)]\\right]$$'
              }</MathJax>
            </div>
            <p className={styles.modelItemDesc}>
              Standard 1-step variational bound: reconstruct observations + regularize posterior to match prior via KL divergence
            </p>
          </div>
          {/* Latent Overshooting */}
          <div
            className={activeLoss === 'overshoot' ? styles.modelItemActive : styles.modelItem}
            onClick={() => setActiveLoss(activeLoss === 'overshoot' ? null : 'overshoot')}
          >
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#a855f7' }} />
              Latent Overshooting
            </h4>
            <div className={styles.formula}>
              <MathJax>{
                '$$\\mathcal{J}_O = \\sum_{t=1}^T \\sum_{d=1}^D \\beta_d \\left[\\mathbb{E}[\\ln p(o_t|s_{t|t-d})] - \\text{KL}[q(s_t|o_{\\leq t}) \\| p(s_t|s_{t-d})]\\right]$$'
              }</MathJax>
            </div>
            <p className={styles.modelItemDesc}>
              Multi-step prediction loss: trains d-step transitions p(s_t|s_{'{t-d}'}) against the same posterior, with weighting factors &beta;_d for consistency
            </p>
          </div>
          {/* CEM Planning Objective */}
          <div
            className={activeLoss === 'cem' ? styles.modelItemActive : styles.modelItem}
            onClick={() => setActiveLoss(activeLoss === 'cem' ? null : 'cem')}
          >
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#10b981' }} />
              CEM Planning Objective
            </h4>
            <div className={styles.formula}>
              <MathJax>{
                '$$a_{1:H}^* = \\arg\\max_{a_{1:H}} \\mathbb{E}_p\\left[\\sum_{\\tau=t}^{t+H} r_\\tau\\right]$$'
              }</MathJax>
            </div>
            <p className={styles.modelItemDesc}>
              Find the action sequence that maximizes expected sum of predicted rewards over planning horizon H, solved approximately via CEM
            </p>
          </div>
        </div>
      </div>

      {/* ===== Paper Lineage ===== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Paper Lineage
        </div>
        <div className={styles.lineageGrid}>
          <div>
            <h4 className={styles.lineageHeading} style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className={styles.lineageList}>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>World Models</div>
                <div className={styles.lineageItemDesc}>
                  Ha &amp; Schmidhuber, 2018 &mdash; Two-stage world model (VAE + MDN-RNN). PlaNet unifies into a single RSSM trained end-to-end.
                </div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>E2C (Embed to Control)</div>
                <div className={styles.lineageItemDesc}>
                  Watter et al., 2015 &mdash; Locally linear latent dynamics model for control from images. PlaNet scales to richer tasks with recurrence.
                </div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>Deep Kalman Filters</div>
                <div className={styles.lineageItemDesc}>
                  Krishnan et al., 2015 &mdash; Deep state space models with variational inference. PlaNet adds the deterministic path for reliable memory.
                </div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>CEM</div>
                <div className={styles.lineageItemDesc}>
                  Rubinstein, 1997 &mdash; The Cross-Entropy Method for optimization. PlaNet uses it for online planning over learned dynamics.
                </div>
              </div>
            </div>
          </div>
          <div>
            <h4 className={styles.lineageHeading} style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className={styles.lineageList}>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>Dreamer</div>
                <div className={styles.lineageItemDesc}>
                  Hafner et al., 2020 &mdash; Replaces CEM planning with learned actor-critic in imagination. Same RSSM, but amortized policy is faster at test time.
                </div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>DreamerV2</div>
                <div className={styles.lineageItemDesc}>
                  Hafner et al., 2021 &mdash; Discrete latent states (32&times;32 categoricals), KL balancing. Achieves human-level Atari from pixels.
                </div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>DreamerV3</div>
                <div className={styles.lineageItemDesc}>
                  Hafner et al., 2023 &mdash; Fixed hyperparameters across domains, symlog predictions. Scales from Atari to Minecraft without tuning.
                </div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>SOLAR</div>
                <div className={styles.lineageItemDesc}>
                  Zhang et al., 2019 &mdash; Stochastic latent actor-critic combining PlaNet-style world model with soft actor-critic for real-robot tasks.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
