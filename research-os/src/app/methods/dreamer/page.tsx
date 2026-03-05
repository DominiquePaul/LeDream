'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './dreamer.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'World Model',
    desc: 'Learn a latent dynamics model from experience to predict future states, rewards, and observations in a compact state space',
    detail: 'Dreamer learns a Recurrent State Space Model (RSSM) that compresses high-dimensional image observations into compact latent states. The model has three components: a representation model (encoder) that maps observations to latent states, a transition model that predicts next states from actions alone (without seeing observations), and a reward model. The key insight is that the transition model enables "imagination" — predicting the future without rendering any images, making it extremely fast.',
  },
  {
    title: 'Latent Imagination',
    desc: 'Imagine thousands of trajectories entirely in the learned latent space — no environment interaction needed during behavior learning',
    detail: 'Instead of planning action-by-action with the world model (like PlaNet does with CEM), Dreamer imagines entire trajectories in the latent space. Starting from encoded real states, it rolls out the transition model for H steps, generating thousands of imagined trajectories in parallel. This is vastly more efficient than environment interaction because: (1) latent states are tiny compared to images, (2) no rendering is needed, and (3) batches of trajectories run in parallel on GPU.',
  },
  {
    title: 'Actor-Critic in Imagination',
    desc: 'Learn an actor and value function by backpropagating analytic gradients of multi-step returns through the learned dynamics',
    detail: 'The action and value models are trained cooperatively on imagined trajectories. The actor maximizes value estimates along trajectories, and crucially, gradients flow back through the learned dynamics via reparameterization. This is what distinguishes Dreamer from model-free actor-critics: instead of using REINFORCE-style gradients (high variance), Dreamer uses analytic gradients through the differentiable world model. The value model (V_Λ) uses exponentially-weighted multi-step returns to balance bias and variance, making the method robust to the imagination horizon length.',
  },
];

const keyFeatures = [
  { title: 'Pixel-to-Control', desc: 'Learns directly from 64×64 image observations, no state access needed' },
  { title: 'Data Efficient', desc: 'Matches D4PG (10⁸ steps) with only 5×10⁶ steps — 20x more efficient' },
  { title: 'Analytic Gradients', desc: 'Backpropagates value gradients through learned dynamics via reparameterization' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  dataset: {
    title: 'Dataset of Experience',
    desc: 'A growing replay buffer of (observation, action, reward) sequences collected by the agent. Initialized with S=5 random seed episodes. After every C=100 training steps, the agent executes its current policy in the environment for 1 episode, adding new experience. The dataset grows over time, improving the world model.',
  },
  rssm: {
    title: 'Recurrent State Space Model (RSSM)',
    desc: 'The world model combines a deterministic recurrent path (GRU) with stochastic latent variables. The deterministic path provides memory, while the stochastic variables capture uncertainty. The CNN encoder processes 64\u00d764 images, and the model is trained via the ELBO objective (reconstruction + KL regularization). Latent states are 30-dimensional diagonal Gaussians.',
  },
  latent: {
    title: 'Compact Latent State',
    desc: 'The latent state s_t is a compact vector combining the deterministic recurrent state h_t and stochastic state z_t. This is much smaller than a 64\u00d764\u00d73 image, enabling fast parallel imagination of thousands of trajectories. The representation model encodes real observations, while the transition model predicts future states without observations.',
  },
  imagination: {
    title: 'Imagined Trajectories',
    desc: 'Starting from real encoded states s_t, Dreamer unrolls the transition model for H=15 steps using the current action model. This generates batches of imagined state-action-reward sequences entirely in the latent space. No images are rendered. These trajectories are used to train the actor and critic via backpropagation through the dynamics.',
  },
  actor: {
    title: 'Action Model (Policy)',
    desc: 'A dense neural network with parameters \u03c6 that outputs a tanh-transformed Gaussian: a_\u03c4 = tanh(\u03bc_\u03c6(s_\u03c4) + \u03c3_\u03c6(s_\u03c4)\u00b7\u03b5). The reparameterization trick allows gradients to flow through sampled actions back into the model. The actor is trained to maximize V_\u03bb estimates along imagined trajectories. Exploration noise Normal(0, 0.3) is added during environment interaction.',
  },
  value: {
    title: 'Value Model (Critic)',
    desc: 'A dense neural network v_\u03c8(s_\u03c4) that estimates the V_\u03bb targets. Trained by regression: min \u03a3 \u00bd||v_\u03c8(s_\u03c4) - V_\u03bb(s_\u03c4)||\u00b2. The value model allows the agent to estimate rewards beyond the imagination horizon H, which is critical for long-horizon tasks. Without it, the agent becomes short-sighted.',
  },
  env: {
    title: 'Environment (DeepMind Control Suite)',
    desc: '20 visual control tasks with 64\u00d764\u00d73 image observations. Tasks include locomotion (Walker, Cheetah, Quadruped), balance (Cartpole, Acrobot), and manipulation (Cup, Finger). Episodes last 1000 steps with rewards in [0,1]. Dreamer uses action repeat R=2 and trains for ~3 hours per 10\u2076 steps on a single V100 GPU.',
  },
  gradients: {
    title: 'Backpropagation Through Dynamics (Key Innovation)',
    desc: 'The defining feature of Dreamer: analytic gradients of value estimates are propagated backwards through the entire imagined trajectory and through the learned dynamics. This is possible because (1) the transition model is differentiable, (2) actions are sampled via reparameterization, and (3) stochastic latent states use the reparameterization trick. This gives much lower variance gradients than REINFORCE/policy gradient methods.',
  },
  vr: {
    title: 'V_R: Pure Reward Sum',
    desc: 'The simplest value estimate: just sum rewards from \u03c4 to \u03c4+H. This ignores any rewards beyond the horizon H. Used as an ablation \u2014 without a value model, the agent is short-sighted and performance degrades for tasks requiring long-horizon credit assignment.',
  },
  vn: {
    title: 'V_N^k: k-step Returns',
    desc: 'Sum k steps of rewards then bootstrap with the value model: V_N^k(s_\u03c4) = \u03a3 \u03b3^(n-\u03c4) r_n + \u03b3^(h-\u03c4) v_\u03c8(s_h). Different values of k trade off bias (from value model errors) and variance (from imagined rewards). k=1 is pure TD, k=H is V_R.',
  },
  vlambda: {
    title: 'V_\u03bb: Exponentially-Weighted Average',
    desc: 'The final value estimate used by Dreamer. Combines all k-step returns with exponentially decaying weights: V_\u03bb = (1-\u03bb) \u03a3 \u03bb^(n-1) V_N^n + \u03bb^(H-1) V_N^H. With \u03bb=0.95, this behaves like GAE (Generalized Advantage Estimation) and smoothly balances bias vs variance. This makes Dreamer robust to the choice of imagination horizon H.',
  },
};

const benchmarkData: Record<string, Record<string, number>> = {
  average: { Dreamer: 823, PlaNet: 332, 'D4PG (1e8)': 786, 'A3C (1e8)': 263 },
  cheetah_run: { Dreamer: 770, PlaNet: 510, 'D4PG (1e8)': 480, 'A3C (1e8)': 250 },
  walker_walk: { Dreamer: 918, PlaNet: 560, 'D4PG (1e8)': 900, 'A3C (1e8)': 200 },
  cartpole_swingup: { Dreamer: 762, PlaNet: 475, 'D4PG (1e8)': 830, 'A3C (1e8)': 400 },
  hopper_hop: { Dreamer: 237, PlaNet: 45, 'D4PG (1e8)': 130, 'A3C (1e8)': 2 },
  quadruped_walk: { Dreamer: 876, PlaNet: 180, 'D4PG (1e8)': 480, 'A3C (1e8)': 50 },
  finger_turn_hard: { Dreamer: 468, PlaNet: 205, 'D4PG (1e8)': 220, 'A3C (1e8)': 80 },
};

const methodColors: Record<string, string> = {
  Dreamer: '#a855f7',
  PlaNet: '#3b82f6',
  'D4PG (1e8)': '#f59e0b',
  'A3C (1e8)': '#888',
};

/* ================================================================
   HELPERS
   ================================================================ */

type MethodKey = 'dreamer' | 'novalue' | 'planet';

function getPerformance(method: MethodKey, H: number): number {
  switch (method) {
    case 'dreamer':
      return 700 + 150 * (1 - Math.exp(-H / 5)) - 20 * Math.max(0, (H - 30) / 30);
    case 'novalue':
      return 200 + 500 * (1 - Math.exp(-H / 8)) * Math.exp(-Math.max(0, H - 20) / 25);
    case 'planet':
      return 500 + 200 * (1 - Math.exp(-H / 3)) - 150 * Math.max(0, (H - 12) / 20);
  }
}

/* ================================================================
   COMPONENT
   ================================================================ */

export default function DreamerPage() {
  // --- Architecture Info ---
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  // --- Model Items ---
  const [activeModel, setActiveModel] = useState<string | null>(null);

  // --- Training Objectives ---
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  // --- Horizon Canvas ---
  const horizonCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [horizon, setHorizon] = useState(15);
  const [selectedMethod, setSelectedMethod] = useState<MethodKey>('dreamer');
  const horizonSize = useCanvasResize(horizonCanvasRef);

  // --- Benchmark ---
  const [selectedTask, setSelectedTask] = useState('average');

  /* ---- Horizon canvas draw ---- */
  const drawHorizonViz = useCallback(() => {
    const canvas = horizonCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = horizonSize.width;
    const h = horizonSize.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const pad = { left: 55, right: 20, top: 20, bottom: 40 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Imagination Horizon H', w / 2, h - 8);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Episode Return', 0, 0);
    ctx.restore();

    // Y axis labels
    ctx.textAlign = 'right';
    ctx.font = '10px sans-serif';
    for (let y = 0; y <= 1000; y += 200) {
      const cy = h - pad.bottom - (y / 1000) * ph;
      ctx.fillStyle = '#555';
      ctx.fillText(y.toString(), pad.left - 8, cy + 4);
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.beginPath();
      ctx.moveTo(pad.left, cy);
      ctx.lineTo(w - pad.right, cy);
      ctx.stroke();
    }

    // X axis labels
    ctx.textAlign = 'center';
    for (let x = 5; x <= 45; x += 5) {
      const cx = pad.left + ((x - 1) / 44) * pw;
      ctx.fillStyle = '#555';
      ctx.fillText(x.toString(), cx, h - pad.bottom + 15);
    }

    const maxH = 45;

    const methods: { key: MethodKey; color: string; label: string }[] = [
      { key: 'planet', color: '#3b82f6', label: 'PlaNet' },
      { key: 'novalue', color: '#10b981', label: 'No value' },
      { key: 'dreamer', color: '#a855f7', label: 'Dreamer' },
    ];

    methods.forEach((m) => {
      ctx.beginPath();
      ctx.strokeStyle = m.key === selectedMethod ? m.color : m.color + '60';
      ctx.lineWidth = m.key === selectedMethod ? 3 : 1.5;
      for (let i = 1; i <= maxH; i++) {
        const perf = getPerformance(m.key, i);
        const cx = pad.left + ((i - 1) / (maxH - 1)) * pw;
        const cy = h - pad.bottom - (Math.max(0, perf) / 1000) * ph;
        i === 1 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      // Label at end
      const endPerf = getPerformance(m.key, maxH);
      const endY = h - pad.bottom - (Math.max(0, endPerf) / 1000) * ph;
      ctx.fillStyle = m.color;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        m.label,
        w - pad.right - 55,
        endY + (m.key === 'novalue' ? -8 : m.key === 'planet' ? 12 : 4)
      );
    });

    // Current horizon line
    const hx = pad.left + ((horizon - 1) / (maxH - 1)) * pw;
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(hx, pad.top);
    ctx.lineTo(hx, h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on selected curve
    const perfAtH = getPerformance(selectedMethod, horizon);
    const dotY = h - pad.bottom - (Math.max(0, perfAtH) / 1000) * ph;
    const dotColor = methods.find((m) => m.key === selectedMethod)!.color;
    ctx.beginPath();
    ctx.arc(hx, dotY, 6, 0, 2 * Math.PI);
    ctx.fillStyle = dotColor;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [horizonSize, horizon, selectedMethod]);

  useEffect(() => {
    drawHorizonViz();
  }, [drawHorizonViz]);

  /* ---- Derived metrics ---- */
  const perfAtH = getPerformance(selectedMethod, horizon);
  const robustnessSpread = (() => {
    const perfs: number[] = [];
    for (let h = 5; h <= 40; h += 5) perfs.push(getPerformance(selectedMethod, h));
    return Math.max(...perfs) - Math.min(...perfs);
  })();
  const robustness = robustnessSpread < 100 ? 'High' : robustnessSpread < 250 ? 'Medium' : 'Low';
  const robustnessColor =
    robustness === 'High' ? '#10b981' : robustness === 'Medium' ? '#f59e0b' : '#ef4444';

  /* ---- Benchmark bars ---- */
  const benchmarkForTask = benchmarkData[selectedTask];
  const benchMaxVal = Math.max(...Object.values(benchmarkForTask), 1);

  /* ================================================================
     RENDER
     ================================================================ */

  return (
    <div className="method-page">
      <h1>Dreamer</h1>
      <p className={styles.subtitle}>
        Dream to Control: Learning Behaviors by Latent Imagination &mdash; Hafner et al., ICLR 2020
      </p>
      <div className={styles.paperLink}>
        <a href="https://danijar.com/dreamer" target="_blank" rel="noopener noreferrer">
          <span className={styles.paperLinkIcon}>&rarr;</span> danijar.com/dreamer
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ====================== World Model Components ====================== */}
      <h2>World Model Components (RSSM)</h2>
      <div className={styles.modelEquations}>
        {[
          { key: 'repr', color: '#10b981', title: 'Representation Model', formula: '\\( p(s_t \\mid s_{t-1}, a_{t-1}, o_t) \\)', desc: 'Encodes observations and past states into compact latent states using the RSSM + CNN encoder' },
          { key: 'trans', color: '#3b82f6', title: 'Transition Model', formula: '\\( q(s_t \\mid s_{t-1}, a_{t-1}) \\)', desc: 'Predicts next latent state without seeing the observation \u2014 enables imagination without rendering' },
          { key: 'reward', color: '#ef4444', title: 'Reward Model', formula: '\\( q(r_t \\mid s_t) \\)', desc: 'Predicts scalar rewards from latent states to evaluate imagined trajectories' },
          { key: 'obs', color: '#a855f7', title: 'Observation Model (for learning)', formula: '\\( q(o_t \\mid s_t) \\)', desc: 'Reconstructs images from latent states \u2014 provides learning signal via pixel reconstruction loss' },
        ].map((m) => (
          <div
            key={m.key}
            className={activeModel === m.key ? styles.modelItemActive : styles.modelItem}
            onClick={() => setActiveModel(activeModel === m.key ? null : m.key)}
          >
            <h4>
              <span className={styles.modelDot} style={{ background: m.color }} />
              {m.title}
            </h4>
            <div className="formula-block">
              <MathJax>{m.formula}</MathJax>
            </div>
            <p className={styles.modelDesc}>{m.desc}</p>
          </div>
        ))}
      </div>

      {/* ====================== Dreamer Architecture SVG ====================== */}
      <h2>Dreamer Architecture</h2>
      <div className="diagram-frame">
        <svg className={styles.pipelineSvg} viewBox="0 0 500 360">
          <defs>
            <marker id="arrowCyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff" />
            </marker>
            <marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
            </marker>
            <marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
            </marker>
            <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
            </marker>
            <marker id="arrowPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
            </marker>
            <marker id="arrowGray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#888" />
            </marker>
          </defs>

          {/* Phase 1: Learn Dynamics */}
          <text x="95" y="15" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">1. Learn Dynamics</text>
          <rect
            x="20" y="25" width="70" height="45" rx="8" fill="#f59e0b"
            className={`${styles.pipelineStage} ${activeComponent === 'dataset' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('dataset')}
          />
          <text x="55" y="45" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" pointerEvents="none">Dataset</text>
          <text x="55" y="57" textAnchor="middle" fill="white" fontSize="8" pointerEvents="none">(o, a, r)</text>

          <rect
            x="110" y="25" width="80" height="45" rx="8" fill="rgba(245,158,11,0.3)" stroke="#f59e0b" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'rssm' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('rssm')}
          />
          <text x="150" y="45" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold" pointerEvents="none">RSSM</text>
          <text x="150" y="57" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">World Model</text>

          <path d="M 90 48 L 108 48" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" className={styles.flowArrow} />

          {/* Phase 2: Learn Behavior */}
          <text x="340" y="15" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold">2. Learn Behavior in Imagination</text>

          <rect
            x="230" y="25" width="70" height="45" rx="8" fill="#3b82f6"
            className={`${styles.pipelineStage} ${activeComponent === 'latent' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('latent')}
          />
          <text x="265" y="42" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" pointerEvents="none">Latent</text>
          <text x="265" y="55" textAnchor="middle" fill="white" fontSize="8" pointerEvents="none">State s_t</text>

          <path d="M 190 48 L 228 48" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" />

          <ellipse
            cx="380" cy="47" rx="85" ry="30" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="2" strokeDasharray="4"
            className={`${styles.pipelineStage} ${activeComponent === 'imagination' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('imagination')}
          />
          <text x="380" y="42" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">Imagined</text>
          <text x="380" y="54" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">Trajectories</text>

          <path d="M 300 48 L 293 48" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" className={styles.flowArrow} />

          {/* Actor */}
          <rect
            x="230" y="110" width="90" height="50" rx="8" fill="#10b981"
            className={`${styles.pipelineStage} ${activeComponent === 'actor' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('actor')}
          />
          <text x="275" y="130" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" pointerEvents="none">Action Model</text>
          <text x="275" y="145" textAnchor="middle" fill="white" fontSize="8" pointerEvents="none">{'q_\u03c6(a_t | s_t)'}</text>

          {/* Value */}
          <rect
            x="350" y="110" width="90" height="50" rx="8" fill="rgba(168,85,247,0.3)" stroke="#a855f7" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'value' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('value')}
          />
          <text x="395" y="130" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">Value Model</text>
          <text x="395" y="145" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">{'v_\u03c8(s_t)'}</text>

          {/* Arrows from Imagination to Actor/Value */}
          <path d="M 350 77 Q 310 95 295 110" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" />
          <path d="M 400 77 Q 400 95 398 108" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />

          {/* Gradient flow box */}
          <rect
            x="270" y="180" width="160" height="35" rx="8" fill="rgba(0,217,255,0.15)" stroke="#00d9ff" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'gradients' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('gradients')}
          />
          <text x="350" y="200" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" pointerEvents="none">Backprop through dynamics</text>

          <path d="M 395 160 L 395 178" stroke="#00d9ff" strokeWidth="2" fill="none" markerEnd="url(#arrowCyan)" />
          <path d="M 275 160 L 310 178" stroke="#00d9ff" strokeWidth="2" fill="none" markerEnd="url(#arrowCyan)" />

          {/* Phase 3: Act in Environment */}
          <text x="130" y="110" textAnchor="middle" fill="#00d9ff" fontSize="10" fontWeight="bold">3. Act in Environment</text>

          <rect
            x="20" y="120" width="80" height="50" rx="8" fill="rgba(0,217,255,0.2)" stroke="#00d9ff" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'env' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('env')}
          />
          <text x="60" y="140" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" pointerEvents="none">Environment</text>
          <text x="60" y="153" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">64x64 images</text>

          <path d="M 230 145 Q 170 150 102 145" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="6" />
          <text x="165" y="140" fill="#888" fontSize="8">actions</text>

          <path d="M 60 120 Q 55 90 55 72" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" strokeDasharray="6" />
          <text x="30" y="95" fill="#888" fontSize="8">data</text>

          {/* Value Estimation Section */}
          <text x="250" y="245" textAnchor="middle" fill="#888" fontSize="10" fontWeight="bold">{'Value Estimation: V_\u03bb'}</text>

          <rect
            x="30" y="255" width="130" height="40" rx="8" fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5"
            className={`${styles.pipelineStage} ${activeComponent === 'vr' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('vr')}
          />
          <text x="95" y="272" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="bold" pointerEvents="none">V_R: Sum rewards</text>
          <text x="95" y="285" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">No value model</text>

          <rect
            x="180" y="255" width="130" height="40" rx="8" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5"
            className={`${styles.pipelineStage} ${activeComponent === 'vn' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('vn')}
          />
          <text x="245" y="272" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">{'V_N^k: k-step returns'}</text>
          <text x="245" y="285" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">Bootstrap at step k</text>

          <rect
            x="330" y="255" width="140" height="40" rx="8" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5"
            className={`${styles.pipelineStage} ${activeComponent === 'vlambda' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('vlambda')}
          />
          <text x="400" y="272" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">{'V_\u03bb: Exponential avg'}</text>
          <text x="400" y="285" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">{'\u03bb-weighted (like GAE)'}</text>

          <path d="M 160 275 L 178 275" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGray)" />
          <path d="M 310 275 L 328 275" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGray)" />

          {/* Training Objectives */}
          <rect x="30" y="315" width="210" height="35" rx="8" fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="1.5" />
          <text x="135" y="335" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">{'Actor: max \u03a3 V_\u03bb(s_\u03c4)'}</text>

          <rect x="260" y="315" width="210" height="35" rx="8" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5" />
          <text x="365" y="335" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">{'Critic: min \u03a3 ||v_\u03c8(s_\u03c4) - V_\u03bb||\u00b2'}</text>
        </svg>
      </div>

      {activeComponent && componentInfo[activeComponent] && (
        <div className="info-panel">
          <h4>{componentInfo[activeComponent].title}</h4>
          <p>{componentInfo[activeComponent].desc}</p>
        </div>
      )}

      {/* ====================== Imagination Horizon ====================== */}
      <h2>Imagination Horizon Effect</h2>
      <div className={styles.sliderContainer}>
        <label>Horizon H:</label>
        <input
          type="range"
          className={styles.slider}
          min={1}
          max={45}
          step={1}
          value={horizon}
          onChange={(e) => setHorizon(parseInt(e.target.value))}
        />
        <span className={styles.sliderValue}>{horizon}</span>
      </div>
      <div className={styles.toggleBtns}>
        {([
          { key: 'dreamer' as MethodKey, label: 'Dreamer (V_\u03bb)' },
          { key: 'novalue' as MethodKey, label: 'No Value Model' },
          { key: 'planet' as MethodKey, label: 'PlaNet (planning)' },
        ]).map((btn) => (
          <button
            key={btn.key}
            className={selectedMethod === btn.key ? styles.toggleBtnActive : styles.toggleBtn}
            onClick={() => setSelectedMethod(btn.key)}
          >
            {btn.label}
          </button>
        ))}
      </div>
      <div className="diagram-frame">
        <canvas ref={horizonCanvasRef} className={styles.imaginationCanvas} />
      </div>
      <div className={styles.metricsRow}>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: '#10b981' }}>
            {Math.round(perfAtH)}
          </div>
          <div className={styles.metricLabel}>Avg Return</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: robustnessColor }}>
            {robustness}
          </div>
          <div className={styles.metricLabel}>Horizon Robustness</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: '#00d9ff' }}>5M</div>
          <div className={styles.metricLabel}>Env Steps</div>
        </div>
      </div>
      <p className={styles.hintText}>
        The value model makes Dreamer robust to the imagination horizon &mdash; it estimates rewards beyond H steps.
      </p>

      {/* ====================== Benchmark Results ====================== */}
      <h2>DeepMind Control Suite Results</h2>
      <div className={styles.comparisonContainer}>
        <p className={styles.benchmarkDesc}>
          Average performance across 20 visual control tasks at 5M environment steps:
        </p>
        {Object.entries(benchmarkForTask).map(([method, score]) => {
          const pct = (Math.max(score, 0) / benchMaxVal) * 100;
          const color = methodColors[method];
          const isDreamer = method === 'Dreamer';
          return (
            <div key={method} className={styles.comparisonRow}>
              <span
                className={styles.comparisonLabel}
                style={{ color, fontWeight: isDreamer ? 700 : 400 }}
              >
                {method}
              </span>
              <div className={styles.comparisonBarTrack}>
                <div
                  className={styles.comparisonBarFill}
                  style={{
                    width: `${pct}%`,
                    background: color,
                    opacity: isDreamer ? 1 : 0.7,
                  }}
                />
              </div>
              <span className={styles.comparisonValue} style={{ color }}>
                {score}
              </span>
            </div>
          );
        })}
        <div className={styles.taskSelectContainer}>
          <label className={styles.taskSelectLabel}>Task:</label>
          <select
            className={styles.taskSelect}
            value={selectedTask}
            onChange={(e) => setSelectedTask(e.target.value)}
          >
            <option value="average">Average (20 tasks)</option>
            <option value="cheetah_run">Cheetah Run</option>
            <option value="walker_walk">Walker Walk</option>
            <option value="cartpole_swingup">Cartpole Swingup</option>
            <option value="hopper_hop">Hopper Hop</option>
            <option value="quadruped_walk">Quadruped Walk</option>
            <option value="finger_turn_hard">Finger Turn Hard</option>
          </select>
        </div>
        <p className={styles.hintTextSmall}>
          Dreamer matches model-free performance (D4PG at 10&#x2078; steps) in 20x fewer environment steps.
        </p>
      </div>

      {/* ====================== Training Objectives ====================== */}
      <h2>Training Objectives</h2>
      <div className={styles.modelEquations}>
        {[
          {
            key: 'dynamics',
            color: '#f59e0b',
            title: 'Dynamics Learning: ELBO Objective',
            formula: '\\( \\mathcal{J}_{\\text{REC}} = \\mathbb{E}_p \\left[ \\sum_t \\ln q(o_t | s_t) + \\ln q(r_t | s_t) - \\beta \\, \\text{KL}\\left[ p(s_t | s_{t-1}, a_{t-1}, o_t) \\,\\|\\, q(s_t | s_{t-1}, a_{t-1}) \\right] \\right] \\)',
            desc: 'Reconstruction of observations + rewards, regularized by KL divergence between posterior and prior (encourages consistent latent dynamics)',
          },
          {
            key: 'actor',
            color: '#10b981',
            title: 'Actor Objective: Maximize Imagined Returns',
            formula: '\\( \\max_\\phi \\; \\mathbb{E}_{q_\\theta, q_\\phi} \\left[ \\sum_{\\tau=t}^{t+H} V_\\lambda(s_\\tau) \\right] \\)',
            desc: 'Maximize value estimates along imagined trajectories \u2014 gradients flow back through the dynamics via reparameterization',
          },
          {
            key: 'critic',
            color: '#a855f7',
            title: 'Critic Objective: Regress V_\u03bb Targets',
            formula: '\\( \\min_\\psi \\; \\mathbb{E}_{q_\\theta, q_\\phi} \\left[ \\sum_{\\tau=t}^{t+H} \\frac{1}{2} \\| v_\\psi(s_\\tau) - V_\\lambda(s_\\tau) \\|^2 \\right] \\)',
            desc: 'The value model learns to predict the \u03bb-weighted returns \u2014 targets are computed from imagined trajectories (gradient stopped)',
          },
        ].map((item) => (
          <div
            key={item.key}
            className={activeLoss === item.key ? styles.modelItemActive : styles.modelItem}
            onClick={() => setActiveLoss(activeLoss === item.key ? null : item.key)}
          >
            <h4>
              <span className={styles.modelDot} style={{ background: item.color }} />
              {item.title}
            </h4>
            <div className="formula-block">
              <MathJax>{item.formula}</MathJax>
            </div>
            <p className={styles.modelDesc}>{item.desc}</p>
          </div>
        ))}
      </div>

      {/* ====================== Paper Lineage ====================== */}
      <h2>Paper Lineage</h2>
      <section className="lineage-section">
        <div className="lineage-grid">
          <div className="lineage-group">
            <h4 style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>PlaNet</h5>
              <p>Hafner et al., 2018 &mdash; Same RSSM world model, but uses online planning (CEM) instead of learned behaviors. Dreamer replaces planning with actor-critic.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>World Models</h5>
              <p>Ha &amp; Schmidhuber, 2018 &mdash; Two-stage world model (VAE + MDN-RNN) with evolved controllers in imagination. Pioneer of learning in latent space.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>SAC</h5>
              <p>Haarnoja et al., 2018 &mdash; The tanh-Gaussian policy parameterization and reparameterization trick used in Dreamer&apos;s action model.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>Dyna</h5>
              <p>Sutton, 1991 &mdash; The classic architecture of learning, planning, and acting with a learned model. Dreamer is a deep learning instantiation of Dyna.</p>
            </div>
          </div>
          <div className="lineage-group">
            <h4 style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>DreamerV2</h5>
              <p>Hafner et al., 2021 &mdash; Discrete latent states (categorical), KL balancing, achieves human-level Atari from pixels with a single GPU.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>DreamerV3</h5>
              <p>Hafner et al., 2023 &mdash; Fixed hyperparameters across domains, symlog predictions, scales from Atari to Minecraft to DMLab without tuning.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>DayDreamer</h5>
              <p>Wu et al., 2022 &mdash; Applies Dreamer to real robots (A1 quadruped, UR5 arm), learning locomotion in ~1 hour of real interaction.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>TD-MPC / TD-MPC2</h5>
              <p>Hansen et al., 2022/2024 &mdash; Combines learned latent dynamics with model-predictive control and TD-learning. Alternative to pure imagination-based approach.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
