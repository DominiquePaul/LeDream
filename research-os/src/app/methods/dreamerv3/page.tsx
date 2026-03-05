'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './dreamerv3.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Symlog Predictions',
    desc: 'Transform targets with symlog(x) = sign(x)·ln(|x|+1) to handle vastly different reward scales without domain-specific tuning',
    detail: 'Predicting rewards and returns across domains is challenging because scales vary by orders of magnitude. Squared loss on large targets causes divergence; absolute/Huber losses stagnate; normalizing by running stats introduces non-stationarity. Symlog is a simple solution: symlog(x) = sign(x)·ln(|x|+1), with inverse symexp(x) = sign(x)·(exp(|x|)-1). It compresses large values while preserving the sign and approximating identity near zero. Applied to decoder targets, reward predictions, and critic predictions. Combined with twohot encoding over exponentially-spaced bins for the critic, this decouples gradient scale from target magnitude.',
  },
  {
    title: 'Fixed Hyperparameters',
    desc: 'Single configuration works across 150+ tasks in 8 diverse domains — no tuning needed for new problems',
    detail: 'Previous RL algorithms require extensive hyperparameter tuning for each new domain. DreamerV3 uses the SAME hyperparameters across: Atari (57 tasks, 200M steps), ProcGen (16 tasks), DMLab (30 tasks), Minecraft (1 task, 100M steps), Atari100k (26 tasks), Proprio Control (18 tasks), Visual Control (20 tasks), and BSuite (23 tasks). This is achieved through robustness techniques: symlog predictions handle different reward scales, free bits prevent KL collapse, return normalization with percentile range makes the actor loss scale-invariant, and separate prediction/dynamics/representation losses with fixed weights (β_pred=1, β_dyn=1, β_rep=0.1).',
  },
  {
    title: 'Return Normalization & Free Bits',
    desc: 'Percentile-based return normalization (5th–95th) for scale-invariant actor loss; free bits clip KL below 1 nat to prevent collapse',
    detail: 'Two critical robustness techniques: (1) Return Normalization: Instead of normalizing advantages by standard deviation (which fails for sparse rewards near zero), normalize returns by the range S = EMA(Per(Rλ, 95) - Per(Rλ, 5), 0.99), with a floor of max(1, S) to avoid amplifying noise. Actor loss: sg((Rλ - v(s)) / max(1, S)) · log π(a|s). (2) Free Bits: Clip dynamics and representation KL losses below 1 nat: L_dyn = max(1, KL[sg(posterior) || prior]), L_rep = max(1, KL[posterior || sg(prior)]). This prevents the world model from wasting capacity on already-learned transitions while focusing on the prediction loss. Together these eliminate the need for per-domain tuning.',
  },
];

const keyFeatures = [
  { title: 'Minecraft Diamonds', desc: 'First algorithm to collect diamonds in Minecraft from scratch — no human data, no curricula' },
  { title: '8 Domains, 150+ Tasks', desc: 'Single configuration outperforms domain-specialized expert algorithms across all benchmarks' },
  { title: 'Predictable Scaling', desc: 'Larger models = better performance + less data needed. 12M to 400M parameters scale robustly' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  rssm: {
    title: 'Recurrent State Space Model (RSSM)',
    desc: 'The world model combines a deterministic recurrent path (GRU) with stochastic categorical latent variables (32 categoricals with 32 classes each). The RSSM encodes observations into compact latent states and predicts future states without observations. In V3, all prediction heads use symlog-transformed targets for scale invariance.',
  },
  reward_head: {
    title: 'Reward Prediction Head (Symlog)',
    desc: 'Predicts rewards using symlog squared loss: ||symlog(r) - symlog(r\u0302)||\u00b2. This compresses large rewards logarithmically while preserving small rewards linearly, allowing the same loss function to work across domains with vastly different reward scales (e.g., Atari scores in thousands vs. control rewards in [0,1]).',
  },
  decoder_head: {
    title: 'Decoder Head (Symlog)',
    desc: 'Reconstructs observations using symlog squared loss on pixel values. The symlog transform prevents the decoder from dominating the gradient when pixel values are large, balancing the learning signal across different observation modalities.',
  },
  continue_head: {
    title: 'Continue Prediction Head',
    desc: 'Binary classifier predicting episode continuation probability using Bernoulli loss. Used during imagination to properly discount returns at episode boundaries, replacing the need for explicit done signals.',
  },
  loss_pred: {
    title: 'Prediction Loss (L_pred)',
    desc: 'Sum of reconstruction, reward, and continue losses: L_pred = -ln p(x_t|z_t,h_t) - ln p(r_t|z_t,h_t) - ln p(c_t|z_t,h_t). Weight \u03b2_pred = 1. This is the main learning signal that drives the world model to produce useful latent representations.',
  },
  loss_dyn: {
    title: 'Dynamics Loss (L_dyn) with Free Bits',
    desc: 'L_dyn = max(1, KL[sg(posterior) || prior]). The stop-gradient on the posterior means this loss only trains the prior (dynamics predictor) to match the posterior. The free bits mechanism (max with 1 nat) prevents the KL from being pushed to zero, which would collapse the latent space. Weight \u03b2_dyn = 1.',
  },
  loss_rep: {
    title: 'Representation Loss (L_rep) with Free Bits',
    desc: 'L_rep = max(1, KL[posterior || sg(prior)]). The stop-gradient on the prior means this loss only trains the encoder to produce posteriors close to the prior. Weight \u03b2_rep = 0.1 (lower to prioritize prediction). Free bits prevent wasting model capacity on already-learned transitions.',
  },
  imagination: {
    title: 'Imagined Trajectories',
    desc: 'Starting from encoded real states, the model unrolls the dynamics predictor for H=15 steps using the current actor policy, generating batches of imagined state-action-reward sequences entirely in the latent space. These trajectories are used to compute \u03bb-returns for training both the actor and critic.',
  },
  actor: {
    title: 'Actor with Return Normalization',
    desc: 'The actor uses REINFORCE with normalized advantages: L(\u03b8) = -sg((R\u03bb - v(s)) / max(1, S)) \u00b7 log \u03c0(a|s) + \u03b7\u00b7H[\u03c0]. The normalization by the 5th-95th percentile range S makes the loss scale-invariant across domains. The entropy bonus \u03b7 encourages exploration. For continuous actions, a tanh-Gaussian is used; for discrete, a categorical.',
  },
  critic: {
    title: 'Distributional Critic with Twohot',
    desc: 'Instead of regressing a scalar value, the critic outputs a categorical distribution over exponentially-spaced bins in symexp space. The target \u03bb-return is encoded as a twohot vector (probability spread across two adjacent bins). This distributional approach, combined with symlog/symexp transforms, makes the critic robust to different value scales without any domain-specific tuning.',
  },
  return_norm: {
    title: 'Return Normalization',
    desc: 'Instead of normalizing advantages by standard deviation (which fails for sparse rewards), DreamerV3 normalizes by the 5th-95th percentile range: S = EMA(Per(R\u03bb, 95) - Per(R\u03bb, 5), 0.99). The floor max(1, S) prevents amplifying noise when all returns are similar. This makes the actor loss scale-invariant: the same learning rate works whether rewards are 0.01 or 10,000.',
  },
  env: {
    title: '8 Diverse Domains',
    desc: 'DreamerV3 is evaluated on 150+ tasks across 8 domains with a single set of hyperparameters: Atari (57 games), ProcGen (16 procedural games), DMLab (30 3D tasks), Minecraft (diamond collection), Atari100k (26 games, 100k steps), Proprio Control (18 tasks), Visual Control (20 tasks), and BSuite (23 diagnostic tasks). This unprecedented breadth demonstrates true generality.',
  },
  scaling: {
    title: 'Predictable Model Scaling',
    desc: 'DreamerV3 exhibits predictable scaling behavior: larger models (12M to 400M parameters) consistently achieve better performance and require less data to reach the same level. This is analogous to scaling laws observed in language models, suggesting that world model-based RL can benefit from simply increasing compute.',
  },
};

interface DomainEntry {
  info: string;
  methods: Record<string, number>;
  maxVal: number;
  labels?: Record<string, string>;
}

const domainData: Record<string, DomainEntry> = {
  atari: { info: 'Atari (57 tasks, 200M steps)', methods: { DreamerV3: 900, MuZero: 800, PPO: 300 }, maxVal: 1000 },
  procgen: { info: 'ProcGen (16 tasks)', methods: { DreamerV3: 55, PPG: 55, PPO: 50 }, maxVal: 70 },
  dmlab: { info: 'DMLab (30 tasks)', methods: { DreamerV3: 65, 'R2D2+': 55, IMPALA: 50 }, maxVal: 80 },
  minecraft: { info: 'Minecraft (1 task)', methods: { DreamerV3: 9, IMPALA: 6, PPO: 0 }, maxVal: 10, labels: { DreamerV3: 'Diamond', IMPALA: 'Iron Pickaxe', PPO: 'Nothing' } },
  atari100k: { info: 'Atari100k (26 tasks, 100k steps)', methods: { DreamerV3: 120, EfficientZero: 130, PPO: 50 }, maxVal: 150 },
  proprio: { info: 'Proprio Control (18 tasks)', methods: { DreamerV3: 850, D4PG: 700, PPO: 500 }, maxVal: 1000 },
  visual: { info: 'Visual Control (20 tasks)', methods: { DreamerV3: 900, 'DrQ-v2': 750, PPO: 500 }, maxVal: 1000 },
  bsuite: { info: 'BSuite (23 tasks)', methods: { DreamerV3: 65, BootDQN: 55, PPO: 45 }, maxVal: 80 },
};

const methodColors: Record<string, string> = {
  DreamerV3: '#a855f7',
  MuZero: '#3b82f6',
  PPO: '#888',
  PPG: '#3b82f6',
  'R2D2+': '#3b82f6',
  IMPALA: '#3b82f6',
  EfficientZero: '#3b82f6',
  D4PG: '#3b82f6',
  'DrQ-v2': '#3b82f6',
  BootDQN: '#3b82f6',
};

const domainOptions = [
  { key: 'atari', label: 'Atari' },
  { key: 'procgen', label: 'ProcGen' },
  { key: 'dmlab', label: 'DMLab' },
  { key: 'minecraft', label: 'Minecraft' },
  { key: 'atari100k', label: 'Atari100k' },
  { key: 'proprio', label: 'Proprio' },
  { key: 'visual', label: 'Visual' },
  { key: 'bsuite', label: 'BSuite' },
];

/* ================================================================
   HELPERS
   ================================================================ */

function symlog(x: number): number {
  return Math.sign(x) * Math.log(Math.abs(x) + 1);
}

/* ================================================================
   COMPONENT
   ================================================================ */

export default function DreamerV3Page() {
  /* ---- state ---- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);
  const [activeModelItem, setActiveModelItem] = useState<string | null>(null);
  const [activeDomain, setActiveDomain] = useState('atari');
  const [symlogX, setSymlogX] = useState(100);

  /* ---- canvas ---- */
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width: cw, height: ch } = useCanvasResize(canvasRef);

  /* ---- draw symlog chart ---- */
  const drawSymlog = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = cw;
    const h = ch;
    if (w === 0 || h === 0) return;

    const pad = { left: 60, right: 20, top: 20, bottom: 40 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    const xMin = -1000, xMax = 1000;
    const yMin = -8, yMax = 8;

    function toCanvasX(val: number) { return pad.left + ((val - xMin) / (xMax - xMin)) * pw; }
    function toCanvasY(val: number) { return h - pad.bottom - ((val - yMin) / (yMax - yMin)) * ph; }

    ctx.clearRect(0, 0, w, h);

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Zero lines
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), pad.top);
    ctx.lineTo(toCanvasX(0), h - pad.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, toCanvasY(0));
    ctx.lineTo(w - pad.right, toCanvasY(0));
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Input x', w / 2, h - 8);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Output', 0, 0);
    ctx.restore();

    // Y axis labels
    ctx.textAlign = 'right';
    ctx.font = '10px sans-serif';
    for (let y = -8; y <= 8; y += 2) {
      const cy = toCanvasY(y);
      ctx.fillStyle = '#555';
      ctx.fillText(y.toString(), pad.left - 8, cy + 4);
      ctx.strokeStyle = 'rgba(255,255,255,0.03)';
      ctx.beginPath();
      ctx.moveTo(pad.left, cy);
      ctx.lineTo(w - pad.right, cy);
      ctx.stroke();
    }

    // X axis labels
    ctx.textAlign = 'center';
    for (let x = -1000; x <= 1000; x += 500) {
      if (x === 0) continue;
      const cx = toCanvasX(x);
      ctx.fillStyle = '#555';
      ctx.fillText(x.toString(), cx, h - pad.bottom + 15);
    }
    ctx.fillText('0', toCanvasX(0), h - pad.bottom + 15);

    const numPoints = 500;

    // Identity line (gray dashed)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(136,136,136,0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    for (let i = 0; i <= numPoints; i++) {
      const x = xMin + (i / numPoints) * (xMax - xMin);
      const y = x / 125;
      const cx = toCanvasX(x);
      const cy = toCanvasY(Math.max(yMin, Math.min(yMax, y)));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(136,136,136,0.6)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('x/125 (identity scaled)', toCanvasX(600), toCanvasY(5.2));

    // log(|x|+1) (red dashed)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(239,68,68,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    for (let i = 0; i <= numPoints; i++) {
      const x = xMin + (i / numPoints) * (xMax - xMin);
      const y = Math.log(Math.abs(x) + 1);
      const cx = toCanvasX(x);
      const cy = toCanvasY(Math.max(yMin, Math.min(yMax, y)));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(239,68,68,0.7)';
    ctx.font = '9px sans-serif';
    ctx.fillText('ln(|x|+1)', toCanvasX(650), toCanvasY(7.2));

    // symlog (blue solid)
    ctx.beginPath();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2.5;
    for (let i = 0; i <= numPoints; i++) {
      const x = xMin + (i / numPoints) * (xMax - xMin);
      const y = symlog(x);
      const cx = toCanvasX(x);
      const cy = toCanvasY(Math.max(yMin, Math.min(yMax, y)));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.fillStyle = '#3b82f6';
    ctx.font = '10px sans-serif';
    ctx.fillText('symlog(x)', toCanvasX(650), toCanvasY(6.5) + 30);

    // Current input marker
    const xVal = symlogX;
    const yVal = symlog(xVal);
    const dotCx = toCanvasX(xVal);
    const dotCy = toCanvasY(Math.max(yMin, Math.min(yMax, yVal)));

    // Vertical dashed line
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(dotCx, toCanvasY(0));
    ctx.lineTo(dotCx, dotCy);
    ctx.stroke();
    // Horizontal dashed line
    ctx.beginPath();
    ctx.moveTo(pad.left, dotCy);
    ctx.lineTo(dotCx, dotCy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot
    ctx.beginPath();
    ctx.arc(dotCx, dotCy, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [cw, ch, symlogX]);

  useEffect(() => {
    drawSymlog();
  }, [drawSymlog]);

  /* ---- derived values for metrics ---- */
  const symlogY = symlog(symlogX);
  const compression = Math.abs(symlogX) > 1
    ? `${(Math.abs(symlogX) / Math.abs(symlogY)).toFixed(1)}x`
    : '~1x';

  /* ---- benchmark data ---- */
  const currentDomain = domainData[activeDomain];

  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className="method-page">
      <h1 className={styles.title}>DreamerV3</h1>
      <p className={styles.subtitle}>
        Mastering Diverse Domains through World Models &mdash; Hafner, Pasukonis, Ba, Lillicrap &middot; 2023
      </p>
      <div className={styles.linkRow}>
        <a href="https://danijar.com/dreamerv3" target="_blank" rel="noopener noreferrer" className={styles.paperLink}>
          <span className={styles.linkIcon}>&rarr;</span> danijar.com/dreamerv3
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Main Grid: World Model + Architecture ==================== */}
      <div className={styles.mainGrid}>
        {/* World Model Components */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            World Model Components (RSSM V3)
          </div>
          <div className={styles.modelEquations}>
            {/* Sequence Model */}
            <div
              className={activeModelItem === 'sequence' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'sequence' ? null : 'sequence')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#3b82f6' }} />
                Sequence Model
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( h_t = f_\\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemNote}>Deterministic recurrent backbone (GRU) that maintains temporal context across steps</p>
            </div>
            {/* Encoder */}
            <div
              className={activeModelItem === 'encoder' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'encoder' ? null : 'encoder')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#10b981' }} />
                Encoder
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( z_t \\sim q_\\phi(z_t \\mid h_t, x_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemNote}>Posterior over discrete categorical latents given recurrent state and observation</p>
            </div>
            {/* Dynamics Predictor */}
            <div
              className={activeModelItem === 'dynamics' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'dynamics' ? null : 'dynamics')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#a855f7' }} />
                Dynamics Predictor
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( \\hat{z}_t \\sim p_\\phi(\\hat{z}_t \\mid h_t) \\)'}</MathJax>
              </div>
              <p className={styles.modelItemNote}>Prior that predicts latent state without observation &mdash; enables imagination</p>
            </div>
            {/* Reward Predictor */}
            <div
              className={activeModelItem === 'reward' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'reward' ? null : 'reward')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#ef4444' }} />
                Reward Predictor
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( \\hat{r}_t \\sim p_\\phi(\\hat{r}_t \\mid h_t, z_t) \\)'}</MathJax>
                {' '}<span className={styles.symlogTag}>[symlog squared loss]</span>
              </div>
              <p className={styles.modelItemNote}>Predicts rewards in symlog space for scale-invariant learning across domains</p>
            </div>
            {/* Continue Predictor */}
            <div
              className={activeModelItem === 'continue' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'continue' ? null : 'continue')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#f59e0b' }} />
                Continue Predictor
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( \\hat{c}_t \\sim p_\\phi(\\hat{c}_t \\mid h_t, z_t) \\)'}</MathJax>
                {' '}<span className={styles.bernoulliTag}>[Bernoulli]</span>
              </div>
              <p className={styles.modelItemNote}>Predicts episode continuation probability for proper discounting</p>
            </div>
            {/* Decoder */}
            <div
              className={activeModelItem === 'decoder' ? styles.modelItemActive : styles.modelItem}
              onClick={() => setActiveModelItem(activeModelItem === 'decoder' ? null : 'decoder')}
            >
              <h4 className={styles.modelItemHeader}>
                <span className={styles.modelDot} style={{ background: '#00d9ff' }} />
                Decoder
              </h4>
              <div className={styles.formula}>
                <MathJax>{'\\( \\hat{x}_t \\sim p_\\phi(\\hat{x}_t \\mid h_t, z_t) \\)'}</MathJax>
                {' '}<span className={styles.symlogTag}>[symlog squared loss]</span>
              </div>
              <p className={styles.modelItemNote}>Reconstructs observations in symlog space &mdash; provides learning signal for the latent space</p>
            </div>
          </div>
        </div>

        {/* DreamerV3 Architecture SVG */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            DreamerV3 Architecture
          </div>
          <svg className={styles.pipelineSvg} viewBox="0 0 500 380">
            {/* Arrow Markers (defs first) */}
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
              <marker id="arrowRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
              </marker>
            </defs>

            {/* World Model Section */}
            <text x="130" y="15" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">World Model (symlog heads)</text>

            {/* RSSM */}
            <rect
              x="20" y="25" width="90" height="50" rx="8"
              fill="rgba(245,158,11,0.3)" stroke="#f59e0b" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'rssm' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'rssm' ? null : 'rssm')}
            />
            <text x="65" y="45" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>RSSM</text>
            <text x="65" y="57" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>h_t, z_t (categorical)</text>

            {/* Symlog Heads */}
            <rect
              x="130" y="25" width="75" height="22" rx="6"
              fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'reward_head' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'reward_head' ? null : 'reward_head')}
            />
            <text x="167" y="40" textAnchor="middle" fill="#ef4444" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Reward (symlog)</text>

            <rect
              x="130" y="52" width="75" height="22" rx="6"
              fill="rgba(0,217,255,0.2)" stroke="#00d9ff" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'decoder_head' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'decoder_head' ? null : 'decoder_head')}
            />
            <text x="167" y="67" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Decoder (symlog)</text>

            <rect
              x="215" y="25" width="75" height="22" rx="6"
              fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'continue_head' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'continue_head' ? null : 'continue_head')}
            />
            <text x="252" y="40" textAnchor="middle" fill="#f59e0b" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Continue (Bern.)</text>

            {/* Arrows RSSM to heads */}
            <path d="M 110 40 L 128 40" stroke="#ef4444" strokeWidth="1.5" fill="none" markerEnd="url(#arrowRed)" />
            <path d="M 110 55 L 128 60" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#arrowCyan)" />
            <path d="M 110 38 L 213 36" stroke="#f59e0b" strokeWidth="1.5" fill="none" markerEnd="url(#arrowOrange)" />

            {/* Three Losses */}
            <text x="250" y="95" textAnchor="middle" fill="#888" fontSize="9" fontWeight="bold">Three Separate Losses</text>

            <rect
              x="20" y="105" width="130" height="35" rx="8"
              fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'loss_pred' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'loss_pred' ? null : 'loss_pred')}
            />
            <text x="85" y="118" textAnchor="middle" fill="#10b981" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'L_pred (\u03b2=1)'}</text>
            <text x="85" y="130" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Reconstruction + Reward + Continue</text>

            <rect
              x="160" y="105" width="130" height="35" rx="8"
              fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'loss_dyn' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'loss_dyn' ? null : 'loss_dyn')}
            />
            <text x="225" y="118" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'L_dyn (\u03b2=1)'}</text>
            <text x="225" y="130" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>max(1, KL[sg(post)||prior])</text>

            <rect
              x="300" y="105" width="130" height="35" rx="8"
              fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth="1.5"
              className={`${styles.pipelineStage} ${activeComponent === 'loss_rep' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'loss_rep' ? null : 'loss_rep')}
            />
            <text x="365" y="118" textAnchor="middle" fill="#3b82f6" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'L_rep (\u03b2=0.1)'}</text>
            <text x="365" y="130" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>max(1, KL[post||sg(prior)])</text>

            {/* Free Bits indicator */}
            <rect x="170" y="145" width="250" height="22" rx="6" fill="rgba(245,158,11,0.1)" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4" />
            <text x="295" y="159" textAnchor="middle" fill="#f59e0b" fontSize="8">Free Bits: KL clipped below 1 nat</text>

            {/* Behavior Learning Section */}
            <text x="250" y="185" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold">Behavior Learning in Imagination</text>

            {/* Imagined Trajectories */}
            <ellipse
              cx="130" cy="215" rx="100" ry="28"
              fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="2" strokeDasharray="4"
              className={`${styles.pipelineStage} ${activeComponent === 'imagination' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'imagination' ? null : 'imagination')}
            />
            <text x="130" y="212" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>Imagined Trajectories</text>
            <text x="130" y="225" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Latent rollouts H=15 steps</text>

            {/* Actor */}
            <rect
              x="260" y="195" width="100" height="45" rx="8" fill="#10b981"
              className={`${styles.pipelineStage} ${activeComponent === 'actor' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'actor' ? null : 'actor')}
            />
            <text x="310" y="213" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>Actor</text>
            <text x="310" y="228" textAnchor="middle" fill="white" fontSize="7" style={{ pointerEvents: 'none' }}>{'\u03c0_\u03b8(a_t | s_t)'}</text>

            {/* Critic */}
            <rect
              x="380" y="195" width="100" height="45" rx="8"
              fill="rgba(168,85,247,0.3)" stroke="#a855f7" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'critic' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'critic' ? null : 'critic')}
            />
            <text x="430" y="213" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>Critic (twohot)</text>
            <text x="430" y="228" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>{'v_\u03c8(s_t) over symexp bins'}</text>

            {/* Arrows */}
            <path d="M 230 215 L 258 215" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" className={styles.flowArrow} />
            <path d="M 360 217 L 378 217" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />

            {/* Return Normalization */}
            <rect
              x="260" y="255" width="220" height="35" rx="8"
              fill="rgba(0,217,255,0.15)" stroke="#00d9ff" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'return_norm' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'return_norm' ? null : 'return_norm')}
            />
            <text x="370" y="270" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>Return Normalization</text>
            <text x="370" y="282" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>{'S = EMA(Per(R\u03bb,95) - Per(R\u03bb,5))'}</text>

            <path d="M 310 240 L 330 253" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#arrowCyan)" />
            <path d="M 430 240 L 410 253" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#arrowCyan)" />

            {/* Environment Loop */}
            <text x="80" y="275" textAnchor="middle" fill="#00d9ff" fontSize="10" fontWeight="bold">Environment</text>

            <rect
              x="20" y="285" width="120" height="45" rx="8"
              fill="rgba(0,217,255,0.2)" stroke="#00d9ff" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'env' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'env' ? null : 'env')}
            />
            <text x="80" y="303" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>8 Domains</text>
            <text x="80" y="318" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>150+ tasks, same config</text>

            {/* Scaling */}
            <rect
              x="260" y="305" width="220" height="35" rx="8"
              fill="rgba(168,85,247,0.1)" stroke="#a855f7" strokeWidth="1"
              className={`${styles.pipelineStage} ${activeComponent === 'scaling' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'scaling' ? null : 'scaling')}
            />
            <text x="370" y="318" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>{'Model Scaling: 12M \u2192 400M params'}</text>
            <text x="370" y="330" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Larger models = better + less data</text>

            {/* Loop arrows */}
            <path d="M 140 307 Q 200 307 258 270" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="6" />
            <text x="200" y="300" fill="#888" fontSize="7">actions</text>

            <path d="M 60 285 Q 55 265 55 248" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" strokeDasharray="6" />
            <text x="30" y="268" fill="#888" fontSize="7">data</text>

            {/* Training Objectives Summary */}
            <rect x="20" y="345" width="460" height="28" rx="8" fill="rgba(168,85,247,0.1)" stroke="#a855f7" strokeWidth="1" />
            <text x="250" y="363" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>
              {'L(\u03c6) = \u03b2_pred\u00b7L_pred + \u03b2_dyn\u00b7L_dyn + \u03b2_rep\u00b7L_rep  |  Actor: REINFORCE + return norm  |  Critic: twohot + symexp'}
            </text>
          </svg>

          {activeComponent && componentInfo[activeComponent] && (
            <div className={styles.infoPanel}>
              <h4 className={styles.infoPanelTitle}>{componentInfo[activeComponent].title}</h4>
              <p className={styles.infoPanelDesc}>{componentInfo[activeComponent].desc}</p>
            </div>
          )}
        </div>
      </div>

      {/* ==================== Symlog + Benchmark Row ==================== */}
      <div className={styles.mainGrid}>
        {/* Interactive Symlog Transform */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Symlog Transform
          </div>
          <div className={styles.sliderContainer}>
            <label>Input x:</label>
            <input
              type="range"
              className={styles.slider}
              min={-1000}
              max={1000}
              step={1}
              value={symlogX}
              onChange={(e) => setSymlogX(parseInt(e.target.value, 10))}
            />
            <span className={styles.sliderValue}>{symlogX}</span>
          </div>
          <div className={styles.canvasContainer}>
            <canvas ref={canvasRef} className={styles.symlogCanvas} />
          </div>
          <div className={styles.metricsRow}>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#888' }}>{symlogX}</div>
              <div className={styles.metricLabel}>Raw x</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#3b82f6' }}>{symlogY.toFixed(2)}</div>
              <div className={styles.metricLabel}>symlog(x)</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue} style={{ color: '#10b981' }}>{compression}</div>
              <div className={styles.metricLabel}>Compression</div>
            </div>
          </div>
          <p className={styles.symlogNote}>
            symlog(x) = sign(x)&middot;ln(|x|+1) compresses large values while preserving sign and approximating identity near zero.
          </p>
        </div>

        {/* Multi-Domain Benchmark */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Multi-Domain Benchmark
          </div>
          <div className={styles.comparisonContainer}>
            <div className={styles.toggleBtns}>
              {domainOptions.map((opt) => (
                <button
                  key={opt.key}
                  className={activeDomain === opt.key ? styles.toggleBtnActive : styles.toggleBtn}
                  onClick={() => setActiveDomain(opt.key)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <div className={styles.domainInfo}>{currentDomain.info}</div>
            <div>
              {Object.entries(currentDomain.methods).map(([method, score]) => {
                const pct = Math.max(score, 0) / currentDomain.maxVal * 100;
                const color = methodColors[method] || '#888';
                const isDreamer = method === 'DreamerV3';
                const extraLabel = currentDomain.labels && currentDomain.labels[method]
                  ? ` (${currentDomain.labels[method]})`
                  : '';
                return (
                  <div className={styles.comparisonRow} key={method}>
                    <span className={styles.comparisonLabel} style={{ color, fontWeight: isDreamer ? 700 : 400 }}>{method}</span>
                    <div className={styles.comparisonBarTrack}>
                      <div
                        className={styles.comparisonBarFill}
                        style={{ width: `${pct}%`, background: color, opacity: isDreamer ? 1 : 0.7 }}
                      />
                    </div>
                    <span className={styles.comparisonValue} style={{ color }}>{score}{extraLabel}</span>
                  </div>
                );
              })}
            </div>
            <p className={styles.benchmarkNote}>
              Same hyperparameters across all 8 domains. DreamerV3 matches or exceeds domain-specialized expert algorithms.
            </p>
          </div>
        </div>
      </div>

      {/* ==================== Training Objectives ==================== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Training Objectives
        </div>
        <div className={styles.modelEquations}>
          {/* World Model */}
          <div className={styles.modelItem}>
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#f59e0b' }} />
              World Model (Three Separate Losses)
            </h4>
            <div className={styles.formula}>
              <MathJax>{'\\( \\mathcal{L}(\\phi) = \\mathbb{E}\\left[\\sum_t \\beta_{\\text{pred}} \\cdot \\mathcal{L}_{\\text{pred}} + \\beta_{\\text{dyn}} \\cdot \\mathcal{L}_{\\text{dyn}} + \\beta_{\\text{rep}} \\cdot \\mathcal{L}_{\\text{rep}}\\right] \\)'}</MathJax>
            </div>
            <div className={styles.formulaSmall}>
              <MathJax>{'\\( \\mathcal{L}_{\\text{pred}} = -\\ln p(x_t | z_t, h_t) - \\ln p(r_t | z_t, h_t) - \\ln p(c_t | z_t, h_t) \\)'}</MathJax>
            </div>
            <div className={styles.formulaSmall}>
              <MathJax>{'\\( \\mathcal{L}_{\\text{dyn}} = \\max\\!\\big(1,\\; \\text{KL}[\\,\\text{sg}(q(z_t|h_t,x_t)) \\,\\|\\, p(z_t|h_t)\\,]\\big) \\)'}</MathJax>
              {' '}&larr; free bits
            </div>
            <div className={styles.formulaSmall}>
              <MathJax>{'\\( \\mathcal{L}_{\\text{rep}} = \\max\\!\\big(1,\\; \\text{KL}[\\,q(z_t|h_t,x_t) \\,\\|\\, \\text{sg}(p(z_t|h_t))\\,]\\big) \\)'}</MathJax>
              {' '}&larr; free bits
            </div>
            <p className={styles.objectiveNote}>
              &beta;<sub>pred</sub>=1, &beta;<sub>dyn</sub>=1, &beta;<sub>rep</sub>=0.1. Free bits (max with 1 nat) prevent KL from collapsing while focusing capacity on prediction loss.
            </p>
          </div>

          {/* Critic */}
          <div className={styles.modelItem}>
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#a855f7' }} />
              Critic (Distributional with Twohot)
            </h4>
            <div className={styles.formula}>
              <MathJax>{'\\( \\mathcal{L}(\\psi) = -\\sum_t \\ln\\, p_\\psi\\!\\left(R^\\lambda_t \\,\\big|\\, s_t\\right) \\)'}</MathJax>
            </div>
            <p className={styles.objectiveNote}>
              Critic outputs a softmax distribution over exponentially-spaced symexp bins. Twohot encoding places probability mass on two adjacent bins. This decouples gradient scale from target magnitude.
            </p>
          </div>

          {/* Actor */}
          <div className={styles.modelItem}>
            <h4 className={styles.modelItemHeader}>
              <span className={styles.modelDot} style={{ background: '#10b981' }} />
              Actor (REINFORCE with Return Normalization)
            </h4>
            <div className={styles.formula}>
              <MathJax>{'\\( \\mathcal{L}(\\theta) = -\\sum_t \\text{sg}\\!\\left(\\frac{R^\\lambda_t - v_\\psi(s_t)}{\\max(1, S)}\\right) \\cdot \\ln \\pi_\\theta(a_t | s_t) + \\eta \\cdot H[\\pi_\\theta(a_t | s_t)] \\)'}</MathJax>
            </div>
            <div className={styles.formulaSmall}>
              <MathJax>{'\\( S = \\text{EMA}\\!\\left(\\text{Per}(R^\\lambda, 95) - \\text{Per}(R^\\lambda, 5),\\; 0.99\\right) \\)'}</MathJax>
            </div>
            <p className={styles.objectiveNote}>
              Advantages normalized by 5th-95th percentile range with floor max(1, S) to avoid amplifying noise. Entropy bonus &eta; encourages exploration.
            </p>
          </div>
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
                <div className={styles.lineageItemTitle}>DreamerV2</div>
                <div className={styles.lineageItemDesc}>Hafner et al., 2021 &mdash; Discrete categoricals and KL balancing. V3 adds robustness techniques for domain generality.</div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>DreamerV1</div>
                <div className={styles.lineageItemDesc}>Hafner et al., 2019 &mdash; Original latent imagination framework with Gaussian latents and dynamics backprop.</div>
              </div>
              <div className={styles.lineageItemBuildsOn}>
                <div className={styles.lineageItemTitle}>Symlog Family</div>
                <div className={styles.lineageItemDesc}>Bi-symmetric logarithmic transforms for handling wide value ranges in neural network predictions.</div>
              </div>
            </div>
          </div>
          <div>
            <h4 className={styles.lineageSectionTitle} style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className={styles.lineageList}>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>DayDreamer</div>
                <div className={styles.lineageItemDesc}>Wu et al., 2022 &mdash; Real robot learning with the Dreamer framework.</div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>TD-MPC2</div>
                <div className={styles.lineageItemDesc}>Hansen et al., 2024 &mdash; Combines world models with model-predictive control, uses similar multi-domain evaluation.</div>
              </div>
              <div className={styles.lineageItemBuiltUpon}>
                <div className={styles.lineageItemTitle}>DIAMOND</div>
                <div className={styles.lineageItemDesc}>Alonso et al., 2024 &mdash; Diffusion world model for Atari, building on Dreamer&#39;s latent imagination framework.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
