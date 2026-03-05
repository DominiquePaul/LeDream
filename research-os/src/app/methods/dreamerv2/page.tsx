'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './dreamerv2.module.css';

/* ================================================================
   Data constants
   ================================================================ */

const coreIdeas = [
  {
    title: 'Discrete Latent Representations',
    desc: 'Replace Gaussian latents with 32 categorical variables × 32 classes, creating a sparse binary vector of 1024 with 32 active bits',
    detail: "The stochastic state in DreamerV2 is a vector of 32 categorical variables, each with 32 classes. Sampling produces a sparse binary vector of length 1024 with exactly 32 active bits. This is fundamentally different from DreamerV1's diagonal Gaussian latents. Benefits: (1) A categorical prior can perfectly fit the aggregate posterior (mixture of categoricals is categorical), while a Gaussian cannot match a mixture of Gaussians. (2) Sparsity from one-hot vectors aids generalization. (3) Straight-through gradients avoid exploding/vanishing gradient issues of reparameterization. (4) Better inductive bias for modeling discrete events in Atari (rooms, items appearing/disappearing).",
  },
  {
    title: 'KL Balancing',
    desc: 'Asymmetric KL loss with α=0.8 trains the prior faster than it regularizes the posterior, preventing collapse',
    detail: 'The KL divergence in the ELBO serves dual purposes: training the prior toward the posterior AND regularizing the posterior toward the prior. Standard KL treats both equally, but learning the transition function is harder than learning representations. KL balancing uses α=0.8 for the prior learning term and (1−α)=0.2 for the posterior regularization term. Implementation: kl_loss = α · KL[sg(posterior) || prior] + (1−α) · KL[posterior || sg(prior)]. This encourages learning an accurate prior over increasing posterior entropy. KL balancing outperforms standard KL on 44 of 55 Atari tasks.',
  },
  {
    title: 'Mixed Actor Gradients',
    desc: 'Combine Reinforce (unbiased, high variance) with straight-through dynamics backprop (biased, low variance) via mixing ratio ρ',
    detail: 'DreamerV2 combines two gradient estimators for the actor: (1) REINFORCE — unbiased but high variance policy gradient using ln π(a|s) · (V^λ − v(s)) as baseline-subtracted advantage; (2) Straight-through dynamics backpropagation — biased but low variance gradients flowing back through the learned dynamics. Combined: L(ψ) = −ρ · ln p(a|s) · sg(V^λ − v(s)) − (1−ρ) · V^λ − η · H[a|s]. For Atari (discrete actions): ρ=1 (pure REINFORCE works better). For continuous control: ρ=0 (dynamics backprop works better). The entropy regularizer η encourages exploration.',
  },
];

const keyFeatures = [
  { title: 'Human-Level Atari', desc: 'First agent to achieve human-level performance on 55 Atari games purely through world model imagination' },
  { title: 'Single GPU', desc: 'Trains in 10 days on a single V100 GPU, 200M frames, 22M parameters' },
  { title: 'Discrete Latents', desc: '32×32 categorical variables: sparse, multimodal, better than Gaussian for complex environments' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  dataset: {
    title: 'Dataset of Experience',
    desc: 'A growing replay buffer of (image, action, reward) sequences collected by the agent. For Atari, images are 64\u00d764 grayscale. The dataset grows as the agent interacts with the environment using its learned policy, continuously improving the world model with new experience.',
  },
  rssm: {
    title: 'Recurrent State Space Model (RSSM) with Discrete Latents',
    desc: 'The RSSM combines a deterministic recurrent path (GRU with 600 units) with stochastic latent variables. In DreamerV2, the stochastic variables are 32 categorical distributions with 32 classes each, replacing the Gaussian distributions of V1. The model state is s_t = (h_t, z_t) where h_t is the deterministic recurrent state and z_t is the discrete stochastic state.',
  },
  categorical: {
    title: '32\u00d732 Categorical Grid',
    desc: 'Each stochastic state consists of 32 categorical variables, each choosing one of 32 classes. This produces a one-hot vector per variable, concatenated into a sparse binary vector of length 1024 with exactly 32 active bits (one per variable). The highlighted cells in the grid represent the active classes. Gradients flow through discrete sampling via the straight-through estimator.',
  },
  klbalance: {
    title: 'KL Balancing (\u03b1=0.8)',
    desc: 'The KL divergence loss is split into two terms with asymmetric weights: kl_loss = \u03b1 \u00b7 KL[sg(posterior) || prior] + (1\u2212\u03b1) \u00b7 KL[posterior || sg(prior)]. With \u03b1=0.8, the prior is trained 4x more aggressively than the posterior is regularized. The stop-gradient (sg) ensures each term only trains one side. This prevents the common failure mode where the posterior collapses to match an undertrained prior.',
  },
  imagination: {
    title: 'Imagined Trajectories',
    desc: 'Starting from encoded real states, the transition predictor unrolls for H=15 steps using the current actor policy. This generates batches of imagined state-action-reward sequences entirely in the latent space. For each batch of 50 real states, 15-step imagined trajectories are generated, creating 750 (s, a, r) tuples per batch for actor-critic training.',
  },
  actor: {
    title: 'Actor with Mixed Gradients',
    desc: 'The actor combines REINFORCE and straight-through dynamics backpropagation. For Atari (discrete actions), \u03c1=1 uses pure REINFORCE with a learned value baseline. For continuous control, \u03c1=0 uses pure dynamics backprop. The entropy bonus \u03b7\u00b7H[a|s] encourages exploration. The actor network is an MLP with 4 layers of 400 units and ELU activations.',
  },
  value: {
    title: 'Critic / Value Model',
    desc: 'The critic v_\u03be(s_t) estimates the \u03bb-return V^\u03bb. Trained by regression: min \u00bd||v_\u03be(s_t) \u2212 sg(V^\u03bb_t)||\u00b2. Stop-gradient on targets prevents the critic loss from affecting the world model. The critic uses the same architecture as the actor: 4 layers of 400 units with ELU. A separate slow-moving target network (EMA) is used for computing V^\u03bb targets.',
  },
  mixedgrad: {
    title: 'Mixed Gradient Estimator',
    desc: 'The key innovation for the actor: combining REINFORCE (unbiased, high variance) with straight-through dynamics backprop (biased, low variance). The mixing ratio \u03c1 controls the balance. REINFORCE uses the advantage V^\u03bb \u2212 v(s) as a baseline to reduce variance. Dynamics backprop flows gradients through the entire imagined trajectory back through the transition model. The optimal \u03c1 depends on the action space.',
  },
  env: {
    title: 'Atari Environment',
    desc: '55 Atari games from the Arcade Learning Environment. Images are downscaled to 64\u00d764 grayscale. The agent uses action repeat of 1 (every frame) and is trained for 200M environment frames on a single NVIDIA V100 GPU over approximately 10 days. DreamerV2 achieves a human-normalized median score of 2.15, surpassing IQN, Rainbow, and other model-free methods.',
  },
  gaussian: {
    title: 'V1: Diagonal Gaussian Latents',
    desc: 'DreamerV1 uses 30-dimensional diagonal Gaussian latent variables z ~ N(\u03bc, \u03c3\u00b2). Gradients flow through sampling via the reparameterization trick: z = \u03bc + \u03c3 \u00b7 \u03b5, \u03b5 ~ N(0,I). While mathematically elegant, this has limitations: (1) the aggregate posterior is a mixture of Gaussians, which a single Gaussian prior cannot match, (2) continuous representations may not be ideal for discrete environments like Atari.',
  },
  v2categorical: {
    title: 'V2: Categorical Latents (32\u00d732)',
    desc: 'DreamerV2 replaces Gaussians with 32 categorical variables of 32 classes each. Gradients flow through the straight-through estimator: forward pass uses sampled one-hot vectors, backward pass uses the softmax probabilities. Advantages: (1) categorical priors can exactly match the aggregate posterior, (2) sparse one-hot vectors provide natural regularization, (3) stable gradients without exploding/vanishing issues, (4) 2^1024 possible discrete states vs continuous Gaussian space.',
  },
};

const benchmarkData: Record<string, Record<string, number>> = {
  gamer_median: { DreamerV2: 2.15, IQN: 1.29, Rainbow: 1.47, C51: 1.09, DQN: 0.65 },
  gamer_mean: { DreamerV2: 11.33, IQN: 8.85, Rainbow: 9.12, C51: 7.7, DQN: 2.84 },
  clipped_record: { DreamerV2: 0.28, IQN: 0.21, Rainbow: 0.17, C51: 0.15, DQN: 0.12 },
};

const methodColors: Record<string, string> = {
  DreamerV2: '#a855f7',
  IQN: '#3b82f6',
  Rainbow: '#10b981',
  C51: '#f59e0b',
  DQN: '#888',
};

/* ================================================================
   Component
   ================================================================ */

export default function DreamerV2Page() {
  /* ----- interactive state ----- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);
  const [activeModelItem, setActiveModelItem] = useState<string | null>(null);
  const [alphaValue, setAlphaValue] = useState(80);
  const [selectedMetric, setSelectedMetric] = useState('gamer_median');

  /* ----- canvas refs ----- */
  const klCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const klSize = useCanvasResize(klCanvasRef);

  /* ----- derived values ----- */
  const alpha = alphaValue / 100;

  /* ----- KL balance metrics ----- */
  const balanceLabel = (() => {
    if (alpha >= 0.7 && alpha <= 0.9) return { text: 'Optimal', color: '#10b981' };
    if (alpha >= 0.5 && alpha < 0.7) return { text: 'Moderate', color: '#f59e0b' };
    if (alpha > 0.9) return { text: 'Aggressive', color: '#ef4444' };
    return { text: 'Over-Reg.', color: '#ef4444' };
  })();

  /* ================================================================
     KL Balancing Canvas drawing
     ================================================================ */
  const drawKLViz = useCallback(() => {
    const canvas = klCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = klSize.width;
    const h = klSize.height;
    if (w === 0 || h === 0) return;

    const pad = { left: 55, right: 20, top: 25, bottom: 40 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    /* Axes */
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
    ctx.fillText('Training Steps', w / 2, h - 8);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Learning Signal', 0, 0);
    ctx.restore();

    /* Y axis labels */
    ctx.textAlign = 'right';
    ctx.font = '10px sans-serif';
    for (let y = 0; y <= 1.0; y += 0.2) {
      const cy = h - pad.bottom - (y / 1.0) * ph;
      ctx.fillStyle = '#555';
      ctx.fillText(y.toFixed(1), pad.left - 8, cy + 4);
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.beginPath();
      ctx.moveTo(pad.left, cy);
      ctx.lineTo(w - pad.right, cy);
      ctx.stroke();
    }

    /* X axis labels */
    ctx.textAlign = 'center';
    const stepLabels = ['0', '50k', '100k', '150k', '200k'];
    stepLabels.forEach((label, i) => {
      const cx = pad.left + (i / (stepLabels.length - 1)) * pw;
      ctx.fillStyle = '#555';
      ctx.fillText(label, cx, h - pad.bottom + 15);
    });

    const nPoints = 100;

    /* Prior learning curve */
    ctx.beginPath();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.5;
    for (let i = 0; i < nPoints; i++) {
      const t = i / (nPoints - 1);
      const priorRate = alpha;
      const val = priorRate * (1 - Math.exp(-t * (3 + 4 * alpha)));
      const cx = pad.left + t * pw;
      const cy = h - pad.bottom - val * ph;
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    /* Posterior regularization curve */
    ctx.beginPath();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    for (let i = 0; i < nPoints; i++) {
      const t = i / (nPoints - 1);
      const postRate = 1 - alpha;
      const val = postRate * (1 - Math.exp(-t * (3 + 4 * (1 - alpha))));
      const cx = pad.left + t * pw;
      const cy = h - pad.bottom - val * ph;
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    /* Optimal zone indicator */
    if (alpha >= 0.7 && alpha <= 0.9) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
      ctx.fillRect(pad.left, pad.top, pw, ph);
    }

    /* Legend */
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#a855f7';
    ctx.fillRect(pad.left + 10, pad.top + 5, 16, 3);
    ctx.fillText('Prior Learning (\u03b1)', pad.left + 32, pad.top + 12);

    ctx.fillStyle = '#10b981';
    ctx.fillRect(pad.left + 10, pad.top + 20, 16, 3);
    ctx.fillText('Posterior Regularization (1\u2212\u03b1)', pad.left + 32, pad.top + 27);

    /* Alpha vertical indicator line */
    const alphaX = pad.left + alpha * pw;
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(alphaX, pad.top);
    ctx.lineTo(alphaX, h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('\u03b1=' + alpha.toFixed(2), alphaX, pad.top - 5);
  }, [alpha, klSize]);

  useEffect(() => {
    drawKLViz();
  }, [drawKLViz]);

  /* ================================================================
     Benchmark bars rendering
     ================================================================ */
  const currentBenchmark = benchmarkData[selectedMetric];
  const maxVal = Math.max(...Object.values(currentBenchmark), 0.01);

  /* ================================================================
     JSX
     ================================================================ */
  return (
    <div className="method-page">
      <div className={styles.container}>
        <h1 className={styles.pageTitle}>DreamerV2</h1>
        <p className={styles.subtitle}>
          Mastering Atari with Discrete World Models &mdash; Hafner, Lillicrap, Norouzi, Ba &middot; ICLR 2021
        </p>
        <div className={styles.paperLink}>
          <a href="https://danijar.com/dreamerv2" target="_blank" rel="noopener noreferrer">
            <span className={styles.linkIcon}>&rarr;</span> danijar.com/dreamerv2
          </a>
        </div>

        <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

        {/* ==================== Main Grid: World Model + Architecture ==================== */}
        <div className={styles.mainGrid}>
          {/* ----- World Model Components ----- */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              World Model Components (RSSM)
            </div>
            <div className={styles.modelEquations}>
              {/* Recurrent Model */}
              <div
                className={activeModelItem === 'recurrent' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'recurrent' ? null : 'recurrent')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#3b82f6' }} />Recurrent Model</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Deterministic GRU backbone that integrates past information into a recurrent state</p>
              </div>
              {/* Representation Model */}
              <div
                className={activeModelItem === 'repr' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'repr' ? null : 'repr')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#10b981' }} />Representation Model</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( z_t \sim q_\phi(z_t \mid h_t, x_t) \quad \text{[32 categoricals} \times \text{32 classes]} \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Encodes observations into discrete latent states &mdash; 32 categorical variables each with 32 classes</p>
              </div>
              {/* Transition Predictor */}
              <div
                className={activeModelItem === 'trans' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'trans' ? null : 'trans')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#f59e0b' }} />Transition Predictor</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t) \quad \text{[predicts without image]} \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Predicts next discrete latent state without seeing the observation &mdash; enables imagination</p>
              </div>
              {/* Image Predictor */}
              <div
                className={activeModelItem === 'image' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'image' ? null : 'image')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#a855f7' }} />Image Predictor</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( \hat{x}_t \sim p_\phi(\hat{x}_t \mid h_t, z_t) \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Reconstructs images from the combined deterministic and stochastic state</p>
              </div>
              {/* Reward Predictor */}
              <div
                className={activeModelItem === 'reward' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'reward' ? null : 'reward')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#ef4444' }} />Reward Predictor</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( \hat{r}_t \sim p_\phi(\hat{r}_t \mid h_t, z_t) \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Predicts scalar rewards from latent states to evaluate imagined trajectories</p>
              </div>
              {/* Discount Predictor */}
              <div
                className={activeModelItem === 'discount' ? styles.modelItemActive : styles.modelItem}
                onClick={() => setActiveModelItem(activeModelItem === 'discount' ? null : 'discount')}
              >
                <h4><span className={styles.modelDot} style={{ background: '#00d9ff' }} />Discount Predictor</h4>
                <div className={styles.formula}>
                  <MathJax inline>{String.raw`\( \hat{\gamma}_t \sim p_\phi(\hat{\gamma}_t \mid h_t, z_t) \)`}</MathJax>
                </div>
                <p className={styles.modelDesc}>Predicts episode continuation probability &mdash; replaces fixed discount with a learned signal</p>
              </div>
            </div>
          </div>

          {/* ----- DreamerV2 Architecture SVG ----- */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              DreamerV2 Architecture
            </div>
            <svg className={styles.pipelineSvg} viewBox="0 0 500 380">
              {/* Arrow Markers */}
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

              {/* Phase 1: World Model Learning */}
              <text x="120" y="15" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">1. World Model Learning</text>

              {/* Dataset */}
              <rect
                x="15" y="25" width="65" height="45" rx="8" fill="#f59e0b"
                className={`${styles.pipelineStage} ${activeComponent === 'dataset' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'dataset' ? null : 'dataset')}
              />
              <text x="47" y="45" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" pointerEvents="none">Dataset</text>
              <text x="47" y="57" textAnchor="middle" fill="white" fontSize="8" pointerEvents="none">(x, a, r)</text>

              {/* RSSM with Discrete */}
              <rect
                x="100" y="25" width="90" height="45" rx="8" fill="rgba(245,158,11,0.3)" stroke="#f59e0b" strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'rssm' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'rssm' ? null : 'rssm')}
              />
              <text x="145" y="42" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold" pointerEvents="none">RSSM</text>
              <text x="145" y="55" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">Discrete Latents</text>

              {/* 32x32 Categorical Grid */}
              <rect
                x="210" y="25" width="55" height="45" rx="6" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'categorical' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'categorical' ? null : 'categorical')}
              />
              <text x="237" y="39" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">32&times;32</text>
              {/* Mini grid to represent categoricals */}
              <g transform="translate(215, 43)" pointerEvents="none">
                <rect x="0" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="6" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="12" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="18" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="24" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="30" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="36" y="0" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="0" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="6" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="12" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="18" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="24" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="30" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="36" y="6" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="0" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="6" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="12" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="18" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.9" />
                <rect x="24" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="30" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
                <rect x="36" y="12" width="4" height="4" rx="1" fill="#a855f7" opacity="0.3" />
              </g>

              {/* Arrow Dataset to RSSM */}
              <path d="M 80 48 L 98 48" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" className={styles.flowArrow} />
              {/* Arrow RSSM to Categorical */}
              <path d="M 190 48 L 208 48" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />

              {/* KL Balancing */}
              <rect
                x="100" y="82" width="90" height="28" rx="6" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'klbalance' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'klbalance' ? null : 'klbalance')}
              />
              <text x="145" y="100" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">KL Balancing (&alpha;=0.8)</text>

              {/* Phase 2: Actor-Critic in Imagination */}
              <text x="380" y="15" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">2. Actor-Critic in Imagination</text>

              {/* Imagination Cloud */}
              <ellipse
                cx="380" cy="50" rx="85" ry="30" fill="rgba(16,185,129,0.12)" stroke="#10b981" strokeWidth="2" strokeDasharray="4"
                className={`${styles.pipelineStage} ${activeComponent === 'imagination' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'imagination' ? null : 'imagination')}
              />
              <text x="380" y="45" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">Imagined</text>
              <text x="380" y="57" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">Trajectories (H=15)</text>

              {/* Arrow Categorical to Imagination */}
              <path d="M 265 48 L 293 48" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" className={styles.flowArrow} />

              {/* Actor */}
              <rect
                x="290" y="100" width="80" height="50" rx="8" fill="#10b981"
                className={`${styles.pipelineStage} ${activeComponent === 'actor' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'actor' ? null : 'actor')}
              />
              <text x="330" y="120" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" pointerEvents="none">Actor</text>
              <text x="330" y="135" textAnchor="middle" fill="white" fontSize="7" pointerEvents="none">REINFORCE</text>
              <text x="330" y="145" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="7" pointerEvents="none">+ Straight-Through</text>

              {/* Value / Critic */}
              <rect
                x="390" y="100" width="80" height="50" rx="8" fill="rgba(168,85,247,0.3)" stroke="#a855f7" strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'value' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'value' ? null : 'value')}
              />
              <text x="430" y="120" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">Critic</text>
              <text x="430" y="135" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">{'v_\u03be(s_t)'}</text>

              {/* Arrows from Imagination to Actor/Critic */}
              <path d="M 340 80 Q 335 92 332 98" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" />
              <path d="M 420 80 Q 428 92 430 98" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />

              {/* Mixed Gradients Box */}
              <rect
                x="290" y="168" width="180" height="32" rx="8" fill="rgba(0,217,255,0.12)" stroke="#00d9ff" strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'mixedgrad' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'mixedgrad' ? null : 'mixedgrad')}
              />
              <text x="380" y="188" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold" pointerEvents="none">Mixed Gradients: &rho;&middot;REINFORCE + (1-&rho;)&middot;Backprop</text>

              <path d="M 330 150 L 345 166" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#arrowCyan)" />
              <path d="M 430 150 L 415 166" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#arrowCyan)" />

              {/* Phase 3: Act in Environment */}
              <text x="80" y="135" textAnchor="middle" fill="#00d9ff" fontSize="10" fontWeight="bold">3. Act in Environment</text>

              {/* Environment */}
              <rect
                x="15" y="145" width="80" height="50" rx="8" fill="rgba(0,217,255,0.2)" stroke="#00d9ff" strokeWidth="2"
                className={`${styles.pipelineStage} ${activeComponent === 'env' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'env' ? null : 'env')}
              />
              <text x="55" y="165" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" pointerEvents="none">Atari</text>
              <text x="55" y="178" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">64&times;64 images</text>

              {/* Arrow: Actor to Env */}
              <path d="M 290 135 Q 190 155 97 170" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="6" />
              <text x="190" y="148" fill="#888" fontSize="8">actions</text>

              {/* Arrow: Env to Dataset */}
              <path d="M 55 145 Q 50 100 48 72" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" strokeDasharray="6" />
              <text x="30" y="115" fill="#888" fontSize="8">data</text>

              {/* Discrete vs Gaussian Comparison */}
              <text x="250" y="230" textAnchor="middle" fill="#888" fontSize="10" fontWeight="bold">Latent Space Comparison</text>

              {/* V1 Gaussian */}
              <rect
                x="30" y="245" width="190" height="50" rx="8" fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'gaussian' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'gaussian' ? null : 'gaussian')}
              />
              <text x="125" y="262" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">V1: Diagonal Gaussian</text>
              <text x="125" y="278" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">{'z ~ N(\u03bc, \u03c3\u00b2) \u00b7 30 dim continuous'}</text>
              <text x="125" y="290" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">Reparameterization gradients</text>

              {/* V2 Categorical */}
              <rect
                x="250" y="245" width="220" height="50" rx="8" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5"
                className={`${styles.pipelineStage} ${activeComponent === 'v2categorical' ? styles.pipelineStageActive : ''}`}
                onClick={() => setActiveComponent(activeComponent === 'v2categorical' ? null : 'v2categorical')}
              />
              <text x="360" y="262" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">V2: 32 Categoricals &times; 32 Classes</text>
              <text x="360" y="278" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">{'z \u2208 {0,1}\u00b9\u2070\u00b2\u2074 \u00b7 32 active bits'}</text>
              <text x="360" y="290" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">Straight-through gradients</text>

              {/* Arrow between them */}
              <path d="M 220 270 L 248 270" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#arrowGray)" />
              <text x="234" y="264" fill="#10b981" fontSize="10" fontWeight="bold">&rarr;</text>

              {/* Training Objectives */}
              <rect x="30" y="315" width="210" height="35" rx="8" fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="1.5" />
              <text x="135" y="330" textAnchor="middle" fill="#10b981" fontSize="8" fontWeight="bold" pointerEvents="none">{'Actor: \u03c1\u00b7REINFORCE + (1-\u03c1)\u00b7V\u03bb'}</text>
              <text x="135" y="343" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">+ entropy reg. &eta;&middot;H[a|s]</text>

              <rect x="260" y="315" width="210" height="35" rx="8" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5" />
              <text x="365" y="330" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">{'Critic: min \u00bd||v_\u03be(s) - sg(V\u03bb)||\u00b2'}</text>
              <text x="365" y="343" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">Stop-gradient on targets</text>
            </svg>

            {activeComponent && componentInfo[activeComponent] && (
              <div className={styles.infoPanel}>
                <h4>{componentInfo[activeComponent].title}</h4>
                <p>{componentInfo[activeComponent].desc}</p>
              </div>
            )}
          </div>
        </div>

        {/* ==================== KL Balancing + Atari Benchmark ==================== */}
        <div className={styles.mainGrid}>
          {/* ----- KL Balancing Interactive ----- */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              KL Balancing Effect
            </div>
            <div className={styles.sliderContainer}>
              <label>&alpha; (balance):</label>
              <input
                type="range"
                className={styles.slider}
                min={0}
                max={100}
                step={1}
                value={alphaValue}
                onChange={(e) => setAlphaValue(Number(e.target.value))}
              />
              <span className={styles.sliderValue}>{alpha.toFixed(2)}</span>
            </div>
            <div className={styles.canvasContainer}>
              <canvas ref={klCanvasRef} className={styles.imaginationCanvas} />
            </div>
            <div className={styles.metricsRow}>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: '#a855f7' }}>{alpha.toFixed(2)}</div>
                <div className={styles.metricLabel}>Prior Learning</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: '#10b981' }}>{(1 - alpha).toFixed(2)}</div>
                <div className={styles.metricLabel}>Posterior Reg.</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricValue} style={{ color: balanceLabel.color }}>{balanceLabel.text}</div>
                <div className={styles.metricLabel}>Balance</div>
              </div>
            </div>
            <p className={styles.klExplainer}>
              &alpha;=0.8 encourages learning an accurate prior while limiting posterior regularization. KL balancing outperforms standard KL on 44 of 55 Atari tasks.
            </p>
          </div>

          {/* ----- Atari Benchmark Results ----- */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>
              Atari Benchmark Results
            </div>
            <div className={styles.comparisonContainer}>
              <p className={styles.benchmarkNote}>Performance across 55 Atari games (200M frames):</p>
              <div>
                {Object.entries(currentBenchmark).map(([method, score]) => {
                  const pct = (Math.max(score, 0) / maxVal) * 100;
                  const color = methodColors[method];
                  const isDV2 = method === 'DreamerV2';
                  return (
                    <div className={styles.comparisonRow} key={method}>
                      <span className={styles.comparisonLabel} style={{ color, fontWeight: isDV2 ? 700 : 400 }}>{method}</span>
                      <div className={styles.comparisonBarTrack}>
                        <div className={styles.comparisonBarFill} style={{ width: `${pct}%`, background: color, opacity: isDV2 ? 1 : 0.7 }} />
                      </div>
                      <span className={styles.comparisonValue} style={{ color }}>{score}</span>
                    </div>
                  );
                })}
              </div>
              <div className={styles.metricSelectContainer}>
                <label>Metric:</label>
                <select
                  className={styles.metricSelect}
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  <option value="gamer_median">Gamer Median</option>
                  <option value="gamer_mean">Gamer Mean</option>
                  <option value="clipped_record">Clipped Record Mean</option>
                </select>
              </div>
              <p className={styles.benchmarkFooter}>
                DreamerV2 is the first agent to achieve human-level Atari performance using a world model, on a single GPU.
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
            {/* World Model ELBO */}
            <div className={styles.modelItem}>
              <h4><span className={styles.modelDot} style={{ background: '#f59e0b' }} />World Model: ELBO / Variational Free Energy</h4>
              <div className={styles.formula}>
                <MathJax inline>{String.raw`\( \mathcal{L}(\phi) = \mathbb{E}\left[\sum_t -\ln p(x_t|h_t,z_t) - \ln p(r_t|h_t,z_t) - \ln p(\gamma_t|h_t,z_t) + \beta \cdot \text{KL}\left[q(z_t|h_t,x_t) \| p(z_t|h_t)\right]\right] \)`}</MathJax>
              </div>
              <p className={styles.modelDesc}>&beta;=0.1 for Atari, &beta;=1.0 for continuous control. KL uses balancing with &alpha;=0.8.</p>
            </div>
            {/* Actor Loss */}
            <div className={styles.modelItem}>
              <h4><span className={styles.modelDot} style={{ background: '#10b981' }} />Actor Loss (combined)</h4>
              <div className={styles.formula}>
                <MathJax inline>{String.raw`\( \mathcal{L}(\psi) = \mathbb{E}\left[\sum_t -\rho \ln p(a_t|s_t) \cdot \text{sg}(V^\lambda_t - v(s_t)) - (1-\rho) \cdot V^\lambda_t - \eta \cdot H[a_t|s_t]\right] \)`}</MathJax>
              </div>
              <p className={styles.modelDesc}>For Atari (discrete): &rho;=1 (pure REINFORCE). For continuous control: &rho;=0 (dynamics backprop). Entropy coefficient &eta; encourages exploration.</p>
            </div>
            {/* Critic Loss */}
            <div className={styles.modelItem}>
              <h4><span className={styles.modelDot} style={{ background: '#a855f7' }} />Critic Loss</h4>
              <div className={styles.formula}>
                <MathJax inline>{String.raw`\( \mathcal{L}(\xi) = \mathbb{E}\left[\sum_t \frac{1}{2}\left(v_\xi(s_t) - \text{sg}(V^\lambda_t)\right)^2\right] \)`}</MathJax>
              </div>
              <p className={styles.modelDesc}>The critic regresses to the &lambda;-return targets with stop-gradient &mdash; standard value function regression.</p>
            </div>
          </div>
        </div>

        {/* ==================== Paper Lineage ==================== */}
        <div className={styles.cardMb}>
          <div className={styles.cardTitle}>
            Paper Lineage
          </div>
          <div className={styles.lineageGrid}>
            <div className={`${styles.lineageColumn} ${styles.lineageBuildsOn}`}>
              <h4>Builds On</h4>
              <div className={styles.lineageItems}>
                <div className={styles.lineageItemBuildsOn}>
                  <div className={styles.lineageItemTitle}>Dreamer (V1)</div>
                  <div className={styles.lineageItemDesc}>Hafner et al., 2019 &mdash; Same RSSM architecture but with Gaussian latents and pure dynamics backprop for actor. V2 switches to discrete latents and mixed gradients.</div>
                </div>
                <div className={styles.lineageItemBuildsOn}>
                  <div className={styles.lineageItemTitle}>PlaNet</div>
                  <div className={styles.lineageItemDesc}>Hafner et al., 2018 &mdash; The original RSSM world model, using online CEM planning instead of learned behaviors.</div>
                </div>
                <div className={styles.lineageItemBuildsOn}>
                  <div className={styles.lineageItemTitle}>Straight-Through Estimator</div>
                  <div className={styles.lineageItemDesc}>Bengio et al., 2013 &mdash; Enables gradient flow through discrete sampling by using softmax probabilities in the backward pass.</div>
                </div>
              </div>
            </div>
            <div className={`${styles.lineageColumn} ${styles.lineageBuiltUpon}`}>
              <h4>Built Upon By</h4>
              <div className={styles.lineageItems}>
                <div className={styles.lineageItemBuiltUpon}>
                  <div className={styles.lineageItemTitle}>DreamerV3</div>
                  <div className={styles.lineageItemDesc}>Hafner et al., 2023 &mdash; Adds symlog predictions, free bits, return normalization. Fixed hyperparameters across 150+ tasks in 8 domains.</div>
                </div>
                <div className={styles.lineageItemBuiltUpon}>
                  <div className={styles.lineageItemTitle}>DayDreamer</div>
                  <div className={styles.lineageItemDesc}>Wu et al., 2022 &mdash; Applies DreamerV2 to real robots, learning locomotion in ~1 hour.</div>
                </div>
                <div className={styles.lineageItemBuiltUpon}>
                  <div className={styles.lineageItemTitle}>IRIS</div>
                  <div className={styles.lineageItemDesc}>Micheli et al., 2023 &mdash; Transformer-based world model with discrete tokens, inspired by DreamerV2{"'"}s discrete latent approach.</div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
