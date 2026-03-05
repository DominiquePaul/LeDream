'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './rlpd.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Symmetric Sampling: 50/50 Online + Offline',
    desc: 'Each batch: 50% from online replay buffer + 50% from offline prior data. No curriculum, no annealing.',
    detail: 'Each training batch is composed of 50% transitions from the online replay buffer and 50% from the offline prior dataset. This is the simplest possible mixing strategy — no curriculum, no annealing, no importance weighting. The key insight: the agent simultaneously learns from both sources, using offline data to bootstrap knowledge of the environment while online data corrects for distributional shift. Unlike prior work (AWAC, IQL) that pre-trains offline first, RLPD starts learning online from the very first step, using the offline data as a persistent prior throughout training. The symmetric split means the agent never "forgets" the offline data, but also never relies on it exclusively.',
  },
  {
    title: 'LayerNorm in Critic',
    desc: 'LayerNorm after every hidden layer in the critic prevents Q-value divergence when mixing data sources.',
    detail: 'When combining offline and online data, the critic network faces a distribution mismatch: offline data may contain (state, action) pairs that the online policy would never visit, and vice versa. Without normalization, the critic can produce wildly overestimated Q-values for out-of-distribution inputs, leading to catastrophic training collapse. LayerNorm in the critic network constrains hidden activations by projecting them onto a hypersphere at each layer (see the LayerNorm page for the geometric view). This bounds the magnitude of internal representations without biasing the learned Q-function — a much softer constraint than conservative Q-learning penalties (CQL). In RLPD\'s experiments, removing LayerNorm from the critic causes complete failure on most tasks.',
  },
  {
    title: 'Large Ensemble + High UTD',
    desc: '10 critic ensemble with UTD=20 extracts maximum learning signal from every interaction.',
    detail: 'RLPD uses 10 critic networks in an ensemble (E=10) and a high update-to-data ratio of G=20 (meaning 20 gradient updates per environment step). This aggressive reuse of data maximizes sample efficiency. The large ensemble provides diverse Q-value estimates that reduce overestimation and improve exploration through optimistic action selection. At inference time, only 2 randomly selected critics are used for the policy update (Clipped Double Q-learning / CDQ), keeping compute manageable. The high UTD ratio is enabled by LayerNorm — without it, training with G=20 diverges. Together, these choices achieve 2.5x improvement over prior methods on standard benchmarks while using a fraction of the environment interactions.',
  },
];

const keyFeatures = [
  { title: '2.5\u00d7 Prior SOTA', desc: 'Outperforms all prior offline-to-online methods by a factor of 2.5 on aggregate benchmark scores' },
  { title: 'Dead Simple', desc: 'Standard SAC + 3 changes: symmetric sampling, LayerNorm, large ensemble. No pre-training, no penalties' },
  { title: 'Foundation for SERL', desc: 'Core algorithm behind SERL \u2014 the leading approach for sample-efficient real-world robot learning' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  online_buffer: {
    title: 'Online Replay Buffer (D_online)',
    desc: 'Standard replay buffer that stores transitions (s, a, r, s\', d) collected by the online policy interacting with the environment. This buffer grows as the agent explores and is used for 50% of each training batch.',
  },
  offline_buffer: {
    title: 'Offline Prior Data (D_prior)',
    desc: 'A fixed dataset of pre-collected transitions, typically from a suboptimal demonstrator or random policy. This data provides the agent with initial knowledge about environment dynamics. RLPD makes no assumptions about the quality of this data — even random exploration data helps bootstrap learning.',
  },
  symmetric_sample: {
    title: 'Symmetric Sampling',
    desc: '50% of each training batch comes from D_online and 50% from D_prior. This is the simplest mixing strategy — no schedules, no importance weights, no curriculum. The batch of size B is composed as B/2 online transitions + B/2 offline transitions, shuffled together for the critic and actor updates.',
  },
  critic_ensemble: {
    title: 'Critic Ensemble (E=10) with LayerNorm',
    desc: 'An ensemble of 10 independent Q-networks, each with LayerNorm after every hidden layer. Each critic receives the same batch but starts from different random initializations, providing diverse value estimates. LayerNorm prevents any single critic from producing unbounded Q-values on out-of-distribution inputs.',
  },
  cdq: {
    title: 'Clipped Double Q-Learning (CDQ)',
    desc: 'For the policy update, RLPD randomly selects 2 critics from the ensemble of 10 and uses the minimum of their Q-values: min(Q_i(s,a), Q_j(s,a)). This prevents overestimation without the full computational cost of evaluating all 10 critics. The random selection adds stochasticity that helps exploration.',
  },
  actor: {
    title: 'SAC Actor (Stochastic Policy)',
    desc: 'A standard SAC actor that outputs a squashed Gaussian policy: \u03c0(a|s) = tanh(\u03bc(s) + \u03c3(s) \u00b7 \u03b5). Trained to maximize Q-values from the CDQ subset minus an entropy penalty \u03b1\u00b7H[\u03c0]. The entropy coefficient \u03b1 is automatically tuned to maintain a target entropy.',
  },
  utd: {
    title: 'High Update-to-Data Ratio (G=20)',
    desc: 'For every environment step, the agent performs G=20 gradient updates on the critics and actor. This extreme reuse of data is only stable because of LayerNorm — without it, G>1 often causes training divergence. The high UTD ratio means RLPD extracts 20x more learning signal from each interaction compared to standard SAC (G=1).',
  },
  env: {
    title: 'Environment Interaction',
    desc: 'The agent interacts with the environment using its current stochastic policy, collecting transitions that are stored in D_online. Standard SAC exploration (entropy bonus) drives exploration. No explicit exploration bonuses or intrinsic rewards are needed.',
  },
};

/* ================================================================
   COMPONENT
   ================================================================ */

export default function RLPDPage() {
  /* ---- state ---- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);
  const [utdRatio, setUtdRatio] = useState(20);
  const [ensembleSize, setEnsembleSize] = useState(10);

  /* ---- canvas refs ---- */
  const samplingCanvasRef = useRef<HTMLCanvasElement>(null);
  const qValueCanvasRef = useRef<HTMLCanvasElement>(null);

  const { width: scw, height: sch } = useCanvasResize(samplingCanvasRef);
  const { width: qcw, height: qch } = useCanvasResize(qValueCanvasRef);

  /* ================================================================
     DRAW: Symmetric Sampling Visualization
     ================================================================ */
  const drawSampling = useCallback(() => {
    const canvas = samplingCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (scw === 0 || sch === 0) return;

    ctx.clearRect(0, 0, scw, sch);

    const pad = { left: 20, right: 20, top: 25, bottom: 20 };
    const midX = scw / 2;

    // Title
    ctx.fillStyle = '#aaa';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Batch Construction (B = 256)', midX, 16);

    // Online buffer (left)
    const bufferY = pad.top + 15;
    const bufferW = (scw - pad.left - pad.right - 40) / 2;
    const bufferH = 70;

    // Online
    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    roundRect(ctx, pad.left, bufferY, bufferW, bufferH, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#3b82f6';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('D_online', pad.left + bufferW / 2, bufferY + 22);
    ctx.fillStyle = '#888';
    ctx.font = '9px sans-serif';
    ctx.fillText('Replay Buffer', pad.left + bufferW / 2, bufferY + 37);
    ctx.fillText('(grows over time)', pad.left + bufferW / 2, bufferY + 50);

    // Offline
    const offX = scw - pad.right - bufferW;
    ctx.fillStyle = 'rgba(245, 158, 11, 0.15)';
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    roundRect(ctx, offX, bufferY, bufferW, bufferH, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f59e0b';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('D_prior', offX + bufferW / 2, bufferY + 22);
    ctx.fillStyle = '#888';
    ctx.font = '9px sans-serif';
    ctx.fillText('Offline Dataset', offX + bufferW / 2, bufferY + 37);
    ctx.fillText('(fixed, suboptimal)', offX + bufferW / 2, bufferY + 50);

    // Arrows from buffers to batch
    const arrowStartY = bufferY + bufferH + 5;
    const batchY = arrowStartY + 40;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    drawArrow(ctx, pad.left + bufferW / 2, arrowStartY, midX - 30, batchY - 5);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('50%', midX - 35, batchY - 15);

    ctx.strokeStyle = '#f59e0b';
    drawArrow(ctx, offX + bufferW / 2, arrowStartY, midX + 30, batchY - 5);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.7)';
    ctx.textAlign = 'left';
    ctx.fillText('50%', midX + 35, batchY - 15);

    // Merged Batch
    const batchW = scw - pad.left - pad.right - 40;
    const batchH = 45;
    const batchX = (scw - batchW) / 2;

    // Draw the batch as half blue, half orange
    ctx.save();
    ctx.beginPath();
    roundRectPath(ctx, batchX, batchY, batchW, batchH, 10);
    ctx.clip();

    // Left half - online
    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    ctx.fillRect(batchX, batchY, batchW / 2, batchH);
    // Right half - offline
    ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
    ctx.fillRect(batchX + batchW / 2, batchY, batchW / 2, batchH);

    ctx.restore();

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    roundRect(ctx, batchX, batchY, batchW, batchH, 10);
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training Batch (B = 256)', midX, batchY + 18);
    ctx.fillStyle = '#888';
    ctx.font = '9px sans-serif';
    ctx.fillText('128 online  +  128 offline  \u2192  shuffle  \u2192  update', midX, batchY + 33);

    // Arrow down to critic/actor
    const updateY = batchY + batchH + 35;
    ctx.strokeStyle = '#10b981';
    drawArrow(ctx, midX, batchY + batchH + 5, midX, updateY - 5);

    // Critic + Actor update box
    const updateW = batchW * 0.7;
    const updateH = 50;
    const updateX = (scw - updateW) / 2;
    ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    roundRect(ctx, updateX, updateY, updateW, updateH, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`G = ${utdRatio} gradient updates per env step`, midX, updateY + 20);
    ctx.fillStyle = '#888';
    ctx.font = '9px sans-serif';
    ctx.fillText(`${ensembleSize} critics (ensemble) + 1 actor + \u03b1 auto-tune`, midX, updateY + 36);
  }, [scw, sch, utdRatio, ensembleSize]);

  useEffect(() => { drawSampling(); }, [drawSampling]);

  /* ================================================================
     DRAW: Q-Value Divergence (with/without LayerNorm)
     ================================================================ */
  const drawQValues = useCallback(() => {
    const canvas = qValueCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (qcw === 0 || qch === 0) return;

    ctx.clearRect(0, 0, qcw, qch);

    const pad = { left: 60, right: 20, top: 30, bottom: 45 };
    const pw = qcw - pad.left - pad.right;
    const ph = qch - pad.top - pad.bottom;

    // X: training steps
    const xMin = 0, xMax = 100;
    // Y: Q-value magnitude
    const yMin = 0, yMax = 60;

    function toX(v: number) { return pad.left + ((v - xMin) / (xMax - xMin)) * pw; }
    function toY(v: number) { return qch - pad.bottom - ((v - yMin) / (yMax - yMin)) * ph; }

    // Axes
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, qch - pad.bottom);
    ctx.lineTo(qcw - pad.right, qch - pad.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#555';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training Steps (k)', qcw / 2, qch - 8);

    ctx.save();
    ctx.translate(14, qch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('|Q-value|', 0, 0);
    ctx.restore();

    // X tick labels
    ctx.textAlign = 'center';
    for (let x = 0; x <= 100; x += 20) {
      ctx.fillStyle = '#555';
      ctx.fillText(`${x}k`, toX(x), qch - pad.bottom + 15);
    }

    // Y tick labels
    ctx.textAlign = 'right';
    for (let y = 0; y <= 60; y += 10) {
      ctx.fillStyle = '#555';
      ctx.fillText(y.toString(), pad.left - 8, toY(y) + 4);
      if (y > 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath();
        ctx.moveTo(pad.left, toY(y));
        ctx.lineTo(qcw - pad.right, toY(y));
        ctx.stroke();
      }
    }

    // "Healthy range" band
    ctx.fillStyle = 'rgba(16, 185, 129, 0.06)';
    ctx.fillRect(pad.left, toY(15), pw, toY(0) - toY(15));
    ctx.fillStyle = 'rgba(16,185,129,0.3)';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Healthy Q-range', pad.left + 5, toY(12));

    const numPts = 200;

    // Without LayerNorm (UTD=20): Q-values diverge
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const noise = Math.sin(x * 0.7) * 1.5 + Math.sin(x * 1.3) * 0.8;
      const y = 5 + (utdRatio / 20) * 0.015 * x * x + noise;
      const cx2 = toX(x);
      const cy2 = toY(Math.min(y, yMax));
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    }
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Without LayerNorm (UTD=1): Q-values slightly overestimate
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const noise = Math.sin(x * 0.9) * 0.5;
      const y = 5 + 0.08 * x + noise;
      const cx2 = toX(x);
      const cy2 = toY(Math.min(y, yMax));
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    }
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // With LayerNorm (UTD=20): stable
    ctx.beginPath();
    for (let i = 0; i <= numPts; i++) {
      const x = xMin + (i / numPts) * (xMax - xMin);
      const noise = Math.sin(x * 0.5) * 0.8 + Math.sin(x * 1.1) * 0.4;
      const y = 5 + 3 * Math.log(1 + x / 20) + noise;
      const cx2 = toX(x);
      const cy2 = toY(Math.min(y, yMax));
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    }
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Labels
    ctx.font = '10px sans-serif';

    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'left';
    ctx.fillText(`No LN, UTD=${utdRatio}`, toX(55), toY(52));
    ctx.fillStyle = 'rgba(239,68,68,0.6)';
    ctx.font = '8px sans-serif';
    ctx.fillText('(diverges \u2192 training collapse)', toX(55), toY(48));

    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.font = '10px sans-serif';
    ctx.fillText('No LN, UTD=1', toX(60), toY(16));

    ctx.fillStyle = '#10b981';
    ctx.font = '10px sans-serif';
    ctx.fillText(`With LN, UTD=${utdRatio}`, toX(55), toY(10));
    ctx.fillStyle = 'rgba(16,185,129,0.6)';
    ctx.font = '8px sans-serif';
    ctx.fillText('(stable + performant)', toX(55), toY(7));
  }, [qcw, qch, utdRatio]);

  useEffect(() => { drawQValues(); }, [drawQValues]);



  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className="method-page">
      <h1 className={styles.title}>RLPD</h1>
      <p className={styles.subtitle}>
        Efficient Online Reinforcement Learning with Offline Data &mdash; Ball, Smith, Kostrikov, Levine &middot; ICML 2023
      </p>
      <div className={styles.linkRow}>
        <a href="https://arxiv.org/abs/2302.02948" target="_blank" rel="noopener noreferrer" className={styles.paperLink}>
          &rarr; arxiv.org/abs/2302.02948
        </a>
        <a href="https://github.com/ikostrikov/rlpd" target="_blank" rel="noopener noreferrer" className={styles.paperLink}>
          &rarr; github.com/ikostrikov/rlpd
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture SVG ==================== */}
      <h2>RLPD Architecture</h2>
      <div className="diagram-frame">
        <svg className={styles.pipelineSvg} viewBox="0 0 480 340">
          <defs>
            <marker id="rlpdArrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#10b981" />
            </marker>
            <marker id="rlpdArrowBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
            </marker>
            <marker id="rlpdArrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
            </marker>
            <marker id="rlpdArrowGray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#888" />
            </marker>
            <marker id="rlpdArrowPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
            </marker>
            <marker id="rlpdArrowCyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff" />
            </marker>
          </defs>

          {/* Data Sources */}
          <text x="130" y="15" textAnchor="middle" fill="#888" fontSize="9" fontWeight="bold">Data Sources</text>

          {/* Online Buffer */}
          <rect
            x="15" y="25" width="110" height="45" rx="8"
            fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'online_buffer' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'online_buffer' ? null : 'online_buffer')}
          />
          <text x="70" y="45" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>D_online</text>
          <text x="70" y="60" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Replay Buffer</text>

          {/* Offline Buffer */}
          <rect
            x="140" y="25" width="110" height="45" rx="8"
            fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'offline_buffer' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'offline_buffer' ? null : 'offline_buffer')}
          />
          <text x="195" y="45" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>D_prior</text>
          <text x="195" y="60" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Offline Data</text>

          {/* Symmetric Sampling */}
          <path d="M 70 72 L 170 92" stroke="#3b82f6" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowBlue)" />
          <path d="M 195 72 L 195 90" stroke="#f59e0b" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowOrange)" />
          <text x="90" y="83" fill="rgba(59,130,246,0.6)" fontSize="7">50%</text>
          <text x="200" y="83" fill="rgba(245,158,11,0.6)" fontSize="7">50%</text>

          <rect
            x="130" y="95" width="130" height="32" rx="8"
            fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'symmetric_sample' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'symmetric_sample' ? null : 'symmetric_sample')}
          />
          <text x="195" y="115" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>Symmetric Sampling</text>

          {/* Arrow to Critic Ensemble */}
          <path d="M 195 129 L 195 147" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#rlpdArrowGreen)" />
          <text x="210" y="142" fill="#888" fontSize="7">batch B=256</text>

          {/* Critic Ensemble */}
          <rect
            x="105" y="150" width="180" height="55" rx="10"
            fill="rgba(168,85,247,0.12)" stroke="#a855f7" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'critic_ensemble' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'critic_ensemble' ? null : 'critic_ensemble')}
          />
          <text x="195" y="170" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold" style={{ pointerEvents: 'none' }}>Critic Ensemble (E=10)</text>
          <text x="195" y="185" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Q&#x2081;, Q&#x2082;, ... Q&#x2081;&#x2080;</text>
          <text x="195" y="197" textAnchor="middle" fill="rgba(16,185,129,0.6)" fontSize="7" style={{ pointerEvents: 'none' }}>+ LayerNorm in every layer</text>

          {/* CDQ subset arrow */}
          <path d="M 285 175 L 320 175" stroke="#a855f7" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowPurple)" />

          {/* CDQ */}
          <rect
            x="325" y="155" width="130" height="40" rx="8"
            fill="rgba(168,85,247,0.1)" stroke="#a855f7" strokeWidth="1.5"
            className={`${styles.pipelineStage} ${activeComponent === 'cdq' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'cdq' ? null : 'cdq')}
          />
          <text x="390" y="172" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>CDQ: min(Q&#x1D62;, Q&#x2C7C;)</text>
          <text x="390" y="186" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Random 2 of 10</text>

          {/* UTD indicator */}
          <rect
            x="10" y="155" width="85" height="45" rx="8"
            fill="rgba(0,217,255,0.1)" stroke="#00d9ff" strokeWidth="1.5"
            className={`${styles.pipelineStage} ${activeComponent === 'utd' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'utd' ? null : 'utd')}
          />
          <text x="52" y="173" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>UTD = G</text>
          <text x="52" y="190" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>20 updates/step</text>

          <path d="M 95 177 L 103 177" stroke="#00d9ff" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowCyan)" />

          {/* Actor */}
          <rect
            x="325" y="215" width="130" height="40" rx="8"
            fill="#10b981"
            className={`${styles.pipelineStage} ${activeComponent === 'actor' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'actor' ? null : 'actor')}
          />
          <text x="390" y="233" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>SAC Actor &#x03C0;(a|s)</text>
          <text x="390" y="247" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="7" style={{ pointerEvents: 'none' }}>max Q &#x2212; &#x03B1;&#x00B7;H[&#x03C0;]</text>

          <path d="M 390 195 L 390 213" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#rlpdArrowGreen)" />

          {/* Environment */}
          <rect
            x="325" y="275" width="130" height="45" rx="8"
            fill="rgba(0,217,255,0.15)" stroke="#00d9ff" strokeWidth="2"
            className={`${styles.pipelineStage} ${activeComponent === 'env' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent(activeComponent === 'env' ? null : 'env')}
          />
          <text x="390" y="295" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" style={{ pointerEvents: 'none' }}>Environment</text>
          <text x="390" y="310" textAnchor="middle" fill="#888" fontSize="7" style={{ pointerEvents: 'none' }}>Adroit / AntMaze / ...</text>

          <path d="M 390 255 L 390 273" stroke="#888" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowGray)" />
          <text x="400" y="267" fill="#888" fontSize="7">actions</text>

          {/* Loop back from env to online buffer */}
          <path d="M 325 300 Q 200 320 70 72" stroke="#3b82f6" strokeWidth="1.5" fill="none" markerEnd="url(#rlpdArrowBlue)" strokeDasharray="6" />
          <text x="170" y="315" fill="rgba(59,130,246,0.6)" fontSize="7">transitions &#x2192; D_online</text>

          {/* Summary bar */}
          <rect x="10" y="318" width="460" height="18" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" strokeWidth="1" />
          <text x="240" y="330" textAnchor="middle" fill="#10b981" fontSize="8" fontWeight="bold" style={{ pointerEvents: 'none' }}>
            No offline pre-training &#x00B7; No conservative penalties &#x00B7; No importance weighting &#x00B7; Just careful design choices
          </text>
        </svg>
      </div>

      {activeComponent && componentInfo[activeComponent] && (
        <div className="info-panel">
          <h4>{componentInfo[activeComponent].title}</h4>
          <p>{componentInfo[activeComponent].desc}</p>
        </div>
      )}

      {/* ==================== Symmetric Sampling Canvas ==================== */}
      <h2>Symmetric Sampling &amp; Update Pipeline</h2>
      <div className="diagram-frame">
        <canvas ref={samplingCanvasRef} className={styles.vizCanvas} />
      </div>
      <div className={styles.sliderContainer}>
        <label>UTD ratio (G):</label>
        <input
          type="range"
          className={styles.slider}
          min={1}
          max={40}
          step={1}
          value={utdRatio}
          onChange={(e) => setUtdRatio(parseInt(e.target.value, 10))}
        />
        <span className={styles.sliderValue}>{utdRatio}</span>
      </div>
      <div className={styles.sliderContainer}>
        <label>Ensemble size (E):</label>
        <input
          type="range"
          className={styles.slider}
          min={2}
          max={20}
          step={1}
          value={ensembleSize}
          onChange={(e) => setEnsembleSize(parseInt(e.target.value, 10))}
        />
        <span className={styles.sliderValue}>{ensembleSize}</span>
      </div>
      <div className={styles.metricsRow}>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: '#3b82f6' }}>50/50</div>
          <div className={styles.metricLabel}>Online / Offline</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: '#f59e0b' }}>{utdRatio}&times;</div>
          <div className={styles.metricLabel}>Updates per Step</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricValue} style={{ color: '#a855f7' }}>{ensembleSize}</div>
          <div className={styles.metricLabel}>Critics</div>
        </div>
      </div>

      {/* ==================== Q-Value Divergence Chart ==================== */}
      <h2>Q-Value Stability: LayerNorm is Critical</h2>
      <div className="diagram-frame">
        <canvas ref={qValueCanvasRef} className={styles.vizCanvasTall} />
      </div>
      <div className={styles.sliderContainer}>
        <label>UTD ratio (G):</label>
        <input
          type="range"
          className={styles.slider}
          min={1}
          max={40}
          step={1}
          value={utdRatio}
          onChange={(e) => setUtdRatio(parseInt(e.target.value, 10))}
        />
        <span className={styles.sliderValue}>{utdRatio}</span>
      </div>
      <div className={styles.annotationCallout}>
        <strong>Key finding:</strong> Without LayerNorm, high UTD ratios cause Q-values to diverge catastrophically.
        With LayerNorm, even UTD=20 remains stable. This is the enabler for RLPD&apos;s sample efficiency.
      </div>

      {/* ==================== Algorithm Pseudocode ==================== */}
      <h2>Algorithm 1: RLPD</h2>
      <div className={styles.pseudocode}>
        <span className={styles.comment}>// Initialize</span><br />
        <span className={styles.keyword}>Input:</span> Prior data <span className={styles.param}>D_prior</span>, ensemble size <span className={styles.param}>E</span>, UTD ratio <span className={styles.param}>G</span><br />
        <span className={styles.keyword}>Init:</span> <span className={styles.param}>D_online &#x2190; &#x2205;</span>, critics <span className={styles.param}>{'{Q_\u03b8\u2081, ..., Q_\u03b8_E}'}</span> with <span className={styles.highlight}>LayerNorm</span>, actor <span className={styles.param}>\u03c0_\u03c6</span>, temp <span className={styles.param}>\u03b1</span><br />
        <br />
        <span className={styles.keyword}>for</span> each environment step <span className={styles.keyword}>do</span><br />
        &nbsp;&nbsp;<span className={styles.comment}>// Collect online transition</span><br />
        &nbsp;&nbsp;<span className={styles.func}>a &#x2190; \u03c0_\u03c6(&#x00B7;|s)</span> &nbsp;&nbsp;&nbsp;<span className={styles.comment}>// sample from stochastic policy</span><br />
        &nbsp;&nbsp;s&apos;, r, d &#x2190; <span className={styles.func}>env.step(a)</span><br />
        &nbsp;&nbsp;<span className={styles.param}>D_online</span> &#x2190; <span className={styles.param}>D_online</span> &#x222A; {'{'} (s, a, r, s&apos;, d) {'}'}<br />
        <br />
        &nbsp;&nbsp;<span className={styles.keyword}>for</span> g = 1 to <span className={styles.param}>G</span> <span className={styles.keyword}>do</span> &nbsp;&nbsp;&nbsp;<span className={styles.comment}>// {`G = ${utdRatio}`} gradient updates per step</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className={styles.comment}>// <span className={styles.highlight}>Symmetric sampling</span></span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;B_on &#x2190; <span className={styles.func}>sample(D_online, B/2)</span> &nbsp;&nbsp;<span className={styles.comment}>// 50% online</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;B_off &#x2190; <span className={styles.func}>sample(D_prior, B/2)</span> &nbsp;&nbsp;<span className={styles.comment}>// 50% offline</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;B &#x2190; <span className={styles.func}>shuffle(B_on &#x222A; B_off)</span><br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className={styles.comment}>// Update all E critics on the mixed batch</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className={styles.keyword}>for</span> i = 1 to <span className={styles.param}>E</span> <span className={styles.keyword}>do</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\u03b8&#x1D62; &#x2190; \u03b8&#x1D62; &#x2212; \u03b7 &#x00B7; &#x2207;<span className={styles.func}>L_SAC(Q_\u03b8&#x1D62;, B)</span><br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className={styles.comment}>// Update actor with CDQ (random 2 of E)</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;i, j &#x2190; <span className={styles.func}>random_pair(1..E)</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;\u03c6 &#x2190; \u03c6 &#x2212; \u03b7 &#x00B7; &#x2207;<span className={styles.func}>L_actor(\u03c0_\u03c6, min(Q_\u03b8&#x1D62;, Q_\u03b8&#x2C7C;))</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;\u03b1 &#x2190; \u03b1 &#x2212; \u03b7 &#x00B7; &#x2207;<span className={styles.func}>L_temp(\u03b1, \u03c0_\u03c6)</span> &nbsp;&nbsp;<span className={styles.comment}>// auto-tune entropy</span>
      </div>
      <div className={styles.annotationCallout} style={{ marginTop: 12 }}>
        <strong>Simplicity is the point.</strong> The entire algorithm is standard SAC with three modifications:
        symmetric sampling (lines with B_on/B_off), LayerNorm in critic architectures, and a large ensemble with high UTD.
        No offline pre-training phase, no conservative penalties, no importance weights.
      </div>

      {/* ==================== Training Objectives ==================== */}
      <h2>Training Objectives (SAC-based)</h2>
      <div className={styles.modelEquations}>
        {/* Critic */}
        <div className={styles.modelItem}>
          <h4 className={styles.modelItemHeader}>
            <span className={styles.modelDot} style={{ background: '#a855f7' }} />
            Critic Loss (per ensemble member)
          </h4>
          <div className="formula-block">
            <MathJax>{'\\( \\mathcal{L}(\\theta_i) = \\mathbb{E}_{(s,a,r,s\') \\sim B}\\!\\left[\\left(Q_{\\theta_i}(s,a) - y\\right)^2\\right] \\)'}</MathJax>
          </div>
          <div className="formula-block">
            <MathJax>{'\\( y = r + \\gamma \\left(\\min_{j \\in \\text{CDQ}} Q_{\\bar{\\theta}_j}(s\', a\') - \\alpha \\log \\pi_\\phi(a\'|s\')\\right), \\quad a\' \\sim \\pi_\\phi(\\cdot | s\') \\)'}</MathJax>
          </div>
          <p className={styles.modelItemNote}>
            CDQ selects 2 random critics from the ensemble for the target. Target networks updated via EMA (&rho; = 0.005). <strong style={{ color: '#10b981' }}>LayerNorm</strong> in every hidden layer of each Q_&theta;.
          </p>
        </div>

        {/* Actor */}
        <div className={styles.modelItem}>
          <h4 className={styles.modelItemHeader}>
            <span className={styles.modelDot} style={{ background: '#10b981' }} />
            Actor Loss (Maximum Entropy)
          </h4>
          <div className="formula-block">
            <MathJax>{'\\( \\mathcal{L}(\\phi) = \\mathbb{E}_{s \\sim B}\\!\\left[\\alpha \\log \\pi_\\phi(a|s) - \\min_{j \\in \\text{CDQ}} Q_{\\theta_j}(s, a)\\right], \\quad a \\sim \\pi_\\phi(\\cdot|s) \\)'}</MathJax>
          </div>
          <p className={styles.modelItemNote}>
            Standard SAC actor: maximize Q-value while maintaining entropy. The CDQ subset provides a less overestimated Q-target for policy improvement.
          </p>
        </div>

        {/* Temperature */}
        <div className={styles.modelItem}>
          <h4 className={styles.modelItemHeader}>
            <span className={styles.modelDot} style={{ background: '#00d9ff' }} />
            Entropy Temperature (Auto-tuned)
          </h4>
          <div className="formula-block">
            <MathJax>{'\\( \\mathcal{L}(\\alpha) = -\\alpha \\cdot \\mathbb{E}_{s \\sim B}\\!\\left[\\log \\pi_\\phi(a|s) + \\bar{H}\\right] \\)'}</MathJax>
          </div>
          <p className={styles.modelItemNote}>
            Target entropy {'\u0048\u0304'} = &minus;dim(A). Automatically balances exploration vs. exploitation. On some environments, entropy backups are used (entropy bonus only in the current-step reward, not bootstrapped).
          </p>
        </div>
      </div>

      {/* ==================== Comparison Table ==================== */}
      <h2>RLPD vs Prior Approaches</h2>
      <table className={styles.comparisonTable}>
        <thead>
          <tr>
            <th>Aspect</th>
            <th>RLPD</th>
            <th>IQL</th>
            <th>AWAC</th>
            <th>CQL &#x2192; Online</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Offline pre-training</td>
            <td className={styles.cellGreen}>None</td>
            <td className={styles.cellAmber}>Required</td>
            <td className={styles.cellAmber}>Required</td>
            <td className={styles.cellRed}>Required (expensive)</td>
          </tr>
          <tr>
            <td>Conservative penalty</td>
            <td className={styles.cellGreen}>None</td>
            <td>Implicit (expectile)</td>
            <td>Advantage-weighted</td>
            <td className={styles.cellAmber}>CQL penalty</td>
          </tr>
          <tr>
            <td>How offline data is used</td>
            <td className={styles.cellGreen}>Persistent 50/50 mixing</td>
            <td>Pre-train only</td>
            <td>Pre-train only</td>
            <td>Pre-train &#x2192; anneal</td>
          </tr>
          <tr>
            <td>Critic architecture</td>
            <td className={styles.cellGreen}>10 critics + LayerNorm</td>
            <td>2 critics</td>
            <td>2 critics</td>
            <td>2 critics + CQL</td>
          </tr>
          <tr>
            <td>UTD ratio</td>
            <td className={styles.cellGreen}>G=20</td>
            <td>G=1</td>
            <td>G=1</td>
            <td>G=1</td>
          </tr>
          <tr>
            <td>Implementation complexity</td>
            <td className={styles.cellGreen}>Minimal (SAC + 3 changes)</td>
            <td>Moderate</td>
            <td>Moderate</td>
            <td className={styles.cellAmber}>High (CQL + online switch)</td>
          </tr>
          <tr>
            <td>Performance</td>
            <td className={styles.cellGreen}>2.5&times; prior SOTA</td>
            <td>Baseline</td>
            <td>Below baseline</td>
            <td>Comparable to IQL</td>
          </tr>
        </tbody>
      </table>

      {/* ==================== Paper Lineage ==================== */}
      <h2>Paper Lineage</h2>
      <section className="lineage-section">
        <div className="lineage-grid">
          <div className="lineage-group">
            <h4 style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>Soft Actor-Critic (SAC)</h5>
              <p>Haarnoja et al., 2018 &mdash; The base algorithm. RLPD is SAC + 3 design choices.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>REDQ / DroQ</h5>
              <p>Chen et al., 2021 / Hiraoka et al., 2022 &mdash; Large ensembles and high UTD ratios for sample-efficient RL.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>LayerNorm (Ba et al., 2016)</h5>
              <p>The key architectural ingredient that enables stable training with high UTD and mixed data.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>IQL / AWAC / CQL</h5>
              <p>Prior offline-to-online methods that RLPD outperforms by avoiding conservative constraints and pre-training.</p>
            </div>
          </div>
          <div className="lineage-group">
            <h4 style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>Cal-QL</h5>
              <p>Nakamoto et al., 2023 &mdash; Calibrated Q-learning that combines RLPD&apos;s architecture with calibrated conservative penalties.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>SERL</h5>
              <p>Luo et al., 2024 &mdash; Sample-Efficient Robot Learning uses RLPD as its core algorithm for real-world robotics.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>BRO / CrossQ</h5>
              <p>Tarasov et al., 2024 / Bhatt et al., 2024 &mdash; Build on RLPD&apos;s insight that LayerNorm + high UTD is a powerful combination.</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>TD7</h5>
              <p>Fujimoto et al., 2023 &mdash; State-of-the-art continuous control incorporating LayerNorm and ensemble ideas from RLPD.</p>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}

/* ================================================================
   CANVAS HELPER FUNCTIONS
   ================================================================ */

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number
) {
  const headLen = 8;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const angle = Math.atan2(dy, dx);

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLen * Math.cos(angle - Math.PI / 6),
    y2 - headLen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 6),
    y2 - headLen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}
