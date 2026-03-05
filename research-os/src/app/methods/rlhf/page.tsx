'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import styles from './rlhf.module.css';

import type { CoreIdea, KeyFeature } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';

/* ------------------------------------------------------------------ */
/*  Data                                                               */
/* ------------------------------------------------------------------ */

const phaseInfo: Record<number, { title: string; desc: string }> = {
  1: {
    title: 'Phase 1: Supervised Fine-Tuning (SFT)',
    desc: 'Start with a pretrained LLM and fine-tune on high-quality demonstrations. This teaches the model the desired format and style of responses. The resulting model \u03C0_SFT serves as both the starting point for RL training and the frozen reference policy.',
  },
  2: {
    title: 'Phase 2: Reward Model Training',
    desc: 'Collect human comparisons: given a prompt, humans rank two responses. Train a reward model using the Bradley-Terry model to predict which response humans prefer. The reward model learns r_\u03C6(x, y) that scores any response.',
  },
  3: {
    title: 'Phase 3: RL Fine-Tuning (PPO)',
    desc: 'Use PPO to maximize the learned reward while staying close to the reference policy (KL penalty). The policy generates responses, gets rewards from r_\u03C6, and updates via policy gradients. The KL term prevents reward hacking.',
  },
};

const coreIdeas: CoreIdea[] = [
  { title: 'Supervised Fine-Tuning', desc: 'Fine-tune pretrained LLM on high-quality demonstration data to learn the task format', detail: phaseInfo[1].desc },
  { title: 'Reward Modeling', desc: 'Train a reward model on human preference comparisons between response pairs', detail: phaseInfo[2].desc },
  { title: 'RL Fine-Tuning', desc: 'Optimize policy with PPO against reward model, with KL penalty to prevent drift', detail: phaseInfo[3].desc },
];

const keyFeatures: KeyFeature[] = [
  { title: 'Learned Reward', desc: 'Reward model captures human preferences without hand-crafted reward functions' },
  { title: 'KL Constraint', desc: 'Prevents reward hacking and maintains language capabilities from pretraining' },
  { title: 'PPO Optimization', desc: 'Stable policy gradient method with clipped objectives for reliable training' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  pretrained: {
    title: 'Pretrained LLM',
    desc: 'Large language model pretrained on massive text corpora (e.g., GPT, LLaMA). Has broad knowledge but may not follow instructions well or align with human preferences.',
  },
  sft: {
    title: 'SFT Model (\u03C0_SFT)',
    desc: 'Supervised fine-tuned model trained on demonstration data. Learns to follow instructions and produce well-formatted responses. Serves as initialization for both the policy and the frozen reference.',
  },
  human: {
    title: 'Human Preferences',
    desc: 'Labelers compare pairs of model responses and indicate which is better. This creates a dataset of (prompt, preferred, rejected) triples used to train the reward model.',
  },
  reward: {
    title: 'Reward Model r_\u03C6(x, y)',
    desc: 'Neural network (often initialized from SFT) that predicts human preference scores. Trained with Bradley-Terry loss to output higher scores for preferred responses. Provides training signal for PPO.',
  },
  reference: {
    title: 'Reference Policy \u03C0_ref',
    desc: 'Frozen copy of the SFT model. Used to compute KL divergence penalty during PPO training. Keeps the policy from drifting too far and "reward hacking".',
  },
  policy: {
    title: 'Policy \u03C0_\u03B8',
    desc: 'The model being trained with PPO. Initialized from SFT, updated to maximize reward while staying close to \u03C0_ref. This is the final aligned model.',
  },
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function RLHFPage() {
  // --- Architecture component info ---
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  // --- Loss item selection ---
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  // --- Preference swap ---
  const [aIsPreferred, setAIsPreferred] = useState(true);

  // --- KL penalty slider ---
  const [beta, setBeta] = useState(0.1);

  // --- Reward Model sliders ---
  const [rw, setRw] = useState(1.5);
  const [rl, setRl] = useState(-0.5);

  // --- Canvas ---
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = useCanvasResize(canvasRef);

  /* ------ KL-derived values ------ */
  const baseKL = 0.5;
  const klAmount = baseKL * (1 - beta * 1.5);
  const reward = 0.5 + klAmount * 0.4;
  const penalty = beta * klAmount * 2;
  const net = reward - penalty;

  let klExplanation: string;
  let policyGradient: string;
  if (beta < 0.05) {
    klExplanation = '\u03B2 \u2248 0: No constraint \u2014 risk of reward hacking!';
    policyGradient = 'linear-gradient(135deg, #dc2626, #ef4444)';
  } else if (beta < 0.15) {
    klExplanation = `\u03B2 = ${beta.toFixed(2)}: Balanced \u2014 allows learning while preventing drift`;
    policyGradient = 'linear-gradient(135deg, #059669, #10b981)';
  } else if (beta < 0.3) {
    klExplanation = `\u03B2 = ${beta.toFixed(2)}: Conservative \u2014 limited policy updates`;
    policyGradient = 'linear-gradient(135deg, #0284c7, #0ea5e9)';
  } else {
    klExplanation = `\u03B2 = ${beta.toFixed(2)}: Very conservative \u2014 policy barely changes`;
    policyGradient = 'linear-gradient(135deg, #1e40af, #3b82f6)';
  }

  /* ------ RM Loss derived values ------ */
  const diff = rw - rl;
  const currentSig = sigmoid(diff);
  const currentLoss = -Math.log(currentSig);

  /* ------ Canvas drawing ------ */
  const drawRmLoss = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvasSize.width;
    const height = canvasSize.height;
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

    // Axis labels
    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('r(y_w) \u2212 r(y_l)', width / 2, height - 8);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    const xMin = -6,
      xMax = 6;
    const yMin = 0,
      yMax = 4;

    function toCanvasX(x: number) {
      return padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
    }
    function toCanvasY(y: number) {
      return height - padding.bottom - ((y - yMin) / (yMax - yMin)) * plotHeight;
    }

    // Grid lines & tick labels
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';

    for (let x = -6; x <= 6; x += 2) {
      const cx = toCanvasX(x);
      ctx.beginPath();
      ctx.moveTo(cx, padding.top);
      ctx.lineTo(cx, height - padding.bottom);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(x.toString(), cx, height - padding.bottom + 12);
    }

    for (let y = 0; y <= 4; y += 1) {
      const cy = toCanvasY(y);
      ctx.beginPath();
      ctx.moveTo(padding.left, cy);
      ctx.lineTo(width - padding.right, cy);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(y.toString(), padding.left - 8, cy + 4);
    }

    // Loss curve: L = -log(sigmoid(x))
    ctx.beginPath();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 3;
    for (let i = 0; i <= plotWidth; i++) {
      const x = xMin + (i / plotWidth) * (xMax - xMin);
      const sig = sigmoid(x);
      const loss = -Math.log(Math.max(sig, 0.0001));
      const cx = toCanvasX(x);
      const cy = toCanvasY(Math.min(loss, yMax));
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Current point
    const d = rw - rl;
    const cSig = sigmoid(d);
    const cLoss = -Math.log(cSig);
    const pointX = toCanvasX(d);
    const pointY = toCanvasY(Math.min(cLoss, yMax));

    // Vertical dashed line
    ctx.strokeStyle = 'rgba(0, 217, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pointX, height - padding.bottom);
    ctx.lineTo(pointX, pointY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot
    ctx.beginPath();
    ctx.arc(pointX, pointY, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#00d9ff';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Reference line x = 0
    const zeroX = toCanvasX(0);
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(zeroX, padding.top);
    ctx.lineTo(zeroX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Region labels
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.textAlign = 'center';
    ctx.fillText('Correct ranking', toCanvasX(3), padding.top + 15);
    ctx.fillStyle = '#ef4444';
    ctx.fillText('Wrong ranking', toCanvasX(-3), padding.top + 15);
  }, [canvasSize, rw, rl]);

  useEffect(() => {
    drawRmLoss();
  }, [drawRmLoss]);

  /* ---------------------------------------------------------------- */
  /*  JSX                                                              */
  /* ---------------------------------------------------------------- */

  return (
    <div className="method-page">
      <h1>Reinforcement Learning from Human Feedback</h1>
      <p className="subtitle">Aligning Language Models with Human Preferences</p>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ===== Architecture Overview ===== */}
      <h2>Architecture Overview</h2>
      <div className="diagram-frame">
        <svg className={styles.pipelineSvg} viewBox="0 0 500 300">
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
            <marker id="arrowPink" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#ec4899" />
            </marker>
          </defs>

          {/* Pretrained LLM */}
          <rect
            x="20" y="20" width="100" height="50" rx="8" fill="#6366f1"
            className={`${styles.pipelineStage} ${activeComponent === 'pretrained' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('pretrained')}
          />
          <text x="70" y="45" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">
            Pretrained
          </text>
          <text x="70" y="58" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            LLM
          </text>

          {/* SFT Model */}
          <rect
            x="180" y="20" width="100" height="50" rx="8" fill="#3b82f6"
            className={`${styles.pipelineStage} ${activeComponent === 'sft' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('sft')}
          />
          <text x="230" y="45" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">
            SFT Model
          </text>
          <text x="230" y="58" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            {'\u03C0_SFT'}
          </text>

          {/* Arrow: Pretrained -> SFT */}
          <path d="M 120 45 L 175 45" stroke="#00d9ff" strokeWidth="2" fill="none" markerEnd="url(#arrowCyan)" className={styles.flowArrow} />
          <text x="147" y="38" fill="#888" fontSize="8">demos</text>

          {/* Reward Model */}
          <rect
            x="340" y="120" width="120" height="60" rx="8" fill="#f59e0b"
            className={`${styles.pipelineStage} ${activeComponent === 'reward' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('reward')}
          />
          <text x="400" y="145" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">
            Reward Model
          </text>
          <text x="400" y="160" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            {'r_\u03C6(x, y)'}
          </text>

          {/* Human Preferences */}
          <rect
            x="340" y="20" width="120" height="50" rx="8" fill="#ec4899"
            className={`${styles.pipelineStage} ${activeComponent === 'human' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('human')}
          />
          <text x="400" y="42" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">
            Human Prefs
          </text>
          <text x="400" y="55" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            {'y_w \u227B y_l'}
          </text>

          {/* Arrow: SFT -> Human */}
          <path d="M 280 45 L 335 45" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" />
          <text x="305" y="38" fill="#888" fontSize="8">sample</text>

          {/* Arrow: Human -> Reward */}
          <path d="M 400 70 L 400 115" stroke="#ec4899" strokeWidth="2" fill="none" markerEnd="url(#arrowPink)" />

          {/* Policy */}
          <rect
            x="20" y="200" width="100" height="60" rx="8" fill="#10b981"
            className={`${styles.pipelineStage} ${activeComponent === 'policy' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('policy')}
          />
          <text x="70" y="225" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">
            Policy
          </text>
          <text x="70" y="240" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            {'\u03C0_\u03B8'}
          </text>

          {/* Reference Policy */}
          <rect
            x="20" y="120" width="100" height="50" rx="8" fill="#1e40af"
            stroke="#3b82f6" strokeWidth="2" strokeDasharray="4"
            className={`${styles.pipelineStage} ${activeComponent === 'reference' ? styles.pipelineStageActive : ''}`}
            onClick={() => setActiveComponent('reference')}
          />
          <text x="70" y="145" textAnchor="middle" fill="white" fontSize="10" pointerEvents="none">
            Reference
          </text>
          <text x="70" y="158" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
            {'\u03C0_ref (frozen)'}
          </text>

          {/* Arrow: SFT -> Reference (copy) */}
          <path d="M 180 60 Q 140 90 120 120" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" strokeDasharray="4" />
          <text x="135" y="95" fill="#888" fontSize="8">copy</text>

          {/* Arrow: SFT -> Policy (init) */}
          <path d="M 180 70 Q 100 130 70 195" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" strokeDasharray="4" />
          <text x="95" y="140" fill="#888" fontSize="8">init</text>

          {/* PPO Update box */}
          <rect x="180" y="190" width="120" height="80" rx="8" fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="2" />
          <text x="240" y="210" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">
            PPO Update
          </text>
          <text x="240" y="230" textAnchor="middle" fill="#888" fontSize="9">
            max r(x,y)
          </text>
          <text x="240" y="245" textAnchor="middle" fill="#888" fontSize="9">
            {'\u2212 \u03B2\u00B7KL(\u03C0||\u03C0_ref)'}
          </text>

          {/* Arrow: Policy -> PPO */}
          <path d="M 120 230 L 175 230" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" className={styles.flowArrow} />

          {/* Arrow: Reward -> PPO */}
          <path d="M 340 150 L 305 190" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" />
          <text x="330" y="175" fill="#888" fontSize="8">reward</text>

          {/* Arrow: Reference -> PPO (KL) */}
          <path d="M 120 160 L 180 200" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" strokeDasharray="4" />
          <text x="140" y="175" fill="#888" fontSize="8">KL</text>

          {/* Arrow: PPO -> Policy */}
          <path d="M 180 250 Q 130 270 90 260" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" />
          <text x="135" y="268" fill="#888" fontSize="8">{'update \u03B8'}</text>
        </svg>
      </div>

      {activeComponent !== null && (
        <div className="info-panel">
          <h4>{componentInfo[activeComponent].title}</h4>
          <p>{componentInfo[activeComponent].desc}</p>
        </div>
      )}

      {/* ===== Loss Functions ===== */}
      <h2>Loss Functions</h2>
      <div className={styles.lossEquations}>
        {/* Reward Model Loss */}
        <div
          className={`${styles.lossItem} ${activeLoss === 'reward' ? styles.lossItemActive : ''}`}
          onClick={() => setActiveLoss(activeLoss === 'reward' ? null : 'reward')}
        >
          <h4 className={styles.lossItemTitle}>
            <span className={styles.lossDot} style={{ background: '#f59e0b' }} />
            Reward Model Loss
          </h4>
          <div className="formula-block">
            <MathJax inline>
              {'\\( \\mathcal{L}_{\\text{RM}} = -\\mathbb{E}_{(x, y_w, y_l)} \\left[ \\log \\sigma \\left( r_\\phi(x, y_w) - r_\\phi(x, y_l) \\right) \\right] \\)'}
            </MathJax>
          </div>
          <p className={styles.lossDesc}>Bradley-Terry model: preferred response should score higher</p>
        </div>

        {/* PPO Objective */}
        <div
          className={`${styles.lossItem} ${activeLoss === 'ppo' ? styles.lossItemActive : ''}`}
          onClick={() => setActiveLoss(activeLoss === 'ppo' ? null : 'ppo')}
        >
          <h4 className={styles.lossItemTitle}>
            <span className={styles.lossDot} style={{ background: '#10b981' }} />
            PPO Objective
          </h4>
          <div className="formula-block">
            <MathJax inline>
              {'\\( \\mathcal{J}(\\theta) = \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_\\theta} \\left[ r_\\phi(x, y) - \\beta \\cdot D_{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}}) \\right] \\)'}
            </MathJax>
          </div>
          <p className={styles.lossDesc}>Maximize reward while staying close to reference policy</p>
        </div>

        {/* KL Penalty */}
        <div
          className={`${styles.lossItem} ${activeLoss === 'kl' ? styles.lossItemActive : ''}`}
          onClick={() => setActiveLoss(activeLoss === 'kl' ? null : 'kl')}
        >
          <h4 className={styles.lossItemTitle}>
            <span className={styles.lossDot} style={{ background: '#3b82f6' }} />
            KL Penalty
          </h4>
          <div className="formula-block">
            <MathJax inline>
              {'\\( D_{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}}) = \\mathbb{E}_{y \\sim \\pi_\\theta} \\left[ \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)} \\right] \\)'}
            </MathJax>
          </div>
          <p className={styles.lossDesc}>Prevents reward hacking and catastrophic forgetting</p>
        </div>
      </div>

      {/* Notation Guide */}
      <div className={styles.notationPanel}>
        <h4>Notation Guide</h4>
        <p className={styles.notationLine}>
          <strong style={{ color: '#f59e0b' }}>
            <MathJax inline>{'\\(y_w, y_l\\)'}</MathJax>
          </strong>{' '}
          &mdash; winning (preferred) and losing (rejected) responses
        </p>
        <p className={styles.notationLine}>
          <strong style={{ color: '#10b981' }}>
            <MathJax inline>{'\\(\\pi_\\theta\\)'}</MathJax>
          </strong>{' '}
          &mdash; policy being trained,{' '}
          <strong style={{ color: '#3b82f6' }}>
            <MathJax inline>{'\\(\\pi_{\\text{ref}}\\)'}</MathJax>
          </strong>{' '}
          &mdash; frozen reference (usually SFT model)
        </p>
        <p className={styles.notationSmall}>
          <MathJax inline>{'\\(\\beta\\)'}</MathJax> controls the strength of the KL penalty. Higher
          {' \u03B2'} = more conservative updates.
        </p>
      </div>

      {/* ===== Human Preference Collection ===== */}
      <h2>Human Preference Collection</h2>
      <div className={styles.preferenceDemo}>
        <div className={styles.promptBox}>
          <label className={styles.promptBoxLabel}>Prompt</label>
          <p className={styles.promptBoxText}>Explain quantum computing to a 10-year-old.</p>
        </div>
        <div className={styles.responsePair}>
          <div
            className={`${styles.response} ${aIsPreferred ? styles.responsePreferred : styles.responseRejected}`}
            onClick={() => setAIsPreferred(!aIsPreferred)}
          >
            <div className={styles.responseLabel}>
              {aIsPreferred ? '\u2713 Preferred' : '\u2717 Rejected'}
            </div>
            <div className={styles.responseText}>
              Imagine you have a magical coin that can be heads AND tails at the same time until
              you look at it. Quantum computers use tiny things called qubits that work like
              these magical coins...
            </div>
          </div>
          <div className={styles.vsBadge}>VS</div>
          <div
            className={`${styles.response} ${!aIsPreferred ? styles.responsePreferred : styles.responseRejected}`}
            onClick={() => setAIsPreferred(!aIsPreferred)}
          >
            <div className={styles.responseLabel}>
              {!aIsPreferred ? '\u2713 Preferred' : '\u2717 Rejected'}
            </div>
            <div className={styles.responseText}>
              Quantum computing leverages quantum mechanical phenomena such as superposition and
              entanglement to perform computations on quantum bits or qubits...
            </div>
          </div>
        </div>
        <p className={styles.hintText}>
          Click responses to swap preference. The reward model learns:{' '}
          <MathJax inline>{'\\(r(x, y_w) > r(x, y_l)\\)'}</MathJax>
        </p>
      </div>

      {/* ===== KL Penalty Effect ===== */}
      <h2>KL Penalty Effect</h2>
      <div className={styles.sliderContainer}>
        <label>{'\u03B2 (KL coef):'}</label>
        <input
          type="range"
          className={styles.slider}
          min="0"
          max="0.5"
          step="0.01"
          value={beta}
          onChange={(e) => setBeta(parseFloat(e.target.value))}
        />
        <span className={styles.sliderValue}>{beta.toFixed(2)}</span>
      </div>
      <div className={styles.klVisualization}>
        <div
          className={styles.policyCircle}
          style={{ background: 'linear-gradient(135deg, #1e40af, #3b82f6)' }}
        >
          {'\u03C0_ref'}
          <br />
          <span style={{ fontSize: '0.7rem', fontWeight: 'normal' }}>frozen</span>
        </div>
        <div className={styles.klArrow}>
          <svg viewBox="0 0 60 24">
            <path d="M 0 12 L 50 12" stroke="#f59e0b" strokeWidth="3" fill="none" />
            <polygon points="50 6, 60 12, 50 18" fill="#f59e0b" />
          </svg>
          <span className={styles.klValueLabel}>KL &asymp; {klAmount.toFixed(2)}</span>
        </div>
        <div className={styles.policyCircle} style={{ background: policyGradient }}>
          {'\u03C0_\u03B8'}
          <br />
          <span style={{ fontSize: '0.7rem', fontWeight: 'normal' }}>training</span>
        </div>
      </div>
      <p className={styles.klExplanation}>{klExplanation}</p>
      <div className={styles.rewardDemo} style={{ marginTop: '15px' }}>
        <div className={styles.rewardBar}>
          <span className={styles.rewardBarLabel}>Reward</span>
          <div className={styles.rewardBarTrack}>
            <div
              className={styles.rewardBarFill}
              style={{
                width: `${reward * 100}%`,
                background: 'linear-gradient(90deg, #10b981, #059669)',
              }}
            />
          </div>
          <span className={styles.rewardBarValue}>+{reward.toFixed(2)}</span>
        </div>
        <div className={styles.rewardBar}>
          <span className={styles.rewardBarLabel}>KL Penalty</span>
          <div className={styles.rewardBarTrack}>
            <div
              className={styles.rewardBarFill}
              style={{
                width: `${penalty * 100}%`,
                background: 'linear-gradient(90deg, #ef4444, #dc2626)',
              }}
            />
          </div>
          <span className={styles.rewardBarValue}>&minus;{penalty.toFixed(2)}</span>
        </div>
        <div className={styles.rewardBar}>
          <span className={styles.rewardBarLabelBold}>Net Objective</span>
          <div className={styles.rewardBarTrack}>
            <div
              className={styles.rewardBarFill}
              style={{
                width: `${Math.max(0, net) * 100}%`,
                background: 'linear-gradient(90deg, #a855f7, #7c3aed)',
              }}
            />
          </div>
          <span className={`${styles.rewardBarValue} ${styles.netValueHighlight}`}>
            {net >= 0 ? `+${net.toFixed(2)}` : net.toFixed(2)}
          </span>
        </div>
      </div>

      {/* ===== Reward Model Loss Visualization ===== */}
      <h2>Reward Model Loss Visualization</h2>
      <div className={styles.rmLossViz}>
        <div className={styles.rmSliders}>
          <div className={styles.rmSliderGroup}>
            <label className={styles.rmSliderGroupLabel}>
              <span>
                r(x, y<sub>w</sub>) &mdash; Preferred
              </span>
              <span className={styles.rmSliderGroupValuePreferred}>
                {rw >= 0 ? `+${rw.toFixed(2)}` : rw.toFixed(2)}
              </span>
            </label>
            <input
              type="range"
              className={`${styles.rmSlider} ${styles.rmSliderPreferred}`}
              min="-3"
              max="3"
              step="0.1"
              value={rw}
              onChange={(e) => setRw(parseFloat(e.target.value))}
            />
          </div>
          <div className={styles.rmSliderGroup}>
            <label className={styles.rmSliderGroupLabel}>
              <span>
                r(x, y<sub>l</sub>) &mdash; Rejected
              </span>
              <span className={styles.rmSliderGroupValueRejected}>
                {rl >= 0 ? `+${rl.toFixed(2)}` : rl.toFixed(2)}
              </span>
            </label>
            <input
              type="range"
              className={`${styles.rmSlider} ${styles.rmSliderRejected}`}
              min="-3"
              max="3"
              step="0.1"
              value={rl}
              onChange={(e) => setRl(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="diagram-frame">
          <canvas ref={canvasRef} className={styles.rmCanvas} />
        </div>

        <div className={styles.rmLossDisplay}>
          <div className={styles.rmMetric}>
            <div
              className={styles.rmMetricValue}
              style={{ color: diff > 0 ? '#10b981' : diff < 0 ? '#ef4444' : '#888' }}
            >
              {diff >= 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2)}
            </div>
            <div className={styles.rmMetricLabel}>
              r(y<sub>w</sub>) &minus; r(y<sub>l</sub>)
            </div>
          </div>
          <div className={styles.rmMetric}>
            <div className={styles.rmMetricValue} style={{ color: '#a855f7' }}>
              {currentSig.toFixed(3)}
            </div>
            <div className={styles.rmMetricLabel}>&sigma;(difference)</div>
          </div>
          <div className={styles.rmMetric}>
            <div
              className={styles.rmMetricValue}
              style={{
                color: currentLoss < 0.5 ? '#10b981' : currentLoss < 1 ? '#f59e0b' : '#ef4444',
              }}
            >
              {currentLoss.toFixed(3)}
            </div>
            <div className={styles.rmMetricLabel}>Loss = &minus;log(&sigma;)</div>
          </div>
        </div>

        <p className={styles.bradleyTerryHint}>
          The Bradley-Terry loss pushes{' '}
          <MathJax inline>{'\\(r(y_w) > r(y_l)\\)'}</MathJax>. When preference is clear (large
          gap), loss &rarr; 0.
          <br />
          When scores are equal, loss = log(2) &asymp; 0.69. Wrong preference &rarr; high loss.
        </p>
      </div>

      {/* ===== Paper Lineage ===== */}
      <h2>Paper Lineage</h2>
      <section className="lineage-section">
        <div className="lineage-grid">
          <div className="lineage-group">
            <h4 style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>TAMER</h5>
              <p>Knox &amp; Stone, 2009 &mdash; Training agents from real-time human feedback signals; early framework for human-in-the-loop RL</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>Deep RL from Human Preferences</h5>
              <p>Christiano et al., 2017 &mdash; Reward learning from pairwise comparisons in Atari/MuJoCo; established the preference-based reward modeling pipeline</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>PPO</h5>
              <p>Schulman et al., 2017 &mdash; Proximal Policy Optimization; stable policy gradient method used as the RL optimizer in RLHF</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#f59e0b' }}>
              <h5>InstructGPT</h5>
              <p>Ouyang et al., 2022 &mdash; Scaled RLHF to GPT-3; demonstrated the SFT &rarr; RM &rarr; PPO pipeline for LLM alignment</p>
            </div>
          </div>
          <div className="lineage-group">
            <h4 style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>DPO</h5>
              <p>Rafailov et al., 2023 &mdash; Direct Preference Optimization; eliminates the reward model by reparameterizing the RLHF objective as a classification loss</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>Constitutional AI (CAI)</h5>
              <p>Bai et al., 2022 &mdash; RLAIF: replaces human labelers with AI feedback guided by a constitution of principles</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>RLHF with KTO</h5>
              <p>Ethayarajh et al., 2024 &mdash; Kahneman-Tversky Optimization; uses binary (good/bad) signals instead of pairwise preferences</p>
            </div>
            <div className="lineage-item" style={{ borderLeftColor: '#10b981' }}>
              <h5>GRPO</h5>
              <p>Shao et al., 2024 &mdash; Group Relative Policy Optimization; replaces the critic with group-level baselines for more efficient RL fine-tuning</p>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
