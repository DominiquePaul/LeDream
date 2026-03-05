'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { MathJax } from 'better-react-mathjax';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import styles from './sac.module.css';

import type { CoreIdea, KeyFeature } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';

const coreIdeas: CoreIdea[] = [
  { title: 'Maximum Entropy RL', desc: 'Augment the standard RL objective with an entropy bonus, encouraging the agent to explore diverse behaviors while maximizing reward' },
  { title: 'Twin Q-Networks', desc: 'Use the minimum of two independent Q-functions to combat overestimation bias in value estimation' },
  { title: 'Automatic Temperature Tuning', desc: 'Learn the entropy coefficient \u03B1 via dual gradient descent to maintain a target entropy level, removing a critical hyperparameter' },
];

const keyFeatures: KeyFeature[] = [
  { title: 'Twin Q-Networks', desc: 'Uses minimum of two Q-values to mitigate overestimation bias (clipped double Q-learning)' },
  { title: 'Stochastic Policy', desc: 'Gaussian policy with reparameterization trick: a = tanh(\u03BC + \u03C3\u00B7\u03B5), \u03B5 ~ N(0,1)' },
  { title: 'Auto Temperature', desc: 'Automatically tunes \u03B1 to maintain target entropy, balancing exploration and exploitation' },
];

/* ------------------------------------------------------------------ */
/*  Static Data                                                        */
/* ------------------------------------------------------------------ */

const componentInfo: Record<string, { title: string; desc: string }> = {
  env: {
    title: 'Environment',
    desc: 'The agent interacts with the environment, receiving states and rewards while executing actions. SAC is designed for continuous action spaces (e.g., robotics, locomotion).',
  },
  buffer: {
    title: 'Replay Buffer',
    desc: "Stores transitions (s, a, r, s', done) for off-policy learning. Random sampling breaks temporal correlations and improves sample efficiency.",
  },
  actor: {
    title: 'Actor Network (Policy)',
    desc: 'Outputs a Gaussian distribution over actions: \u03bc(s) and \u03c3(s). Actions are sampled via reparameterization: a = tanh(\u03bc + \u03c3\u00b7\u03b5). The squashing ensures bounded actions.',
  },
  critic1: {
    title: 'Critic Q\u2081',
    desc: 'First Q-network estimating Q(s,a). Takes state-action pairs and outputs value estimates. Trained to minimize soft Bellman residual.',
  },
  critic2: {
    title: 'Critic Q\u2082',
    desc: 'Second Q-network for double Q-learning. Using min(Q\u2081, Q\u2082) reduces overestimation bias that plagues standard actor-critic methods.',
  },
  target1: {
    title: 'Target Network Q\u0304\u2081',
    desc: 'Slowly-updated copy of Q\u2081 used for stable target computation. Updated via Polyak averaging: \u03c6\u0304 \u2190 \u03c4\u03c6 + (1\u2212\u03c4)\u03c6\u0304 with \u03c4 \u2248 0.005.',
  },
  target2: {
    title: 'Target Network Q\u0304\u2082',
    desc: 'Slowly-updated copy of Q\u2082. Target networks prevent the moving target problem during training.',
  },
  alpha: {
    title: 'Temperature \u03b1',
    desc: 'Entropy coefficient controlling exploration. Higher \u03b1 encourages random exploration; lower \u03b1 focuses on exploitation. Can be automatically tuned to maintain target entropy H\u0304 = -dim(A).',
  },
};

const stepDetailsExpanded: Record<number, string> = {
  1: 'Uniformly sample a mini-batch B of N transitions from the replay buffer D. This is the only source of data for learning (off-policy).',
  2: "Sample fresh action \u00e3' ~ \u03c0(\u00b7|s') from current policy, evaluate target critics on (s', \u00e3'), compute soft Bellman target: y = r + \u03b3(1-d)[min Q\u0304(s',\u00e3') - \u03b1 log \u03c0(\u00e3'|s')]",
  3: 'Train both critics independently by minimizing MSE to target: J_Q = \u00bd(Q(s,a) - y)\u00b2. Independence comes from separate parameters and optimizer noise.',
  4: 'Sample \u00e3 ~ \u03c0(\u00b7|s) from current policy, plug into current critics. Minimize: \u03b1 log \u03c0(\u00e3|s) - min(Q\u2081,Q\u2082)(s,\u00e3). This is the policy improvement step.',
  5: 'Update \u03b1 by minimizing: J(\u03b1) = -\u03b1 \u00b7 E[log \u03c0(\u00e3|s) + H\u0304]. Keeps entropy near target H\u0304 = -dim(A). Optional but standard in modern SAC.',
  6: 'Soft update targets: \u03c6\u0304\u1d62 \u2190 \u03c4\u03c6\u1d62 + (1\u2212\u03c4)\u03c6\u0304\u1d62 for i \u2208 {1,2}. Stabilises bootstrapping by slowly tracking online networks.',
};

interface StepData {
  number: number;
  title: string;
  defaultDesc: string;
}

const steps: StepData[] = [
  { number: 1, title: 'Sample from Replay Buffer', defaultDesc: "Draw mini-batch of transitions (s, a, r, s', d)" },
  { number: 2, title: 'Compute Target Q-value', defaultDesc: "Sample \u00e3' ~ \u03c0(\u00b7|s'), evaluate targets, subtract entropy" },
  { number: 3, title: 'Update Critics', defaultDesc: 'Train independently via MSE to soft Bellman target' },
  { number: 4, title: 'Update Actor', defaultDesc: 'Sample \u00e3 ~ \u03c0(\u00b7|s), maximize Q \u2212 \u03b1\u00b7log \u03c0' },
  { number: 5, title: 'Update Temperature \u03b1', defaultDesc: 'Adjust entropy coefficient automatically' },
  { number: 6, title: 'Soft Update Targets', defaultDesc: '\u03c6\u0304 \u2190 \u03c4\u03c6 + (1\u2212\u03c4)\u03c6\u0304' },
];

interface Experience {
  isNew: boolean;
  sampled: boolean;
}

/* ------------------------------------------------------------------ */
/*  Drawing helper                                                     */
/* ------------------------------------------------------------------ */

function drawDistribution(
  canvas: HTMLCanvasElement,
  alpha: number,
  width: number,
  height: number,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx || width === 0 || height === 0) return;

  ctx.clearRect(0, 0, width, height);

  // Axes
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, height - 30);
  ctx.lineTo(width - 20, height - 30);
  ctx.moveTo(40, height - 30);
  ctx.lineTo(40, 20);
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '11px sans-serif';
  ctx.fillText('Action', width / 2 - 20, height - 10);
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('\u03c0(a|s)', -20, 0);
  ctx.restore();

  // Gaussian
  const sigma = 0.1 + alpha * 0.8;
  const mu = 0;
  const plotWidth = width - 80;
  const plotHeight = height - 60;
  const xOffset = 50;

  ctx.beginPath();
  ctx.strokeStyle = '#a855f7';
  ctx.lineWidth = 3;

  const maxY = 1 / (sigma * Math.sqrt(2 * Math.PI));

  for (let i = 0; i <= plotWidth; i++) {
    const x = (i / plotWidth - 0.5) * 4;
    const y =
      Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) /
      (sigma * Math.sqrt(2 * Math.PI));
    const canvasX = xOffset + i;
    const canvasY = height - 30 - (y / maxY) * plotHeight * 0.9;

    if (i === 0) ctx.moveTo(canvasX, canvasY);
    else ctx.lineTo(canvasX, canvasY);
  }
  ctx.stroke();

  // Fill
  ctx.lineTo(xOffset + plotWidth, height - 30);
  ctx.lineTo(xOffset, height - 30);
  ctx.closePath();
  ctx.fillStyle = 'rgba(168, 85, 247, 0.2)';
  ctx.fill();

  // Entropy annotation
  const entropy = 0.5 * Math.log(2 * Math.PI * Math.E * sigma * sigma);
  ctx.fillStyle = '#00d9ff';
  ctx.font = 'bold 12px sans-serif';
  ctx.fillText('H(\u03c0) \u2248 ' + entropy.toFixed(2), width - 110, 35);
  ctx.fillStyle = '#888';
  ctx.font = '10px sans-serif';
  ctx.fillText('\u03c3 = ' + sigma.toFixed(2), width - 110, 50);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SACPage() {
  /* ---- Architecture interaction ---- */
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  /* ---- Algorithm steps ---- */
  const [activeStep, setActiveStep] = useState<number | null>(null);

  /* ---- Loss items ---- */
  const [activeLoss, setActiveLoss] = useState<string | null>(null);

  /* ---- Entropy slider ---- */
  const [alphaVal, setAlphaVal] = useState(0.2);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = useCanvasResize(canvasRef);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawDistribution(canvas, alphaVal, canvasSize.width, canvasSize.height);
  }, [alphaVal, canvasSize]);

  /* ---- Replay Buffer ---- */
  const MAX_EXPERIENCES = 150;
  const [experiences, setExperiences] = useState<Experience[]>(() =>
    Array.from({ length: 60 }, () => ({ isNew: false, sampled: false })),
  );

  const addExperience = useCallback(() => {
    setExperiences((prev) => {
      const cleared = prev.map((e) => ({ ...e, isNew: false, sampled: false }));
      if (cleared.length >= MAX_EXPERIENCES) cleared.shift();
      cleared.push({ isNew: true, sampled: false });
      return cleared;
    });
  }, []);

  const sampleBatch = useCallback(() => {
    setExperiences((prev) => {
      if (prev.length < 8) return prev;
      const cleared = prev.map((e) => ({ ...e, isNew: false, sampled: false }));
      const indices = new Set<number>();
      while (indices.size < 8) {
        indices.add(Math.floor(Math.random() * cleared.length));
      }
      indices.forEach((i) => {
        cleared[i] = { ...cleared[i], sampled: true };
      });
      return cleared;
    });
  }, []);

  const clearBuffer = useCallback(() => {
    setExperiences([]);
  }, []);

  /* ---- Helper to get experience color ---- */
  const getExpColor = (exp: Experience, idx: number, total: number) => {
    if (exp.sampled) return '#f59e0b';
    const age = (total - idx) / total;
    const r = Math.round(16 + (59 - 16) * age);
    const g = Math.round(185 + (130 - 185) * age);
    const b = Math.round(129 + (246 - 129) * age);
    return `rgb(${r}, ${g}, ${b})`;
  };

  /* ---- Render ---- */
  return (
    <div className="method-page">
      <h1>Soft Actor-Critic (SAC)</h1>
      <p className="subtitle">
        Maximum Entropy Reinforcement Learning Algorithm &mdash; Haarnoja et al., 2018
      </p>
      <div className={styles.githubLink}>
        <a
          href="https://github.com/haarnoja/sac"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span style={{ marginRight: 5 }}>&rarr;</span> github.com/haarnoja/sac
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ============ Architecture Diagram ============ */}
      <section>
        <h2>Network Architecture</h2>
        <div className="diagram-frame">
          <svg className={styles.architectureSvg} viewBox="0 0 500 400">
            {/* Defs (arrow markers) */}
            <defs>
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
                id="arrowPink"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#ec4899" />
              </marker>
            </defs>

            {/* Environment */}
            <rect
              x="200" y="10" width="100" height="50" rx="8"
              fill="#10b981"
              className={`${styles.networkBox}${activeComponent === 'env' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('env')}
            />
            <text x="250" y="40" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold" pointerEvents="none">
              Environment
            </text>

            {/* Replay Buffer */}
            <rect
              x="350" y="100" width="120" height="60" rx="8"
              fill="#f59e0b"
              className={`${styles.networkBox}${activeComponent === 'buffer' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('buffer')}
            />
            <text x="410" y="125" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Replay Buffer
            </text>
            <text x="410" y="145" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              (s, a, r, s&apos;, d)
            </text>

            {/* Actor (Policy) */}
            <rect
              x="30" y="170" width="120" height="70" rx="8"
              fill="#a855f7"
              className={`${styles.networkBox}${activeComponent === 'actor' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('actor')}
            />
            <text x="90" y="195" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Actor {'\u03c0'}_{'{\u03b8}'}
            </text>
            <text x="90" y="215" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              Gaussian Policy
            </text>
            <text x="90" y="230" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              {'\u03bc'}(s), {'\u03c3'}(s)
            </text>

            {/* Critic 1 */}
            <rect
              x="190" y="170" width="100" height="70" rx="8"
              fill="#3b82f6"
              className={`${styles.networkBox}${activeComponent === 'critic1' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('critic1')}
            />
            <text x="240" y="200" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Critic Q_{'\u03c6\u2081'}
            </text>
            <text x="240" y="220" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              Q(s, a)
            </text>

            {/* Critic 2 */}
            <rect
              x="310" y="170" width="100" height="70" rx="8"
              fill="#3b82f6"
              className={`${styles.networkBox}${activeComponent === 'critic2' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('critic2')}
            />
            <text x="360" y="200" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              Critic Q_{'\u03c6\u2082'}
            </text>
            <text x="360" y="220" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">
              Q(s, a)
            </text>

            {/* Target Critic 1 */}
            <rect
              x="190" y="290" width="100" height="50" rx="8"
              fill="#1e40af" stroke="#3b82f6" strokeWidth="2" strokeDasharray="4"
              className={`${styles.networkBox}${activeComponent === 'target1' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('target1')}
            />
            <text x="240" y="315" textAnchor="middle" fill="white" fontSize="10" pointerEvents="none">
              Target Q_{'\u03c6\u0304\u2081'}
            </text>

            {/* Target Critic 2 */}
            <rect
              x="310" y="290" width="100" height="50" rx="8"
              fill="#1e40af" stroke="#3b82f6" strokeWidth="2" strokeDasharray="4"
              className={`${styles.networkBox}${activeComponent === 'target2' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('target2')}
            />
            <text x="360" y="315" textAnchor="middle" fill="white" fontSize="10" pointerEvents="none">
              Target Q_{'\u03c6\u0304\u2082'}
            </text>

            {/* Alpha (Temperature) */}
            <rect
              x="30" y="290" width="100" height="50" rx="8"
              fill="#ec4899"
              className={`${styles.networkBox}${activeComponent === 'alpha' ? ` ${styles.networkBoxActive}` : ''}`}
              onClick={() => setActiveComponent('alpha')}
            />
            <text x="80" y="315" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold" pointerEvents="none">
              {'\u03b1'} (Temperature)
            </text>

            {/* Flow Arrows */}
            {/* Environment to Buffer */}
            <path
              d="M 300 35 Q 400 35 410 100"
              stroke="#10b981" strokeWidth="2" fill="none"
              markerEnd="url(#arrowGreen)"
              className={styles.flowArrow}
            />

            {/* Actor to Environment */}
            <path
              d="M 90 170 Q 90 90 200 35"
              stroke="#a855f7" strokeWidth="2" fill="none"
              markerEnd="url(#arrowPurple)"
              className={styles.flowArrow}
            />
            <text x="100" y="110" fill="#a855f7" fontSize="9">
              action a
            </text>

            {/* Buffer to Critics */}
            <path d="M 350 130 L 290 170" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" />
            <path d="M 410 160 L 360 170" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" />

            {/* Buffer to Actor */}
            <path d="M 350 130 Q 200 130 150 175" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" />

            {/* Critics to Targets (soft update) */}
            <path d="M 240 240 L 240 290" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" strokeDasharray="4" />
            <path d="M 360 240 L 360 290" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" strokeDasharray="4" />
            <text x="270" y="270" fill="#3b82f6" fontSize="8">
              {'\u03c4'} soft update
            </text>

            {/* Alpha to Actor */}
            <path d="M 90 290 L 90 240" stroke="#ec4899" strokeWidth="2" fill="none" markerEnd="url(#arrowPink)" />
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
          Learning loop (per update). Acting + collecting transitions happens outside this loop.
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
          {/* Critic Loss */}
          <div
            className={`${styles.lossItem}${activeLoss === 'critic' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'critic' ? null : 'critic')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#3b82f6' }} />
              Critic Loss (Soft Bellman)
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( J_Q(\\phi) = \\mathbb{E}_{(s,a,r,s\') \\sim \\mathcal{D}} \\left[ \\frac{1}{2} \\left( Q_\\phi(s,a) - y \\right)^2 \\right] \\)'}
              </MathJax>
            </div>
            <div className="formula-block">
              <MathJax inline>
                {'\\( y = r + \\gamma (1-d) \\left( \\min_{j=1,2} Q_{\\bar{\\phi}_j}(s\', \\tilde{a}\') - \\alpha \\log \\pi_\\theta(\\tilde{a}\'|s\') \\right) \\)'}
              </MathJax>
            </div>
            <p className={styles.lossHint}>
              Critics are trained independently (separate parameters + optimizer states)
            </p>
          </div>

          {/* Actor Loss */}
          <div
            className={`${styles.lossItem}${activeLoss === 'actor' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'actor' ? null : 'actor')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#a855f7' }} />
              Actor Loss
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( J_\\pi(\\theta) = \\mathbb{E}_{s \\sim \\mathcal{D}, \\tilde{a} \\sim \\pi_\\theta} \\left[ \\alpha \\log \\pi_\\theta(\\tilde{a}|s) - \\min_{j=1,2} Q_{\\phi_j}(s, \\tilde{a}) \\right] \\)'}
              </MathJax>
            </div>
          </div>

          {/* Temperature Loss */}
          <div
            className={`${styles.lossItem}${activeLoss === 'alpha' ? ` ${styles.lossItemActive}` : ''}`}
            onClick={() => setActiveLoss(activeLoss === 'alpha' ? null : 'alpha')}
          >
            <h4>
              <span className={styles.lossDot} style={{ background: '#ec4899' }} />
              Temperature Loss
            </h4>
            <div className="formula-block">
              <MathJax inline>
                {'\\( J(\\alpha) = \\mathbb{E}_{\\tilde{a} \\sim \\pi_\\theta} \\left[ -\\alpha \\left( \\log \\pi_\\theta(\\tilde{a}|s) + \\bar{\\mathcal{H}} \\right) \\right] \\)'}
              </MathJax>
            </div>
            <p className={styles.lossHint}>
              where{' '}
              <MathJax inline>{'\\( \\bar{\\mathcal{H}} \\)'}</MathJax>{' '}
              is the target entropy (typically{' '}
              <MathJax inline>{'\\( -\\dim(\\mathcal{A}) \\)'}</MathJax>)
            </p>
          </div>
        </div>

        {/* Notation Guide */}
        <div className={styles.notationPanel}>
          <h4>Notation Guide</h4>
          <p>
            <strong style={{ color: '#00d9ff' }}>Prime (&apos;)</strong> &mdash; next
            timestep:{' '}
            <MathJax inline>{'\\(s\'\\)'}</MathJax> = next state,{' '}
            <MathJax inline>{'\\(a\'\\)'}</MathJax> = next action
          </p>
          <p>
            <strong style={{ color: '#a855f7' }}>Tilde (~)</strong> &mdash; freshly
            sampled from current policy:{' '}
            <MathJax inline>{'\\(\\tilde{a} \\sim \\pi_\\theta(\\cdot|s)\\)'}</MathJax>
          </p>
          <p style={{ fontSize: '0.85rem', color: '#888' }}>
            The stored action{' '}
            <MathJax inline>{'\\(a\\)'}</MathJax> is from the replay buffer (old
            policy), while{' '}
            <MathJax inline>{"\\(\\tilde{a}'\\)"}</MathJax> is sampled from the
            current policy to compute the bootstrap target.
          </p>
        </div>
      </section>

      {/* ============ Entropy & Exploration ============ */}
      <section>
        <h2>Entropy &amp; Exploration</h2>
        <div className={styles.entropyViz}>
          <div className={styles.entropySliderContainer}>
            <label>Temperature {'\u03b1'}:</label>
            <input
              type="range"
              className={styles.entropySlider}
              min="0.01"
              max="1"
              step="0.01"
              value={alphaVal}
              onChange={(e) => setAlphaVal(parseFloat(e.target.value))}
            />
            <span className={styles.entropyValue}>{alphaVal.toFixed(2)}</span>
          </div>
          <canvas ref={canvasRef} className={styles.distributionCanvas} />
          <p className={styles.entropyHint}>
            Higher {'\u03b1'} &rarr; More exploration (wider distribution)
            <br />
            Lower {'\u03b1'} &rarr; More exploitation (peaked distribution)
          </p>
        </div>
      </section>

      {/* ============ Replay Buffer ============ */}
      <section>
        <h2>Experience Replay Buffer</h2>
        <div className={styles.replayBufferViz}>
          {experiences.map((exp, idx) => (
            <div
              key={idx}
              className={
                `${styles.experience}` +
                (exp.isNew ? ` ${styles.experienceNew}` : '') +
                (exp.sampled ? ` ${styles.experienceSampled}` : '')
              }
              style={{ background: getExpColor(exp, idx, experiences.length) }}
            />
          ))}
        </div>
        <div className={styles.legend}>
          <div className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: '#10b981' }} />
            <span>Recent experiences</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: '#3b82f6' }} />
            <span>Older experiences</span>
          </div>
          <div className={styles.legendItem}>
            <div
              className={styles.legendDot}
              style={{ background: '#f59e0b', boxShadow: '0 0 8px #f59e0b' }}
            />
            <span>Sampled for training</span>
          </div>
        </div>
        <div className={styles.controls}>
          <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={addExperience}>
            Add Experience
          </button>
          <button className={`${styles.btn} ${styles.btnSecondary}`} onClick={sampleBatch}>
            Sample Batch
          </button>
          <button className={`${styles.btn} ${styles.btnSecondary}`} onClick={clearBuffer}>
            Clear Buffer
          </button>
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
                <h5>DDPG</h5>
                <p>
                  Lillicrap et al., 2015 &mdash; Continuous control with deep RL;
                  deterministic actor-critic that SAC extends with stochastic policies
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>TD3</h5>
                <p>
                  Fujimoto et al., 2018 &mdash; Twin critics + delayed policy updates; SAC
                  adopts twin Q-networks to address overestimation
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#f59e0b' }}>
                <h5>Soft Q-Learning</h5>
                <p>
                  Haarnoja et al., 2017 &mdash; Maximum entropy framework for RL; direct
                  predecessor using energy-based policies
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
                <h5>CQL</h5>
                <p>
                  Kumar et al., 2020 &mdash; Adds conservative regularizer to SAC for
                  offline RL
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>REDQ</h5>
                <p>
                  Chen et al., 2021 &mdash; Ensemble of Q-functions with random subset
                  selection; improves SAC&apos;s sample efficiency
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>DrQ / DrQ-v2</h5>
                <p>
                  Kostrikov et al., 2020/2021 &mdash; SAC + data augmentation for
                  vision-based RL; state-of-the-art on pixel-based continuous control
                </p>
              </div>
              <div className={styles.lineageItem} style={{ borderLeftColor: '#10b981' }}>
                <h5>SAC-N / EDAC</h5>
                <p>
                  An et al., 2021 &mdash; SAC with N critics and penalized Q-ensemble for
                  offline RL without explicit conservatism
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
