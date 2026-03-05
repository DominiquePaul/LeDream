'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MathJax } from 'better-react-mathjax';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import { ToggleButtons } from '@/components/ToggleButtons/ToggleButtons';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import styles from './cql.module.css';

import type { CoreIdea, KeyFeature } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';

/* ================================================================
   DATA & CONSTANTS
   ================================================================ */

const ideaInfo: Record<string, { title: string; desc: string }> = {
  '1': {
    title: 'The Overestimation Problem in Offline RL',
    desc: 'In standard RL, if the Q-function overestimates the value of some actions, the agent can try those actions and correct the error. In offline RL, there is no such correction mechanism. The policy greedily selects actions with the highest Q-values, which tend to be out-of-distribution actions with erroneously inflated values. This causes the policy to fail catastrophically at test time.',
  },
  '2': {
    title: 'Conservative Q-Function Estimation',
    desc: 'CQL learns a Q-function whose expected value under the policy is a lower bound on the true policy value. This means the policy is always optimized against a pessimistic estimate \u2014 if it performs well under the conservative Q, it will perform at least as well in reality. The key insight is that only the expected value needs to be a lower bound, not every individual Q(s,a).',
  },
  '3': {
    title: 'Push Down OOD, Pull Up In-Distribution',
    desc: "The CQL regularizer has two opposing forces: (1) it minimizes Q-values under the current policy distribution (pushing down values for actions the policy might take but weren\u2019t in the data), and (2) it maximizes Q-values under the dataset distribution (pulling up values for actions that were actually observed). This creates a gap that naturally steers the policy toward in-distribution actions.",
  },
};

const coreIdeas: CoreIdea[] = [
  { title: 'The Overestimation Problem in Offline RL', desc: 'Standard offline RL overestimates Q-values for out-of-distribution (OOD) actions, causing catastrophic policy failure', detail: ideaInfo['1'].desc },
  { title: 'Conservative Q-Function Estimation', desc: 'CQL learns a Q-function that lower-bounds the true value, preventing over-optimistic policy updates', detail: ideaInfo['2'].desc },
  { title: 'Push Down OOD, Pull Up In-Distribution', desc: 'Minimize Q-values under the policy (push down OOD), maximize Q-values under the dataset (pull up known)', detail: ideaInfo['3'].desc },
];

const keyFeatures: KeyFeature[] = [
  { title: 'Fully Offline', desc: 'Learns entirely from static datasets with no environment interaction needed' },
  { title: 'Simple to Implement', desc: 'Just ~20 lines of code added to standard SAC or DQN implementations' },
  { title: 'No Behavior Cloning', desc: 'Unlike prior methods, CQL does not require fitting a separate behavior policy estimator' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  dataset: {
    title: 'Static Dataset D',
    desc: 'A fixed dataset of (state, action, reward, next_state) tuples collected by one or more behavior policies. CQL never collects new data \u2014 it learns entirely from this offline dataset. The dataset can come from multiple sources (expert demos, random exploration, previous policies).',
  },
  qnet: {
    title: 'Q-Network Q_\u03b8(s, a)',
    desc: 'Neural network approximating the action-value function. In the actor-critic variant, twin Q-networks are used (like SAC). For discrete actions, a single QR-DQN network is used. The Q-network receives both standard Bellman updates and the CQL regularization.',
  },
  bellman: {
    title: 'Bellman Backup Target',
    desc: 'Standard temporal difference target: y = r + \u03b3 \u00b7 Q_target(s\u2019, a\u2019). For actor-critic, a\u2019 comes from the current policy. For Q-learning, a\u2019 = argmax Q. The Bellman error provides the standard RL learning signal, pulling Q-values toward bootstrapped estimates.',
  },
  cql_reg: {
    title: 'CQL Regularizer',
    desc: 'The key innovation. Two terms: (1) Push Down: minimize E_\u03bc[Q(s,a)] where \u03bc is the policy (or logsumexp over all actions). This penalizes high Q-values for actions the policy might choose. (2) Pull Up: maximize E_\u03c0\u03b2[Q(s,a)] over dataset actions. This prevents the Q-function from collapsing everywhere.',
  },
  loss: {
    title: 'Total CQL Loss',
    desc: 'The final training objective combines the standard Bellman error with the CQL regularizer scaled by \u03b1: L = \u00bd\u00b7Bellman_error + \u03b1\u00b7CQL_reg. The coefficient \u03b1 can be fixed or automatically tuned via Lagrangian dual gradient descent to maintain a target level of conservatism.',
  },
  policy: {
    title: 'Policy \u03c0',
    desc: 'In the actor-critic variant, the policy is trained to maximize the conservative Q-values (like standard SAC). Because the Q-values are conservative, the policy naturally avoids OOD actions without needing an explicit policy constraint. In the Q-learning variant, the policy is implicitly defined as the greedy policy over Q.',
  },
};

interface ObjVariant {
  title: string;
  formula: string;
  desc: string;
  color: string;
}

const objVariants: Record<string, ObjVariant> = {
  basic: {
    title: 'CQL (Eq. 1): Pointwise Lower Bound',
    formula: String.raw`\( \text{CQL reg.} \;=\; \alpha \; \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \mu(a|s)}} \!\left[Q(s,a)\right] \)`,
    desc: 'Simplest form \u2014 only pushes Q-values down under \u03bc(a|s), no pull-up term. Gives a pointwise lower bound: Q\u0302(s,a) \u2264 Q\u03c0(s,a) for all (s,a). Can be overly conservative since it penalizes even in-distribution actions.',
    color: '#ef4444',
  },
  tight: {
    title: 'CQL (Eq. 2): Expected Lower Bound (Tighter)',
    formula: String.raw`\( \text{CQL reg.} \;=\; \alpha \!\left( \underbrace{ \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \mu(a|s)}}\!\left[Q(s,a)\right] }_{\color{#ef4444}{\text{push down}}} \;-\; \underbrace{ \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \hat{\pi}_\beta(a|s)}}\!\left[Q(s,a)\right] }_{\color{#10b981}{\text{pull up}}} \right) \)`,
    desc: 'Adds the pull-up term under \u03c0\u03b2(a|s). Only the expected value V\u0302\u03c0(s) is lower-bounded, not every individual Q(s,a). This avoids unnecessary pessimism while remaining safe.',
    color: '#10b981',
  },
  cqlh: {
    title: 'CQL(H): Practical Variant (what you implement)',
    formula: String.raw`\( \text{CQL reg.} \;=\; \alpha \!\left( \underbrace{ \mathbb{E}_{s \sim \mathcal{D}} \!\left[ \log \sum_{a \in \mathcal{A}} \exp Q(s,a) \right] }_{\color{#ef4444}{\text{push down (logsumexp over all actions)}}} \;-\; \underbrace{ \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \hat{\pi}_\beta(a|s)}} \!\left[Q(s,a)\right] }_{\color{#10b981}{\text{pull up under } \hat{\pi}_\beta}} \right) \)`,
    desc: 'Sets \u03bc = softmax(Q) via the logsumexp trick \u2014 the log\u2211exp over all actions is a smooth upper bound on max Q(s,a). No need to sample from \u03bc explicitly. This is the variant used in practice \u2014 just ~20 lines of code on top of SAC or DQN.',
    color: '#a855f7',
  },
};


/* ================================================================
   COMPONENT
   ================================================================ */

export default function CQLPage() {
  /* ---------- State ---------- */
  const [objVariant, setObjVariant] = useState<string>('basic');
  const [alpha, setAlpha] = useState<number>(1.0);
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  /* ---------- Canvas Refs ---------- */
  const problemCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const regCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const problemSize = useCanvasResize(problemCanvasRef);
  const regSize = useCanvasResize(regCanvasRef);

  /* ---------- Drawing: Problem Canvas ---------- */
  const drawProblemViz = useCallback(() => {
    const canvas = problemCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = problemSize.width;
    const h = problemSize.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const pad = { left: 50, right: 20, top: 25, bottom: 35 };
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
    ctx.fillText('Action space', w / 2, h - 8);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Q-value', 0, 0);
    ctx.restore();

    const xMin = -3, xMax = 3;
    const yMin = -1.5, yMax = 5;

    function toX(x: number) { return pad.left + ((x - xMin) / (xMax - xMin)) * pw; }
    function toY(y: number) { return h - pad.bottom - ((y - yMin) / (yMax - yMin)) * ph; }

    // Dataset coverage region
    const dataLeft = -1.2, dataRight = 1.2;
    ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
    ctx.fillRect(toX(dataLeft), pad.top, toX(dataRight) - toX(dataLeft), ph);

    ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(dataLeft), pad.top);
    ctx.lineTo(toX(dataLeft), h - pad.bottom);
    ctx.moveTo(toX(dataRight), pad.top);
    ctx.lineTo(toX(dataRight), h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Dataset', toX(0), pad.top + 15);

    // True Q-function
    ctx.beginPath();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      const y = 2.0 * Math.exp(-0.5 * x * x) + 0.3 * Math.sin(x * 2);
      const cx = toX(x), cy = toY(y);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Standard Q-Learning: overestimated Q
    ctx.beginPath();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      const trueQ = 2.0 * Math.exp(-0.5 * x * x) + 0.3 * Math.sin(x * 2);
      let learnedQ;
      if (x < dataLeft || x > dataRight) {
        const dist = x < dataLeft ? (dataLeft - x) : (x - dataRight);
        learnedQ = trueQ + 1.5 * dist + 0.8 * Math.sin(x * 3);
      } else {
        learnedQ = trueQ + 0.15 * Math.sin(x * 5);
      }
      const cx = toX(x), cy = toY(learnedQ);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Overestimated!', toX(2.2), toY(4.2));
    ctx.fillText('Overestimated!', toX(-2.2), toY(3.8));

    // CQL: conservative Q
    ctx.beginPath();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2.5;
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      const trueQ = 2.0 * Math.exp(-0.5 * x * x) + 0.3 * Math.sin(x * 2);
      let learnedQ;
      if (x < dataLeft || x > dataRight) {
        const dist = x < dataLeft ? (dataLeft - x) : (x - dataRight);
        learnedQ = trueQ - 0.8 * dist;
      } else {
        learnedQ = trueQ - 0.15;
      }
      const cx = toX(x), cy = toY(learnedQ);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.fillStyle = '#3b82f6';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Conservative', toX(2.2), toY(-0.5));

    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    const legendY = pad.top + 12;
    const legendX = w - pad.right - 160;

    ctx.fillStyle = '#10b981';
    ctx.fillRect(legendX, legendY - 5, 12, 3);
    ctx.fillText('True Q', legendX + 18, legendY);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(legendX, legendY + 12, 12, 3);
    ctx.fillText('Standard Q', legendX + 18, legendY + 17);

    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(legendX, legendY + 29, 12, 3);
    ctx.fillText('CQL Q', legendX + 18, legendY + 34);
  }, [problemSize]);

  useEffect(() => { drawProblemViz(); }, [drawProblemViz]);

  /* ---------- Drawing: Regularizer Canvas ---------- */
  const drawRegViz = useCallback(() => {
    const canvas = regCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = regSize.width;
    const h = regSize.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const pad = { left: 50, right: 20, top: 20, bottom: 35 };
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
    ctx.fillText('Action', w / 2, h - 8);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Q(s,a)', 0, 0);
    ctx.restore();

    const nActions = 8;
    const barWidth = pw / (nActions * 2.5);
    const dataActions = [2, 3, 4, 5];
    const oodActions = [0, 1, 6, 7];

    const baseQ = [1.8, 2.5, 1.5, 2.0, 1.8, 1.6, 2.8, 2.2];
    const trueQ = [0.5, 0.8, 1.5, 2.0, 1.8, 1.6, 0.3, 0.6];

    const cqlQ = baseQ.map((q, i) => {
      if (oodActions.includes(i)) {
        return q - alpha * 0.6;
      } else {
        return q + alpha * 0.15;
      }
    });

    const yMax = 4;
    const yMin = -0.5;
    function toY(y: number) { return h - pad.bottom - ((y - yMin) / (yMax - yMin)) * ph; }

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath();
    ctx.moveTo(pad.left, toY(0));
    ctx.lineTo(w - pad.right, toY(0));
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let y = 0; y <= 3; y++) {
      ctx.fillText(y.toString(), pad.left - 8, toY(y) + 4);
    }

    // Draw bars
    const actionLabels = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'];
    for (let i = 0; i < nActions; i++) {
      const cx = pad.left + (i + 0.5) * (pw / nActions);
      const val = cqlQ[i];
      const barTop = toY(Math.max(val, 0));
      const isOOD = oodActions.includes(i);

      // Bar
      ctx.fillStyle = isOOD ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)';
      ctx.fillRect(cx - barWidth / 2, barTop, barWidth, h - pad.bottom - barTop);

      // True Q marker
      const trueY = toY(trueQ[i]);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - barWidth / 2 - 4, trueY);
      ctx.lineTo(cx + barWidth / 2 + 4, trueY);
      ctx.stroke();

      // Arrow showing push/pull direction
      if (alpha > 0.2) {
        const arrowStartY = toY(baseQ[i]);
        const arrowEndY = toY(cqlQ[i]);
        if (Math.abs(arrowStartY - arrowEndY) > 5) {
          ctx.strokeStyle = isOOD ? '#ef4444' : '#10b981';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(cx, arrowStartY);
          ctx.lineTo(cx, arrowEndY);
          ctx.stroke();
          // Arrowhead
          const dir = arrowEndY > arrowStartY ? 1 : -1;
          ctx.beginPath();
          ctx.moveTo(cx - 4, arrowEndY - dir * 6);
          ctx.lineTo(cx, arrowEndY);
          ctx.lineTo(cx + 4, arrowEndY - dir * 6);
          ctx.stroke();
        }
      }

      // Label
      ctx.fillStyle = isOOD ? '#ef4444' : '#10b981';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(actionLabels[i], cx, h - pad.bottom + 12);
      ctx.fillText(isOOD ? 'OOD' : 'Data', cx, h - pad.bottom + 22);
    }

    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('\u2014 True Q', w - pad.right - 80, pad.top + 15);

    // suppress unused warnings
    void dataActions;
  }, [regSize, alpha]);

  useEffect(() => { drawRegViz(); }, [drawRegViz]);

  /* ---------- Computed Metrics ---------- */
  const oodCount = 4;
  const dataCount = 4;
  const nActions = 8;
  const pushDownVal = alpha * 0.6 * oodCount / nActions;
  const pullUpVal = alpha * 0.15 * dataCount / nActions;
  const netVal = pullUpVal - pushDownVal;

  /* ---------- Current Variant ---------- */
  const currentVariant = objVariants[objVariant];

  /* ================================================================
     RENDER
     ================================================================ */

  return (
    <div className="method-page">
      <h1>Conservative Q-Learning (CQL)</h1>
      <p className={styles.subtitle}>
        Offline RL via Conservative Q-Function Estimation &mdash; Kumar et al., 2020
      </p>
      <div className={styles.githubLink}>
        <a href="https://github.com/aviralkumar2907/CQL" target="_blank" rel="noopener noreferrer">
          &rarr; github.com/aviralkumar2907/CQL
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== The Overestimation Problem ==================== */}
      <section>
        <h2>The Overestimation Problem</h2>
        <div className={styles.problemViz}>
          <div className={styles.problemDiagram}>
            <canvas ref={problemCanvasRef} className={styles.problemCanvas} />
          </div>
          <p className={styles.problemExplanation}>
            Standard offline RL overestimates Q-values for out-of-distribution actions (red line),
            causing the policy to select bad actions. CQL learns conservative Q-values (blue line)
            that stay close to the true values, especially for in-distribution actions within the dataset coverage.
          </p>
        </div>
      </section>

      {/* ==================== CQL Objective ==================== */}
      <section>
        <h2>CQL Objective</h2>

        {/* Full unified objective */}
        <div className={styles.objectiveSection}>
          <h4>Full Training Objective</h4>
          <p className={styles.objectiveIntro}>
            CQL adds a single regularizer to the standard Bellman error. The complete objective minimized at each iteration:
          </p>
          <div className={styles.fullObjectiveBox}>
            <MathJax>{String.raw`\[
              \hat{Q}^{k+1} \leftarrow \arg\min_Q \;\;
              \underbrace{
                \frac{1}{2}\, \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \!\left[\Big( Q(s,a) - \big(r + \gamma \hat{Q}^k(s', \pi(s'))\big) \Big)^2 \right]
              }_{\text{Standard Bellman error (same as SAC / DQN)}}
              \;\;+\;\;
              \underbrace{
                \color{#a855f7}{ \alpha \!\left(
                  \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \mu(a|s)}}\!\left[Q(s,a)\right]
                  \;-\;
                  \mathbb{E}_{\substack{s \sim \mathcal{D} \\ a \sim \hat{\pi}_\beta(a|s)}}\!\left[Q(s,a)\right]
                \right) }
              }_{\color{#a855f7}{\textbf{CQL regularizer (added by CQL)}}}
            \]`}</MathJax>
          </div>
          <div className={styles.objectiveAnnotations}>
            <div className={`${styles.annotationBox} ${styles.annotationBellman}`}>
              <div className={styles.annotationLabel} style={{ color: '#3b82f6' }}>Bellman Error</div>
              <div className={styles.annotationDesc}>
                Standard TD backup &mdash; identical to what SAC or DQN already compute. Drives Q toward bootstrapped targets.
              </div>
            </div>
            <div className={`${styles.annotationBox} ${styles.annotationCql}`}>
              <div className={styles.annotationLabel} style={{ color: '#a855f7' }}>CQL Regularizer</div>
              <div className={styles.annotationDesc}>
                <strong style={{ color: '#ef4444' }}>Push down</strong> Q-values under &mu;(a|s) &mdash;{' '}
                <strong style={{ color: '#10b981' }}>Pull up</strong> Q-values for dataset actions under &pi;<sub>&beta;</sub>. Net effect: conservative lower bound.
              </div>
            </div>
          </div>
        </div>

        {/* Variant selector */}
        <p className={styles.variantIntro}>
          The Bellman error is the same in all variants. These tabs show only the{' '}
          <strong style={{ color: '#a855f7' }}>CQL regularizer</strong> that gets added to it:
        </p>
        <div className={styles.variantTabs}>
          <ToggleButtons
            options={[
              { key: 'basic', label: 'CQL (Eq. 1)' },
              { key: 'tight', label: 'CQL (Eq. 2)' },
              { key: 'cqlh', label: 'CQL(H) \u2014 Practical' },
            ]}
            active={objVariant}
            onChange={setObjVariant}
          />
        </div>

        <div className={styles.variantPanel}>
          <h4 className={styles.variantTitle} style={{ color: currentVariant.color }}>
            {currentVariant.title}
          </h4>
          <div
            className={styles.formula}
            style={{
              borderColor: currentVariant.color + '40',
              background: currentVariant.color + '15',
            }}
          >
            <MathJax>{currentVariant.formula}</MathJax>
          </div>
          <p className={styles.variantDesc}>{currentVariant.desc}</p>
        </div>

        {/* Notation Guide */}
        <InfoPanel title="Notation Guide" className={styles.notationGuide}>
          <p>
            <strong style={{ color: '#f59e0b' }}><MathJax inline>{'\\(\\hat{\\pi}_\\beta\\)'}</MathJax></strong> &mdash; empirical behavior policy (from the dataset)
          </p>
          <p>
            <strong style={{ color: '#10b981' }}><MathJax inline>{'\\(\\mu(a|s)\\)'}</MathJax></strong> &mdash; distribution to push Q-values down under (often set to current policy &pi;)
          </p>
          <p>
            <strong style={{ color: '#a855f7' }}><MathJax inline>{'\\(\\alpha\\)'}</MathJax></strong> &mdash; conservatism coefficient. Higher &alpha; = more conservative lower bound. Can be auto-tuned via Lagrangian dual descent.
          </p>
          <p>
            <strong style={{ color: '#3b82f6' }}><MathJax inline>{'\\(r + \\gamma \\hat{Q}^k(s\', \\pi(s\'))\\)'}</MathJax></strong> &mdash; TD target: reward + discounted next-state value (uses s&prime; from the dataset tuple)
          </p>
          <p>
            The full objective is what you implement &mdash; just add the purple term to your existing Bellman update code.
          </p>
        </InfoPanel>
      </section>

      {/* ==================== CQL Regularizer Effect ==================== */}
      <section>
        <h2>CQL Regularizer Effect</h2>
        <div className={styles.cqlRegViz}>
          <div className={styles.sliderRow}>
            <label>&alpha; (CQL coef):</label>
            <input
              type="range"
              min={0}
              max={5}
              step={0.1}
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
            />
            <span className={styles.sliderValue}>{alpha.toFixed(1)}</span>
          </div>
          <div className={styles.regCanvasContainer}>
            <canvas ref={regCanvasRef} className={styles.regCanvas} />
          </div>
          <div className={styles.regMetrics}>
            <div className={styles.regMetric}>
              <div className={styles.regMetricValue} style={{ color: '#ef4444' }}>
                {(-pushDownVal).toFixed(2)}
              </div>
              <div className={styles.regMetricLabel}>Push Down (OOD)</div>
            </div>
            <div className={styles.regMetric}>
              <div className={styles.regMetricValue} style={{ color: '#10b981' }}>
                +{pullUpVal.toFixed(2)}
              </div>
              <div className={styles.regMetricLabel}>Pull Up (Dataset)</div>
            </div>
            <div className={styles.regMetric}>
              <div className={styles.regMetricValue} style={{ color: netVal <= 0 ? '#a855f7' : '#ef4444' }}>
                {netVal.toFixed(2)}
              </div>
              <div className={styles.regMetricLabel}>Net Conservatism</div>
            </div>
          </div>
          <p className={styles.regExplanation}>
            CQL penalizes Q-values for actions likely under the policy but unlikely in the dataset, while preserving Q-values for observed actions.
          </p>
        </div>
      </section>

      {/* ==================== How CQL Works - Architecture ==================== */}
      <section>
        <h2>How CQL Works</h2>
        <div className="diagram-frame">
          <svg className={styles.pipelineSvg} viewBox="0 0 500 300">
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
            </defs>

            {/* Static Dataset */}
            <rect
              x="20" y="20" width="120" height="55" rx="8" fill="#f59e0b"
              className={`${styles.pipelineStage} ${activeComponent === 'dataset' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'dataset' ? null : 'dataset')}
            />
            <text x="80" y="42" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">Static Dataset D</text>
            <text x="80" y="58" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">(s, a, r, s&apos;)</text>

            {/* Q-Network */}
            <rect
              x="200" y="20" width="120" height="55" rx="8" fill="#3b82f6"
              className={`${styles.pipelineStage} ${activeComponent === 'qnet' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'qnet' ? null : 'qnet')}
            />
            <text x="260" y="42" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">Q-Network</text>
            <text x="260" y="58" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">{'Q_\u03b8(s, a)'}</text>

            {/* Arrow: Dataset to Q-Net */}
            <path d="M 140 48 L 195 48" stroke="#f59e0b" strokeWidth="2" fill="none" markerEnd="url(#arrowOrange)" className={styles.flowArrow} />
            <text x="167" y="40" fill="#888" fontSize="8">sample</text>

            {/* Bellman Backup */}
            <rect
              x="370" y="20" width="110" height="55" rx="8" fill="rgba(59,130,246,0.3)" stroke="#3b82f6" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'bellman' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'bellman' ? null : 'bellman')}
            />
            <text x="425" y="42" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" pointerEvents="none">Bellman</text>
            <text x="425" y="58" textAnchor="middle" fill="#3b82f6" fontSize="9" pointerEvents="none">Target</text>

            {/* Arrow: Q to Bellman */}
            <path d="M 320 48 L 365 48" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" />

            {/* CQL Regularizer Box */}
            <rect
              x="20" y="120" width="200" height="80" rx="8" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'cql_reg' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'cql_reg' ? null : 'cql_reg')}
            />
            <text x="120" y="145" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold" pointerEvents="none">CQL Regularizer</text>
            <text x="120" y="165" textAnchor="middle" fill="#888" fontSize="9" pointerEvents="none">{'Push down: E_\u03bc[Q(s,a)]'}</text>
            <text x="120" y="180" textAnchor="middle" fill="#888" fontSize="9" pointerEvents="none">{'Pull up: E_\u03c0\u03b2[Q(s,a)]'}</text>

            {/* Arrow: Q to CQL Reg */}
            <path d="M 230 75 Q 180 100 150 120" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />
            <text x="180" y="105" fill="#888" fontSize="8">Q-values</text>

            {/* Combined Loss */}
            <rect
              x="280" y="130" width="120" height="60" rx="8" fill="rgba(0,217,255,0.2)" stroke="#00d9ff" strokeWidth="2"
              className={`${styles.pipelineStage} ${activeComponent === 'loss' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'loss' ? null : 'loss')}
            />
            <text x="340" y="155" textAnchor="middle" fill="#00d9ff" fontSize="10" fontWeight="bold" pointerEvents="none">Total Loss</text>
            <text x="340" y="170" textAnchor="middle" fill="#888" fontSize="9" pointerEvents="none">{'Bellman + \u03b1 CQL'}</text>

            {/* Arrows to Combined Loss */}
            <path d="M 425 75 Q 425 110 400 130" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrowBlue)" />
            <path d="M 220 160 L 275 160" stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowPurple)" />

            {/* Policy */}
            <rect
              x="180" y="230" width="120" height="55" rx="8" fill="#10b981"
              className={`${styles.pipelineStage} ${activeComponent === 'policy' ? styles.pipelineStageActive : ''}`}
              onClick={() => setActiveComponent(activeComponent === 'policy' ? null : 'policy')}
            />
            <text x="240" y="252" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" pointerEvents="none">{'Policy \u03c0'}</text>
            <text x="240" y="268" textAnchor="middle" fill="white" fontSize="9" pointerEvents="none">max E[Q(s,a)]</text>

            {/* Arrow: Loss updates Q */}
            <path d="M 340 190 Q 340 215 290 235" stroke="#00d9ff" strokeWidth="2" fill="none" markerEnd="url(#arrowCyan)" />
            <text x="330" y="218" fill="#888" fontSize="8">update</text>

            {/* Arrow: Policy to Q (provides mu) */}
            <path d="M 240 230 Q 270 180 280 75" stroke="#10b981" strokeWidth="2" fill="none" markerEnd="url(#arrowGreen)" strokeDasharray="4" />
            <text x="275" y="150" fill="#888" fontSize="8">&mu;</text>

            {/* No Environment box */}
            <rect x="380" y="230" width="100" height="55" rx="8" fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth="2" strokeDasharray="6" />
            <text x="430" y="252" textAnchor="middle" fill="#ef4444" fontSize="10" fontWeight="bold">Environment</text>
            <text x="430" y="268" textAnchor="middle" fill="#ef4444" fontSize="9">No interaction!</text>
            <line x1="380" y1="230" x2="480" y2="285" stroke="#ef4444" strokeWidth="2" opacity="0.5" />
            <line x1="480" y1="230" x2="380" y2="285" stroke="#ef4444" strokeWidth="2" opacity="0.5" />
          </svg>
        </div>

        <InfoPanel
          title={activeComponent ? componentInfo[activeComponent]?.title : undefined}
          visible={!!activeComponent}
        >
          {activeComponent && componentInfo[activeComponent] && (
            <p>{componentInfo[activeComponent].desc}</p>
          )}
        </InfoPanel>
      </section>




      {/* ==================== Paper Lineage ==================== */}
      <section>
        <h2>Paper Lineage</h2>
        <div className={styles.lineageGrid}>
          <div className={styles.lineageColumn}>
            <h4 style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className={styles.lineageList}>
              {[
                { title: 'SAC', desc: 'Haarnoja et al., 2018 \u2014 Soft Actor-Critic; CQL(H) is built directly on top of SAC\u2019s actor-critic framework' },
                { title: 'BCQ', desc: 'Fujimoto et al., 2019 \u2014 Batch-Constrained Q-learning; first to identify the distribution shift problem in offline RL and constrain the policy' },
                { title: 'BEAR', desc: 'Kumar et al., 2019 \u2014 Bootstrapping Error Accumulation Reduction; constrains policy via MMD to behavior policy support' },
                { title: 'BRAC', desc: 'Wu et al., 2019 \u2014 Behavior Regularized Actor Critic; unifies offline RL methods via policy regularization; CQL instead regularizes the Q-function' },
              ].map((item) => (
                <div key={item.title} className={`${styles.lineageItem} ${styles.lineageItemBuildsOn}`}>
                  <div className={styles.lineageItemTitle}>{item.title}</div>
                  <div className={styles.lineageItemDesc}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
          <div className={styles.lineageColumn}>
            <h4 style={{ color: '#10b981' }}>Built Upon By</h4>
            <div className={styles.lineageList}>
              {[
                { title: 'IQL', desc: 'Kostrikov et al., 2022 \u2014 Implicit Q-Learning; avoids querying OOD actions entirely via expectile regression, simplifying CQL\u2019s approach' },
                { title: 'Cal-QL', desc: 'Nakamoto et al., 2023 \u2014 Calibrated CQL; fixes over-conservatism by calibrating the lower bound to the behavior policy value' },
                { title: 'TD3+BC', desc: 'Fujimoto & Gu, 2021 \u2014 Minimalist offline RL: just TD3 + a BC regularization term; competitive with CQL at a fraction of the complexity' },
                { title: 'Decision Transformer', desc: 'Chen et al., 2021 \u2014 Reframes offline RL as sequence modeling; an alternative paradigm to CQL\u2019s value-based conservatism' },
              ].map((item) => (
                <div key={item.title} className={`${styles.lineageItem} ${styles.lineageItemBuiltUpon}`}>
                  <div className={styles.lineageItemTitle}>{item.title}</div>
                  <div className={styles.lineageItemDesc}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
