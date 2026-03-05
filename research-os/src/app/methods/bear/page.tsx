'use client';

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { MathJax } from 'better-react-mathjax';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { SliderControl } from '@/components/SliderControl/SliderControl';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import styles from './bear.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Bootstrapping Error',
    desc: 'Q-learning on static data accumulates errors by bootstrapping from out-of-distribution (OOD) actions whose Q-values are never corrected',
    detail: 'In offline RL, the Bellman backup computes targets by maximizing over all actions — including those absent from the dataset. Function approximation produces arbitrary Q-values for these OOD actions, and the max operator selects the most overestimated. Unlike online RL, the agent cannot try the action and correct the estimate. Errors propagate through successive backups: high-error regions leak into nearby in-distribution states, eventually corrupting the entire Q-function. BEAR shows this is the primary failure mode of naïve off-policy methods on static data.',
  },
  {
    title: 'Support Constraint (not Distribution)',
    desc: 'Constrain the learned policy\'s support to match the behavior policy\'s — but allow different probabilities within that support',
    detail: 'BCQ constrains the learned policy distribution to be close to the behavior policy. BEAR makes a weaker but better constraint: only require the policy to have support where the behavior policy does (where β(a|s) ≥ ε). The policy can assign any probabilities to those actions. This avoids the conservatism trap: if the behavior policy is near-uniform, distribution-matching forces the learned policy to also be near-uniform, preventing improvement. Support matching keeps actions in-distribution for safe Q-estimation while allowing the policy to concentrate on high-value actions.',
  },
  {
    title: 'MMD for Support Matching',
    desc: 'Maximum Mean Discrepancy measures support overlap without requiring density estimation — key for continuous action spaces',
    detail: 'BEAR uses the sampled Maximum Mean Discrepancy (MMD) between the actor π and a model of the data distribution πdata. MMD is a kernel-based distance between distributions that can be estimated from samples alone — no density evaluation needed. Crucially, with limited samples, the MMD between a distribution and a uniform distribution over its support is similar to the MMD between the distribution and itself. This means minimizing MMD approximately constrains to the same support rather than matching the exact shape. Laplacian and Gaussian kernels both work; the threshold ε ≈ 0.05 is set automatically via dual gradient descent on a Lagrange multiplier α.',
  },
];

const keyFeatures = [
  { title: 'Robust to Data Quality', desc: 'Unlike BCQ which struggles with random data, BEAR learns consistently well from expert, medium-quality, and random datasets with the same hyperparameters' },
  { title: 'Theoretical Guarantees', desc: 'Formal bounds on suboptimality via the concentrability coefficient C(Π) and suboptimality constant α(Π), showing the tradeoff between staying close to data and policy quality' },
  { title: 'Adaptive Constraint', desc: 'The Lagrange multiplier α automatically tunes the constraint strength — no manual threshold selection needed. Dual gradient descent balances exploration of the action space with data support' },
];

/* ================================================================
   MMD HELPERS
   ================================================================ */

const DATA_SAMPLES = [-1.4, -0.9, -0.5, -0.1, 0.2, 0.5, 0.8, 1.3];
const POLICY_BASE_SAMPLES = [-1.2, -0.7, -0.3, 0.0, 0.3, 0.6, 1.0, 1.4];
const KERNEL_BW = 0.7;

function gKernel(a: number, b: number): number {
  return Math.exp(-((a - b) ** 2) / (2 * KERNEL_BW * KERNEL_BW));
}

function gaussianPdf(x: number, mu: number): number {
  return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-((x - mu) ** 2) / 2);
}

function computeMMD2(xs: number[], ys: number[]) {
  const n = xs.length, m = ys.length;
  let xx = 0, cross = 0, yy = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) xx += gKernel(xs[i], xs[j]);
  xx /= n * n;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) cross += gKernel(xs[i], ys[j]);
  cross = -(2 / (n * m)) * cross;
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) yy += gKernel(ys[i], ys[j]);
  yy /= m * m;
  return { mmd2: xx + cross + yy, xx, cross, yy };
}

/* ================================================================
   COMPONENT
   ================================================================ */

export default function BEARPage() {
  const [policyOffset, setPolicyOffset] = useState(1.5);

  /* --- MMD canvas --- */
  const mmdCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mmdSize = useCanvasResize(mmdCanvasRef);

  const mmdResult = useMemo(() => {
    const ys = POLICY_BASE_SAMPLES.map(x => x + policyOffset);
    return computeMMD2(DATA_SAMPLES, ys);
  }, [policyOffset]);

  const drawMMD = useCallback(() => {
    const canvas = mmdCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = mmdSize.width, h = mmdSize.height;
    if (w === 0 || h === 0) return;
    ctx.clearRect(0, 0, w, h);

    const pad = { left: 50, right: 20, top: 20, bottom: 45 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;
    const xMin = -4, xMax = 6, yMax = 0.45;
    const toX = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * pw;
    const toY = (v: number) => pad.top + ph * (1 - v / yMax);
    const baseline = toY(0);

    /* axes */
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, baseline);
    ctx.lineTo(w - pad.right, baseline);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Action space', w / 2, h - 5);

    ctx.save();
    ctx.translate(14, (pad.top + baseline) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Density', 0, 0);
    ctx.restore();

    /* data distribution: fill then stroke */
    ctx.beginPath();
    ctx.moveTo(toX(xMin), baseline);
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      ctx.lineTo(toX(x), toY(gaussianPdf(x, 0)));
    }
    ctx.lineTo(toX(xMax), baseline);
    ctx.closePath();
    ctx.fillStyle = 'rgba(168,85,247,0.1)';
    ctx.fill();

    ctx.beginPath();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      const cx = toX(x), cy = toY(gaussianPdf(x, 0));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    /* policy distribution: fill then stroke */
    ctx.beginPath();
    ctx.moveTo(toX(xMin), baseline);
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      ctx.lineTo(toX(x), toY(gaussianPdf(x, policyOffset)));
    }
    ctx.lineTo(toX(xMax), baseline);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,217,255,0.1)';
    ctx.fill();

    ctx.beginPath();
    ctx.strokeStyle = '#00d9ff';
    ctx.lineWidth = 2;
    for (let i = 0; i <= pw; i++) {
      const x = xMin + (i / pw) * (xMax - xMin);
      const cx = toX(x), cy = toY(gaussianPdf(x, policyOffset));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    /* sample dots — data (purple) */
    DATA_SAMPLES.forEach(x => {
      ctx.beginPath();
      ctx.arc(toX(x), baseline + 10, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#a855f7';
      ctx.fill();
    });

    /* sample dots — policy (cyan) */
    const policySamples = POLICY_BASE_SAMPLES.map(x => x + policyOffset);
    policySamples.forEach(x => {
      ctx.beginPath();
      ctx.arc(toX(x), baseline + 22, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#00d9ff';
      ctx.fill();
    });

    /* sample row labels */
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#a855f7';
    ctx.fillText('\u03C0data', pad.left - 5, baseline + 14);
    ctx.fillStyle = '#00d9ff';
    ctx.fillText('\u03C0\u03C6', pad.left - 5, baseline + 26);

    /* legend */
    const lx = w - pad.right - 85;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#a855f7';
    ctx.fillRect(lx, pad.top + 4, 8, 8);
    ctx.fillText('\u03C0data (data)', lx + 12, pad.top + 12);
    ctx.fillStyle = '#00d9ff';
    ctx.fillRect(lx, pad.top + 18, 8, 8);
    ctx.fillText('\u03C0\u03C6 (policy)', lx + 12, pad.top + 26);
  }, [mmdSize, policyOffset]);

  useEffect(() => { drawMMD(); }, [drawMMD]);

  return (
    <div className="method-page">
      <h1>BEAR: Bootstrapping Error Accumulation Reduction</h1>
      <p className={styles.subtitle}>
        Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction &mdash; Kumar, Fu, Tucker &amp; Levine, NeurIPS 2019
      </p>
      <div className={styles.githubLink}>
        <a href="https://arxiv.org/abs/1906.00949" target="_blank" rel="noopener noreferrer">
          <span style={{ marginRight: 5 }}>&rarr;</span> arxiv.org/abs/1906.00949
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture ==================== */}
      <section>
        <h2>BEAR Architecture</h2>
        <p className={styles.archSubtitle}>
          Actor-critic (SAC backbone) with twin Q-networks and an MMD support constraint.
          Learns from a static offline dataset &mdash; no environment interaction.
        </p>

        {/* --- Network cards --- */}
        <div className={styles.networkGrid}>
          <div className={styles.networkCard} style={{ borderColor: '#00d9ff' }}>
            <div className={styles.networkCardTitle} style={{ color: '#00d9ff' }}>Actor &pi;&#x03C6;</div>
            <div className={styles.networkCardType}>Policy network</div>
            <div className={styles.networkCardBody}>
              s &rarr; &pi;&#x03C6;(a|s) &nbsp; (tanh Gaussian)<br />
              Stochastic policy trained to maximize Q subject to MMD support constraint. Free to concentrate probability on high-value actions within the data support.
            </div>
          </div>
          <div className={styles.networkCard} style={{ borderColor: '#3b82f6' }}>
            <div className={styles.networkCardTitle} style={{ color: '#3b82f6' }}>Q&#x03B8;&#x2081;, Q&#x03B8;&#x2082;</div>
            <div className={styles.networkCardType}>Twin critics (K = 2)</div>
            <div className={styles.networkCardBody}>
              (s, a) &rarr; &#x211D; &nbsp; (two separate networks)<br />
              Independently initialized, trained on the same targets. Conservative estimate uses &lambda;&middot;min + (1&minus;&lambda;)&middot;max across the pair to penalize OOD actions.
            </div>
          </div>
          <div className={styles.networkCard} style={{ borderColor: '#a855f7' }}>
            <div className={styles.networkCardTitle} style={{ color: '#a855f7' }}>&pi;data</div>
            <div className={styles.networkCardType}>Behavior model</div>
            <div className={styles.networkCardBody}>
              s &rarr; tanh N(&mu;(s), &sigma;(s))<br />
              Pre-trained via MLE on dataset actions. Only used to provide samples for the MMD constraint &mdash; not part of the loss directly.
            </div>
          </div>
        </div>
        <p className={styles.targetNote}>
          + target copies &pi;&prime;&#x03C6;, Q&prime;&#x2081;, Q&prime;&#x2082; updated via soft Polyak averaging (&tau; = 0.005)
        </p>

        {/* --- Loss computation --- */}
        <h3 style={{ marginTop: 24, marginBottom: 12 }}>Loss Computation</h3>
        <div className={styles.lossGrid}>
          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>1</span>
              <span className={styles.lossCardTitle} style={{ color: '#3b82f6' }}>Critic Loss (each Q&#x03B8;&#x1D62;)</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ \hat{a}' \sim \pi'_\phi(\cdot|s'), \qquad y = r + \gamma \bigl[ \lambda \min_j Q'_j(s', \hat{a}') + (1{-}\lambda) \max_j Q'_j(s', \hat{a}') \bigr] \]`}</MathJax>
              <MathJax>{String.raw`\[ \mathcal{L}_{\text{critic}} = \bigl( Q_{\theta_i}(s, a) - y \bigr)^2 \]`}</MathJax>
              <p className={styles.lossNote}>
                Target actions from target actor &pi;&prime;&#x03C6;. The &lambda;-weighted min/max penalizes OOD actions where the two critics disagree.
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>2</span>
              <span className={styles.lossCardTitle} style={{ color: '#00d9ff' }}>Actor Loss (constrained policy improvement)</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ \mathcal{L}_{\text{actor}} = -\,\mathbb{E}_{\hat{a} \sim \pi_\phi}\!\bigl[\min_j Q_j(s, \hat{a})\bigr] \;+\; \alpha \cdot \text{MMD}\bigl(\pi_{\text{data}}(\cdot|s),\; \pi_\phi(\cdot|s)\bigr) \]`}</MathJax>
              <p className={styles.lossNote}>
                Maximize conservative Q-value while staying within data support. The Lagrange multiplier &alpha; auto-balances the two objectives.
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>3</span>
              <span className={styles.lossCardTitle} style={{ color: '#10b981' }}>Dual Variable Update</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ \log \alpha \;\leftarrow\; \log \alpha \;+\; \eta_\alpha \cdot \bigl(\text{MMD}(\pi_{\text{data}}, \pi_\phi) - \varepsilon\bigr), \qquad \log \alpha \in [-5,\, 10] \]`}</MathJax>
              <p className={styles.lossNote}>
                When MMD &gt; &epsilon;, &alpha; increases to tighten the constraint. When MMD &lt; &epsilon;, &alpha; decreases to allow more optimization freedom. No manual threshold tuning needed.
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>4</span>
              <span className={styles.lossCardTitle} style={{ color: '#888' }}>Soft Target Update</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ \theta' \leftarrow \tau\,\theta + (1 - \tau)\,\theta', \qquad \phi' \leftarrow \tau\,\phi + (1 - \tau)\,\phi' \]`}</MathJax>
              <p className={styles.lossNote}>
                All target networks (Q&prime;&#x2081;, Q&prime;&#x2082;, &pi;&prime;&#x03C6;) updated with Polyak averaging, &tau; = 0.005.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ==================== MMD Visualization ==================== */}
      <section>
        <h2>MMD: Measuring Distribution Distance</h2>
        <p className={styles.mmdIntro}>
          Maximum Mean Discrepancy compares two distributions using only samples &mdash;
          no density estimation needed. Drag the slider to see how MMD&sup2; changes as
          the policy moves away from the data support.
        </p>
        <div className="diagram-frame">
          <canvas ref={mmdCanvasRef} className={styles.mmdCanvas} />
        </div>
        <div className={styles.mmdControls}>
          <SliderControl
            label="Policy offset"
            value={policyOffset}
            min={0}
            max={3}
            step={0.1}
            onChange={setPolicyOffset}
            displayValue={policyOffset.toFixed(1)}
            color="#00d9ff"
          />
        </div>
        <div className={styles.mmdTerms}>
          <div className={styles.mmdTerm}>
            <span className={styles.mmdTermLabel}>k(x,x)</span>
            <span className={styles.mmdTermValue} style={{ color: '#a855f7' }}>
              {mmdResult.xx.toFixed(3)}
            </span>
          </div>
          <span className={styles.mmdTermOp}>+</span>
          <div className={styles.mmdTerm}>
            <span className={styles.mmdTermLabel}>&minus;2k(x,y)</span>
            <span className={styles.mmdTermValue} style={{ color: '#f59e0b' }}>
              {mmdResult.cross.toFixed(3)}
            </span>
          </div>
          <span className={styles.mmdTermOp}>+</span>
          <div className={styles.mmdTerm}>
            <span className={styles.mmdTermLabel}>k(y,y)</span>
            <span className={styles.mmdTermValue} style={{ color: '#00d9ff' }}>
              {mmdResult.yy.toFixed(3)}
            </span>
          </div>
          <span className={styles.mmdTermOp}>=</span>
          <div className={`${styles.mmdTerm} ${styles.mmdTermTotal}`}>
            <span className={styles.mmdTermLabel}>MMD&sup2;</span>
            <span
              className={styles.mmdTermValue}
              style={{ color: mmdResult.mmd2 < 0.05 ? '#10b981' : mmdResult.mmd2 < 0.2 ? '#f59e0b' : '#ef4444' }}
            >
              {mmdResult.mmd2.toFixed(3)}
            </span>
          </div>
        </div>
        <p className={styles.mmdNote}>
          When distributions overlap, the cross-term k(x,y) is large and pulls MMD&sup2; toward 0.
          As they separate, k(x,y) shrinks and MMD&sup2; grows &mdash; signaling the policy has left the data support.
        </p>
      </section>

      {/* ==================== Training Algorithm ==================== */}
      <section>
        <h2>Training Loop</h2>

        <div className="formula-block">
          <h4>BEAR Policy Improvement (constrained actor update)</h4>
          <MathJax>{String.raw`\[ \pi_\phi := \max_{\pi \in \Delta_{|\mathcal{S}|}} \mathbb{E}_{s \sim \mathcal{D}} \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \min_{j=1,...,K} \hat{Q}_j(s,a) \right] \quad \text{s.t.} \; \mathbb{E}_{s \sim \mathcal{D}}\!\left[\text{MMD}(\mathcal{D}(s),\, \pi(\cdot|s))\right] \leq \varepsilon \]`}</MathJax>
        </div>

        <div className="formula-block">
          <h4>MMD (kernel two-sample test)</h4>
          <MathJax>{String.raw`\[ \text{MMD}^2 = \frac{1}{n^2}\sum_{i,i'} k(x_i, x_{i'}) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j) + \frac{1}{m^2}\sum_{j,j'} k(y_j, y_{j'}) \]`}</MathJax>
          <p className={styles.formulaNote}>
            k(&#x00B7;,&#x00B7;) is a Laplacian or Gaussian kernel &mdash; no density estimation needed
          </p>
        </div>

        <ol className="algo-steps">
          <li>
            <strong>Sample</strong> minibatch of N transitions (s, a, r, s&prime;) from static dataset D
          </li>
          <li>
            <strong>Q-update:</strong>{' '}
            sample p actions <span style={{whiteSpace: 'nowrap'}}>&#x00E2;&prime; ~ &#x03C0;<sub>&#x03C6;&prime;</sub>(&#x00B7;|s&prime;),</span>{' '}
            form target <span style={{whiteSpace: 'nowrap'}}>y = r + &#x03B3;[&#x03BB;&#xB7;min<sub>j</sub> Q&prime;<sub>j</sub> + (1&minus;&#x03BB;)&#xB7;max<sub>j</sub> Q&prime;<sub>j</sub>],</span>{' '}
            minimize <span style={{whiteSpace: 'nowrap'}}>(Q<sub>&#x03B8;i</sub>(s,a) &minus; y)&sup2;</span>
          </li>
          <li>
            <strong>Sample actor actions</strong> {'{'}&#x00E2;<sub>i</sub> ~ &#x03C0;<sub>&#x03C6;</sub>(&#x00B7;|s){'}'}<sub>m</sub> and <strong>data actions</strong> {'{'}a<sub>j</sub> ~ D(s){'}'}<sub>n</sub>
          </li>
          <li>
            <strong>Policy update:</strong> maximize min<sub>j</sub> Q<sub>j</sub>(s, &#x00E2;) via gradient ascent, penalized by &#x03B1; &middot; MMD(D(s), &#x03C0;<sub>&#x03C6;</sub>(&#x00B7;|s))
          </li>
          <li>
            <strong>Dual update:</strong> update Lagrange multiplier &#x03B1; &mdash; increase if MMD &gt; &#x03B5;, decrease otherwise
          </li>
          <li>
            <strong>Soft update</strong> all target networks: &#x03B8;&prime; &#x2190; &#x03C4;&#x03B8; + (1&minus;&#x03C4;)&#x03B8;&prime;
          </li>
        </ol>

        <h4 style={{ marginTop: 28, marginBottom: 10 }}>Pseudocode</h4>
        <div className={styles.pseudocode}>
          <span className={styles.comment}>// Initialize</span><br />
          <span className={styles.keyword}>Input:</span> Static dataset <span className={styles.param}>D</span>, kernel bandwidth <span className={styles.param}>&sigma;</span>, threshold <span className={styles.param}>&epsilon;</span><br />
          <span className={styles.keyword}>Init:</span> Q-ensemble <span className={styles.param}>{'{Q_\u03b8\u2081, ..., Q_\u03b8\u2096}'}</span>, actor <span className={styles.param}>&pi;_&phi;</span>, data model <span className={styles.param}>&pi;data</span>, Lagrange mult. <span className={styles.param}>&alpha;</span><br />
          <br />
          <span className={styles.comment}>// Pre-train data distribution model via MLE on D</span><br />
          <span className={styles.param}>&pi;data</span> &larr; <span className={styles.func}>fit</span> tanh <span className={styles.func}>N(&mu;(s), &sigma;(s))</span> on <span className={styles.param}>D</span><br />
          <br />
          <span className={styles.keyword}>for</span> each training step <span className={styles.keyword}>do</span><br />
          &nbsp;&nbsp;<span className={styles.comment}>// Sample minibatch</span><br />
          &nbsp;&nbsp;B &larr; <span className={styles.func}>sample</span>(<span className={styles.param}>D</span>, N) &nbsp;&nbsp;<span className={styles.comment}>// N transitions (s, a, r, s&prime;)</span><br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Q-update: Bellman backup with conservative target</span><br />
          &nbsp;&nbsp;<span className={styles.keyword}>for</span> i = 1 to <span className={styles.param}>K</span> <span className={styles.keyword}>do</span><br />
          &nbsp;&nbsp;&nbsp;&nbsp;&acirc;&prime; &sim; <span className={styles.func}>&pi;_&phi;&prime;(&middot;|s&prime;)</span> &nbsp;&nbsp;<span className={styles.comment}>// sample p target-policy actions</span><br />
          &nbsp;&nbsp;&nbsp;&nbsp;y &larr; r + <span className={styles.param}>&gamma;</span> [<span className={styles.param}>&lambda;</span>&middot;<span className={styles.func}>min</span><sub>j</sub> Q&prime;<sub>j</sub>(s&prime;, &acirc;&prime;) + (1&minus;<span className={styles.param}>&lambda;</span>)&middot;<span className={styles.func}>max</span><sub>j</sub> Q&prime;<sub>j</sub>(s&prime;, &acirc;&prime;)]<br />
          &nbsp;&nbsp;&nbsp;&nbsp;&theta;&#x1D62; &larr; &theta;&#x1D62; &minus; &eta; &middot; &nabla;(<span className={styles.func}>Q_&theta;&#x1D62;</span>(s,a) &minus; y)&sup2;<br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Sample for <span className={styles.highlight}>MMD</span> computation</span><br />
          &nbsp;&nbsp;{'{'}&acirc;&#x2081;, ..., &acirc;&#x2098;{'}'} &sim; <span className={styles.func}>&pi;_&phi;(&middot;|s)</span> &nbsp;&nbsp;&nbsp;<span className={styles.comment}>// m actor samples</span><br />
          &nbsp;&nbsp;{'{'}a&#x2081;, ..., a&#x2099;{'}'} &sim; <span className={styles.func}>&pi;data(&middot;|s)</span> &nbsp;&nbsp;<span className={styles.comment}>// n data-model samples</span><br />
          &nbsp;&nbsp;mmd &larr; <span className={styles.func}>MMD&sup2;</span>(&acirc;, a; kernel=<span className={styles.param}>k_&sigma;</span>)<br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Policy update: <span className={styles.highlight}>constrained improvement</span></span><br />
          &nbsp;&nbsp;&phi; &larr; &phi; + &eta; &middot; &nabla;[<span className={styles.func}>min</span><sub>j</sub> <span className={styles.func}>Q</span><sub>j</sub>(s, &acirc;) &minus; <span className={styles.param}>&alpha;</span> &middot; mmd]<br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Dual update: auto-tune constraint strength</span><br />
          &nbsp;&nbsp;log <span className={styles.param}>&alpha;</span> &larr; log <span className={styles.param}>&alpha;</span> + &eta;<sub>&alpha;</sub> &middot; (mmd &minus; <span className={styles.param}>&epsilon;</span>) &nbsp;&nbsp;<span className={styles.comment}>// clip log &alpha; &isin; [&minus;5, 10]</span><br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Soft update all target networks</span><br />
          &nbsp;&nbsp;&theta;&prime;&#x1D62; &larr; <span className={styles.param}>&tau;</span>&theta;&#x1D62; + (1&minus;<span className={styles.param}>&tau;</span>)&theta;&prime;&#x1D62; &nbsp;&nbsp;<span className={styles.keyword}>for all</span> i<br />
          &nbsp;&nbsp;&phi;&prime; &larr; <span className={styles.param}>&tau;</span>&phi; + (1&minus;<span className={styles.param}>&tau;</span>)&phi;&prime;
        </div>

        <InfoPanel title="Support vs Distribution Constraint">
          <p>
            <strong style={{ color: '#10b981' }}>BEAR (support constraint)</strong> &mdash; only zero out actions where &#x03B2;(a|s) &lt; &#x03B5;. Within the support, the policy is free to assign any probabilities. This lets BEAR concentrate probability on high-Q actions even when the behavior policy spread probability uniformly.
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#a855f7' }}>BCQ (distribution constraint)</strong> &mdash; forces the learned policy to stay close to the behavior distribution. If &#x03B2; is near-uniform (e.g., random exploration), BCQ is forced to be near-uniform too, which can prevent learning a good policy.
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#ef4444' }}>Na&iuml;ve Q-learning (no constraint)</strong> &mdash; selects actions anywhere in the action space. OOD actions have unreliable Q-values, leading to bootstrapping error accumulation and divergence.
          </p>
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
                { title: 'BCQ', desc: 'Fujimoto et al., 2019 \u2014 First to identify extrapolation error in offline RL; BEAR generalizes its distribution constraint to a weaker support constraint' },
                { title: 'SAC', desc: 'Haarnoja et al., 2018 \u2014 BEAR uses SAC\u2019s stochastic actor-critic framework as its backbone algorithm' },
                { title: 'MMD (Kernel Two-Sample Test)', desc: 'Gretton et al., 2012 \u2014 The Maximum Mean Discrepancy provides the sample-based support matching mechanism without requiring density estimation' },
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
                { title: 'CQL', desc: 'Kumar et al., 2020 \u2014 Same lab (Levine); replaces explicit support constraints with conservative Q-value regularization, a simpler and more scalable approach' },
                { title: 'BRAC', desc: 'Wu et al., 2019 \u2014 Unifies offline RL methods including BEAR under a common policy-regularization framework with KL and MMD variants' },
                { title: 'IQL', desc: 'Kostrikov et al., 2022 \u2014 Avoids querying OOD actions entirely via expectile regression, sidestepping the constraint design problem BEAR addresses' },
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
