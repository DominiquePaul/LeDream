'use client';

import { useState } from 'react';
import { MathJax } from 'better-react-mathjax';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './bcq.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Extrapolation Error',
    desc: 'Standard off-policy algorithms (DQN, DDPG) catastrophically fail on fixed batch data because they estimate unrealistic Q-values for unseen actions',
    detail: 'When a policy selects an action not in the batch, Q-learning must estimate its value by extrapolation. With function approximation this estimate can be arbitrarily wrong \u2014 and combined with the max operator, errors are amplified into persistent overestimation. In online RL, the agent would try the action and correct the error; in batch RL, there is no correction mechanism, so value estimates diverge and the policy collapses.',
  },
  {
    title: 'Batch-Constrained Policy',
    desc: 'Only select actions that are similar to those in the batch \u2014 if the data never took an action, don\'t trust its Q-value',
    detail: "BCQ constrains the policy to actions with high likelihood under the batch distribution. This is formalized as batch-constrained: a policy \u03C0 \u2208 \u03A0B such that it only selects actions (s,a) where (s,a) \u2208 B. For deterministic MDPs, this eliminates extrapolation error entirely (Theorem 2) and guarantees the learned policy matches or outperforms the behavior policy from any state in the batch.",
  },
  {
    title: 'Generate, Perturb, Select',
    desc: 'Use a VAE to generate plausible actions from the batch, perturb them slightly, then pick the highest Q-value among candidates',
    detail: 'BCQ uses a conditional VAE G\u03C9(s) trained on the batch to model P(a|s). At inference, it samples n candidate actions, each is adjusted by a small perturbation model \u03BE\u03C6(s,a,\u03A6) bounded in [\u2212\u03A6,\u03A6], and the twin Q-networks score each candidate. The highest-valued perturbed action is selected. This creates a smooth interpolation between behavioral cloning (\u03A6=0, n=1) and full Q-learning (\u03A6\u2192\u221E, n\u2192\u221E).',
  },
];

const keyFeatures = [
  { title: 'Fully Offline', desc: 'First continuous-control deep RL algorithm that learns from arbitrary fixed batch data without any environment interaction' },
  { title: 'Unified Approach', desc: 'Works across expert imitation, suboptimal data, and noisy demonstrations with a single set of hyperparameters' },
  { title: 'Stable Q-Values', desc: 'Unlike DDPG/DQN which diverge offline, BCQ produces stable value estimates by constraining actions to the batch' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  batch: {
    title: 'Fixed Batch B',
    desc: "A static dataset of (s, a, r, s') transitions collected by one or more behavior policies. BCQ never collects new data \u2014 it learns entirely offline. The batch can come from expert demonstrations, suboptimal exploratory policies, or mixed sources. All four networks are trained only on this fixed data.",
  },
  vae: {
    title: 'Conditional VAE G\u03C9(s)',
    desc: "A state-conditioned Variational Auto-Encoder that models the action distribution in the batch. The encoder E\u03C91(s,a) maps to latent (\u03BC,\u03C3), the decoder D\u03C92(s,z) reconstructs the action. Trained with reconstruction loss + KL divergence. During inference, latent z is sampled from N(0,1) and clipped to [\u22120.5, 0.5] to stay near observed actions. Two hidden layers of size 750.",
  },
  perturb: {
    title: 'Perturbation Model \u03BE\u03C6(s, a, \u03A6)',
    desc: "A small residual network that adjusts VAE-generated actions within [\u2212\u03A6, \u03A6] (\u03A6 is typically small, e.g. 0.05). Trained via the deterministic policy gradient to maximize Q: \u03C6 \u2190 argmax \u03A3 Q\u03B81(s, a + \u03BE\u03C6(s,a,\u03A6)). Uses tanh output scaled by \u03A6. This lets the policy fine-tune actions beyond what the VAE samples, without straying far from the data.",
  },
  twinq: {
    title: 'Twin Q-Networks Q\u03B81, Q\u03B82',
    desc: "Two Q-networks trained with the same target to reduce overestimation (from Clipped Double Q-learning / TD3). The value target uses a weighted minimum: \u03BB\u00B7min(Q1,Q2) + (1\u2212\u03BB)\u00B7max(Q1,Q2) with \u03BB=0.75. This penalizes high-variance (uncertain) regions, pushing the policy toward states with reliable data. Both use standard architecture: (s,a) \u2192 400 \u2192 300 \u2192 1.",
  },
  targets: {
    title: 'Target Networks (Q\'1, Q\'2, \u03BE\u03C6\')',
    desc: "Slowly-updated copies of both Q-networks and the perturbation model. Updated via soft Polyak averaging: \u03B8' \u2190 \u03C4\u03B8 + (1\u2212\u03C4)\u03B8' with \u03C4=0.005. The target perturbation model \u03BE\u03C6' and target Q-networks are used together to compute the value target, ensuring stable learning.",
  },
};

/* ================================================================
   COMPONENT
   ================================================================ */

export default function BCQPage() {
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  return (
    <div className="method-page">
      <h1>Batch-Constrained Q-Learning (BCQ)</h1>
      <p className={styles.subtitle}>
        Off-Policy Deep Reinforcement Learning without Exploration &mdash; Fujimoto et al., 2019
      </p>
      <div className={styles.githubLink}>
        <a href="https://github.com/sfujim/BCQ" target="_blank" rel="noopener noreferrer">
          <span style={{ marginRight: 5 }}>&rarr;</span> github.com/sfujim/BCQ
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture ==================== */}
      <section>
        <h2>BCQ Architecture</h2>
        <div className="diagram-frame">
          <svg className={styles.archSvg} viewBox="0 0 500 440">
            <defs>
              <marker id="bcqA" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#00d9ff" />
              </marker>
              <marker id="bcqAG" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#10b981" />
              </marker>
              <marker id="bcqAB" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#3b82f6" />
              </marker>
              <marker id="bcqAO" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#f59e0b" />
              </marker>
              <marker id="bcqAP" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#a855f7" />
              </marker>
              <marker id="bcqAR" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#ef4444" />
              </marker>
            </defs>

            {/* === Fixed Batch === */}
            <rect
              x="180" y="8" width="140" height="40" rx="8"
              fill="rgba(245,158,11,0.15)" stroke="#f59e0b" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'batch' ? null : 'batch')}
            />
            <text x="250" y="26" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold" pointerEvents="none">Fixed Batch B</text>
            <text x="250" y="40" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">(s, a, r, s&#x2032;) &mdash; no new data</text>

            {/* Arrows from batch */}
            <path d="M 210 48 L 100 72" stroke="#f59e0b" strokeWidth="1.5" markerEnd="url(#bcqAO)" />
            <path d="M 290 48 L 400 72" stroke="#f59e0b" strokeWidth="1.5" markerEnd="url(#bcqAO)" />

            {/* === VAE === */}
            <rect
              x="15" y="74" width="170" height="62" rx="10"
              fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'vae' ? null : 'vae')}
            />
            <text x="100" y="94" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="bold" pointerEvents="none">VAE G&#x03C9;(s)</text>
            <text x="100" y="108" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Encoder E(s,a) &#x2192; (&#x03BC;,&#x03C3;)</text>
            <text x="100" y="121" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Decoder D(s,z) &#x2192; &#x00E2;</text>

            {/* === Twin Q === */}
            <rect
              x="315" y="74" width="170" height="62" rx="10"
              fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'twinq' ? null : 'twinq')}
            />
            <text x="400" y="94" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" pointerEvents="none">Twin Q-Networks</text>
            <text x="400" y="108" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Q&#x03B8;&#x2081;(s, a) &amp; Q&#x03B8;&#x2082;(s, a)</text>
            <text x="400" y="121" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">weighted min for target</text>

            {/* VAE generates candidates */}
            <path d="M 100 136 L 100 160" stroke="#a855f7" strokeWidth="1.5" markerEnd="url(#bcqAP)" />

            {/* Candidate actions box */}
            <rect x="15" y="162" width="170" height="35" rx="8" fill="rgba(168,85,247,0.08)" stroke="#a855f7" strokeWidth="1" strokeDasharray="4 3" />
            <text x="100" y="178" textAnchor="middle" fill="#a855f7" fontSize="8" pointerEvents="none">n candidate actions</text>
            <text x="100" y="190" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">{'{'}a&#x2081;, a&#x2082;, ..., a&#x2099;{'}'} ~ G&#x03C9;(s)</text>

            {/* Arrow to perturbation */}
            <path d="M 100 197 L 100 220" stroke="#a855f7" strokeWidth="1.5" markerEnd="url(#bcqAP)" />

            {/* === Perturbation Model === */}
            <rect
              x="15" y="222" width="170" height="55" rx="10"
              fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'perturb' ? null : 'perturb')}
            />
            <text x="100" y="242" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">Perturbation &#x03BE;&#x03C6;(s, a, &#x03A6;)</text>
            <text x="100" y="258" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">a&#x0303; = a + &#x03BE;&#x03C6;(s, a, &#x03A6;)</text>
            <text x="100" y="270" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">bounded in [&#x2212;&#x03A6;, &#x03A6;]</text>

            {/* Perturbed actions to Q-networks */}
            <path d="M 185 250 L 310 108" stroke="#10b981" strokeWidth="1.5" markerEnd="url(#bcqAG)" />
            <text x="255" y="170" textAnchor="middle" fill="#10b981" fontSize="7" transform="rotate(-30, 255, 170)">perturbed actions &#x00E3;&#x1D62;</text>

            {/* Q-networks select best */}
            <path d="M 400 136 L 400 162" stroke="#3b82f6" strokeWidth="1.5" markerEnd="url(#bcqAB)" />

            {/* Selection box */}
            <rect x="315" y="164" width="170" height="40" rx="8" fill="rgba(0,217,255,0.1)" stroke="#00d9ff" strokeWidth="1.5" />
            <text x="400" y="181" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold" pointerEvents="none">argmax Q(s, &#x00E3;&#x1D62;)</text>
            <text x="400" y="196" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">select highest-valued candidate</text>

            {/* Output action */}
            <path d="M 400 204 L 400 228" stroke="#00d9ff" strokeWidth="1.5" markerEnd="url(#bcqA)" />
            <rect x="340" y="230" width="120" height="30" rx="8" fill="rgba(0,217,255,0.06)" stroke="#00d9ff" strokeWidth="1" />
            <text x="400" y="249" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold" pointerEvents="none">Action &#x03C0;(s)</text>

            {/* DPG arrow: Q -> Perturbation */}
            <path d="M 315 120 Q 250 140 185 245" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="4 3" fill="none" markerEnd="url(#bcqAO)" />
            <text x="230" y="200" textAnchor="middle" fill="#f59e0b" fontSize="6.5" transform="rotate(-55, 230, 200)">&#x2207;&#x03C6; via DPG</text>

            {/* === Separator === */}
            <line x1="20" y1="280" x2="480" y2="280" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

            {/* === Target Networks === */}
            <rect
              x="100" y="295" width="300" height="45" rx="10"
              fill="rgba(59,130,246,0.06)" stroke="#3b82f6" strokeWidth="1.2" strokeDasharray="6 3"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'targets' ? null : 'targets')}
            />
            <text x="250" y="312" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">Target Networks (soft update &#x03C4; = 0.005)</text>
            <text x="250" y="328" textAnchor="middle" fill="#666" fontSize="7.5" pointerEvents="none">Q&#x2032;&#x03B8;&#x2081;, Q&#x2032;&#x03B8;&#x2082;, &#x03BE;&#x2032;&#x03C6; &mdash; used for stable TD target computation</text>

            {/* Soft update arrows */}
            <path d="M 370 136 Q 330 220 300 295" stroke="#3b82f6" strokeWidth="1" strokeDasharray="3 3" fill="none" markerEnd="url(#bcqAB)" />
            <path d="M 140 277 Q 170 285 190 295" stroke="#10b981" strokeWidth="1" strokeDasharray="3 3" fill="none" markerEnd="url(#bcqAG)" />

            {/* Value target formula */}
            <rect x="50" y="360" width="400" height="34" rx="8" fill="rgba(239,68,68,0.06)" stroke="rgba(239,68,68,0.2)" strokeWidth="1" />
            <text x="250" y="374" textAnchor="middle" fill="#ef4444" fontSize="8" fontWeight="bold">Value target</text>
            <text x="250" y="387" textAnchor="middle" fill="#888" fontSize="7.5">y = r + &#x03B3; max&#x2090;&#x1D62; [ &#x03BB; min(Q&#x2032;&#x2081;, Q&#x2032;&#x2082;) + (1&#x2212;&#x03BB;) max(Q&#x2032;&#x2081;, Q&#x2032;&#x2082;) ]</text>
            <path d="M 250 340 L 250 358" stroke="#ef4444" strokeWidth="1" strokeDasharray="3 2" fill="none" markerEnd="url(#bcqAR)" />

            {/* Click hint */}
            <text x="250" y="425" textAnchor="middle" fill="#555" fontSize="8" fontStyle="italic">Click any component for details</text>
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

      {/* ==================== Training Algorithm ==================== */}
      <section>
        <h2>Training Loop</h2>

        <div className="formula-block">
          <h4>BCQ Policy (action selection)</h4>
          <MathJax>{String.raw`\[ \pi(s) = \underset{a_i + \xi_\phi(s, a_i, \Phi)}{\arg\max}\; Q_\theta\!\left(s,\; a_i + \xi_\phi(s, a_i, \Phi)\right), \quad \{a_i \sim G_\omega(s)\}_{i=1}^{n} \]`}</MathJax>
        </div>

        <div className="formula-block">
          <h4>Weighted Clipped Double Q target</h4>
          <MathJax>{String.raw`\[ y = r + \gamma \max_{\tilde{a}_i} \Big[ \lambda \min_{j=1,2} Q_{\theta'_j}(s', \tilde{a}_i) + (1\!-\!\lambda) \max_{j=1,2} Q_{\theta'_j}(s', \tilde{a}_i) \Big] \]`}</MathJax>
          <p className={styles.formulaNote}>
            &#x03BB; = 0.75 biases toward the minimum, penalizing uncertain OOD states
          </p>
        </div>

        <ol className="algo-steps">
          <li>
            <strong>Sample</strong> minibatch of N transitions (s, a, r, s&prime;) from fixed batch B
          </li>
          <li>
            <strong>Train VAE</strong> G&#x03C9;: reconstruct actions via encoder-decoder with KL regularization
          </li>
          <li>
            <strong>Generate</strong> n actions per next-state s&prime; from VAE: {'{'}a<sub>i</sub> ~ G&#x03C9;(s&prime;){'}'}
          </li>
          <li>
            <strong>Perturb</strong> each action: &#x00E3;<sub>i</sub> = a<sub>i</sub> + &#x03BE;<sub>&#x03C6;&prime;</sub>(s&prime;, a<sub>i</sub>, &#x03A6;) using target perturbation network
          </li>
          <li>
            <strong>Compute target</strong> y using weighted min over target Q-networks (&#x03BB; = 0.75)
          </li>
          <li>
            <strong>Update Q-networks</strong> &#x03B8;<sub>1</sub>, &#x03B8;<sub>2</sub> by minimizing (y &minus; Q(s,a))&sup2;
          </li>
          <li>
            <strong>Update perturbation</strong> &#x03C6; via DPG: max Q&#x03B8;&#x2081;(s, a + &#x03BE;&#x03C6;(s,a,&#x03A6;))
          </li>
          <li>
            <strong>Soft update</strong> all target networks: &#x03B8;&prime; &#x2190; &#x03C4;&#x03B8; + (1&minus;&#x03C4;)&#x03B8;&prime; &nbsp;(&#x03C4; = 0.005)
          </li>
        </ol>

        <InfoPanel title="The &#x03A6; Knob: Imitation &#x2194; RL">
          <p>
            <strong style={{ color: '#10b981' }}>&#x03A6; = 0, n = 1</strong> &mdash; the policy is pure behavioral cloning (just reproduce what the VAE generates).
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#a855f7' }}>&#x03A6; small, n = 10</strong> &mdash; the default BCQ regime: stay close to batch actions but optimize Q within that neighborhood.
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#ef4444' }}>&#x03A6; &#x2192; &#x221E;, n &#x2192; &#x221E;</strong> &mdash; approaches unconstrained Q-learning (greedy over entire action space). This is what DDPG does, and it fails offline.
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
                { title: 'DQN / DDPG', desc: 'Mnih et al., 2013 / Lillicrap et al., 2015 \u2014 BCQ diagnoses why these methods catastrophically fail when applied to fixed batch data (extrapolation error)' },
                { title: 'VAE', desc: 'Kingma & Welling, 2014 \u2014 The conditional VAE generative model is central to BCQ\u2019s action proposal mechanism' },
                { title: 'TD3', desc: 'Fujimoto et al., 2018 \u2014 Twin Q-networks and the clipped double-Q trick are directly adopted; the same first author' },
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
                { title: 'BEAR', desc: 'Kumar et al., 2019 \u2014 Replaces distribution matching with a weaker support constraint via MMD, improving on BCQ\u2019s conservatism with random data' },
                { title: 'CQL', desc: 'Kumar et al., 2020 \u2014 Learns a conservative Q-function that lower-bounds the true value, avoiding BCQ\u2019s explicit generative model' },
                { title: 'TD3+BC', desc: 'Fujimoto & Gu, 2021 \u2014 A minimalist offline RL method: just TD3 plus a behavioral cloning term, avoiding BCQ\u2019s VAE complexity' },
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
