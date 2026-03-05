'use client';

import { useState } from 'react';
import { MathJax } from 'better-react-mathjax';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './dqn.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'CNN as Q-Function Approximator',
    desc: 'A convolutional neural network maps raw 84×84 pixel frames directly to Q-values for every discrete action in one forward pass',
    detail: "Prior RL methods used hand-crafted features or required a separate forward pass per action. DQN's architecture outputs all Q(s,a) values simultaneously — one unit per valid action — enabling efficient action selection via argmax. The network takes 4 stacked grayscale frames (84×84×4) as input, allowing it to perceive motion and velocity from differences between frames.",
  },
  {
    title: 'Experience Replay',
    desc: "Store transitions (s, a, r, s') in a large buffer and train on random minibatches, breaking temporal correlations between samples",
    detail: 'In online RL, consecutive training samples are highly correlated (similar states in sequence), causing unstable gradient updates. Experience replay stores the last 1M transitions and samples uniformly at random, decorrelating the training data. This stabilizes learning, improves data efficiency (each transition is reused many times), and enables off-policy learning since the buffer contains transitions from older policies.',
  },
  {
    title: 'Off-Policy Q-Learning',
    desc: "Learn the optimal Q* by bootstrapping on max Q-values, using data collected by a separate ε-greedy exploration policy",
    detail: "Q-learning estimates Q* regardless of how data was collected — it always bootstraps on the greedy action max_a' Q(s',a'). This off-policy property is essential for experience replay (training on transitions from past policies). The behavior policy is ε-greedy: with probability ε it takes a random action, otherwise it exploits the current Q-values. Epsilon anneals from 1.0 to 0.1 over the first million frames.",
  },
];

const keyFeatures = [
  { title: 'Raw Pixels In', desc: 'Learns directly from 84×84 pixel frames with no hand-crafted features' },
  { title: 'Model-Free', desc: "No dynamics model — learns Q-values purely from (s, a, r, s') experience tuples" },
  { title: 'One Architecture', desc: 'Same CNN, same hyperparameters across all 7 Atari games — no per-game tuning' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  input: {
    title: 'Preprocessed Input (84×84×4)',
    desc: "Raw 210×160 RGB Atari frames are converted to grayscale, downsampled to 84×84, and 4 consecutive frames are stacked into a single tensor. Stacking lets the network infer velocity and direction of objects — a single frame is ambiguous (is the ball going left or right?).",
  },
  conv1: {
    title: 'Conv Layer 1: 16 filters, 8×8, stride 4',
    desc: 'Applies 16 filters of size 8×8 with stride 4 followed by ReLU. This aggressively downsamples the spatial dimensions (84→20) while detecting low-level visual features like edges and object boundaries.',
  },
  conv2: {
    title: 'Conv Layer 2: 32 filters, 4×4, stride 2',
    desc: 'Applies 32 filters of size 4×4 with stride 2 followed by ReLU (20→9). Combines low-level features into higher-level representations — detecting game objects, their positions, and spatial relationships.',
  },
  fc: {
    title: 'Fully Connected: 256 ReLU Units',
    desc: 'Flattens the convolutional features (32×9×9 = 2592 values) into 256 ReLU units. This layer integrates spatial information from across the entire visual field into a compact state representation for value estimation.',
  },
  qout: {
    title: 'Q-Value Output Layer (K units)',
    desc: 'A linear output layer with one unit per valid action (4–18 depending on the Atari game). Each output estimates Q(s, aᵢ) — the expected discounted return of taking action aᵢ from state s. The agent selects argmax.',
  },
  replay: {
    title: 'Replay Buffer (N = 1,000,000)',
    desc: "A circular buffer storing the last 1M transitions (s, a, r, s'). At each training step, a random minibatch of 32 transitions is sampled uniformly. This breaks temporal correlations, prevents catastrophic forgetting, and allows each experience to be reused across many gradient updates.",
  },
};

/* ================================================================
   COMPONENT
   ================================================================ */

export default function DQNPage() {
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  return (
    <div className="method-page">
      <h1>Deep Q-Network (DQN)</h1>
      <p className={styles.subtitle}>
        Playing Atari with Deep Reinforcement Learning &mdash; Mnih et al., 2013
      </p>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture Diagram ==================== */}
      <section>
        <h2>DQN Architecture</h2>
        <div className="diagram-frame">
          <svg className={styles.archSvg} viewBox="0 0 520 370">
            <defs>
              <marker id="dqnArr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#00d9ff" />
              </marker>
              <marker id="dqnArrAmber" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#f59e0b" />
              </marker>
              <marker id="dqnArrGreen" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#10b981" />
              </marker>
              <marker id="dqnArrPurple" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#a855f7" />
              </marker>
            </defs>

            {/* === Q-Network Pipeline === */}
            <text x="260" y="16" textAnchor="middle" fill="#888" fontSize="10" fontWeight="bold">Q-Network Architecture</text>

            {/* Input */}
            <rect
              x="8" y="28" width="78" height="62" rx="8"
              fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'input' ? null : 'input')}
            />
            <text x="47" y="50" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">84×84×4</text>
            <text x="47" y="64" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">Pixels</text>
            <text x="47" y="78" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">(4 frames)</text>

            <path d="M 86 59 L 113 59" stroke="#00d9ff" strokeWidth="1.5" markerEnd="url(#dqnArr)" />

            {/* Conv1 */}
            <rect
              x="115" y="28" width="78" height="62" rx="8"
              fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'conv1' ? null : 'conv1')}
            />
            <text x="154" y="48" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">Conv1</text>
            <text x="154" y="62" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">16 × 8×8</text>
            <text x="154" y="76" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">stride 4, ReLU</text>

            <path d="M 193 59 L 220 59" stroke="#00d9ff" strokeWidth="1.5" markerEnd="url(#dqnArr)" />

            {/* Conv2 */}
            <rect
              x="222" y="28" width="78" height="62" rx="8"
              fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'conv2' ? null : 'conv2')}
            />
            <text x="261" y="48" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">Conv2</text>
            <text x="261" y="62" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">32 × 4×4</text>
            <text x="261" y="76" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">stride 2, ReLU</text>

            <path d="M 300 59 L 327 59" stroke="#00d9ff" strokeWidth="1.5" markerEnd="url(#dqnArr)" />

            {/* FC */}
            <rect
              x="329" y="28" width="68" height="62" rx="8"
              fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'fc' ? null : 'fc')}
            />
            <text x="363" y="48" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">FC</text>
            <text x="363" y="62" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">256 units</text>
            <text x="363" y="76" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">ReLU</text>

            <path d="M 397 59 L 424 59" stroke="#00d9ff" strokeWidth="1.5" markerEnd="url(#dqnArr)" />

            {/* Q Output */}
            <rect
              x="426" y="22" width="84" height="80" rx="8"
              fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'qout' ? null : 'qout')}
            />
            <text x="468" y="42" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="bold" pointerEvents="none">Q-Values</text>
            <text x="468" y="58" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Q(s, a&#x2081;)</text>
            <text x="468" y="70" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Q(s, a&#x2082;)</text>
            <text x="468" y="80" textAnchor="middle" fill="#666" fontSize="9" pointerEvents="none">&#x22EE;</text>
            <text x="468" y="92" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">Q(s, a&#x2096;)</text>

            <text x="468" y="116" textAnchor="middle" fill="#ef4444" fontSize="8" fontWeight="bold">argmax &#x2192; action</text>

            {/* === Separator === */}
            <line x1="15" y1="140" x2="505" y2="140" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

            {/* === Training Loop === */}
            <text x="260" y="162" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">Training Loop</text>

            {/* Environment */}
            <rect x="10" y="178" width="95" height="50" rx="8" fill="rgba(168,85,247,0.15)" stroke="#a855f7" strokeWidth="1.5" />
            <text x="57" y="200" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">Environment</text>
            <text x="57" y="214" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">(Atari Emulator)</text>

            <path d="M 105 203 L 138 203" stroke="#a855f7" strokeWidth="1.5" markerEnd="url(#dqnArrPurple)" />
            <text x="121" y="196" textAnchor="middle" fill="#666" fontSize="6.5">(s,a,r,s&apos;)</text>

            {/* Replay Buffer */}
            <rect
              x="140" y="175" width="120" height="56" rx="8"
              fill="rgba(245,158,11,0.15)" stroke="#f59e0b" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'replay' ? null : 'replay')}
            />
            <text x="200" y="197" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold" pointerEvents="none">Replay Buffer D</text>
            <text x="200" y="211" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">N = 1,000,000</text>
            <text x="200" y="223" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">uniform random sample</text>

            <path d="M 260 203 L 298 203" stroke="#f59e0b" strokeWidth="1.5" markerEnd="url(#dqnArrAmber)" />
            <text x="279" y="196" textAnchor="middle" fill="#666" fontSize="6.5">minibatch</text>

            {/* Q-Network training */}
            <rect x="300" y="178" width="100" height="50" rx="8" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.5" />
            <text x="350" y="200" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">Q-Network</text>
            <text x="350" y="214" textAnchor="middle" fill="#888" fontSize="7.5" pointerEvents="none">SGD update</text>

            <path d="M 400 203 L 427 203" stroke="#3b82f6" strokeWidth="1.5" markerEnd="url(#dqnArr)" />

            {/* TD Loss */}
            <rect x="429" y="178" width="82" height="50" rx="8" fill="rgba(0,217,255,0.12)" stroke="#00d9ff" strokeWidth="1.5" />
            <text x="470" y="197" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold" pointerEvents="none">TD Loss</text>
            <text x="470" y="210" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">(y &#x2212; Q(s,a))&#xB2;</text>

            {/* Action loop back to environment */}
            <path d="M 350 178 Q 350 150 200 150 Q 57 150 57 178" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4 3" fill="none" markerEnd="url(#dqnArrGreen)" />
            <text x="200" y="144" textAnchor="middle" fill="#10b981" fontSize="7">&#x03B5;-greedy action selection</text>

            {/* Key insight box */}
            <rect x="30" y="252" width="460" height="52" rx="10" fill="rgba(0,217,255,0.06)" stroke="rgba(0,217,255,0.2)" strokeWidth="1" />
            <text x="260" y="271" textAnchor="middle" fill="#00d9ff" fontSize="9" fontWeight="bold">Key architectural insight</text>
            <text x="260" y="287" textAnchor="middle" fill="#888" fontSize="8">
              One forward pass outputs Q-values for ALL actions &#x2192; no per-action computation
            </text>
            <text x="260" y="299" textAnchor="middle" fill="#888" fontSize="8">
              This only works for discrete action spaces (4&#x2013;18 actions in Atari)
            </text>

            {/* Click hint */}
            <text x="260" y="345" textAnchor="middle" fill="#555" fontSize="8" fontStyle="italic">Click any component for details</text>
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
        <h2>Training Algorithm</h2>

        <div className="formula-block">
          <h4>TD Loss (Q-learning)</h4>
          <MathJax>{String.raw`\[ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \!\left[\Big( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \Big)^{2} \right] \]`}</MathJax>
        </div>

        <ol className="algo-steps">
          <li><strong>Observe</strong> preprocessed state &#x03C6;(s) from emulator (84&times;84&times;4 stacked frames)</li>
          <li><strong>Select action</strong> via &#x03B5;-greedy: random with prob &#x03B5;, else a = argmax<sub>a</sub> Q(s, a; &#x03B8;)</li>
          <li><strong>Store</strong> transition (s, a, r, s&prime;) in replay buffer D</li>
          <li><strong>Sample</strong> random minibatch of 32 transitions from D</li>
          <li><strong>Compute</strong> TD target: y = r + &#x03B3; max<sub>a&prime;</sub> Q(s&prime;, a&prime;; &#x03B8;) &nbsp;(or y = r if terminal)</li>
          <li><strong>Update</strong> &#x03B8; by gradient descent on (y &minus; Q(s, a; &#x03B8;))&sup2;</li>
        </ol>

        <InfoPanel title="The 2015 Nature Version" className={styles.notePanel}>
          <p>
            The 2015 follow-up (&ldquo;Human-level control through deep reinforcement learning&rdquo;, <em>Nature</em>)
            added a <strong style={{ color: '#f59e0b' }}>target network</strong> &#x03B8;<sup>&minus;</sup> that
            is periodically copied from the online network. The TD target becomes
            y = r + &#x03B3; max Q(s&prime;, a&prime;; &#x03B8;<sup>&minus;</sup>), preventing moving-target instability.
            DDPG later refined this into <em>soft</em> target updates.
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
                { title: 'Q-Learning', desc: 'Watkins, 1989 \u2014 The foundational off-policy temporal-difference algorithm that DQN scales to high-dimensional inputs with neural networks' },
                { title: 'TD-Gammon', desc: 'Tesauro, 1995 \u2014 Demonstrated that neural networks could learn game-playing from self-play via TD learning; DQN generalized this across many games' },
                { title: 'Experience Replay', desc: 'Lin, 1992 \u2014 Originally proposed storing and re-using past transitions to improve sample efficiency; DQN showed it also stabilizes neural network training' },
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
                { title: 'Double DQN', desc: 'van Hasselt et al., 2016 \u2014 Decouples action selection from evaluation to fix DQN\u2019s systematic Q-value overestimation' },
                { title: 'DDPG', desc: 'Lillicrap et al., 2015 \u2014 Extends DQN\u2019s key ideas (replay, target networks) to continuous action spaces via an explicit actor network' },
                { title: 'Rainbow', desc: 'Hessel et al., 2018 \u2014 Combines six DQN improvements (double, dueling, prioritized replay, multi-step, distributional, noisy nets) into one agent' },
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
