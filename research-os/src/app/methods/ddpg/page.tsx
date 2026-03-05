'use client';

import { useState } from 'react';
import { MathJax } from 'better-react-mathjax';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './ddpg.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Actor-Critic Split',
    desc: 'Separate the policy (Actor) from the value function (Critic) — the actor outputs a continuous action, the critic evaluates it',
    detail: "DQN outputs Q-values for all discrete actions and picks argmax. With continuous actions this is impossible — you can't enumerate infinite actions. DDPG solves this by learning a separate actor network \u03BC(s|\u03B8\u03BC) that directly outputs the continuous action vector. The critic Q(s,a|\u03B8Q) then evaluates how good that action is. The actor is trained via the deterministic policy gradient: backpropagate through the critic to find which direction in action space increases Q.",
  },
  {
    title: 'Soft Target Updates',
    desc: 'Instead of periodically copying the network (DQN), slowly blend target weights: \u03B8\' \u2190 \u03C4\u03B8 + (1\u2212\u03C4)\u03B8\' with \u03C4 \u226A 1',
    detail: "DQN's 2015 version copies the online network to the target network every C steps, causing abrupt target changes. DDPG instead uses Polyak averaging: at every step, the target weights move a tiny fraction \u03C4 (typically 0.001) toward the online weights. This provides much smoother, more stable targets. Both the actor and critic have their own target networks (\u03BC' and Q'), so the TD target y = r + \u03B3Q'(s', \u03BC'(s')) uses only slowly-moving parameters.",
  },
  {
    title: 'Continuous Exploration via Noise',
    desc: 'Add temporally-correlated Ornstein-Uhlenbeck noise to the actor\'s output for smooth exploration in physical control tasks',
    detail: "DQN uses \u03B5-greedy: with probability \u03B5 pick a random discrete action. This doesn't work for continuous spaces \u2014 a random action vector would be erratic. DDPG instead adds noise directly to the actor's output: a = \u03BC(s) + N, where N is an Ornstein-Uhlenbeck process that generates temporally correlated noise. This produces smooth, momentum-respecting exploration trajectories that are more effective for physical control tasks with inertia.",
  },
];

const keyFeatures = [
  { title: 'Continuous Control', desc: 'Handles high-dimensional continuous action spaces \u2014 joint torques, motor commands, steering' },
  { title: 'Simple & General', desc: 'Same architecture and hyperparameters solve 20+ physics tasks from cartpole to locomotion' },
  { title: 'Pixel-Ready', desc: 'Can learn directly from raw pixel observations using convolutional layers, like DQN' },
];

const componentInfo: Record<string, { title: string; desc: string }> = {
  actor: {
    title: 'Actor Network \u03BC(s|\u03B8\u03BC)',
    desc: 'Maps state to a deterministic continuous action vector. Architecture: state \u2192 400 ReLU \u2192 300 ReLU \u2192 tanh output (bounded actions). Trained to maximize Q by following the gradient \u2207\u03B8\u03BC Q(s, \u03BC(s)) \u2014 the chain rule through the critic tells the actor how to adjust its output to increase value.',
  },
  critic: {
    title: 'Critic Network Q(s, a|\u03B8Q)',
    desc: "Estimates the Q-value of a state-action pair. Architecture: state \u2192 400 ReLU \u2192 [concat action at layer 2] \u2192 300 ReLU \u2192 linear output. Trained with standard TD learning on minibatches from the replay buffer. Actions aren't input until the second hidden layer \u2014 this gives the network a chance to process state features before combining with the action.",
  },
  target_actor: {
    title: "Target Actor \u03BC'(s|\u03B8\u03BC')",
    desc: "A slowly-updated copy of the actor used to compute TD targets. Updated via soft Polyak averaging: \u03B8\u03BC' \u2190 \u03C4\u03B8\u03BC + (1\u2212\u03C4)\u03B8\u03BC' with \u03C4 = 0.001. Provides a stable action for computing the next-state value in the Bellman target.",
  },
  target_critic: {
    title: "Target Critic Q'(s, a|\u03B8Q')",
    desc: "A slowly-updated copy of the critic used to compute TD targets: y = r + \u03B3Q'(s', \u03BC'(s')). Updated via soft Polyak averaging: \u03B8Q' \u2190 \u03C4\u03B8Q + (1\u2212\u03C4)\u03B8Q'. The slow tracking prevents the moving-target problem that destabilizes naive Q-learning with neural networks.",
  },
  replay: {
    title: 'Replay Buffer R',
    desc: "Identical in concept to DQN's experience replay. A large buffer (10\u2076 transitions) stores (s, a, r, s') tuples. Random minibatches break temporal correlations. Because DDPG is off-policy, it can learn from transitions collected by older versions of the policy \u2014 this is what makes replay possible.",
  },
  noise: {
    title: 'Exploration Noise (OU Process)',
    desc: 'An Ornstein-Uhlenbeck process with \u03B8=0.15 and \u03C3=0.2 generates temporally correlated noise. Unlike white noise, OU noise has inertia \u2014 it produces smooth exploration trajectories that are effective for physical control tasks where jerky random actions would be inefficient.',
  },
};

const dqnComparison = [
  { aspect: 'Action space', dqn: 'Discrete (4\u201318 actions)', ddpg: 'Continuous (\u211D\u1D3A)' },
  { aspect: 'Policy', dqn: 'Implicit: argmax Q(s, \u00B7)', ddpg: 'Explicit actor: \u03BC(s|\u03B8\u03BC)' },
  { aspect: 'Q output', dqn: 'K values (one per action)', ddpg: 'Single scalar Q(s, a)' },
  { aspect: 'Target update', dqn: 'Periodic hard copy', ddpg: 'Soft Polyak: \u03C4\u03B8 + (1\u2212\u03C4)\u03B8\'' },
  { aspect: 'Exploration', dqn: '\u03B5-greedy (random action)', ddpg: '\u03BC(s) + OU noise' },
];

/* ================================================================
   COMPONENT
   ================================================================ */

export default function DDPGPage() {
  const [activeComponent, setActiveComponent] = useState<string | null>(null);

  return (
    <div className="method-page">
      <h1>Deep Deterministic Policy Gradient (DDPG)</h1>
      <p className={styles.subtitle}>
        Continuous Control with Deep Reinforcement Learning &mdash; Lillicrap et al., 2015
      </p>

      {/* ==================== From DQN to DDPG ==================== */}
      <section>
        <h2>From DQN to DDPG</h2>
        <p className={styles.comparisonIntro}>
          DQN solves high-dimensional observation spaces (pixels) but only handles <strong style={{ color: '#ef4444' }}>discrete</strong> actions.
          DDPG adapts DQN&apos;s key ideas to <strong style={{ color: '#10b981' }}>continuous</strong> action spaces
          by replacing the implicit argmax policy with an explicit actor network.
        </p>

        <div className={styles.comparisonTable}>
          <div className={styles.comparisonHeader}>
            <span className={styles.comparisonAspect}></span>
            <span className={styles.comparisonDqn}>DQN</span>
            <span className={styles.comparisonDdpg}>DDPG</span>
          </div>
          {dqnComparison.map((row) => (
            <div key={row.aspect} className={styles.comparisonRow}>
              <span className={styles.comparisonAspect}>{row.aspect}</span>
              <span className={styles.comparisonDqn}>{row.dqn}</span>
              <span className={styles.comparisonDdpg}>{row.ddpg}</span>
            </div>
          ))}
        </div>

        <div className="diagram-frame">
          <svg viewBox="0 0 520 140" className={styles.comparisonSvg}>
            {/* DQN side */}
            <text x="130" y="16" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">DQN (Discrete)</text>
            <rect x="20" y="24" width="60" height="36" rx="6" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.2" />
            <text x="50" y="40" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">State s</text>
            <text x="50" y="52" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">pixels</text>

            <path d="M 80 42 L 98 42" stroke="#00d9ff" strokeWidth="1.2" markerEnd="url(#ddpgArr)" />

            <rect x="100" y="24" width="60" height="36" rx="6" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.2" />
            <text x="130" y="40" textAnchor="middle" fill="#3b82f6" fontSize="8" fontWeight="bold" pointerEvents="none">Q-Net</text>
            <text x="130" y="52" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">&#x2192; K values</text>

            <path d="M 160 42 L 178 42" stroke="#00d9ff" strokeWidth="1.2" markerEnd="url(#ddpgArr)" />

            <rect x="180" y="28" width="55" height="28" rx="6" fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth="1.2" />
            <text x="207" y="46" textAnchor="middle" fill="#ef4444" fontSize="8" fontWeight="bold" pointerEvents="none">argmax</text>

            <text x="130" y="82" textAnchor="middle" fill="#ef4444" fontSize="8">&#x274C; Can&apos;t enumerate &#x221E; actions</text>

            {/* Divider */}
            <line x1="260" y1="5" x2="260" y2="135" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="4 3" />

            {/* DDPG side */}
            <defs>
              <marker id="ddpgArr" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#00d9ff" />
              </marker>
              <marker id="ddpgArrGreen" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#10b981" />
              </marker>
            </defs>

            <text x="390" y="16" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">DDPG (Continuous)</text>
            <rect x="280" y="24" width="60" height="36" rx="6" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.2" />
            <text x="310" y="40" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">State s</text>
            <text x="310" y="52" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">pixels / vec</text>

            <path d="M 340 34 L 358 34" stroke="#10b981" strokeWidth="1.2" markerEnd="url(#ddpgArrGreen)" />
            <path d="M 340 52 L 358 52" stroke="#00d9ff" strokeWidth="1.2" markerEnd="url(#ddpgArr)" />

            <rect x="360" y="20" width="60" height="24" rx="6" fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="1.2" />
            <text x="390" y="36" textAnchor="middle" fill="#10b981" fontSize="8" fontWeight="bold" pointerEvents="none">Actor &#x03BC;</text>

            <rect x="360" y="48" width="60" height="24" rx="6" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth="1.2" />
            <text x="390" y="64" textAnchor="middle" fill="#3b82f6" fontSize="8" fontWeight="bold" pointerEvents="none">Critic Q</text>

            <path d="M 420 32 L 453 32 Q 460 32 460 44 L 460 48" stroke="#10b981" strokeWidth="1.2" markerEnd="url(#ddpgArrGreen)" />
            <text x="440" y="26" fill="#888" fontSize="6.5">a &#x2208; &#x211D;&#x1D3A;</text>

            <path d="M 420 60 L 460 60 L 460 56" stroke="#3b82f6" strokeWidth="1.2" />

            <rect x="445" y="44" width="55" height="22" rx="6" fill="rgba(0,217,255,0.12)" stroke="#00d9ff" strokeWidth="1.2" />
            <text x="472" y="58" textAnchor="middle" fill="#00d9ff" fontSize="7" fontWeight="bold" pointerEvents="none">Q(s, a)</text>

            <text x="390" y="96" textAnchor="middle" fill="#10b981" fontSize="8">&#x2705; Actor <tspan fontStyle="italic">learns</tspan> the argmax</text>

            {/* Bottom insight */}
            <rect x="70" y="108" width="380" height="26" rx="8" fill="rgba(245,158,11,0.08)" stroke="rgba(245,158,11,0.2)" strokeWidth="1" />
            <text x="260" y="125" textAnchor="middle" fill="#f59e0b" fontSize="8">
              DDPG = DQN&apos;s replay buffer + target networks + actor-critic for continuous actions
            </text>
          </svg>
        </div>
      </section>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture ==================== */}
      <section>
        <h2>DDPG Architecture</h2>
        <div className="diagram-frame">
          <svg className={styles.archSvg} viewBox="0 0 500 400">
            <defs>
              <marker id="ddpgA" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#00d9ff" />
              </marker>
              <marker id="ddpgAG" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#10b981" />
              </marker>
              <marker id="ddpgAB" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#3b82f6" />
              </marker>
              <marker id="ddpgAA" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#f59e0b" />
              </marker>
              <marker id="ddpgAP" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#a855f7" />
              </marker>
            </defs>

            {/* === State Input === */}
            <rect x="190" y="8" width="120" height="40" rx="8" fill="rgba(168,85,247,0.2)" stroke="#a855f7" strokeWidth="1.5" />
            <text x="250" y="26" textAnchor="middle" fill="#a855f7" fontSize="9" fontWeight="bold" pointerEvents="none">State s</text>
            <text x="250" y="40" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">(low-dim or pixels)</text>

            {/* Arrows from state to actor and critic */}
            <path d="M 220 48 L 130 68" stroke="#10b981" strokeWidth="1.5" markerEnd="url(#ddpgAG)" />
            <path d="M 280 48 L 370 68" stroke="#3b82f6" strokeWidth="1.5" markerEnd="url(#ddpgAB)" />

            {/* === Actor Network === */}
            <rect
              x="40" y="70" width="170" height="65" rx="10"
              fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'actor' ? null : 'actor')}
            />
            <text x="125" y="92" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold" pointerEvents="none">Actor &#x03BC;(s|&#x03B8;&#x1D5A;)</text>
            <text x="125" y="108" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">s &#x2192; 400 &#x2192; 300 &#x2192; tanh</text>
            <text x="125" y="122" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">outputs continuous action a &#x2208; &#x211D;&#x1D3A;</text>

            {/* === Critic Network === */}
            <rect
              x="290" y="70" width="170" height="65" rx="10"
              fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth="2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'critic' ? null : 'critic')}
            />
            <text x="375" y="92" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold" pointerEvents="none">Critic Q(s, a|&#x03B8;&#x1D44;)</text>
            <text x="375" y="108" textAnchor="middle" fill="#888" fontSize="8" pointerEvents="none">s &#x2192; 400 &#x2192; [+a] &#x2192; 300 &#x2192; 1</text>
            <text x="375" y="122" textAnchor="middle" fill="#888" fontSize="7" pointerEvents="none">action input at 2nd layer</text>

            {/* Actor -> action -> Critic */}
            <path d="M 210 102 L 285 102" stroke="#10b981" strokeWidth="1.5" markerEnd="url(#ddpgAG)" />
            <text x="247" y="97" textAnchor="middle" fill="#10b981" fontSize="7" fontWeight="bold">action a</text>

            {/* Policy gradient arrow: Critic -> Actor */}
            <path d="M 305 70 Q 250 55 195 70" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3" fill="none" markerEnd="url(#ddpgAA)" />
            <text x="250" y="58" textAnchor="middle" fill="#f59e0b" fontSize="7">&#x2207;&#x03B8;&#x03BC; J = &#x2207;&#x2090;Q &#xB7; &#x2207;&#x03B8;&#x03BC;&#x03BC;</text>

            {/* === Exploration Noise === */}
            <rect
              x="40" y="155" width="170" height="35" rx="8"
              fill="rgba(168,85,247,0.12)" stroke="#a855f7" strokeWidth="1.2"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'noise' ? null : 'noise')}
            />
            <text x="125" y="174" textAnchor="middle" fill="#a855f7" fontSize="8" fontWeight="bold" pointerEvents="none">+ OU Noise &#x1D4A9;</text>
            <text x="125" y="184" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">a = &#x03BC;(s) + &#x1D4A9;&#x209C;</text>

            <path d="M 125 135 L 125 153" stroke="#a855f7" strokeWidth="1.2" markerEnd="url(#ddpgAP)" />

            {/* === Replay Buffer === */}
            <rect
              x="290" y="155" width="170" height="35" rx="8"
              fill="rgba(245,158,11,0.12)" stroke="#f59e0b" strokeWidth="1.5"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'replay' ? null : 'replay')}
            />
            <text x="375" y="174" textAnchor="middle" fill="#f59e0b" fontSize="8" fontWeight="bold" pointerEvents="none">Replay Buffer R</text>
            <text x="375" y="184" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">(s, a, r, s&#x2032;) &#x2192; random minibatch</text>

            <path d="M 375 135 L 375 153" stroke="#f59e0b" strokeWidth="1.2" markerEnd="url(#ddpgAA)" />

            {/* === Separator === */}
            <line x1="20" y1="210" x2="480" y2="210" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

            {/* === Target Networks === */}
            <text x="250" y="232" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">Target Networks (Soft Update)</text>

            {/* Target Actor */}
            <rect
              x="40" y="245" width="170" height="50" rx="10"
              fill="rgba(16,185,129,0.08)" stroke="#10b981" strokeWidth="1.2" strokeDasharray="6 3"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'target_actor' ? null : 'target_actor')}
            />
            <text x="125" y="266" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold" pointerEvents="none">Target Actor &#x03BC;&#x2032;</text>
            <text x="125" y="282" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">&#x03B8;&#x03BC;&#x2032; &#x2190; &#x03C4;&#x03B8;&#x03BC; + (1&#x2212;&#x03C4;)&#x03B8;&#x03BC;&#x2032;</text>

            {/* Target Critic */}
            <rect
              x="290" y="245" width="170" height="50" rx="10"
              fill="rgba(59,130,246,0.08)" stroke="#3b82f6" strokeWidth="1.2" strokeDasharray="6 3"
              className={styles.archStage}
              onClick={() => setActiveComponent(activeComponent === 'target_critic' ? null : 'target_critic')}
            />
            <text x="375" y="266" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold" pointerEvents="none">Target Critic Q&#x2032;</text>
            <text x="375" y="282" textAnchor="middle" fill="#666" fontSize="7" pointerEvents="none">&#x03B8;Q&#x2032; &#x2190; &#x03C4;&#x03B8;Q + (1&#x2212;&#x03C4;)&#x03B8;Q&#x2032;</text>

            {/* Soft update arrows */}
            <path d="M 125 135 Q 60 200 100 245" stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 3" fill="none" markerEnd="url(#ddpgAG)" />
            <path d="M 375 190 Q 440 220 400 245" stroke="#3b82f6" strokeWidth="1.2" strokeDasharray="3 3" fill="none" markerEnd="url(#ddpgAB)" />

            <text x="55" y="198" fill="#10b981" fontSize="6.5">&#x03C4; = 0.001</text>
            <text x="442" y="222" fill="#3b82f6" fontSize="6.5">&#x03C4; = 0.001</text>

            {/* TD Target box */}
            <rect x="130" y="315" width="240" height="35" rx="8" fill="rgba(0,217,255,0.08)" stroke="rgba(0,217,255,0.25)" strokeWidth="1" />
            <text x="250" y="330" textAnchor="middle" fill="#00d9ff" fontSize="8" fontWeight="bold">TD Target</text>
            <text x="250" y="343" textAnchor="middle" fill="#888" fontSize="8">y = r + &#x03B3; Q&#x2032;(s&#x2032;, &#x03BC;&#x2032;(s&#x2032;))</text>

            <path d="M 125 295 Q 125 310 170 318" stroke="#10b981" strokeWidth="1" strokeDasharray="3 2" fill="none" markerEnd="url(#ddpgAG)" />
            <path d="M 375 295 Q 375 310 330 318" stroke="#3b82f6" strokeWidth="1" strokeDasharray="3 2" fill="none" markerEnd="url(#ddpgAB)" />

            {/* Click hint */}
            <text x="250" y="390" textAnchor="middle" fill="#555" fontSize="8" fontStyle="italic">Click any component for details</text>
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

      {/* ==================== Algorithm ==================== */}
      <section>
        <h2>Training Algorithm</h2>

        <div className="formula-block">
          <h4>Critic Loss (TD error)</h4>
          <MathJax>{String.raw`\[ L(\theta^Q) = \mathbb{E} \!\left[\Big( r + \gamma\, Q'(s', \mu'(s')) - Q(s, a) \Big)^{2} \right] \]`}</MathJax>
        </div>

        <div className="formula-block">
          <h4>Actor Update (deterministic policy gradient)</h4>
          <MathJax>{String.raw`\[ \nabla_{\theta^\mu} J \approx \mathbb{E}\!\left[ \nabla_a Q(s, a)\big|_{a=\mu(s)} \;\cdot\; \nabla_{\theta^\mu} \mu(s) \right] \]`}</MathJax>
        </div>

        <ol className="algo-steps">
          <li>
            <strong>Select</strong> action a = &#x03BC;(s|&#x03B8;&#x1D5A;) + &#x1D4A9;<sub>t</sub> &nbsp;(actor + OU noise)
          </li>
          <li>
            <strong>Store</strong> (s, a, r, s&prime;) in replay buffer R
          </li>
          <li>
            <strong>Sample</strong> random minibatch of N transitions from R
          </li>
          <li>
            <strong>Update critic</strong> by minimizing TD loss: L = (1/N) &#x2211;(y<sub>i</sub> &minus; Q(s<sub>i</sub>, a<sub>i</sub>))&sup2;
          </li>
          <li>
            <strong>Update actor</strong> via policy gradient: &#x2207;<sub>&#x03B8;&#x03BC;</sub>J using &#x2207;<sub>a</sub>Q &#xB7; &#x2207;<sub>&#x03B8;&#x03BC;</sub>&#x03BC;
          </li>
          <li>
            <strong>Soft update</strong> both target networks: &#x03B8;&prime; &#x2190; &#x03C4;&#x03B8; + (1&minus;&#x03C4;)&#x03B8;&prime;
          </li>
        </ol>

        {/* Borrowed from DQN */}
        <InfoPanel title="What DDPG Borrows from DQN">
          <p>
            <strong style={{ color: '#f59e0b' }}>Replay buffer</strong> &mdash;
            identical concept: store transitions, sample random minibatches for decorrelated training.
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#f59e0b' }}>Target networks</strong> &mdash;
            same idea but refined: soft Polyak averaging (&#x03C4; = 0.001) instead of periodic hard copy.
            Applied to <em>both</em> actor and critic.
          </p>
          <p style={{ marginTop: 8 }}>
            <strong style={{ color: '#f59e0b' }}>Off-policy learning</strong> &mdash;
            replay is only possible because both DQN and DDPG learn off-policy (from data collected by older policies).
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
                { title: 'DQN', desc: 'Mnih et al., 2013/2015 \u2014 DDPG directly adapts DQN\u2019s replay buffer, target networks, and off-policy learning to continuous action spaces' },
                { title: 'DPG (Deterministic Policy Gradient)', desc: 'Silver et al., 2014 \u2014 Provides the theoretical foundation: the deterministic policy gradient theorem that DDPG implements with deep networks' },
                { title: 'Actor-Critic Methods', desc: 'Konda & Tsitsiklis, 2000 \u2014 The actor-critic architecture separating policy from value function, which DDPG scales to high-dimensional continuous control' },
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
                { title: 'TD3', desc: 'Fujimoto et al., 2018 \u2014 Fixes DDPG\u2019s overestimation with twin critics, delayed policy updates, and target policy smoothing' },
                { title: 'SAC', desc: 'Haarnoja et al., 2018 \u2014 Replaces DDPG\u2019s deterministic policy with a stochastic one and adds entropy regularization for more robust exploration' },
                { title: 'D4PG', desc: 'Barth-Maron et al., 2018 \u2014 Distributional DDPG: replaces the scalar critic with a distributional one and adds prioritized replay and N-step returns' },
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
