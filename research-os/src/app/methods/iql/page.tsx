'use client';

import { MathJax } from 'better-react-mathjax';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import { InfoPanel } from '@/components/InfoPanel/InfoPanel';
import styles from './iql.module.css';

/* ================================================================
   DATA
   ================================================================ */

const coreIdeas = [
  {
    title: 'Never Query OOD Actions',
    desc: 'Learn optimal Q-values without ever evaluating actions outside the dataset — completely sidesteps the extrapolation problem',
    detail: 'Prior offline RL methods (BCQ, BEAR, CQL) all need to query Q(s,a) for unseen actions during training — either to compute Bellman targets via max_a Q(s\',a\') or to regularize the Q-function. This inevitably requires handling out-of-distribution actions. IQL avoids this entirely: both the value function loss and Q-function loss only evaluate actions that appear in the dataset. The key trick is using expectile regression on V to implicitly approximate the max without ever computing it.',
  },
  {
    title: 'Expectile Regression for Implicit Maximization',
    desc: 'Approximate max_a Q(s,a) by fitting V(s) to the upper expectile of Q-values — no explicit maximization needed',
    detail: 'The τ-expectile of a random variable is the solution to an asymmetric least squares problem: L²_τ(u) = |τ - 𝟙(u < 0)| · u². For τ = 0.5, this gives the mean (standard MSE). As τ → 1, the expectile approaches the maximum. IQL fits V_ψ(s) to the τ-expectile of Q_θ̂(s,a) over dataset actions at state s. This means V(s) ≈ max_a Q(s,a) without ever computing Q for unseen actions. The value τ interpolates between SARSA (τ=0.5) and Q-learning (τ→1). In practice τ ∈ {0.7, 0.9} works well; higher τ is better for tasks requiring "stitching" of suboptimal trajectories.',
  },
  {
    title: 'Two-Stage: TD Learning then Policy Extraction',
    desc: 'First learn V and Q via modified Bellman backups, then extract a policy via advantage-weighted regression — fully decoupled',
    detail: 'IQL\'s training has two independent stages. Stage 1 (TD learning): alternate between fitting V_ψ via expectile regression on Q_θ̂, and fitting Q_θ via standard Bellman backup using V_ψ(s\') instead of max_a Q(s\',a). This avoids OOD queries entirely. Stage 2 (policy extraction): train π_φ via advantage-weighted behavioral cloning — actions with high advantage A(s,a) = Q(s,a) - V(s) get higher weight. The temperature β controls how greedy the policy is. Crucially, the policy does not influence value learning at all, so the two stages can even run concurrently.',
  },
];

const keyFeatures = [
  { title: 'Simple Implementation', desc: 'Only requires adding an asymmetric L2 loss to a standard SARSA-style TD algorithm — ~20 lines of code on top of standard RL' },
  { title: 'Computationally Efficient', desc: '1M updates in ~20 minutes on a single GTX 1080. ~4x faster than CQL due to no logsumexp or policy sampling in the value update' },
  { title: 'Strong Fine-tuning', desc: 'Advantage-weighted policy extraction provides excellent initialization for online RL fine-tuning, significantly outperforming CQL and AWAC' },
];

/* ================================================================
   COMPONENT
   ================================================================ */

export default function IQLPage() {
  return (
    <div className="method-page">
      <h1>IQL: Implicit Q-Learning</h1>
      <p className={styles.subtitle}>
        Offline Reinforcement Learning with Implicit Q-Learning &mdash; Kostrikov, Nair &amp; Levine, ICLR 2022
      </p>
      <div className={styles.githubLink}>
        <a href="https://arxiv.org/abs/2110.06169" target="_blank" rel="noopener noreferrer">
          <span style={{ marginRight: 5 }}>&rarr;</span> arxiv.org/abs/2110.06169
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Architecture ==================== */}
      <section>
        <h2>IQL Architecture</h2>
        <p className={styles.archSubtitle}>
          Not actor-critic &mdash; the policy is fully decoupled from value learning.
          V and Q are trained via modified Bellman backups; the policy is extracted separately via advantage-weighted regression.
        </p>

        {/* --- Network cards --- */}
        <div className={styles.networkGrid}>
          <div className={styles.networkCard} style={{ borderColor: '#10b981' }}>
            <div className={styles.networkCardTitle} style={{ color: '#10b981' }}>V&#x03C8;</div>
            <div className={styles.networkCardType}>Value network</div>
            <div className={styles.networkCardBody}>
              s &rarr; &#x211D;<br />
              Trained via expectile regression on Q&#x0302;(s,a) over dataset actions.
              Approximates the upper expectile of Q &mdash; implicitly estimates max<sub>a</sub> Q(s,a) without querying OOD actions.
            </div>
          </div>
          <div className={styles.networkCard} style={{ borderColor: '#3b82f6' }}>
            <div className={styles.networkCardTitle} style={{ color: '#3b82f6' }}>Q&#x03B8;&#x2081;, Q&#x03B8;&#x2082;</div>
            <div className={styles.networkCardType}>Twin Q-networks</div>
            <div className={styles.networkCardBody}>
              (s, a) &rarr; &#x211D; &nbsp; (clipped double Q-learning)<br />
              Standard Bellman backup using V&#x03C8;(s&prime;) as the next-state value &mdash;
              never computes max<sub>a&prime;</sub> Q(s&prime;, a&prime;). Only evaluated on dataset (s,a) pairs.
            </div>
          </div>
          <div className={styles.networkCard} style={{ borderColor: '#a855f7' }}>
            <div className={styles.networkCardTitle} style={{ color: '#a855f7' }}>&pi;&#x03C6;</div>
            <div className={styles.networkCardType}>Policy (extracted separately)</div>
            <div className={styles.networkCardBody}>
              s &rarr; &pi;&#x03C6;(a|s)<br />
              Trained via advantage-weighted regression after V and Q converge. Actions with high advantage
              A = Q &minus; V get exponentially more weight. Does not influence value learning.
            </div>
          </div>
        </div>
        <p className={styles.targetNote}>
          + target Q-networks Q&#x0302;&#x2081;, Q&#x0302;&#x2082; updated via Polyak averaging (&alpha; = 0.005). No target V or target policy needed.
        </p>

        {/* --- Loss computation --- */}
        <h3 style={{ marginTop: 24, marginBottom: 12 }}>Loss Computation</h3>
        <div className={styles.lossGrid}>
          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>1</span>
              <span className={styles.lossCardTitle} style={{ color: '#10b981' }}>Value Loss (expectile regression)</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[ L_2^\tau\!\bigl(Q_{\hat{\theta}}(s,a) - V_\psi(s)\bigr) \right] \]`}</MathJax>
              <MathJax>{String.raw`\[ \text{where} \quad L_2^\tau(u) = |\tau - \mathbb{1}(u < 0)|\, u^2 \]`}</MathJax>
              <p className={styles.lossNote}>
                Asymmetric L2 loss: for &tau; &gt; 0.5, positive residuals (where Q &gt; V) are upweighted,
                pushing V toward the upper expectile of Q. As &tau; &rarr; 1, V &rarr; max<sub>a</sub> Q(s,a).
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>2</span>
              <span className={styles.lossCardTitle} style={{ color: '#3b82f6' }}>Q Loss (standard Bellman with V target)</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ L_Q(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\!\left[\bigl(r(s,a) + \gamma\, V_\psi(s') - Q_\theta(s,a)\bigr)^2\right] \]`}</MathJax>
              <p className={styles.lossNote}>
                Looks like SARSA but uses V&#x03C8;(s&prime;) instead of Q(s&prime;,a&prime;).
                Since V already approximates the max, this performs multi-step dynamic programming without ever querying OOD actions.
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>3</span>
              <span className={styles.lossCardTitle} style={{ color: '#a855f7' }}>Policy Extraction (advantage-weighted regression)</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ L_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\exp\!\bigl(\beta\,(Q_{\hat{\theta}}(s,a) - V_\psi(s))\bigr)\;\log \pi_\phi(a|s)\right] \]`}</MathJax>
              <p className={styles.lossNote}>
                Weighted behavioral cloning: actions with high advantage A = Q &minus; V get exponentially more weight.
                &beta; is an inverse temperature &mdash; small &beta; &rarr; behavior cloning, large &beta; &rarr; greedy on Q.
              </p>
            </div>
          </div>

          <div className={styles.lossCard}>
            <div className={styles.lossCardHeader}>
              <span className={styles.lossStep}>4</span>
              <span className={styles.lossCardTitle} style={{ color: '#888' }}>Target Q Update</span>
            </div>
            <div className={styles.lossCardBody}>
              <MathJax>{String.raw`\[ \hat{\theta} \leftarrow (1 - \alpha)\,\hat{\theta} + \alpha\,\theta \]`}</MathJax>
              <p className={styles.lossNote}>
                Only the Q-networks have target copies. No target V or target policy &mdash; simpler than actor-critic methods.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ==================== Training Algorithm ==================== */}
      <section>
        <h2>Training Loop</h2>

        <div className="formula-block">
          <h4>The expectile trick: implicit maximization</h4>
          <MathJax>{String.raw`\[ \tau = 0.5 \;\Rightarrow\; V(s) \approx \mathbb{E}_a[Q(s,a)] \;\text{(SARSA)} \qquad\qquad \tau \to 1 \;\Rightarrow\; V(s) \approx \max_a Q(s,a) \;\text{(Q-learning)} \]`}</MathJax>
          <p className={styles.formulaNote}>
            &tau; interpolates between SARSA and Q-learning. In practice &tau; &isin; {'{'}0.7, 0.9{'}'} works well;
            higher &tau; is needed for tasks requiring trajectory stitching (e.g. Ant Maze).
          </p>
        </div>

        <h4 style={{ marginTop: 28, marginBottom: 10 }}>Pseudocode</h4>
        <div className={styles.pseudocode}>
          <span className={styles.comment}>// Initialize</span><br />
          <span className={styles.keyword}>Input:</span> Dataset <span className={styles.param}>D</span>, expectile <span className={styles.param}>&tau;</span>, temperature <span className={styles.param}>&beta;</span><br />
          <span className={styles.keyword}>Init:</span> V<sub>&psi;</sub>, Q<sub>&theta;1</sub>, Q<sub>&theta;2</sub>, target Q&#x0302;<sub>1</sub>, Q&#x0302;<sub>2</sub>, policy &pi;<sub>&phi;</sub><br />
          <br />
          <span className={styles.comment}>// Stage 1: TD Learning (alternating V and Q updates)</span><br />
          <span className={styles.keyword}>for</span> each gradient step <span className={styles.keyword}>do</span><br />
          &nbsp;&nbsp;<span className={styles.comment}>// Value update &mdash; expectile regression</span><br />
          &nbsp;&nbsp;&psi; &larr; &psi; &minus; &eta; &middot; &nabla;<span className={styles.func}>L<sub>V</sub></span>(&psi;) &nbsp;&nbsp;<span className={styles.comment}>// L&sup2;<sub>&tau;</sub>(Q&#x0302;(s,a) &minus; V(s))</span><br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Q update &mdash; standard MSE Bellman with V target</span><br />
          &nbsp;&nbsp;&theta; &larr; &theta; &minus; &eta; &middot; &nabla;<span className={styles.func}>L<sub>Q</sub></span>(&theta;) &nbsp;&nbsp;<span className={styles.comment}>// (r + &gamma;V(s&prime;) &minus; Q(s,a))&sup2;</span><br />
          <br />
          &nbsp;&nbsp;<span className={styles.comment}>// Target update</span><br />
          &nbsp;&nbsp;&theta;&#x0302; &larr; (1 &minus; <span className={styles.param}>&alpha;</span>)&theta;&#x0302; + <span className={styles.param}>&alpha;</span>&theta;<br />
          <span className={styles.keyword}>end for</span><br />
          <br />
          <span className={styles.comment}>// Stage 2: Policy Extraction (advantage-weighted regression)</span><br />
          <span className={styles.keyword}>for</span> each gradient step <span className={styles.keyword}>do</span><br />
          &nbsp;&nbsp;&phi; &larr; &phi; &minus; &eta; &middot; &nabla;<span className={styles.func}>L<sub>&pi;</sub></span>(&phi;) &nbsp;&nbsp;<span className={styles.comment}>// exp(&beta;(Q&#x0302;(s,a) &minus; V(s))) &middot; log &pi;(a|s)</span><br />
          <span className={styles.keyword}>end for</span>
        </div>

        <InfoPanel title="Why &ldquo;Implicit&rdquo; Q-Learning?">
          <p>
            Standard Q-learning computes <strong>max<sub>a</sub> Q(s,a)</strong> explicitly in the Bellman target.
            IQL never computes this max &mdash; instead, the expectile regression on V <em>implicitly</em> approximates
            it through the asymmetric loss. The Q-function only sees dataset actions, yet still converges to near-optimal
            values via the V &harr; Q alternation.
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
                { title: 'CQL', desc: 'Kumar et al., 2020 \u2014 Conservative Q-learning; IQL eliminates the need for the CQL regularizer by avoiding OOD queries entirely' },
                { title: 'BEAR', desc: 'Kumar et al., 2019 \u2014 Support constraint via MMD; IQL sidesteps the constraint design problem altogether' },
                { title: 'AWR / AWAC', desc: 'Peng et al., 2019; Nair et al., 2020 \u2014 Advantage-weighted regression for policy extraction, used by IQL in its second stage' },
                { title: 'Expectile Regression', desc: 'Koenker & Hallock, 2001 \u2014 Asymmetric least squares for estimating distribution expectiles; the statistical core of IQL' },
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
                { title: 'Cal-QL', desc: 'Nakamoto et al., 2023 \u2014 Calibrated offline RL combining CQL-style conservatism with IQL-style efficiency for better online fine-tuning' },
                { title: 'IDQL', desc: 'Hansen-Estruch et al., 2023 \u2014 Replaces IQL\u2019s Gaussian policy with a diffusion model for more expressive policy extraction' },
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
