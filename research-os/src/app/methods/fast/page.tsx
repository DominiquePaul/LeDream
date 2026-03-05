'use client';

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './fast.module.css';

const coreIdeas = [
  {
    title: 'Frequency-Domain Action Compression',
    desc: 'Apply the Discrete Cosine Transform (DCT) to action chunks, exploiting the fact that robot actions are temporally smooth and most energy concentrates in low-frequency coefficients \u2014 the same principle behind JPEG.',
    detail: 'Robot manipulation actions are inherently smooth signals. By transforming each action dimension independently into frequency space via DCT, most of the signal energy is captured by just the first few coefficients. High-frequency coefficients are near-zero and can be quantized away, yielding a sparse matrix that compresses dramatically with BPE tokenization.',
  },
  {
    title: 'Embodiment-Agnostic Tokenization',
    desc: 'Quantile-normalize actions to [-1, 1] and tokenize in frequency space, making the representation robust across different robot embodiments, action dimensions, and control frequencies.',
    detail: 'By normalizing each action dimension to a standard range and operating in the frequency domain, FAST decouples the tokenization from specific robot hardware. The same tokenizer works whether the robot has 6 or 12 joints, runs at 10 Hz or 50 Hz. BPE further adapts the vocabulary to common patterns in the training data.',
  },
];

const keyFeatures = [
  { title: 'DCT + BPE Pipeline', desc: 'Normalize \u2192 DCT \u2192 quantize \u2192 flatten \u2192 BPE yields dramatic token compression over naive per-timestep tokenization' },
  { title: 'Sparsity from Smoothness', desc: 'Robot actions are smooth, so high-frequency DCT coefficients are near-zero \u2014 quantization produces a sparse matrix that compresses well' },
  { title: 'Fewer Tokens = Faster Inference', desc: 'Compressed action tokens mean shorter sequences for autoregressive models, giving faster inference and a stronger learning signal per token' },
];

/* ================================================================
   MATH HELPERS
   ================================================================ */

function dct1d(signal: number[]): number[] {
  const N = signal.length;
  const out: number[] = [];
  for (let k = 0; k < N; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += signal[n] * Math.cos((Math.PI / N) * (n + 0.5) * k);
    }
    out.push(sum * (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N)));
  }
  return out;
}

function idct1d(coeffs: number[]): number[] {
  const N = coeffs.length;
  const out: number[] = [];
  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const scale = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
      sum += scale * coeffs[k] * Math.cos((Math.PI / N) * (n + 0.5) * k);
    }
    out.push(sum);
  }
  return out;
}

/**
 * Generate manipulation-like action signals.
 * dimIndex controls the "personality" of each dimension:
 *   0,1  — piecewise-constant with sudden jumps (gripper / binary-ish)
 *   2    — fast ramp-and-hold (like a reach then stop)
 *   rest — smoother but still with sharp kinks (joint angles)
 */
function generateSignal(length: number, seed: number, dimIndex = 0): number[] {
  let s = seed;
  const rand = () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s / 2147483647) * 2 - 1;
  };
  // consume a few values to decorrelate across seeds
  rand(); rand(); rand();

  const signal: number[] = [];
  const mode = dimIndex % 5;

  if (mode <= 1) {
    // ---- gripper / binary: hold a value, then snap to another ----
    const numSegments = 2 + Math.floor((rand() + 1) * 1.5); // 2-4 segments
    const levels: number[] = [];
    for (let j = 0; j < numSegments; j++) levels.push(rand() * 0.9);
    const breakpoints = [0];
    for (let j = 1; j < numSegments; j++) {
      breakpoints.push(Math.floor((j / numSegments) * length) + Math.floor(rand() * 1.5));
    }
    breakpoints.push(length);
    for (let i = 0; i < length; i++) {
      let seg = 0;
      for (let j = 1; j < breakpoints.length; j++) {
        if (i >= breakpoints[j]) seg = j; else break;
      }
      signal.push(Math.max(-1, Math.min(1, levels[Math.min(seg, levels.length - 1)])));
    }
  } else if (mode === 2) {
    // ---- ramp-and-hold: fast move then plateau ----
    const start = rand() * 0.5;
    const end = rand() * 0.8;
    const rampEnd = Math.floor(length * (0.2 + (rand() + 1) * 0.15));
    for (let i = 0; i < length; i++) {
      if (i <= rampEnd) {
        const t = rampEnd > 0 ? i / rampEnd : 1;
        signal.push(Math.max(-1, Math.min(1, start + (end - start) * t)));
      } else {
        // small high-freq jitter on the plateau (controller noise)
        signal.push(Math.max(-1, Math.min(1, end + rand() * 0.04)));
      }
    }
  } else {
    // ---- piecewise-linear with sharp corners (joint trajectory) ----
    const numWaypoints = 3 + Math.floor((rand() + 1) * 1.5);
    const waypoints: { t: number; v: number }[] = [{ t: 0, v: rand() * 0.6 }];
    for (let j = 1; j < numWaypoints; j++) {
      waypoints.push({
        t: j / (numWaypoints - 1),
        v: rand() * 0.8,
      });
    }
    for (let i = 0; i < length; i++) {
      const t = i / Math.max(length - 1, 1);
      // find surrounding waypoints
      let lo = 0;
      for (let j = 1; j < waypoints.length; j++) {
        if (waypoints[j].t <= t) lo = j; else break;
      }
      const hi = Math.min(lo + 1, waypoints.length - 1);
      const segLen = waypoints[hi].t - waypoints[lo].t;
      const frac = segLen > 0 ? (t - waypoints[lo].t) / segLen : 0;
      const v = waypoints[lo].v + (waypoints[hi].v - waypoints[lo].v) * frac;
      signal.push(Math.max(-1, Math.min(1, v)));
    }
  }
  return signal;
}

function simulateBPE(sequence: number[]): number[] {
  const tokens: number[] = [];
  let i = 0;
  while (i < sequence.length) {
    if (sequence[i] === 0) {
      let count = 0;
      while (i < sequence.length && sequence[i] === 0 && count < 8) {
        count++;
        i++;
      }
      tokens.push(500 + count);
    } else {
      tokens.push(sequence[i] + 256);
      i++;
    }
  }
  return tokens;
}

/* ================================================================
   COLOR HELPERS — all rounded to avoid hydration mismatches
   ================================================================ */

function coeffBg(value: number, max: number): string {
  if (value === 0) return 'rgba(255,255,255,0.02)';
  const t = Math.min(Math.abs(value) / Math.max(max, 1), 1);
  const a = Math.round((0.1 + t * 0.4) * 100) / 100;
  return value > 0 ? `rgba(196,136,77,${a})` : `rgba(96,165,250,${a})`;
}

function coeffText(value: number, max: number): string {
  if (Math.abs(value) > max * 0.15) return value > 0 ? '#c4884d' : '#60a5fa';
  return '#555';
}

/* ================================================================
   COMPONENT
   ================================================================ */

export default function FASTPage() {
  const [scaleGamma, setScaleGamma] = useState(10);
  const [numTimesteps, setNumTimesteps] = useState(8);
  const [numDimensions, setNumDimensions] = useState(5);
  const [seed, setSeed] = useState(42);

  const signalCanvasRef = useRef<HTMLCanvasElement>(null);
  const freqCanvasRef = useRef<HTMLCanvasElement>(null);
  const { width: sigW, height: sigH } = useCanvasResize(signalCanvasRef);
  const { width: freqW, height: freqH } = useCanvasResize(freqCanvasRef);

  /* ---- derived data ---- */
  const actionChunk = useMemo(() => {
    const chunk: number[][] = [];
    for (let d = 0; d < numDimensions; d++) {
      chunk.push(generateSignal(numTimesteps, seed * (d + 1) + d * 137, d));
    }
    return chunk;
  }, [numTimesteps, numDimensions, seed]);

  const dctCoeffs = useMemo(() => actionChunk.map((dim) => dct1d(dim)), [actionChunk]);

  const quantized = useMemo(
    () => dctCoeffs.map((dim) => dim.map((c) => Math.round(c * scaleGamma))),
    [dctCoeffs, scaleGamma],
  );

  const flattened = useMemo(() => {
    const flat: number[] = [];
    for (let t = 0; t < numTimesteps; t++) {
      for (let d = 0; d < numDimensions; d++) {
        flat.push(quantized[d]?.[t] ?? 0);
      }
    }
    return flat;
  }, [quantized, numTimesteps, numDimensions]);

  const bpeTokens = useMemo(() => simulateBPE(flattened), [flattened]);

  const reconstructed = useMemo(
    () => quantized.map((dim) => idct1d(dim.map((v) => v / scaleGamma))),
    [quantized, scaleGamma],
  );

  /* ---- stats ---- */
  const naiveTokenCount = numTimesteps * numDimensions;
  const fastTokenCount = bpeTokens.length;
  const compressionRatio = naiveTokenCount / Math.max(fastTokenCount, 1);
  const nonZeroCount = flattened.filter((v) => v !== 0).length;
  const sparsity = ((1 - nonZeroCount / flattened.length) * 100).toFixed(0);

  const maxQuantized = useMemo(() => {
    let mx = 0;
    for (const row of quantized) for (const v of row) mx = Math.max(mx, Math.abs(v));
    return mx;
  }, [quantized]);

  const maxDctCoeff = useMemo(() => {
    let mx = 0;
    for (const row of dctCoeffs) for (const v of row) mx = Math.max(mx, Math.abs(v));
    return mx || 1;
  }, [dctCoeffs]);

  const DIM_COLORS = ['#c4884d', '#3b82f6', '#10b981', '#ef4444', '#8b7ec8', '#06b6d4', '#ec4899'];

  /* ---- draw signal canvas ---- */
  const drawSignal = useCallback(() => {
    const canvas = signalCanvasRef.current;
    if (!canvas || sigW === 0 || sigH === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const pad = { left: 40, right: 16, top: 14, bottom: 24 };
    const pw = sigW - pad.left - pad.right;
    const ph = sigH - pad.top - pad.bottom;
    ctx.clearRect(0, 0, sigW, sigH);

    // grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let y = -1; y <= 1; y += 0.5) {
      const cy = pad.top + ph * (1 - (y + 1) / 2);
      ctx.beginPath();
      ctx.moveTo(pad.left, cy);
      ctx.lineTo(sigW - pad.right, cy);
      ctx.stroke();
    }

    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    for (const y of [-1, 0, 1]) {
      ctx.fillText(y.toString(), pad.left - 6, pad.top + ph * (1 - (y + 1) / 2) + 3);
    }
    ctx.textAlign = 'center';
    for (let t = 0; t < numTimesteps; t++) {
      ctx.fillText(`t${t}`, pad.left + (t / Math.max(numTimesteps - 1, 1)) * pw, sigH - 6);
    }

    for (let d = 0; d < numDimensions; d++) {
      const color = DIM_COLORS[d % DIM_COLORS.length];

      // original
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.8;
      for (let t = 0; t < numTimesteps; t++) {
        const cx = pad.left + (t / Math.max(numTimesteps - 1, 1)) * pw;
        const cy = pad.top + ph * (1 - (actionChunk[d][t] + 1) / 2);
        t === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      // reconstructed (dashed)
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.3;
      ctx.setLineDash([4, 3]);
      for (let t = 0; t < numTimesteps; t++) {
        const cx = pad.left + (t / Math.max(numTimesteps - 1, 1)) * pw;
        const val = reconstructed[d]?.[t] ?? 0;
        const cy = pad.top + ph * (1 - (Math.max(-1, Math.min(1, val)) + 1) / 2);
        t === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // dots
      for (let t = 0; t < numTimesteps; t++) {
        const cx = pad.left + (t / Math.max(numTimesteps - 1, 1)) * pw;
        const cy = pad.top + ph * (1 - (actionChunk[d][t] + 1) / 2);
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }

    // legend
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    let ly = pad.top + 10;
    for (let d = 0; d < Math.min(numDimensions, 5); d++) {
      ctx.fillStyle = DIM_COLORS[d % DIM_COLORS.length];
      ctx.fillRect(pad.left + 4, ly - 4, 10, 2);
      ctx.fillText(`dim ${d}`, pad.left + 18, ly);
      ly += 12;
    }
    ctx.fillStyle = '#555';
    ctx.fillText('dashed = reconstructed', pad.left + 4, ly);
  }, [sigW, sigH, actionChunk, reconstructed, numTimesteps, numDimensions, DIM_COLORS]);

  /* ---- draw freq canvas ---- */
  const drawFreq = useCallback(() => {
    const canvas = freqCanvasRef.current;
    if (!canvas || freqW === 0 || freqH === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const pad = { left: 40, right: 16, top: 14, bottom: 24 };
    const pw = freqW - pad.left - pad.right;
    const ph = freqH - pad.top - pad.bottom;
    ctx.clearRect(0, 0, freqW, freqH);

    const yCenter = pad.top + ph / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, yCenter);
    ctx.lineTo(freqW - pad.right, yCenter);
    ctx.stroke();

    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const freqLabels = ['DC', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15'];
    for (let k = 0; k < numTimesteps; k++) {
      ctx.fillText(freqLabels[k] || `f${k}`, pad.left + ((k + 0.5) / numTimesteps) * pw, freqH - 6);
    }

    ctx.textAlign = 'right';
    ctx.fillText(`${maxDctCoeff.toFixed(1)}`, pad.left - 6, pad.top + 6);
    ctx.fillText('0', pad.left - 6, yCenter + 3);
    ctx.fillText(`-${maxDctCoeff.toFixed(1)}`, pad.left - 6, pad.top + ph);

    const barWidth = (pw / numTimesteps) / (numDimensions + 1);
    for (let d = 0; d < numDimensions; d++) {
      ctx.fillStyle = DIM_COLORS[d % DIM_COLORS.length];
      ctx.globalAlpha = 0.7;
      for (let k = 0; k < numTimesteps; k++) {
        const coeff = dctCoeffs[d][k];
        const cx = pad.left + ((k + 0.5) / numTimesteps) * pw - (numDimensions * barWidth) / 2 + d * barWidth;
        const barH = (Math.abs(coeff) / maxDctCoeff) * (ph / 2);
        ctx.fillRect(cx, coeff >= 0 ? yCenter - barH : yCenter, barWidth - 1, barH);
      }
      ctx.globalAlpha = 1;
    }
  }, [freqW, freqH, dctCoeffs, numTimesteps, numDimensions, maxDctCoeff, DIM_COLORS]);

  useEffect(() => { drawSignal(); }, [drawSignal]);
  useEffect(() => { drawFreq(); }, [drawFreq]);

  /* ================================================================
     RENDER — all steps visible at once
     ================================================================ */
  return (
    <div className="method-page">
      <h1 className={styles.title}>FAST Tokenization</h1>
      <p className={styles.subtitle}>
        Frequency-space Action Sequence Tokenization &mdash; Pertsch, Stachowicz et al. &middot; 2025
      </p>
      <div className={styles.linkRow}>
        <a href="https://pi.website/research/fast" target="_blank" rel="noopener noreferrer" className={styles.paperLink}>
          <span className={styles.linkIcon}>&rarr;</span> pi.website/research/fast
        </a>
      </div>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ==================== Controls ==================== */}
      <div className={styles.card}>
        <div className={styles.controls}>
          <div className={styles.controlGroup}>
            <div className={styles.controlLabel}>
              Timesteps (H) <span className={styles.controlValue}>{numTimesteps}</span>
            </div>
            <input type="range" className={styles.controlSlider} min={4} max={16} step={1} value={numTimesteps}
              onChange={(e) => setNumTimesteps(parseInt(e.target.value, 10))} />
          </div>
          <div className={styles.controlGroup}>
            <div className={styles.controlLabel}>
              Action Dimensions (D) <span className={styles.controlValue}>{numDimensions}</span>
            </div>
            <input type="range" className={styles.controlSlider} min={2} max={7} step={1} value={numDimensions}
              onChange={(e) => setNumDimensions(parseInt(e.target.value, 10))} />
          </div>
          <div className={styles.controlGroup}>
            <div className={styles.controlLabel}>
              Scale &gamma; <span className={styles.controlValue}>{scaleGamma}</span>
            </div>
            <input type="range" className={styles.controlSlider} min={1} max={30} step={1} value={scaleGamma}
              onChange={(e) => setScaleGamma(parseInt(e.target.value, 10))} />
          </div>
          <div className={styles.controlGroup} style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button className={styles.randomBtn} onClick={() => setSeed((s) => s + 7)}>
              New random signal
            </button>
          </div>
        </div>
      </div>

      {/* ==================== STEP 1 — Normalize ==================== */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          <div className={styles.stepBadge}>1</div>
          Normalize action chunk to [-1, 1]
        </div>
        <p className={styles.cardDescription}>
          Each action dimension is quantile-normalized to [-1, 1]. This makes the tokenizer robust across different robot embodiments. Below: {numDimensions} dimensions &times; {numTimesteps} timesteps. Dashed lines show the reconstruction after compression.
        </p>
        <div className={styles.signalCanvasWrap}>
          <canvas ref={signalCanvasRef} className={styles.signalCanvas} />
        </div>
      </div>

      {/* ==================== STEP 2 — DCT ==================== */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          <div className={styles.stepBadge}>2</div>
          Discrete Cosine Transform
        </div>
        <p className={styles.cardDescription}>
          Each dimension is independently transformed into frequency components. Low frequencies (left) capture the overall motion shape. Because robot actions are smooth, most energy concentrates in the first few coefficients — the same principle behind JPEG compression.
        </p>
        <div className={styles.freqCanvasWrap}>
          <canvas ref={freqCanvasRef} className={styles.freqCanvas} />
        </div>
        <div className={styles.signalLabel}>DCT coefficient matrix (D &times; H)</div>
        <div className={styles.matrixContainer}>
          <div className={styles.matrixWrapper}>
            <div className={styles.matrixColumnLabels}>
              {Array.from({ length: numTimesteps }, (_, k) => (
                <div key={k} className={styles.matrixColumnLabel}>{k === 0 ? 'DC' : `f${k}`}</div>
              ))}
            </div>
            {dctCoeffs.map((dim, d) => (
              <div key={d} className={styles.matrixRow}>
                <div className={styles.matrixRowLabel}>d{d}</div>
                {dim.map((val, k) => (
                  <div key={k} className={styles.matrixCell}
                    style={{ background: coeffBg(val, maxDctCoeff), color: coeffText(val, maxDctCoeff) }}>
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ==================== STEP 3 — Quantize ==================== */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          <div className={styles.stepBadge}>3</div>
          Scale by &gamma;={scaleGamma} &amp; round to integers
        </div>
        <p className={styles.cardDescription}>
          Coefficients are multiplied by &gamma; and rounded. Small values become zero, creating a sparse matrix. Higher &gamma; preserves more detail but produces more tokens. Currently <strong>{sparsity}%</strong> of entries are zero.
        </p>
        <div className={styles.matrixContainer}>
          <div className={styles.matrixWrapper}>
            <div className={styles.matrixColumnLabels}>
              {Array.from({ length: numTimesteps }, (_, k) => (
                <div key={k} className={styles.matrixColumnLabel}>{k === 0 ? 'DC' : `f${k}`}</div>
              ))}
            </div>
            {quantized.map((dim, d) => (
              <div key={d} className={styles.matrixRow}>
                <div className={styles.matrixRowLabel}>d{d}</div>
                {dim.map((val, k) => (
                  <div key={k}
                    className={val === 0 ? styles.matrixCellZero : styles.matrixCellNonZero}
                    style={val !== 0 ? { background: coeffBg(val, maxQuantized), color: val > 0 ? '#c4884d' : '#60a5fa' } : undefined}>
                    {val}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className={styles.statsBar}>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#c4884d' }}>{nonZeroCount}</div>
            <div className={styles.statLabel}>Non-zero</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#555' }}>{flattened.length - nonZeroCount}</div>
            <div className={styles.statLabel}>Zero</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#10b981' }}>{sparsity}%</div>
            <div className={styles.statLabel}>Sparsity</div>
          </div>
        </div>
      </div>

      {/* ==================== STEP 4 — Flatten ==================== */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          <div className={styles.stepBadge}>4</div>
          Flatten column-first (low frequencies first)
        </div>
        <p className={styles.cardDescription}>
          The matrix is read column-by-column, interleaving dimensions. Low-frequency components come first, giving the autoregressive model the overall trajectory shape before details. {flattened.length} integers total.
        </p>
        <div className={styles.flatSequence}>
          {flattened.slice(0, 60).map((val, i) => (
            <div key={i} className={val === 0 ? styles.flatTokenZero : styles.flatTokenNonZero}>
              {val}
            </div>
          ))}
          {flattened.length > 60 && <span className={styles.flatEllipsis}>&hellip; +{flattened.length - 60}</span>}
        </div>
      </div>

      {/* ==================== STEP 5 — BPE ==================== */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          <div className={styles.stepBadge}>5</div>
          Byte Pair Encoding &rarr; final action tokens
        </div>
        <p className={styles.cardDescription}>
          BPE merges repeated zeros and common patterns into single tokens. {flattened.length} integers compress to {bpeTokens.length} tokens &mdash; <strong>{compressionRatio.toFixed(1)}&times;</strong> compression. Fewer tokens = faster inference + stronger learning signal per token.
        </p>
        <div className={styles.bpeTokens}>
          {bpeTokens.map((tok, i) => (
            <div key={i} className={styles.bpeToken}>{tok}</div>
          ))}
        </div>

        <div className={styles.statsBar}>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#888' }}>{naiveTokenCount}</div>
            <div className={styles.statLabel}>Naive tokens</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#c4884d' }}>{fastTokenCount}</div>
            <div className={styles.statLabel}>FAST tokens</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue} style={{ color: '#10b981' }}>{compressionRatio.toFixed(1)}&times;</div>
            <div className={styles.statLabel}>Compression</div>
          </div>
        </div>

        {/* comparison bars */}
        <div className={styles.comparisonBar}>
          <div className={styles.comparisonRow}>
            <span className={styles.comparisonLabel}>Naive</span>
            <div className={styles.comparisonBarTrack}>
              <div className={styles.comparisonBarFill} style={{ width: '100%', background: '#555' }} />
            </div>
            <span className={styles.comparisonValue}>{naiveTokenCount}</span>
          </div>
          <div className={styles.comparisonRow}>
            <span className={styles.comparisonLabel} style={{ color: '#c4884d' }}>FAST</span>
            <div className={styles.comparisonBarTrack}>
              <div className={styles.comparisonBarFill}
                style={{ width: `${Math.round((fastTokenCount / naiveTokenCount) * 100)}%`, background: '#c4884d' }} />
            </div>
            <span className={styles.comparisonValue} style={{ color: '#c4884d' }}>{fastTokenCount}</span>
          </div>
        </div>
      </div>

      {/* ==================== Paper Lineage ==================== */}
      <section>
        <h2>Paper Lineage</h2>
        <div className={styles.lineageGrid}>
          <div className={styles.lineageColumn}>
            <h4 style={{ color: '#f59e0b' }}>Builds On</h4>
            <div className={styles.lineageList}>
              {[
                { title: 'DCT / JPEG', desc: 'Ahmed et al., 1974 \u2014 The Discrete Cosine Transform that FAST applies to action sequences exploits the same smoothness principle behind JPEG image compression' },
                { title: 'BPE (Byte Pair Encoding)', desc: 'Sennrich et al., 2016 \u2014 The subword tokenization algorithm that FAST uses to further compress quantized DCT coefficients into compact token sequences' },
                { title: 'Action Chunking (ACT)', desc: 'Zhao et al., 2023 \u2014 Predicting chunks of future actions rather than single timesteps; FAST compresses these chunks in frequency space' },
                { title: 'RT-2 / Octo', desc: 'Brohan et al., 2023 / Ghosh et al., 2024 \u2014 Autoregressive vision-language-action models that FAST\u2019s tokenizer is designed to plug into' },
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
                { title: '\u03C0\u2080 (pi-zero)', desc: 'Physical Intelligence, 2024 \u2014 Uses FAST tokenization as part of a large-scale foundation model for robot manipulation' },
                { title: '\u03C0\u2080.5', desc: 'Physical Intelligence, 2025 \u2014 Extends the \u03C0\u2080 foundation model with vision-language reasoning, retaining FAST action tokens for dexterous control' },
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
