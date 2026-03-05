'use client';

import { useState, useRef, useEffect } from 'react';
import { MathJax } from 'better-react-mathjax';
import { useCanvasResize } from '@/hooks/useCanvasResize';
import { useAnimationFrame } from '@/hooks/useAnimationFrame';
import { CoreIdeasAndFeatures } from '@/components/CoreIdeasAndFeatures/CoreIdeasAndFeatures';
import styles from './robot-controllers.module.css';

const coreIdeas = [
  {
    title: 'Impedance Control as Virtual Mass-Spring-Damper',
    desc: 'Impedance control does not enforce position \u2014 it enforces a mechanical relationship between displacement and force. Small errors produce gentle forces, enabling safe contact-rich manipulation.',
    detail: 'Unlike position control which drives joints to exact targets (behaving like infinite stiffness), impedance control makes the robot behave like a virtual mass-spring-damper around a target pose. The key equation F = K(x_des - x) + D(dx_des - dx) + M(ddx_des - ddx) lets you tune how stiff or compliant the robot is. This is essential for contact-rich tasks where rigid controllers cause force spikes and damage.',
  },
  {
    title: 'Controller Choice Dominates RL Performance',
    desc: 'The low-level controller can matter more than the learned policy itself. Wrong controller makes even a perfect policy fail on contact-rich manipulation tasks.',
    detail: 'SERL demonstrated that for contact-rich tasks like PCB insertion, an overly stiff controller bends fragile pins while an overly compliant one cannot position the part. Re-tuning stiffness K alone (without retraining the policy) could flip a task from 0% to >90% success. The controller runs at 1 kHz inner loop while the RL policy runs at 10-50 Hz outer loop \u2014 this 1:100 ratio demands careful coordination.',
  },
  {
    title: 'Error Clamping for Force Safety',
    desc: 'SERL clamps the pose error |e| \u2264 \u0394 before it reaches the impedance controller, providing a hard upper bound on interaction force without sacrificing gain accuracy.',
    detail: 'When the RL policy sends large setpoint jumps, the error e = p - p_ref can become dangerously large, generating excessive forces on contact. SERL\u2019s solution: clamp the error so |e| \u2264 \u0394. This gives a guaranteed force bound |F_max| \u2264 k_p\u00b7|\u0394| + 2\u00b7k_d\u00b7|\u0394|\u00b7f, where f is the control frequency. By choosing \u0394, you get safety without reducing controller gains.',
  },
];

const keyFeatures = [
  { title: 'Impedance \u2260 Position', desc: 'Impedance control regulates the force-motion relationship, not the position. This enables safe contact.' },
  { title: 'K Dominates', desc: 'Stiffness gain K is the single most impactful parameter for contact-rich RL performance.' },
  { title: 'Controller > Policy', desc: 'Wrong controller can make even a perfect policy fail. Get the low-level right first.' },
];

/* ================================================================
   Controller type info data
   ================================================================ */
const ctrlInfo: Record<string, { title: string; desc: string }> = {
  position: {
    title: 'Position Control',
    desc: 'The simplest mode: the controller drives joints to reach exact target positions (or Cartesian poses). Very stiff inner loop \u2014 the robot resists any deviation. Great for pick-and-place in free space, but brittle on contact: any position error during insertion causes force spikes. Essentially impedance control with K \u2192 \u221e.',
  },
  velocity: {
    title: 'Velocity Control',
    desc: 'The controller tracks a desired velocity rather than a position. Useful when the task specifies motion rates (e.g. wiping, scanning). On contact the robot doesn\u2019t build up large position error \u2014 but it also has no notion of a \u201ctarget pose\u201d to spring back to. Often combined with an outer position loop.',
  },
  torque: {
    title: 'Torque / Force Control',
    desc: 'Directly commands joint torques or Cartesian forces. Maximum flexibility \u2014 you have full control over every force the robot exerts. Required for gravity compensation, dexterous manipulation, and human-safe robots. However, it demands a very accurate dynamics model and runs at high frequency (\u22651 kHz) to remain stable.',
  },
  impedance: {
    title: 'Impedance Control',
    desc: 'Makes the robot behave like a virtual mass\u2013spring\u2013damper around a target pose. The key insight: it does not enforce position, it enforces a mechanical relationship between displacement and force. Small errors produce gentle forces; large errors produce stronger restoring forces \u2014 all tunable via K, D, M. This is the go-to for contact-rich manipulation in SERL.',
  },
};

/* ================================================================
   SERL factor info data
   ================================================================ */
const serlInfo: Record<string, { title: string; desc: string }> = {
  type: {
    title: 'OEM Controller Type',
    desc: 'Most robot arms ship with a default position or velocity controller. Swapping to torque or impedance control often requires firmware changes or a custom driver. SERL found that impedance control is critical for contact-rich tasks \u2014 the OEM position controller made the same policy fail consistently on insertion tasks that the impedance controller handled easily.',
  },
  params: {
    title: 'Impedance Parameters (K, D)',
    desc: 'Even with impedance control enabled, the stiffness and damping values must be carefully tuned per task. PCB insertion needs lower K to avoid bending pins; peg-in-hole needs moderate K to maintain alignment force. SERL showed that re-tuning K alone (without retraining the policy) could flip a task from 0% to >90% success.',
  },
  rate: {
    title: 'Update Rate & Inner-Loop Latency',
    desc: 'The impedance controller runs in a fast inner loop (typically 1 kHz), while the RL policy runs at a slower outer loop (10\u201350 Hz). If the inner loop is too slow or has jitter, the robot cannot react to contact forces in time, causing instability. The ratio between policy rate and controller rate determines how smoothly learned actions are executed.',
  },
};

/* ================================================================
   Physics helpers
   ================================================================ */
interface SimState {
  x: number;
  v: number;
  contactForce: number;
}

const WALL_X = 0.6;
const TARGET_X = 0.8;
const DT = 1 / 60;
const MASS = 1.0;

function physicsStep(state: SimState, K: number, D: number) {
  const error = TARGET_X - state.x;
  const velError = 0 - state.v;
  let F = K * error + D * velError;

  state.contactForce = 0;
  if (state.x >= WALL_X) {
    const wallF = 2000 * (state.x - WALL_X);
    F -= wallF;
    state.contactForce = wallF;
  }

  const a = F / MASS;
  state.v += a * DT;
  state.v *= 0.995;
  state.x += state.v * DT;

  if (state.x < 0) { state.x = 0; state.v = 0; }
  if (state.x > 1) { state.x = 1; state.v = 0; }
}

function drawSim(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  state: SimState,
) {
  const pad = { left: 30, right: 30, top: 40, bottom: 50 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;
  const toX = (v: number) => pad.left + v * pw;
  const cy = pad.top + ph * 0.5;

  ctx.clearRect(0, 0, w, h);

  // Floor line
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, cy + 30);
  ctx.lineTo(w - pad.right, cy + 30);
  ctx.stroke();

  // Wall
  const wallPx = toX(WALL_X);
  ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
  ctx.fillRect(wallPx, pad.top, w - pad.right - wallPx, ph);
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(wallPx, pad.top);
  ctx.lineTo(wallPx, pad.top + ph);
  ctx.stroke();
  // Hatching
  ctx.strokeStyle = 'rgba(239,68,68,0.3)';
  ctx.lineWidth = 1;
  for (let i = 0; i < ph; i += 10) {
    ctx.beginPath();
    ctx.moveTo(wallPx, pad.top + i);
    ctx.lineTo(wallPx + 12, pad.top + i + 10);
    ctx.stroke();
  }
  ctx.fillStyle = '#ef4444';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Wall (obstacle)', wallPx + 35, pad.top + 15);

  // Target (dashed line)
  const targetPx = toX(TARGET_X);
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(targetPx, pad.top);
  ctx.lineTo(targetPx, pad.top + ph);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b';
  ctx.font = '10px sans-serif';
  ctx.fillText('x_des (target)', targetPx, pad.top + ph + 15);

  // Spring visualization
  const robPx = toX(state.x);
  const springStartX = robPx + 20;
  const springEndX = Math.min(targetPx, wallPx) - 5;
  if (springEndX > springStartX + 10) {
    const nCoils = 8;
    const coilW = (springEndX - springStartX) / nCoils;
    const coilAmp = 8;
    ctx.strokeStyle = 'rgba(16,185,129,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(springStartX, cy);
    for (let i = 0; i < nCoils; i++) {
      const sx = springStartX + i * coilW;
      ctx.lineTo(sx + coilW * 0.25, cy - coilAmp);
      ctx.lineTo(sx + coilW * 0.75, cy + coilAmp);
    }
    ctx.lineTo(springEndX, cy);
    ctx.stroke();
  }

  // Robot (circle)
  ctx.beginPath();
  ctx.arc(robPx, cy, 18, 0, Math.PI * 2);
  ctx.fillStyle = '#3b82f6';
  ctx.fill();
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = 'white';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('EE', robPx, cy + 4);

  // Contact force arrow
  if (state.contactForce > 1) {
    const arrowLen = Math.min(state.contactForce * 0.3, 80);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(wallPx, cy);
    ctx.lineTo(wallPx - arrowLen, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(wallPx - arrowLen, cy);
    ctx.lineTo(wallPx - arrowLen + 8, cy - 5);
    ctx.lineTo(wallPx - arrowLen + 8, cy + 5);
    ctx.closePath();
    ctx.fillStyle = '#ef4444';
    ctx.fill();
    ctx.fillStyle = '#ef4444';
    ctx.font = '10px sans-serif';
    ctx.fillText(state.contactForce.toFixed(0) + ' N', wallPx - arrowLen / 2, cy - 12);
  }

  // Labels
  ctx.fillStyle = '#3b82f6';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('robot end-effector', robPx, cy + 40);
}

/* ================================================================
   Rigid / Springy canvas drawing
   ================================================================ */
function drawRigidCanvas(ctx: CanvasRenderingContext2D, w: number, h: number, animT: number) {
  ctx.clearRect(0, 0, w, h);
  const cy = h / 2;
  const wallX = w * 0.65;

  // Wall
  ctx.fillStyle = 'rgba(239,68,68,0.1)';
  ctx.fillRect(wallX, 10, w - wallX - 10, h - 20);
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(wallX, 10);
  ctx.lineTo(wallX, h - 10);
  ctx.stroke();

  // Rigid rod
  const rodTip = wallX - 2;
  const rodBase = 30;
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(rodBase, cy);
  ctx.lineTo(rodTip, cy);
  ctx.stroke();
  ctx.lineCap = 'butt';

  // Impact sparks
  const sparkPhase = (animT * 3) % 1;
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI - Math.PI / 2;
    const dist = 5 + sparkPhase * 20;
    const sx = wallX + Math.cos(angle) * dist;
    const sy = cy + Math.sin(angle) * dist;
    ctx.fillStyle = `rgba(239, 68, 68, ${1 - sparkPhase})`;
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Force arrow (large, constant)
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(wallX + 5, cy);
  ctx.lineTo(wallX + 40, cy);
  ctx.stroke();
  ctx.fillStyle = '#ef4444';
  ctx.beginPath();
  ctx.moveTo(wallX + 40, cy);
  ctx.lineTo(wallX + 34, cy - 5);
  ctx.lineTo(wallX + 34, cy + 5);
  ctx.closePath();
  ctx.fill();
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('HIGH F', wallX + 25, cy - 10);

  // Circle at tip
  ctx.beginPath();
  ctx.arc(rodTip, cy, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#666';
  ctx.fill();
}

function drawSpringyCanvas(ctx: CanvasRenderingContext2D, w: number, h: number, animT: number) {
  ctx.clearRect(0, 0, w, h);
  const cy = h / 2;
  const wallX = w * 0.65;

  // Wall
  ctx.fillStyle = 'rgba(16,185,129,0.05)';
  ctx.fillRect(wallX, 10, w - wallX - 10, h - 20);
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(wallX, 10);
  ctx.lineTo(wallX, h - 10);
  ctx.stroke();

  // Spring arm
  const rodBase = 30;
  const springStart = w * 0.35;
  const tipX = wallX - 2 - Math.sin(animT * 2) * 8;

  // Rigid part
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(rodBase, cy);
  ctx.lineTo(springStart, cy);
  ctx.stroke();
  ctx.lineCap = 'butt';

  // Spring coils
  const nCoils = 6;
  const coilW = (tipX - springStart) / nCoils;
  const coilAmp = 7;
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(springStart, cy);
  for (let i = 0; i < nCoils; i++) {
    const sx = springStart + i * coilW;
    ctx.lineTo(sx + coilW * 0.25, cy - coilAmp);
    ctx.lineTo(sx + coilW * 0.75, cy + coilAmp);
  }
  ctx.lineTo(tipX, cy);
  ctx.stroke();

  // Tip circle
  ctx.beginPath();
  ctx.arc(tipX, cy, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#10b981';
  ctx.fill();

  // Gentle force arrow (small)
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(wallX + 3, cy);
  ctx.lineTo(wallX + 16, cy);
  ctx.stroke();
  ctx.fillStyle = '#10b981';
  ctx.beginPath();
  ctx.moveTo(wallX + 16, cy);
  ctx.lineTo(wallX + 12, cy - 3);
  ctx.lineTo(wallX + 12, cy + 3);
  ctx.closePath();
  ctx.fill();
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('low F', wallX + 12, cy - 10);
}

/* ================================================================
   Clamp canvas drawing
   ================================================================ */
const CLAMP_KP = 80;
const CLAMP_KD = 20;
const CLAMP_FREQ = 1000;

function drawClampViz(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  delta: number,
) {
  const pad = { left: 55, right: 20, top: 20, bottom: 35 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;

  const eMax = 0.12;
  const fMaxPlot = CLAMP_KP * 0.12 + 2 * CLAMP_KD * 0.12 * CLAMP_FREQ;

  const toX = (e: number) => pad.left + (e / eMax) * pw;
  const toY = (f: number) => h - pad.bottom - (f / fMaxPlot) * ph;

  ctx.clearRect(0, 0, w, h);

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
  ctx.fillText('Pose error |e| (m)', w / 2, h - 6);

  ctx.save();
  ctx.translate(14, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Force (N)', 0, 0);
  ctx.restore();

  // X axis ticks
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  for (let e = 0; e <= 0.12; e += 0.02) {
    const x = toX(e);
    ctx.fillStyle = '#555';
    ctx.fillText((e * 1000).toFixed(0) + 'mm', x, h - pad.bottom + 14);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, h - pad.bottom);
    ctx.stroke();
  }

  // Unclamped force line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(239,68,68,0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  for (let i = 0; i <= pw; i++) {
    const e = (i / pw) * eMax;
    const f = CLAMP_KP * e + 2 * CLAMP_KD * e * CLAMP_FREQ;
    const cx2 = toX(e);
    const cy2 = toY(f);
    i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Clamped force line
  const fBound = CLAMP_KP * delta + 2 * CLAMP_KD * delta * CLAMP_FREQ;
  ctx.beginPath();
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 3;
  for (let i = 0; i <= pw; i++) {
    const e = (i / pw) * eMax;
    const f = e <= delta
      ? CLAMP_KP * e + 2 * CLAMP_KD * e * CLAMP_FREQ
      : fBound;
    const cx2 = toX(e);
    const cy2 = toY(f);
    i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
  }
  ctx.stroke();

  // Delta vertical line
  const deltaX = toX(delta);
  ctx.strokeStyle = '#ec4899';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(deltaX, pad.top);
  ctx.lineTo(deltaX, h - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#ec4899';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('|e| = \u0394', deltaX, pad.top + 14);

  // Force bound horizontal line
  const fBoundY = toY(fBound);
  ctx.strokeStyle = 'rgba(16,185,129,0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(pad.left, fBoundY);
  ctx.lineTo(w - pad.right, fBoundY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#10b981';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('F_max = ' + fBound.toFixed(0) + ' N', pad.left + 5, fBoundY - 6);

  // Danger zone shading
  ctx.fillStyle = 'rgba(239,68,68,0.06)';
  ctx.fillRect(deltaX, pad.top, w - pad.right - deltaX, fBoundY - pad.top);

  // Legend
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  const lx = w - pad.right - 145;
  const ly = pad.top + 15;

  ctx.strokeStyle = 'rgba(239,68,68,0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(lx + 20, ly);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ef4444';
  ctx.fillText('Unclamped', lx + 25, ly + 4);

  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(lx, ly + 16);
  ctx.lineTo(lx + 20, ly + 16);
  ctx.stroke();
  ctx.fillStyle = '#10b981';
  ctx.fillText('Clamped (|e|\u2264\u0394)', lx + 25, ly + 20);

  return fBound;
}

/* ================================================================
   Main Page Component
   ================================================================ */
export default function RobotControllersPage() {
  /* ---- interactive state ---- */
  const [activeCtrl, setActiveCtrl] = useState('position');
  const [activeFactor, setActiveFactor] = useState('type');
  const [K, setK] = useState(80);
  const [D, setD] = useState(20);
  const [delta, setDelta] = useState(0.02);

  /* ---- display metrics (updated from animation frame) ---- */
  const [posError, setPosError] = useState('0.000');
  const [forceDisplay, setForceDisplay] = useState('0.0 N');
  const [behaviorLabel, setBehaviorLabel] = useState('Compliant');
  const [behaviorColor, setBehaviorColor] = useState('#10b981');
  const [spectrumPct, setSpectrumPct] = useState(16);
  const [fMaxDisplay, setFMaxDisplay] = useState('0.0 N');
  const [deltaDisplay, setDeltaDisplay] = useState('20.0 mm');
  const [safetyLabel, setSafetyLabel] = useState('Safe');
  const [safetyColor, setSafetyColor] = useState('#10b981');

  /* ---- canvas refs ---- */
  const simCanvasRef = useRef<HTMLCanvasElement>(null);
  const rigidCanvasRef = useRef<HTMLCanvasElement>(null);
  const springyCanvasRef = useRef<HTMLCanvasElement>(null);
  const clampCanvasRef = useRef<HTMLCanvasElement>(null);

  /* ---- responsive canvas sizing ---- */
  /* No aspect ratio → uses CSS height from .simCanvas / .intuitionItem canvas / .clampCanvas */
  const simSize = useCanvasResize(simCanvasRef);
  const rigidSize = useCanvasResize(rigidCanvasRef);
  const springySize = useCanvasResize(springyCanvasRef);
  const clampSize = useCanvasResize(clampCanvasRef);

  /* ---- physics state refs (no re-renders at 60 fps) ---- */
  const simStateRef = useRef<SimState>({ x: 0.15, v: 0, contactForce: 0 });
  const animTRef = useRef(0);

  /* reset sim when K or D changes */
  const kRef = useRef(K);
  const dRef = useRef(D);
  useEffect(() => { kRef.current = K; }, [K]);
  useEffect(() => { dRef.current = D; }, [D]);
  const prevK = useRef(K);
  const prevD = useRef(D);
  useEffect(() => {
    if (K !== prevK.current || D !== prevD.current) {
      simStateRef.current = { x: 0.15, v: 0, contactForce: 0 };
      prevK.current = K;
      prevD.current = D;
    }
  }, [K, D]);

  /* ---- delta ref ---- */
  const deltaRef = useRef(delta);
  useEffect(() => { deltaRef.current = delta; }, [delta]);

  /* ---- main animation frame (all canvases) ---- */
  /* Throttle React state updates to ~10fps to avoid excessive re-renders */
  const metricsCounterRef = useRef(0);

  useAnimationFrame(() => {
    const currentK = kRef.current;
    const currentD = dRef.current;
    const st = simStateRef.current;

    // Physics: 3 sub-steps
    for (let i = 0; i < 3; i++) physicsStep(st, currentK, currentD);

    // Draw impedance sim canvas
    const simCtx = simCanvasRef.current?.getContext('2d');
    if (simCtx && simSize.width > 0) {
      drawSim(simCtx, simSize.width, simSize.height, st);
    }

    // Draw rigid / springy canvases
    animTRef.current += 0.016;
    const t = animTRef.current;

    const rigidCtx = rigidCanvasRef.current?.getContext('2d');
    if (rigidCtx && rigidSize.width > 0) {
      drawRigidCanvas(rigidCtx, rigidSize.width, rigidSize.height, t);
    }

    const springyCtx = springyCanvasRef.current?.getContext('2d');
    if (springyCtx && springySize.width > 0) {
      drawSpringyCanvas(springyCtx, springySize.width, springySize.height, t);
    }

    // Draw clamp canvas
    const clampCtx = clampCanvasRef.current?.getContext('2d');
    let fBound = 0;
    if (clampCtx && clampSize.width > 0) {
      fBound = drawClampViz(clampCtx, clampSize.width, clampSize.height, deltaRef.current) || 0;
    }

    // Update React metrics ~10fps
    metricsCounterRef.current++;
    if (metricsCounterRef.current % 6 === 0) {
      // Sim metrics
      const pe = Math.abs(TARGET_X - st.x);
      setPosError(pe.toFixed(3));
      setForceDisplay(st.contactForce.toFixed(1) + ' N');
      let bLabel = 'Compliant';
      if (currentK > 300) bLabel = 'Very Stiff';
      else if (currentK > 150) bLabel = 'Stiff';
      else if (currentK > 60) bLabel = 'Medium';
      setBehaviorLabel(bLabel);
      setBehaviorColor(
        currentK > 300 ? '#ef4444' : currentK > 150 ? '#f59e0b' : currentK > 60 ? '#f59e0b' : '#10b981',
      );
      setSpectrumPct(Math.min(currentK / 500, 1) * 100);

      // Clamp metrics
      const d = deltaRef.current;
      const fb = CLAMP_KP * d + 2 * CLAMP_KD * d * CLAMP_FREQ;
      setFMaxDisplay(fb.toFixed(0) + ' N');
      const dMm = d * 1000;
      setDeltaDisplay(dMm.toFixed(0) === '0' ? d.toFixed(3) + ' m' : dMm.toFixed(1) + ' mm');
      const sl = fb < 500 ? 'Safe' : fb < 2000 ? 'Caution' : 'Danger';
      const sc = fb < 500 ? '#10b981' : fb < 2000 ? '#f59e0b' : '#ef4444';
      setSafetyLabel(sl);
      setSafetyColor(sc);
    }
  });

  /* ================================================================
     JSX
     ================================================================ */
  return (
    <div className="method-page">
      <h1 className={styles.title}>Robot Controllers</h1>
      <p className={styles.subtitle}>
        Low-level control laws for manipulation &mdash; from position control to impedance tuning for contact-rich RL
      </p>

      <CoreIdeasAndFeatures coreIdeas={coreIdeas} keyFeatures={keyFeatures} />

      {/* ========== What Is a Controller ========== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          What Is a Controller?
        </div>
        <p className={styles.descText}>
          The <strong style={{ color: '#c4884d' }}>controller</strong> is the low-level robot control law that converts desired motion &mdash; from a planner, policy, or RL agent &mdash; into actuator commands (motor torques or currents). The RL policy outputs a <em>target</em>; the controller figures out the physical forces needed to reach it.
        </p>

        {/* Pipeline SVG */}
        <svg className={styles.pipelineSvg} viewBox="0 0 700 200">
          <defs>
            <marker id="arrW" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" /></marker>
            <marker id="arrG" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#10b981" /></marker>
            <marker id="arrB" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" /></marker>
            <marker id="arrR" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" /></marker>
          </defs>
          {/* RL Policy */}
          <rect x="20" y="55" width="130" height="60" rx="10" fill="#a855f7" />
          <text x="85" y="82" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">RL Policy &#x3C0;</text>
          <text x="85" y="100" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="10">outputs x_des, v_des</text>
          {/* Arrow */}
          <path d="M 150 85 L 210 85" stroke="#f59e0b" strokeWidth="2.5" fill="none" markerEnd="url(#arrW)" className={styles.flowArrow} />
          <text x="180" y="75" textAnchor="middle" fill="#f59e0b" fontSize="9">desired pose</text>
          {/* Controller */}
          <rect x="215" y="40" width="170" height="90" rx="10" fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="2" />
          <text x="300" y="65" textAnchor="middle" fill="#f59e0b" fontSize="12" fontWeight="bold">Controller</text>
          <text x="300" y="82" textAnchor="middle" fill="#888" fontSize="9">e.g. Impedance</text>
          <text x="300" y="98" textAnchor="middle" fill="#888" fontSize="9">F = K(x_des &minus; x) + D(&middot;)</text>
          <text x="300" y="114" textAnchor="middle" fill="#888" fontSize="9">runs at ~1 kHz</text>
          {/* Arrow */}
          <path d="M 385 85 L 445 85" stroke="#10b981" strokeWidth="2.5" fill="none" markerEnd="url(#arrG)" className={styles.flowArrow} />
          <text x="415" y="75" textAnchor="middle" fill="#10b981" fontSize="9">torques &#x3C4;</text>
          {/* Robot */}
          <rect x="450" y="55" width="130" height="60" rx="10" fill="#10b981" />
          <text x="515" y="82" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Robot Arm</text>
          <text x="515" y="100" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="10">joints + actuators</text>
          {/* Feedback arrow */}
          <path d="M 515 115 Q 515 170 300 170 Q 85 170 85 115" stroke="#3b82f6" strokeWidth="2" fill="none" markerEnd="url(#arrB)" strokeDasharray="6 3" />
          <text x="300" y="162" textAnchor="middle" fill="#3b82f6" fontSize="9">state feedback (x, v, F)</text>
          {/* Environment */}
          <rect x="610" y="55" width="75" height="60" rx="10" fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth="2" />
          <text x="647" y="82" textAnchor="middle" fill="#ef4444" fontSize="11" fontWeight="bold">Object</text>
          <text x="647" y="98" textAnchor="middle" fill="#ef4444" fontSize="9">contact</text>
          {/* Contact arrow */}
          <path d="M 580 85 L 607 85" stroke="#ef4444" strokeWidth="2.5" fill="none" markerEnd="url(#arrR)" />
        </svg>

        <p className={styles.captionCenter}>
          The policy decides <em>where</em> to go. The controller decides <em>how hard</em> to push to get there.
        </p>
      </div>

      {/* ========== Controller Types + Impedance Control ========== */}
      <div className={styles.mainGrid}>
        {/* Controller Types */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Controller Types
          </div>
          <div className={styles.ctrlTypes}>
            {([
              { key: 'position', label: 'Position', desc: 'Track a target joint / Cartesian position' },
              { key: 'velocity', label: 'Velocity', desc: 'Track a desired velocity profile' },
              { key: 'torque', label: 'Torque / Force', desc: 'Directly command joint torques or Cartesian forces' },
              { key: 'impedance', label: 'Impedance', desc: 'Virtual mass\u2013spring\u2013damper relationship' },
            ] as const).map((ct, index) => (
              <div
                key={ct.key}
                className={activeCtrl === ct.key ? styles.ctrlTypeActive : styles.ctrlType}
                onClick={() => setActiveCtrl(ct.key)}
              >
                <div className={styles.ideaNum} style={{ margin: '0 auto 10px' }}>
                  {index + 1}
                </div>
                <h4 className={styles.ctrlTypeH4}>{ct.label}</h4>
                <p className={styles.ctrlTypeP}>{ct.desc}</p>
              </div>
            ))}
          </div>
          <div className={styles.infoPanel}>
            <h4 className={styles.infoPanelH4}>{ctrlInfo[activeCtrl].title}</h4>
            <p className={styles.infoPanelP}>{ctrlInfo[activeCtrl].desc}</p>
          </div>
        </div>

        {/* Impedance Control */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Impedance Control
          </div>
          <p className={styles.impedanceDesc}>
            An impedance controller makes the robot behave like a <strong style={{ color: '#c4884d' }}>virtual mass&ndash;spring&ndash;damper</strong> relative to a target pose. It does <em>not</em> strictly enforce position &mdash; it enforces a <em>mechanical relationship</em> between motion and force.
          </p>
          <div className={styles.equationBox}>
            <div className={styles.equationCenter}>
              <MathJax>{`\\[
                F = \\underbrace{K (x_{\\text{des}} - x)}_{\\text{spring}} + \\underbrace{D (\\dot{x}_{\\text{des}} - \\dot{x})}_{\\text{damper}} + \\underbrace{M (\\ddot{x}_{\\text{des}} - \\ddot{x})}_{\\text{inertia}}
              \\]`}</MathJax>
            </div>
          </div>
          <div className={styles.paramGrid}>
            <div className={styles.paramCardK}>
              <div className={styles.paramLabel} style={{ color: '#10b981' }}>K &mdash; Stiffness</div>
              <div className={styles.paramDesc}>How strongly the spring pulls back to target. Units: N/m</div>
            </div>
            <div className={styles.paramCardD}>
              <div className={styles.paramLabel} style={{ color: '#3b82f6' }}>D &mdash; Damping</div>
              <div className={styles.paramDesc}>Resists velocity differences. Prevents oscillation. Units: Ns/m</div>
            </div>
            <div className={styles.paramCardM}>
              <div className={styles.paramLabel} style={{ color: '#8b7ec8' }}>M &mdash; Inertia</div>
              <div className={styles.paramDesc}>Virtual mass shaping. Often set to zero in practice.</div>
            </div>
          </div>
        </div>
      </div>

      {/* ========== Interactive Impedance Sim + Stiffness Spectrum ========== */}
      <div className={styles.mainGrid}>
        {/* Spring-Damper Simulation */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Interactive: Feel the Impedance
          </div>
          <p className={styles.simDesc}>
            Drag the sliders to see how stiffness K and damping D change the robot&apos;s response when it contacts a wall. The robot tries to reach the dashed target (x<sub>des</sub>).
          </p>

          <div className={`${styles.sliderRow} ${styles.sliderK}`}>
            <label style={{ color: '#10b981' }}>K (stiffness):</label>
            <input
              type="range"
              min="5"
              max="500"
              step="5"
              value={K}
              onChange={(e) => setK(Number(e.target.value))}
            />
            <span className={styles.sliderVal} style={{ color: '#10b981' }}>{K}</span>
          </div>
          <div className={`${styles.sliderRow} ${styles.sliderD}`}>
            <label style={{ color: '#3b82f6' }}>D (damping):</label>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={D}
              onChange={(e) => setD(Number(e.target.value))}
            />
            <span className={styles.sliderVal} style={{ color: '#3b82f6' }}>{D}</span>
          </div>

          <div className={styles.simCanvasContainer}>
            <canvas ref={simCanvasRef} className={styles.simCanvas} />
          </div>

          <div className={styles.metricsRow}>
            <div className={styles.metricItem}>
              <div className={styles.metricValue} style={{ color: '#10b981' }}>{posError}</div>
              <div className={styles.metricLabel}>Position error</div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricValue} style={{ color: '#ef4444' }}>{forceDisplay}</div>
              <div className={styles.metricLabel}>Contact force</div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricValue} style={{ color: behaviorColor }}>{behaviorLabel}</div>
              <div className={styles.metricLabel}>Behavior</div>
            </div>
          </div>
        </div>

        {/* Stiffness Tradeoffs */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            Stiffness Tradeoff
          </div>
          <p className={styles.tradeoffDesc}>
            The stiffness gain K is the single most important tuning parameter for contact-rich tasks. Getting it wrong dominates all other sources of error &mdash; sometimes more than the learned policy itself.
          </p>

          <div className={styles.spectrumBar}>
            <div className={styles.spectrumGradient} />
            <div
              className={styles.spectrumMarker}
              style={{ left: `${spectrumPct}%` }}
              data-label="sweet spot"
            />
          </div>
          <div className={styles.spectrumLabels}>
            <span>Compliant (low K)</span>
            <span>Medium</span>
            <span>Very Stiff (high K)</span>
          </div>

          <div className={styles.tradeoffGrid} style={{ marginTop: 18 }}>
            <div
              className={styles.tradeoffCard}
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
            >
              <h4 style={{ color: '#ef4444' }}>Too Stiff</h4>
              <ul>
                <li style={{ color: '#ccc' }}>Contacts become brittle</li>
                <li style={{ color: '#ccc' }}>Small pose errors &rarr; force spikes</li>
                <li style={{ color: '#ccc' }}>Chatter, jamming, or damage</li>
                <li style={{ color: '#ccc' }}>RL policies become unstable</li>
              </ul>
            </div>
            <div
              className={styles.tradeoffCard}
              style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}
            >
              <h4 style={{ color: '#10b981' }}>Too Compliant</h4>
              <ul>
                <li style={{ color: '#ccc' }}>Robot yields too much</li>
                <li style={{ color: '#ccc' }}>Insertion or alignment fails</li>
                <li style={{ color: '#ccc' }}>Insufficient contact force</li>
                <li style={{ color: '#ccc' }}>Sluggish, imprecise motion</li>
              </ul>
            </div>
          </div>
          <p className={styles.tradeoffCaption}>
            Contact-rich tasks need the <strong style={{ color: '#c4884d' }}>sweet spot</strong> &mdash; compliant enough to absorb errors, stiff enough to accomplish the task.
          </p>
        </div>
      </div>

      {/* ========== Intuition: Rigid vs Springy ========== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Intuition: What Does Stiffness Feel Like?
        </div>
        <div className={styles.intuitionRow}>
          <div className={styles.intuitionItem} style={{ border: '2px solid rgba(239,68,68,0.3)' }}>
            <canvas ref={rigidCanvasRef} />
            <h4 className={styles.intuitionItemH4} style={{ color: '#ef4444' }}>Stiff Controller</h4>
            <p className={styles.intuitionItemP}>
              Robot behaves like a <strong>rigid metal rod</strong>. Pushes through obstacles with full force. Any contact mismatch = damage.
            </p>
          </div>
          <div className={styles.intuitionItem} style={{ border: '2px solid rgba(16,185,129,0.3)' }}>
            <canvas ref={springyCanvasRef} />
            <h4 className={styles.intuitionItemH4} style={{ color: '#10b981' }}>Compliant Controller</h4>
            <p className={styles.intuitionItemP}>
              Robot behaves like a <strong>springy wrist</strong>. Absorbs contact gently. Can adapt to misalignment. Needed for contact-rich tasks.
            </p>
          </div>
        </div>
      </div>

      {/* ========== SERL Hierarchical Controller ========== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          SERL: Hierarchical Control &amp; Force Bounding
        </div>
        <p className={styles.impedanceDesc} style={{ marginBottom: 15 }}>
          SERL uses a <strong style={{ color: '#ec4899' }}>two-layered control hierarchy</strong>: the RL policy &#x3C0;(a|s) sends setpoint targets at ~10&nbsp;Hz, while a downstream impedance controller tracks them at 1&nbsp;kHz. Each RL timestep blocks 100 inner-loop steps. This mismatch creates a critical safety problem.
        </p>

        {/* SERL Hierarchy SVG */}
        <svg className={styles.serlHierarchySvg} viewBox="0 0 750 170">
          <defs>
            <marker id="arrPink" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#ec4899" /></marker>
            <marker id="arrAmber" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" /></marker>
            <marker id="arrGreen2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#10b981" /></marker>
          </defs>
          {/* RL Policy (outer loop) */}
          <rect x="15" y="30" width="160" height="65" rx="10" fill="#a855f7" />
          <text x="95" y="55" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">RL Policy &#x3C0;(a|s)</text>
          <text x="95" y="72" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="10">~10 Hz (outer loop)</text>
          {/* Arrow RL to clamp */}
          <path d="M 175 62 L 230 62" stroke="#ec4899" strokeWidth="2.5" fill="none" markerEnd="url(#arrPink)" className={styles.flowArrow} />
          <text x="202" y="54" textAnchor="middle" fill="#ec4899" fontSize="9">p_ref</text>
          {/* Error Clamp Box */}
          <rect x="235" y="25" width="130" height="75" rx="10" fill="rgba(236,72,153,0.15)" stroke="#ec4899" strokeWidth="2" />
          <text x="300" y="48" textAnchor="middle" fill="#ec4899" fontSize="11" fontWeight="bold">Error Clamp</text>
          <text x="300" y="64" textAnchor="middle" fill="#888" fontSize="10">|e| &le; &#x394;</text>
          <text x="300" y="80" textAnchor="middle" fill="#888" fontSize="9">bounds max force</text>
          {/* Arrow clamp to impedance */}
          <path d="M 365 62 L 420 62" stroke="#f59e0b" strokeWidth="2.5" fill="none" markerEnd="url(#arrAmber)" className={styles.flowArrow} />
          <text x="392" y="54" textAnchor="middle" fill="#f59e0b" fontSize="9">e (clamped)</text>
          {/* Impedance Controller */}
          <rect x="425" y="20" width="175" height="85" rx="10" fill="rgba(245,158,11,0.2)" stroke="#f59e0b" strokeWidth="2" />
          <text x="512" y="43" textAnchor="middle" fill="#f59e0b" fontSize="11" fontWeight="bold">Impedance Controller</text>
          <text x="512" y="60" textAnchor="middle" fill="#888" fontSize="9">F = k_p&middot;e + k_d&middot;&#x1E0B; + F_ff + F_cor</text>
          <text x="512" y="76" textAnchor="middle" fill="#888" fontSize="9">1 kHz (inner loop)</text>
          <text x="512" y="90" textAnchor="middle" fill="#888" fontSize="9">&#x3C4; = J&#x1D40; F + &#x3C4;_null</text>
          {/* Arrow impedance to robot */}
          <path d="M 600 62 L 655 62" stroke="#10b981" strokeWidth="2.5" fill="none" markerEnd="url(#arrGreen2)" className={styles.flowArrow} />
          <text x="627" y="54" textAnchor="middle" fill="#10b981" fontSize="9">&#x3C4;</text>
          {/* Robot */}
          <rect x="660" y="35" width="75" height="55" rx="10" fill="#10b981" />
          <text x="697" y="58" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Robot</text>
          <text x="697" y="72" textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="9">joints</text>
          {/* Frequency annotations */}
          <rect x="15" y="110" width="160" height="30" rx="6" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.3)" strokeWidth="1" />
          <text x="95" y="130" textAnchor="middle" fill="#a855f7" fontSize="10">1 step = 100 ms</text>
          <rect x="425" y="110" width="175" height="30" rx="6" fill="rgba(245,158,11,0.15)" stroke="rgba(245,158,11,0.3)" strokeWidth="1" />
          <text x="512" y="130" textAnchor="middle" fill="#f59e0b" fontSize="10">100 steps per RL step</text>
          {/* Big bracket showing ratio */}
          <path d="M 180 120 Q 302 155 425 120" stroke="#888" strokeWidth="1" fill="none" strokeDasharray="4 3" />
          <text x="302" y="155" textAnchor="middle" fill="#888" fontSize="9">1:100 ratio</text>
        </svg>

        <div className={styles.serlSubGrid}>
          {/* SERL impedance equation */}
          <div className={styles.serlPanel}>
            <h4 className={styles.serlPanelH4} style={{ color: '#c4884d' }}>SERL Impedance Objective</h4>
            <div className={styles.serlEquationBox}>
              <div className={styles.serlEquationCenter}>
                <MathJax>{`\\[
                  F = \\underbrace{k_p \\cdot e}_{\\text{spring}} + \\underbrace{k_d \\cdot \\dot{e}}_{\\text{damper}} + \\underbrace{F_{ff}}_{\\text{feed-fwd}} + \\underbrace{F_{cor}}_{\\text{Coriolis}}
                \\]`}</MathJax>
              </div>
            </div>
            <p className={styles.serlPanelDesc}>
              where <strong style={{ color: '#ccc' }}>e = p &minus; p<sub>ref</sub></strong>, p is the measured pose, p<sub>ref</sub> is the target from the RL policy. F<sub>ff</sub> is feed-forward force, F<sub>cor</sub> is Coriolis compensation. This is then converted to joint torques: <strong style={{ color: '#ccc' }}>&tau; = J<sup>T</sup>F + &tau;<sub>null</sub></strong>.
            </p>
          </div>

          {/* The force bounding problem + solution */}
          <div className={styles.serlPanel}>
            <h4 className={styles.serlPanelH4} style={{ color: '#ec4899' }}>The Force Bounding Problem</h4>
            <p className={styles.serlPanelBodyDesc}>
              If p<sub>ref</sub> is far from the current pose (e.g. RL sends a big jump), the error e becomes large, generating <strong style={{ color: '#ef4444' }}>dangerous forces</strong> on contact. Reducing gains k<sub>p</sub> hurts accuracy. SERL&apos;s solution: <strong style={{ color: '#ec4899' }}>clamp the error</strong> so |e| &le; &Delta;.
            </p>
            <div className={styles.serlEquationBoxPink}>
              <div className={styles.serlEquationCenter}>
                <MathJax>{`\\[
                  |F_{\\max}| \\;\\leq\\; k_p \\cdot |\\Delta| \\;+\\; 2\\, k_d \\cdot |\\Delta| \\cdot f
                \\]`}</MathJax>
              </div>
            </div>
            <p className={styles.serlPanelDesc}>
              where <strong style={{ color: '#ccc' }}>f</strong> is the control frequency (1 kHz). By choosing &Delta; you get a <em>hard upper bound</em> on interaction force &mdash; without sacrificing gain accuracy.
            </p>
          </div>
        </div>

        {/* Interactive Delta clamp visualization */}
        <div className={styles.clampSection}>
          <h4 className={styles.clampH4}>Interactive: Error Clamping &amp; Force Bound</h4>
          <p className={styles.clampDesc}>
            Adjust &Delta; to see how clamping the pose error limits the maximum force. Gains k<sub>p</sub>=80, k<sub>d</sub>=20, f=1000&nbsp;Hz.
          </p>

          <div className={`${styles.sliderRow} ${styles.sliderDelta}`} style={{ marginBottom: 4 }}>
            <label style={{ color: '#ec4899', minWidth: 110 }}>&Delta; (clamp):</label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={delta}
              onChange={(e) => setDelta(Number(e.target.value))}
            />
            <span className={styles.sliderVal} style={{ color: '#ec4899', minWidth: 70 }}>{deltaDisplay}</span>
          </div>

          <div className={styles.clampCanvasWrap}>
            <canvas ref={clampCanvasRef} className={styles.clampCanvas} />
          </div>

          <div className={styles.metricsRow} style={{ marginTop: 10 }}>
            <div className={styles.metricItem}>
              <div className={styles.metricValueSmall} style={{ color: '#ec4899' }}>{fMaxDisplay}</div>
              <div className={styles.metricLabel}>Max force F<sub>max</sub></div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricValueSmall} style={{ color: '#c4884d' }}>{deltaDisplay}</div>
              <div className={styles.metricLabel}>&Delta; (clamp radius)</div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricValueSmall} style={{ color: safetyColor }}>{safetyLabel}</div>
              <div className={styles.metricLabel}>Contact safety</div>
            </div>
          </div>
        </div>
      </div>

      {/* ========== Why SERL Cares ========== */}
      <div className={styles.cardMb}>
        <div className={styles.cardTitle}>
          Why Controller Choice Dominates RL Performance (SERL)
        </div>
        <p className={styles.serlCaresDesc}>
          SERL learns from <em>real</em> interaction on physical robots. The same RL policy can perform very differently depending on the low-level controller. For contact-rich manipulation, controller choice and impedance tuning can dominate performance &mdash; sometimes more than the learned policy itself. For example, in PCB insertion, an overly stiff controller bends fragile pins, while an overly compliant one cannot position the part.
        </p>

        <div className={styles.serlFactors}>
          {([
            { key: 'type', label: 'Controller Type', desc: 'Position vs torque vs impedance' },
            { key: 'params', label: 'Impedance Params', desc: 'Stiffness K, damping D tuning' },
            { key: 'rate', label: 'Update Rate', desc: 'Inner-loop frequency & latency' },
          ] as const).map((sf, index) => (
            <div
              key={sf.key}
              className={activeFactor === sf.key ? styles.serlFactorActive : styles.serlFactor}
              onClick={() => setActiveFactor(sf.key)}
            >
              <div className={styles.ideaNum} style={{ margin: '0 auto 10px' }}>
                {index + 1}
              </div>
              <h4 className={styles.serlFactorH4}>{sf.label}</h4>
              <p className={styles.serlFactorP}>{sf.desc}</p>
            </div>
          ))}
        </div>
        <div className={styles.serlInfoPanel}>
          <h4 className={styles.serlInfoH4}>{serlInfo[activeFactor].title}</h4>
          <p className={styles.infoPanelP}>{serlInfo[activeFactor].desc}</p>
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
                { title: 'Impedance Control', desc: 'Hogan, 1985 \u2014 The foundational framework for controlling the dynamic relationship between force and motion, enabling compliant robot behavior' },
                { title: 'Operational Space Control', desc: 'Khatib, 1987 \u2014 Task-space formulation that lets controllers reason about end-effector pose rather than joint angles, used in most modern manipulation stacks' },
                { title: 'Compliance Control Survey', desc: 'Villani & De Schutter, 2016 \u2014 Comprehensive survey of force/compliance control methods that underpin the controller design choices discussed here' },
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
                { title: 'SERL', desc: 'Luo, Xu et al., 2024 \u2014 Demonstrates that controller choice (especially stiffness tuning) dominates RL policy performance on contact-rich real-robot tasks' },
                { title: 'DayDreamer / DreamerV3', desc: 'Wu et al., 2022 / Hafner et al., 2023 \u2014 World-model RL agents that rely on impedance controllers for safe real-robot deployment' },
                { title: 'Diffusion Policy', desc: 'Chi et al., 2023 \u2014 Uses impedance-controlled robots as the hardware substrate for diffusion-based visuomotor policy learning' },
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
