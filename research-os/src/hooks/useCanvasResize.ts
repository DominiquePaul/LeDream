import { useEffect, useState, useRef, RefObject } from 'react';

interface CanvasSize {
  width: number;
  height: number;
}

/**
 * Handles DPR-aware canvas resizing via ResizeObserver.
 * Returns the CSS (logical) width/height so drawing code can use them.
 *
 * When aspectRatio is provided, height = width / aspectRatio.
 * When not provided, uses the canvas element's CSS height (set via stylesheet).
 *
 * To avoid infinite resize loops, we only react to width changes.
 */
export function useCanvasResize(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  aspectRatio?: number
): CanvasSize {
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });
  const lastWidthRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    function resize() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const parent = canvas.parentElement;
      if (!parent) return;

      const dpr = window.devicePixelRatio || 1;
      const cs = getComputedStyle(parent);
      const w = parent.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);

      // Only proceed if width actually changed (prevents feedback loops)
      if (Math.abs(w - lastWidthRef.current) < 1 && lastWidthRef.current > 0) return;
      lastWidthRef.current = w;

      // Determine CSS height: from aspect ratio, or from the canvas's CSS-specified height
      let h: number;
      if (aspectRatio) {
        h = w / aspectRatio;
      } else {
        // Use the CSS-computed height of the canvas itself
        h = canvas.clientHeight || w * 0.5;
      }

      // Set the backing-store size (physical pixels)
      canvas.width = w * dpr;
      canvas.height = h * dpr;

      // Set the display size (CSS pixels) — only set width; let CSS handle height
      // unless we're computing from aspect ratio
      canvas.style.width = `${w}px`;
      if (aspectRatio) {
        canvas.style.height = `${h}px`;
      }

      const ctx = canvas.getContext('2d');
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      setSize({ width: w, height: h });
    }

    const ro = new ResizeObserver(resize);
    ro.observe(parent);
    resize();

    return () => ro.disconnect();
  }, [canvasRef, aspectRatio]);

  return size;
}
