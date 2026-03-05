import { useEffect, useRef } from 'react';

/**
 * Runs a rAF loop calling `callback` every frame.
 * Automatically cleans up on unmount.
 * The callback receives delta time (ms) since the last frame.
 */
export function useAnimationFrame(callback: (dt: number) => void) {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    let rafId: number;
    let lastTime = performance.now();

    function loop(now: number) {
      const dt = now - lastTime;
      lastTime = now;
      callbackRef.current(dt);
      rafId = requestAnimationFrame(loop);
    }

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, []);
}
