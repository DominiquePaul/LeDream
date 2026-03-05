'use client';

import { MathJax } from 'better-react-mathjax';
import { CSSProperties } from 'react';

interface MathBlockProps {
  /** Raw LaTeX string (wrap in \( ... \) for inline or \[ ... \] for display) */
  formula: string;
  /** If true, centers the formula in a block container */
  display?: boolean;
  style?: CSSProperties;
  className?: string;
}

export function MathBlock({
  formula,
  display = false,
  style,
  className = '',
}: MathBlockProps) {
  return (
    <div
      className={className}
      style={{
        textAlign: display ? 'center' : undefined,
        overflowX: 'auto',
        padding: display ? '15px 0' : undefined,
        ...style,
      }}
    >
      <MathJax>{formula}</MathJax>
    </div>
  );
}
