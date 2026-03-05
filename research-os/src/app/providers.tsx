'use client';

import { MathJaxContext } from 'better-react-mathjax';
import { mathjaxConfig } from '@/lib/mathjax-config';
import { ThemeProvider } from '@/hooks/useTheme';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <MathJaxContext
        version={3}
        config={mathjaxConfig}
        src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
      >
        {children}
      </MathJaxContext>
    </ThemeProvider>
  );
}
