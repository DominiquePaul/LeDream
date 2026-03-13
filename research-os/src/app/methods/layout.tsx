"use client";

import { MathJaxContext } from "better-react-mathjax";

const mathjaxConfig = {
  loader: { load: ["[tex]/ams", "[tex]/color"] },
  tex: {
    packages: { "[+]": ["ams", "color"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
  },
};

export default function MethodsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <MathJaxContext
      version={3}
      config={mathjaxConfig}
      src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
    >
      {children}
    </MathJaxContext>
  );
}
