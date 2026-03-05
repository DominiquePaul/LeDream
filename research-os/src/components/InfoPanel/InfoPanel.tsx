'use client';

import { ReactNode } from 'react';

interface InfoPanelProps {
  title?: string;
  children: ReactNode;
  visible?: boolean;
  className?: string;
}

export function InfoPanel({
  title,
  children,
  visible = true,
  className = '',
}: InfoPanelProps) {
  if (!visible) return null;

  return (
    <div className={`info-panel ${className}`}>
      {title && <h4>{title}</h4>}
      {typeof children === 'string' ? <p>{children}</p> : children}
    </div>
  );
}
