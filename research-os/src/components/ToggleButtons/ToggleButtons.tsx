'use client';

import styles from './ToggleButtons.module.css';

interface ToggleOption {
  key: string;
  label: string;
}

interface ToggleButtonsProps {
  options: ToggleOption[];
  active: string;
  onChange: (key: string) => void;
  className?: string;
}

export function ToggleButtons({ options, active, onChange, className = '' }: ToggleButtonsProps) {
  return (
    <div className={`${styles.group} ${className}`}>
      {options.map((opt) => (
        <button
          key={opt.key}
          className={`${styles.btn} ${active === opt.key ? styles.active : ''}`}
          onClick={() => onChange(opt.key)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
