'use client';

import styles from './SliderControl.module.css';

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  displayValue?: string;
  unit?: string;
  color?: string;
}

export function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  unit = '',
  color,
}: SliderControlProps) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <label className={styles.label}>{label}</label>
        <span className={styles.value} style={color ? { color } : undefined}>
          {displayValue ?? value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={styles.slider}
      />
    </div>
  );
}
