'use client';

import { useState } from 'react';
import styles from './CoreIdeasAndFeatures.module.css';

export interface CoreIdea {
  title: string;
  desc: string;
  detail?: string;
}

export interface KeyFeature {
  title: string;
  desc: string;
}

interface Props {
  coreIdeas: CoreIdea[];
  keyFeatures: KeyFeature[];
}

export function CoreIdeasAndFeatures({ coreIdeas, keyFeatures }: Props) {
  const [activeIdea, setActiveIdea] = useState<number | null>(null);

  return (
    <div className={styles.container}>
      {/* Core Ideas */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.heading}>Core Ideas</h2>
          <span className={styles.tooltip}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
              <path d="M8 7v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <circle cx="8" cy="5" r="0.75" fill="currentColor" />
            </svg>
            <span className={styles.tooltipText}>
              The fundamental innovations this paper introduces &mdash; what makes it novel and different from prior work.
            </span>
          </span>
        </div>
        <ul className={styles.list}>
          {coreIdeas.map((idea, idx) => (
            <li
              key={idx}
              className={`${styles.item} ${idea.detail ? styles.clickable : ''} ${activeIdea === idx ? styles.active : ''}`}
              onClick={idea.detail ? () => setActiveIdea(activeIdea === idx ? null : idx) : undefined}
            >
              <strong>{idea.title}</strong> &mdash; {idea.desc}
            </li>
          ))}
        </ul>
        {activeIdea !== null && coreIdeas[activeIdea]?.detail && (
          <div className={styles.detail}>
            <p>{coreIdeas[activeIdea].detail}</p>
          </div>
        )}
      </div>

      {/* Key Features */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.heading}>Key Features</h2>
          <span className={styles.tooltip}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
              <path d="M8 7v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <circle cx="8" cy="5" r="0.75" fill="currentColor" />
            </svg>
            <span className={styles.tooltipText}>
              Practical takeaways &mdash; what this method achieves and why it matters in practice.
            </span>
          </span>
        </div>
        <ul className={styles.list}>
          {keyFeatures.map((feat, idx) => (
            <li key={idx} className={styles.item}>
              <strong>{feat.title}</strong> &mdash; {feat.desc}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
