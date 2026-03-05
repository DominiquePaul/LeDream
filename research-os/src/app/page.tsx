import Link from 'next/link';
import { methods, categories, getMethodsByCategory } from '@/lib/methods-registry';
import styles from './page.module.css';

export default function HomePage() {
  return (
    <div>
      {/* Hero */}
      <div className={styles.hero}>
        <h1>ML Method Visualizations</h1>
        <p>
          Interactive visualizations to help understand machine learning algorithms and
          methods from research papers. Click on any method to explore its architecture,
          key concepts, and step-by-step explanations.
        </p>

        <div className={styles.stats}>
          <div className={styles.stat}>
            <div className={styles.statValue}>{methods.length}</div>
            <div className={styles.statLabel}>Methods</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>{categories.length}</div>
            <div className={styles.statLabel}>Categories</div>
          </div>
        </div>
      </div>

      {/* Category Sections */}
      {categories.map((cat) => {
        const catMethods = getMethodsByCategory(cat.id);
        return (
          <div className={styles.categorySection} key={cat.id}>
            <h2 className={styles.sectionTitle}>{cat.title}</h2>
            <div className={styles.methodsGrid}>
              {catMethods.map((m) => (
                <Link
                  href={`/methods/${m.slug}`}
                  className={styles.methodCard}
                  key={m.slug}
                >
                  <div className={styles.methodCardHeader}>
                    <div
                      className={styles.methodIcon}
                      data-category={m.iconClass}
                    >
                      {m.abbrev}
                    </div>
                    <div>
                      <h3>{m.title}</h3>
                      <span className={styles.category}>{cat.title}</span>
                    </div>
                  </div>
                  <p>{m.description}</p>
                  <div className={styles.methodTags}>
                    {m.tags.map((tag) => (
                      <span className="tag" key={tag}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </Link>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
