import { LineageGraph } from './LineageGraph';
import s from './lineage.module.css';

export const metadata = {
  title: 'Paper Lineage — ML Method Visualizations',
  description:
    'Interactive dependency graph showing how machine learning papers build on each other.',
};

export default function LineagePage() {
  return (
    <div className={s.page}>
      <h1>Paper Lineage</h1>
      <p className={s.subtitle}>
        How the papers on this site relate to each other and the broader
        literature. Solid nodes are on-site pages you can click into; dashed
        nodes are external references.
      </p>
      <LineageGraph />
    </div>
  );
}
