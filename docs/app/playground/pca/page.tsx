import type { Metadata } from 'next';
import Link from 'next/link';
import { PcaPlayground } from '@/components/playground/PcaPlayground';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'PCA Visualization — Interactive Principal Component Analysis in JavaScript' },
  description: 'Explore principal component analysis with an interactive PCA visualization. Paste data, inspect explained variance and loadings, and run PCA in JavaScript with @kanaries/ml.',
  keywords: ['PCA visualization', 'principal component analysis visualization', 'PCA JavaScript', 'interactive PCA', 'dimensionality reduction'],
  alternates: { canonical: `${siteUrl}/playground/pca` },
  openGraph: {
    title: 'Interactive PCA Visualization in JavaScript',
    description: 'Project multidimensional data, drag a valid loading basis, and reconstruct observations in your browser.',
    url: `${siteUrl}/playground/pca`,
  },
};

const faq: ToolFaq[] = [
  { question: 'What does a PCA visualization show?', answer: 'A PCA visualization maps observations from many correlated features into a smaller set of orthogonal principal components. Nearby points have similar combinations of the original features, while loading arrows show which features influence each direction.' },
  { question: 'What is explained variance in PCA?', answer: 'Explained variance measures how much of the dataset’s total variation a principal component captures. The first component captures the largest possible share, and each later component captures the largest remaining share while staying orthogonal to earlier components.' },
  { question: 'Why can the loading vectors rotate in this playground?', answer: 'The playground applies one rigid rotation to the entire two-dimensional PCA coordinate system. Scores and loadings rotate together, so distances, angles, orthogonality, and reconstruction stay unchanged. It is a different view of the same fitted subspace.' },
  { question: 'Can PCA recover the original data exactly?', answer: 'Only when all information-bearing components are retained. A two-component reconstruction is an approximation when the original data has more than two independent dimensions. The displayed reconstruction error quantifies what the selected 2D view omits.' },
  { question: 'Does this PCA tool upload my data?', answer: 'No. Parsing, fitting, projection, and inverse transformation all happen locally in your browser with @kanaries/ml. The page does not require an account or a server request for calculation.' },
];

export default function PcaPage() {
  return (
    <ToolPageLayout
      name="PCA visualization"
      description="Paste multidimensional data and watch principal component analysis turn correlated features into an inspectable 2D projection. Explore variance, loadings, and inverse transformation without leaving your browser."
      pathname="/playground/pca"
      sectionName="Playground"
      sectionPath="/playground"
      eyebrow="Interactive dimensionality reduction"
      activityLabel="Live PCA model"
      tool={<PcaPlayground />}
      faq={faq}
      related={[
        { href: '/docs/apis/decomposition/pca', title: 'PCA JavaScript API', description: 'Constructor, methods, returned components, and implementation details.' },
        { href: '/playground/kmeans', title: 'K-Means visualization', description: 'Cluster observations after learning how projections expose structure.' },
        { href: '/playground/knn', title: 'KNN visualization', description: 'Explore a local classifier and its nonlinear decision regions.' },
      ]}
    >
      <h2>What principal component analysis does</h2>
      <p>Principal component analysis, usually shortened to PCA, is a linear dimensionality-reduction method. It replaces the original feature axes with new, mutually perpendicular axes called principal components. The first component follows the direction of greatest variation in the centered data. The second follows the greatest remaining variation while staying orthogonal to the first, and the pattern continues for later components.</p>
      <p>This transformation is useful when a dataset contains many correlated measurements. Height, weight, and several body-size measurements may all carry overlapping information; product usage metrics such as sessions, time, and actions often do too. PCA compresses that redundancy into fewer coordinates. Analysts use the result for exploratory plots, noise reduction, preprocessing, anomaly inspection, and faster downstream models.</p>

      <h2>An interactive PCA visualization in JavaScript</h2>
      <p>The visualization above fits PCA with <code>@kanaries/ml</code>, a scikit-learn-style machine-learning library for JavaScript and TypeScript. Enter a numeric CSV table, and the model centers the columns, computes the principal directions, and projects every row into two dimensions. Everything runs on the client, which makes the playground useful in a browser lesson, an internal frontend tool, or a Node.js workflow without sending data to a Python service.</p>
      <p>The points are PCA scores: the coordinates of observations on the two retained components. The red arrows are feature loadings. A long arrow means that a feature has a strong relationship with the displayed component plane. Arrows pointing in similar directions suggest positive correlation; arrows pointing in opposite directions suggest negative correlation; nearly perpendicular arrows suggest a weaker linear relationship in this projection.</p>

      <h2>How to read and manipulate the biplot</h2>
      <p>Start with the flower sample and look for observations that separate along PC1. Then select a point. The table below the plot maps its two-dimensional score back into the original feature space using <code>inverseTransform</code>. Comparing original and reconstructed values makes dimensionality reduction concrete: a small error means two components represent that observation well, while a larger error signals information in omitted directions.</p>
      <p>You can also drag any loading endpoint. This does not refit PCA or move one feature independently. Instead, it rotates the entire two-dimensional basis, including all observations and loading vectors. A rigid coordinate rotation is mathematically valid because it preserves the fitted subspace, pairwise distances, angles, and reconstruction. Use Reset to return to the conventional PC1-horizontal view.</p>

      <h2>Explained variance and choosing components</h2>
      <p>The explained-variance percentages answer a practical compression question: how much of the original variation survives in this view? A high combined percentage means the first two components give a faithful overview. A low percentage warns that a flat scatterplot hides important structure. There is no universal cutoff. Visualization may tolerate more loss than forecasting, and a small-variance direction can still contain a rare but meaningful signal.</p>
      <p>PCA is sensitive to feature scale. A column measured in thousands can dominate another measured between zero and one. Standardize features first when units differ and absolute variance is not intrinsically meaningful. Also remember that PCA is linear: curved manifolds and discrete class boundaries may need methods such as UMAP, t-SNE, or task-specific feature engineering.</p>

      <h2>JavaScript PCA compared with scikit-learn</h2>
      <p>The code tabs connect this interactive view to reusable programs. Both APIs follow the same fit-transform workflow: create a two-component estimator, fit and project a matrix, inspect components and explained variance, then optionally inverse-transform scores. Component signs can differ across correct implementations because an eigenvector and its negative describe the same axis. Compare reconstructed values and subspaces rather than expecting every sign to match.</p>
      <p>For production work, validate missing values, decide how to scale columns, and fit preprocessing only on training data to prevent leakage. Read the <Link href="/docs/apis/decomposition/pca">PCA JavaScript API guide</Link> for detailed method behavior, then use the <Link href="/playground/kmeans">K-Means playground</Link> or <Link href="/playground/knn">KNN playground</Link> to see how unsupervised and supervised algorithms respond to geometry.</p>

      <h2>A practical PCA workflow</h2>
      <p>Begin by defining which columns are legitimate model features. Remove identifiers, post-outcome fields, and values that would not exist when the transformation is used. Split data before fitting preprocessing, impute missing values from training statistics, and standardize when units should contribute equally. Fit PCA on the training matrix, retain the fitted mean and components, and apply that unchanged transformation to validation and future observations.</p>
      <p>Choose the component count with a combination of cumulative explained variance and downstream validation. A scree plot can reveal a bend, but predictive accuracy, reconstruction quality, latency, and interpretability are often better decision criteria. Monitor incoming feature distributions because a fixed projection can become misleading after product behavior or measurement systems drift. PCA components are combinations, not causal factors, and a visually separated group is a hypothesis to investigate rather than proof of a real population. This browser visualization is best used to build intuition, inspect preprocessing choices, and communicate the geometry before the same fitted workflow is embedded in an application.</p>
    </ToolPageLayout>
  );
}
