import type { Metadata } from 'next';
import Link from 'next/link';
import { KnnPlayground } from '@/components/playground/KnnPlayground';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'KNN Visualization — Interactive k-Nearest Neighbors in JavaScript' },
  description: 'Explore k-nearest neighbors with an interactive KNN visualization. Draw training points, move a query, compare k and distance metrics, and inspect every vote.',
  keywords: ['KNN visualization', 'k nearest neighbors visualization', 'KNN JavaScript', 'KNN decision boundary', 'interactive machine learning'],
  alternates: { canonical: `${siteUrl}/playground/knn` },
  openGraph: { title: 'Interactive KNN Visualization in JavaScript', description: 'Draw labeled points and see the @kanaries/ml KNN decision boundary update instantly.', url: `${siteUrl}/playground/knn` },
};

const faq: ToolFaq[] = [
  { question: 'How does the k-nearest neighbors algorithm classify a point?', answer: 'KNN measures the distance from a query to stored training samples, selects the k closest samples, and assigns the class that wins their vote. Distance weighting can give closer neighbors more influence than farther neighbors.' },
  { question: 'How should I choose k in KNN?', answer: 'A small k produces flexible boundaries but can overfit noise. A larger k smooths the boundary but can erase small classes or local structure. Use cross-validation on representative data, and prefer an odd k for binary classification when you want fewer ties.' },
  { question: 'What is the difference between Euclidean and Manhattan distance?', answer: 'Euclidean distance is the straight-line L2 distance and forms circular neighborhoods in two dimensions. Manhattan distance adds absolute coordinate differences and forms diamond-shaped L1 neighborhoods. The best choice depends on feature geometry and meaning.' },
  { question: 'Does KNN need feature scaling?', answer: 'Usually, yes. A feature with a large numeric range can dominate the distance even if it is not more important. Standardization or another meaningful scaling method should be fit on the training set before KNN.' },
  { question: 'Is this decision boundary calculated by @kanaries/ml?', answer: 'Yes. Each cell in the colored decision grid and the draggable query prediction comes from a fitted @kanaries/ml KNearestNeighbors classifier running in the browser.' },
];

export default function KnnPage() {
  return (
    <ToolPageLayout
      name="KNN visualization"
      description="Draw labeled samples, drag a query point, and see exactly how k-nearest neighbors turns local distances into a classification. Compare neighborhoods and decision regions live in your browser."
      pathname="/playground/knn"
      sectionName="Playground"
      sectionPath="/playground"
      eyebrow="Interactive classification playground"
      activityLabel="Live KNN classifier"
      tool={<KnnPlayground />}
      faq={faq}
      related={[
        { href: '/docs/apis/neighbors/knn', title: 'KNN JavaScript API', description: 'Fit, predict, weighting, and distance options for browser or Node.js.' },
        { href: '/playground/kmeans', title: 'K-Means visualization', description: 'Compare supervised neighborhoods with unsupervised centroid clusters.' },
        { href: '/tools/logistic-regression-calculator', title: 'Logistic regression calculator', description: 'Inspect probabilities and a linear classification boundary.' },
      ]}
    >
      <h2>What k-nearest neighbors does</h2>
      <p>K-nearest neighbors, or KNN, is a supervised learning method that predicts from local examples. Training is deliberately simple: the classifier stores feature vectors and their labels. When a new sample arrives, it measures the distance to the training set, finds the closest <em>k</em> samples, and combines their labels in a vote. The intuition is that observations near one another are likely to share an outcome.</p>
      <p>KNN can represent nonlinear class boundaries without learning coefficients or constructing a tree. That makes it a useful baseline for classification, a teaching model for distance-based learning, and a practical method when datasets are modest and local similarity is meaningful. The same neighborhood idea also supports regression, imputation, recommendation, and anomaly scoring.</p>

      <h2>A complete KNN visualization in JavaScript</h2>
      <p>The interactive map fits <code>Neighbors.KNearestNeighbors</code> from <code>@kanaries/ml</code>. Choose blobs, interlocking moons, or an XOR pattern; then change <em>k</em>, the distance metric, or the weighting rule. Every colored cell is a real browser-side prediction. There is no pre-rendered illustration and no backend Python process.</p>
      <p>Click inside the map to add a labeled training observation. Use the class controls to decide its label. Drag the diamond-shaped query point and dashed lines identify its current neighbors. The table exposes their rank, distance, class, and effective vote weight, connecting the final color to the individual evidence behind it.</p>

      <h2>How k changes bias and variance</h2>
      <p>Set <em>k</em> to one and the classifier gives each training sample its own territory. This can capture fine structure, but one mislabeled or noisy sample can create an island in the decision map. Increase <em>k</em> and votes average over a broader area. The boundary becomes smoother and less sensitive to individual samples, but a large neighborhood may overwhelm minority classes or merge genuinely separate regions.</p>
      <p>This is the classic bias-variance tradeoff in a visible form. A very flexible boundary has low training bias and high sensitivity; a heavily smoothed boundary makes stronger assumptions. Select <em>k</em> with cross-validation rather than by judging the training plot alone. In binary tasks, an odd value can reduce ties, though class imbalance and distance weighting still matter.</p>

      <h2>Distance metrics and weighted voting</h2>
      <p>Euclidean distance measures the straight-line separation between points. Manhattan distance adds horizontal and vertical differences. Change the selector on the XOR data and watch the regions respond: L2 neighborhoods are circular, while L1 neighborhoods have diamond geometry. Neither is universally superior. The metric should reflect how feature differences combine in the application.</p>
      <p>Uniform voting gives every selected neighbor one vote. Distance voting increases the influence of close observations, which can preserve local detail even with a moderately large <em>k</em>. An exact match needs special handling to avoid division by zero; the library resolves this case while the table caps its displayed reciprocal at a safe numeric denominator.</p>

      <h2>Scaling, performance, and responsible interpretation</h2>
      <p>Because KNN depends on distances, features must be comparable. A yearly-income column measured in thousands will dominate a zero-to-one score unless you scale or deliberately weight them. Fit the scaler on training data only, then apply the same transformation to validation and production samples. Missing values, categorical variables, and irrelevant dimensions also need thoughtful preprocessing.</p>
      <p>Prediction compares a query with stored observations, so basic KNN becomes slower as the dataset grows. Spatial indexes can help in low dimensions, while approximate-neighbor methods are common at large scale. High-dimensional spaces introduce another challenge: distances become less discriminative, often called the curse of dimensionality. Feature selection or a method such as <Link href="/playground/pca">PCA</Link> can help, provided validation confirms that useful class information is retained.</p>

      <h2>JavaScript KNN and scikit-learn</h2>
      <p>The code tabs show parallel APIs. In JavaScript, construct <code>KNearestNeighbors</code> with positional settings for <em>k</em>, weights, and metric, call <code>fit</code>, then <code>predict</code>. Scikit-learn expresses the same choices as named arguments. Both implement the familiar estimator workflow, making it straightforward to prototype in Python and move an interactive experience into a browser or Node.js application.</p>
      <p>Use the <Link href="/docs/apis/neighbors/knn">KNN JavaScript API guide</Link> for method details. Compare this local, nonparametric boundary with the linear probabilities in the <Link href="/tools/logistic-regression-calculator">logistic regression calculator</Link>, or open the <Link href="/playground/kmeans">K-Means visualization</Link> to see what changes when labels are unavailable.</p>

      <h2>A practical KNN evaluation workflow</h2>
      <p>Reserve validation data before choosing scaling, features, <em>k</em>, weights, or a metric. Put preprocessing and KNN in one repeatable pipeline so every fold learns scaling only from its training portion. Evaluate more than overall accuracy when classes are imbalanced: per-class recall, precision, F1, a confusion matrix, and calibrated decision requirements can reveal failures hidden by an average score.</p>
      <p>Inspect errors in feature space and ask whether nearby samples should genuinely share a label. Duplicate records can make validation look unrealistically good, while time-dependent or user-dependent rows may require grouped or chronological splits. At prediction time, monitor distance to the neighborhood as well as the winning label. A query far from every training observation is an extrapolation even if KNN returns a confident-looking majority. Production systems should define an abstention or fallback rule for such cases, measure latency as the reference set grows, and document which training records are retained. The playground isolates geometry; a dependable application adds data governance, representative evaluation, and monitoring around it.</p>
    </ToolPageLayout>
  );
}
