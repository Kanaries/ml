import type { Metadata } from 'next';
import Link from 'next/link';
import { KMeansPlayground } from '@/components/playground/KMeansPlayground';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'K-Means Clustering Visualization — Interactive JavaScript Playground' },
  description: 'Step through K-Means clustering in an interactive visualization. Change k, add points, watch centroids move, and verify the result with @kanaries/ml.',
  keywords: ['K Means visualization', 'K-Means clustering visualization', 'KMeans JavaScript', 'interactive clustering', 'Lloyd algorithm'],
  alternates: { canonical: `${siteUrl}/playground/kmeans` },
  openGraph: { title: 'Interactive K-Means Clustering Visualization', description: 'Watch assignments, centroids, and inertia change one Lloyd iteration at a time.', url: `${siteUrl}/playground/kmeans` },
};

const faq: ToolFaq[] = [
  { question: 'How does K-Means clustering work?', answer: 'K-Means alternates two steps: assign each observation to its nearest centroid, then replace each centroid with the mean of its assigned observations. It stops when centers stabilize, inertia changes very little, or the iteration limit is reached.' },
  { question: 'What does k mean in K-Means?', answer: 'k is the number of clusters and centroids the algorithm must produce. It is selected before fitting. Domain knowledge, silhouette analysis, stability, and an elbow plot of inertia can inform the choice.' },
  { question: 'What is inertia?', answer: 'Inertia is the sum of squared distances from observations to their assigned centroids. Lloyd updates never increase it, but a lower value is automatic when k grows and does not by itself prove that clusters are meaningful.' },
  { question: 'Why can K-Means return different results?', answer: 'Its objective has local optima, so different initial centers can lead to different partitions. K-means++ initialization, multiple restarts, and a fixed random seed improve reliability and reproducibility.' },
  { question: 'When is K-Means a poor choice?', answer: 'K-Means struggles with curved, non-spherical, differently dense, or heavily outlier-contaminated groups. The moons and uneven presets make these assumptions visible.' },
];

export default function KMeansPage() {
  return (
    <ToolPageLayout
      name="K-Means clustering visualization"
      description="Watch K-Means alternate between assigning points and moving centroids. Add observations, change the number of clusters, and inspect inertia one iteration at a time."
      pathname="/playground/kmeans"
      sectionName="Playground"
      sectionPath="/playground"
      eyebrow="Interactive clustering playground"
      activityLabel="Live K-Means model"
      tool={<KMeansPlayground />}
      faq={faq}
      related={[
        { href: '/docs/apis/clusters/kmeans', title: 'K-Means JavaScript API', description: 'Constructor options, fitPredict, centroids, inertia, and reproducibility.' },
        { href: '/playground/pca', title: 'PCA visualization', description: 'Project high-dimensional observations before exploring cluster structure.' },
        { href: '/playground/knn', title: 'KNN visualization', description: 'Compare centroid assignment with labeled nearest-neighbor voting.' },
      ]}
    >
      <h2>What K-Means clustering does</h2>
      <p>K-Means partitions unlabeled observations into a chosen number of groups. Each group is represented by a centroid, the coordinate-wise mean of its members. The algorithm seeks a partition with small within-cluster squared distances, formalized as inertia or within-cluster sum of squares. It is one of the most widely used baselines for exploratory segmentation.</p>
      <p>Typical applications include grouping customers by behavior, compressing image colors, organizing documents after embedding, initializing other models, and summarizing a large set with representative centers. K-Means is fast and easy to interpret when its geometric assumptions fit the data, but the simplicity can also conceal important limitations.</p>

      <h2>An interactive K-Means visualization in JavaScript</h2>
      <p>The playground records each Lloyd iteration: assign every point to the nearest center, recompute centers from assigned points, and repeat. Select a preset, change <em>k</em>, add observations, or advance one iteration at a time. Colors show current assignments and the large cross-marked circles show centroids.</p>
      <p>The visible history is computed explicitly so every transition can be inspected. Alongside it, the page fits <code>Clusters.KMeans</code> from <code>@kanaries/ml</code> with the same initial centers. The displayed centroid delta checks that the educational steps and production library converge to the same result. All fitting happens locally in the browser.</p>

      <h2>Assignment and centroid update steps</h2>
      <p>During assignment, each observation joins the centroid with the smallest squared Euclidean distance. This divides the plane into Voronoi regions with straight boundaries. During update, each centroid moves to the mean of its new members. That mean is the point minimizing the sum of squared distances for a fixed assignment, so the objective cannot increase after an update.</p>
      <p>These alternating improvements eventually stabilize, but they guarantee only a local optimum. Reset the visualization to see deterministic farthest-first seeds used for clarity. Production K-Means commonly uses k-means++, which spreads initial centers probabilistically, then repeats fitting several times and retains the lowest-inertia run.</p>

      <h2>Choosing k without fooling yourself</h2>
      <p>The algorithm cannot discover how many groups you intended; <em>k</em> is an input. Inertia always falls or stays equal as <em>k</em> grows, reaching zero when every distinct point can become a center. Therefore, selecting the model with the smallest raw inertia would simply favor the largest allowed <em>k</em>.</p>
      <p>An elbow plot looks for diminishing improvements, while the silhouette score compares cohesion with separation. Stability across resamples or initializations is another useful signal. Most importantly, clusters should support a real decision. A mathematically tidy partition can be useless when it does not align with actionable differences or when sensitive attributes create harmful segments.</p>

      <h2>Assumptions revealed by the presets</h2>
      <p>The blobs preset matches K-Means well: groups are compact, separated, and roughly spherical. The moons preset violates that shape assumption. Even when two curved bands are visually obvious, nearest-centroid regions stay convex and cut across them. Density-based or graph-based clustering is often a better fit for such geometry.</p>
      <p>The uneven preset combines different cluster sizes and spreads. Squared distance gives faraway observations substantial influence, and a large diffuse group may be divided while small groups are merged. Outliers can pull means dramatically because a centroid is not a robust statistic. Scaling matters too: high-range features dominate distance unless units are standardized or deliberately weighted.</p>

      <h2>JavaScript K-Means and scikit-learn</h2>
      <p>The code comparison uses matching concepts: number of clusters, tolerance, maximum iterations, random seed, and multiple initializations. The JavaScript estimator exposes <code>fitPredict</code>, <code>getCentroids</code>, and <code>getInertia</code>; scikit-learn provides the analogous fitted attributes. Small differences can arise from initialization sequences, tie handling, or stopping rules, so compare objective quality and aligned centers rather than raw label numbers.</p>
      <p>Read the <Link href="/docs/apis/clusters/kmeans">K-Means JavaScript API guide</Link> for detailed options. Use the <Link href="/playground/pca">PCA visualization</Link> to inspect high-dimensional structure before clustering, or compare this unsupervised centroid rule with labeled voting in the <Link href="/playground/knn">KNN visualization</Link>.</p>

      <h2>A practical clustering workflow</h2>
      <p>Define an observation and feature set that matches the decision the clusters will support. Remove identifiers and leakage, handle missing values, and scale features according to meaningful differences. Fit several seeds for each candidate <em>k</em>, then compare inertia, silhouette, stability, cluster sizes, and sensitivity to outliers. Visualize more than a single convenient projection, because a two-dimensional view can hide separation or invent apparent overlap.</p>
      <p>After selecting a solution, profile clusters using variables that were not allowed to dominate fitting. Give each group a descriptive interpretation, but avoid treating an algorithmic assignment as a natural or permanent identity. Test whether the segmentation changes an outcome through a controlled intervention. In deployment, store preprocessing and centers together, assign new observations with the same distance rule, and monitor feature drift, centroid distance, and cluster proportions. A surge in faraway points can mean the model no longer represents the population. Retraining should repeat validation rather than silently replacing centers, especially when cluster IDs drive customer experiences or operational policies.</p>
    </ToolPageLayout>
  );
}
