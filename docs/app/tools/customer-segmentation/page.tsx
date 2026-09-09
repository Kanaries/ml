import type { Metadata } from 'next';
import Link from 'next/link';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';
import { WorkflowCalculator } from '@/components/tools/WorkflowCalculator';
const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const title = 'Customer Segmentation Tool';
const description =
    'Create customer groups from CSV features or transaction-based RFM data. Standardize features, inspect K-Means group profiles, and export customer labels.';
export const metadata: Metadata = {
    title: { absolute: `${title} — Free CSV Tool` },
    description,
    alternates: { canonical: `${siteUrl}/tools/customer-segmentation` },
    openGraph: { title, description, url: `${siteUrl}/tools/customer-segmentation` },
};
const faq: ToolFaq[] = [
    {
        question: 'How is RFM calculated?',
        answer: 'Recency is days since the last transaction relative to an explicit UTC date. Frequency counts transaction rows, and monetary value sums nonnegative amounts. Use one currency and deduplicate orders first.',
    },
    {
        question: 'Does group 0 mean low-value customers?',
        answer: 'No. Cluster numbers are arbitrary labels. Read the group sizes and original-unit feature means before assigning a business description.',
    },
    {
        question: 'Why standardize the features?',
        answer: 'Population standardization puts numeric features on comparable scales. A constant feature becomes zero. This does not fix skew, redundant features or a poorly chosen observation window.',
    },
    {
        question: 'Is my CSV uploaded?',
        answer: 'No. Parsing, fitting and chart rendering run in this browser tab. Files are limited to 1 MB, 2,000 rows and 30 columns; clustering also limits the input to 500 customers and 12 features.',
    },
];
const related = [
    {
        href: '/docs/guides/customer-segmentation-csv',
        title: 'Customer Segmentation from a CSV: An RFM Walkthrough',
        description:
            'Turn dated transactions into recency, frequency and monetary features, then inspect reproducible K-Means customer groups in your browser.',
    },
    {
        href: '/docs/guides/rfm-vs-kmeans',
        title: 'RFM Scoring vs K-Means: Which Customer Groups Can You Explain?',
        description:
            'Compare explicit RFM score rules with distance-based customer clusters, including ties, skew, stability and how to validate a segment.',
    },
    {
        href: '/tools',
        title: 'All data tools',
        description: 'Explore calculators and browser-based machine learning workflows.',
    },
];
export default function Page() {
    return (
        <ToolPageLayout
            compact
            name={title}
            description={description}
            pathname="/tools/customer-segmentation"
            tool={<WorkflowCalculator kind="customer" />}
            faq={faq}
            related={related}
        >
            <h2>Build segments from a checkable customer table</h2>
            <p>
                Use feature mode for one unique customer per row. Every non-ID column must be a numeric feature, so
                remove names, notes and labels before fitting. Alternatively, choose transaction mode and map customer
                ID, date and amount columns to build recency, frequency and monetary totals.
            </p>
            <h2>Inspect profiles before naming groups</h2>
            <p>
                The model standardizes features and fits K-Means with seed 42, five initializations and up to 60
                iterations. Read customer counts and average raw features for every group. The chart shows the first
                feature against cluster ID; it is a partial view of the fitted feature space, not a complete separation
                plot.
            </p>
            <h2>Keep the observation window consistent</h2>
            <p>
                RFM depends on the reference date and input history. Deduplicate orders, use a single currency and
                handle refunds before importing. Export prepared feature rows with customer IDs and labels for review.
                The generated code starts from that prepared matrix; JavaScript and scikit-learn can choose different
                local minima despite equivalent settings.
            </p>
            <h2>Work through a practical example</h2>
            <p>
                <Link href="/docs/guides/customer-segmentation-csv">
                    Customer Segmentation from a CSV: An RFM Walkthrough
                </Link>{' '}
                explains the input, assumptions and expected output. Continue with{' '}
                <Link href="/docs/guides/rfm-vs-kmeans">
                    RFM Scoring vs K-Means: Which Customer Groups Can You Explain?
                </Link>{' '}
                to compare methods and interpret results.
            </p>
        </ToolPageLayout>
    );
}
