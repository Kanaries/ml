import type { Metadata } from 'next';
import Link from 'next/link';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';
import { WorkflowCalculator } from '@/components/tools/WorkflowCalculator';
const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const title = 'Outlier Calculator';
const description =
    'Find outliers in CSV data with IQR, Z-score or modified Z-score. Review original rows, export flags and charts, and reproduce the calculation locally.';
export const metadata: Metadata = {
    title: { absolute: `${title} — Free CSV Tool` },
    description,
    alternates: { canonical: `${siteUrl}/tools/outlier-calculator` },
    openGraph: { title, description, url: `${siteUrl}/tools/outlier-calculator` },
};
const faq: ToolFaq[] = [
    {
        question: 'Which rules are available?',
        answer: 'IQR uses 1.5 times the interquartile range with linear-interpolated quartiles. Z-score uses population standard deviation and an absolute threshold above 3. Modified Z-score uses median absolute deviation and a threshold above 3.5.',
    },
    {
        question: 'Does a flagged row need to be removed?',
        answer: 'No. Original records remain in the results. Investigate units, data quality and the population before deciding whether to correct, retain or exclude a value.',
    },
    {
        question: 'What if all values are identical?',
        answer: 'IQR flags none. Z-scores are undefined with zero standard deviation, and modified Z-scores are undefined with zero MAD. The tool explains these cases.',
    },
    {
        question: 'Is my CSV uploaded?',
        answer: 'No. Parsing, fitting and chart rendering run in this browser tab. Files are limited to 1 MB, 2,000 rows and 30 columns; clustering also limits the input to 500 customers and 12 features.',
    },
];
const related = [
    {
        href: '/docs/guides/find-outliers-csv',
        title: 'How to Find Outliers in a CSV File Without Python',
        description:
            'Review unusual CSV values with IQR, Z-score and modified Z-score, preserve original records, and export flags without uploading your data.',
    },
    {
        href: '/docs/guides/outlier-methods',
        title: 'IQR vs Z-Score vs Isolation Forest: Compare the Same Dataset',
        description:
            'See why a robust fence, a mean-based score and a multivariate anomaly detector can disagree, with a reproducible CSV example.',
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
            pathname="/tools/outlier-calculator"
            tool={<WorkflowCalculator kind="outlier" />}
            faq={faq}
            related={related}
        >
            <h2>Review unusual observations without losing context</h2>
            <p>
                Paste a CSV with a header row, select a numeric column and calculate. Text IDs and other fields stay
                attached to each observation. Switch the detection rule to compare how assumptions change the flagged
                rows. The chart highlights flagged points in red; the export keeps both flagged and unflagged records.
            </p>
            <h2>Understand the default example</h2>
            <p>
                The sample contains nine values between 10 and 17 and one value of 80. Linear-interpolated quartiles are
                12 and 15.75, producing IQR fences of 6.375 and 21.375. IQR flags 80. The strict population Z-score rule
                does not flag it because its score remains below 3. This is a useful demonstration of how a large value
                changes its own reference mean and spread.
            </p>
            <h2>Choose the unit of analysis</h2>
            <p>
                A single-column rule cannot identify every unusual combination of features. Review comparable
                populations together and keep units consistent. The JavaScript Isolation Forest guide covers a
                multivariate alternative. A statistical flag is not a diagnosis of fraud or proof of a data error.
            </p>
            <h2>Work through a practical example</h2>
            <p>
                <Link href="/docs/guides/find-outliers-csv">How to Find Outliers in a CSV File Without Python</Link>{' '}
                explains the input, assumptions and expected output. Continue with{' '}
                <Link href="/docs/guides/outlier-methods">
                    IQR vs Z-Score vs Isolation Forest: Compare the Same Dataset
                </Link>{' '}
                to compare methods and interpret results.
            </p>
        </ToolPageLayout>
    );
}
