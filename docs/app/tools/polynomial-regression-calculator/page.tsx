import type { Metadata } from 'next';
import Link from 'next/link';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';
import { WorkflowCalculator } from '@/components/tools/WorkflowCalculator';
const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const title = 'Polynomial Regression Calculator';
const description =
    'Fit a polynomial curve to CSV data, compare holdout error, inspect residuals, and export predictions with reproducible JavaScript and Python code.';
export const metadata: Metadata = {
    title: { absolute: `${title} — Free CSV Tool` },
    description,
    alternates: { canonical: `${siteUrl}/tools/polynomial-regression-calculator` },
    openGraph: { title, description, url: `${siteUrl}/tools/polynomial-regression-calculator` },
};
const faq: ToolFaq[] = [
    {
        question: 'How is validation calculated?',
        answer: 'Every fifth row, starting with the first data row, is held out. A separate model fits scaling and coefficients on the remaining rows. The displayed curve is fitted to all rows.',
    },
    {
        question: 'What does the equation use?',
        answer: 'The polynomial is expressed in z = (x - center) / scale to improve numerical conditioning. Keep this transformation with the coefficients when predicting.',
    },
    {
        question: 'Can I use this for a time-series forecast?',
        answer: 'The fixed row split is intended for exploratory comparisons. For time series, use a chronological validation procedure. High-degree polynomials can extrapolate poorly outside the observed range.',
    },
    {
        question: 'Is my CSV uploaded?',
        answer: 'No. Parsing, fitting and chart rendering run in this browser tab. Files are limited to 1 MB, 2,000 rows and 30 columns; clustering also limits the input to 500 customers and 12 features.',
    },
];
const related = [
    {
        href: '/docs/guides/choose-polynomial-degree',
        title: 'How to Choose a Polynomial Degree Without Overfitting',
        description:
            'Compare polynomial fits using held-out errors and residuals instead of choosing the curve with the highest training R\u00b2.',
    },
    {
        href: '/docs/guides/outliers-regression',
        title: 'How Outliers Change a Regression Line',
        description:
            'Inspect leverage and residuals with paired fits before deciding whether an unusual observation should be corrected or retained.',
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
            pathname="/tools/polynomial-regression-calculator"
            tool={<WorkflowCalculator kind="polynomial" />}
            faq={faq}
            related={related}
        >
            <h2>Fit a curve and inspect what it misses</h2>
            <p>
                Choose different X and Y columns, set a degree from 1 to 6, and calculate. The curve, equation, fitted
                predictions and residual chart describe a least-squares model fitted with @kanaries/ml. The residual is
                observed minus predicted Y. At least ten rows are required for the separate holdout comparison.
            </p>
            <h2>Use validation to compare degree</h2>
            <p>
                Training error often falls as polynomial degree increases. Holdout error measures how a separate fitted
                model performs on excluded rows. Compare degrees on the same data and split, then favor a simpler model
                when performance is similar. A constant target has undefined R²; the tool reports that explicitly.
            </p>
            <h2>Reproduce the full transformation</h2>
            <p>
                The calculation centers and scales X before generating polynomial powers. JavaScript expands those
                features and uses Linear.LinearRegression; Python combines StandardScaler, PolynomialFeatures and
                LinearRegression. The code panel includes current data and reproduces the holdout split. Keep
                full-precision code for reproduction rather than copying the rounded display equation.
            </p>
            <h2>Work through a practical example</h2>
            <p>
                <Link href="/docs/guides/choose-polynomial-degree">
                    How to Choose a Polynomial Degree Without Overfitting
                </Link>{' '}
                explains the input, assumptions and expected output. Continue with{' '}
                <Link href="/docs/guides/outliers-regression">How Outliers Change a Regression Line</Link> to compare
                methods and interpret results.
            </p>
        </ToolPageLayout>
    );
}
