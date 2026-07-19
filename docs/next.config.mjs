import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  transpilePackages: ['@kanaries/ml'],
  async headers() {
    return [
      {
        source: '/docs/:path*.md',
        headers: [
          {
            key: 'Content-Type',
            value: 'text/markdown; charset=utf-8',
          },
          {
            key: 'X-Robots-Tag',
            value: 'noindex, follow',
          },
        ],
      },
      ...['/llms.txt', '/llm.txt'].map((source) => ({
        source,
        headers: [
          {
            key: 'Content-Type',
            value: 'text/plain; charset=utf-8',
          },
        ],
      })),
    ];
  },
  async redirects() {
    return [
      {
        source: '/docs/apis/linear/lasso',
        destination: '/docs/apis/linear/lassoRegression',
        permanent: true,
      },
      {
        source: '/docs/apis/linear/ridge',
        destination: '/docs/apis/linear/ridgeRegression',
        permanent: true,
      },
      {
        source: '/docs/apis/ensemble/adaboost',
        destination: '/docs/apis/ensemble/adaboostRegressor',
        permanent: true,
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'img.shields.io',
      },
      {
        protocol: 'https',
        hostname: 'kanaries.net',
      },
      {
        protocol: 'https',
        hostname: '*.vercel.app',
      },
    ],
  },
};

export default withMDX(config);
