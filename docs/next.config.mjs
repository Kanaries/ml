import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  transpilePackages: ['@kanaries/ml'],
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
