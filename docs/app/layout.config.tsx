import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

/**
 * Shared layout configurations
 *
 * you can customise layouts individually from:
 * Home Layout: app/(home)/layout.tsx
 * Docs Layout: app/docs/layout.tsx
 */
export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <>
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
          aria-label="@kanaries/ml logo"
        >
          <rect x={0.6} y={0.6} width={22.8} height={22.8} rx={4} fill="none" stroke="currentColor" strokeWidth={1.4} />
          <line x1={4} y1={20} x2={20} y2={4} stroke="currentColor" strokeWidth={1.4} strokeDasharray="2.4 2.2" opacity={0.7} />
          <circle cx={7.5} cy={15.5} r={2.2} fill="currentColor" />
          <rect x={14.2} y={6.3} width={4} height={4} fill="currentColor" />
        </svg>
        <span style={{ fontWeight: 600, letterSpacing: '-0.01em' }}>@kanaries/ml</span>
      </>
    ),
  },
  // see https://fumadocs.dev/docs/ui/navigation/links
  links: [
    {
      text: 'Docs Home',
      url: '/docs',
    },
    {
      text: 'API Reference',
      url: '/docs/apis',
    },
    {
      text: 'GitHub',
      url: 'https://github.com/Kanaries/ml',
    },
  ],
};
