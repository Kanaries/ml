import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const docsRoot = path.resolve(__dirname, '..');
const contentRoot = path.join(docsRoot, 'content', 'docs');
const publicDir = path.join(docsRoot, 'public');
const defaultSiteUrl = 'https://ml.kanaries.net';
const baseSiteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? defaultSiteUrl).replace(/\/$/, '');

/**
 * Recursively walk through the content directory and collect URL paths.
 *
 * @param {string} directory Current directory to scan.
 * @param {string[]} segments Accumulated route segments.
 * @param {Set<string>} routes Collector for discovered routes.
 */
async function collectRoutes(directory, segments, routes) {
  let entries;
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return;
    }
    throw error;
  }

  await Promise.all(
    entries.map(async (entry) => {
      if (entry.name.startsWith('.')) {
        return;
      }

      if (entry.isDirectory()) {
        await collectRoutes(path.join(directory, entry.name), [...segments, entry.name], routes);
        return;
      }

      if (!entry.isFile()) {
        return;
      }

      const ext = path.extname(entry.name).toLowerCase();
      if (ext !== '.md' && ext !== '.mdx') {
        return;
      }

      const name = path.basename(entry.name, ext);
      const routeSegments = name === 'index' ? segments : [...segments, name];
      if (routeSegments.length === 0) {
        routes.add('/');
      } else {
        routes.add(`/${routeSegments.join('/')}`);
      }
    })
  );
}

async function generateSitemap() {
  const routes = new Set(['/']);

  await collectRoutes(contentRoot, ['docs'], routes);

  // Ensure the docs landing page is always included
  routes.add('/docs');

  const sortedRoutes = Array.from(routes).sort((a, b) => {
    if (a === b) return 0;
    if (a === '/') return -1;
    if (b === '/') return 1;
    return a.localeCompare(b);
  });

  const urls = sortedRoutes
    .map((route) => {
      const url = new URL(route, `${baseSiteUrl}/`).toString();
      return `  <url>\n    <loc>${url}</loc>\n  </url>`;
    })
    .join('\n');

  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;

  await fs.mkdir(publicDir, { recursive: true });
  const sitemapPath = path.join(publicDir, 'sitemap.xml');
  await fs.writeFile(sitemapPath, sitemap, 'utf8');

  console.log(`Sitemap written to ${sitemapPath}`);
}

generateSitemap().catch((error) => {
  console.error('Failed to generate sitemap:', error);
  process.exitCode = 1;
});
