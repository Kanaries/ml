import { getMarkdownPath } from '@/lib/agent-docs';
import { docsSearch } from '@/lib/search';
import { source } from '@/lib/source';
import { NextRequest } from 'next/server';

const DEFAULT_LIMIT = 5;
const MAX_LIMIT = 20;
const MAX_QUERY_LENGTH = 200;

function parseLimit(value: string | null): number {
  if (!value) return DEFAULT_LIMIT;

  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return DEFAULT_LIMIT;
  return Math.min(Math.max(parsed, 1), MAX_LIMIT);
}

export async function GET(request: NextRequest): Promise<Response> {
  const query = request.nextUrl.searchParams.get('q')?.trim() ?? '';
  if (!query) {
    return Response.json(
      { error: 'The q query parameter is required.' },
      { status: 400 },
    );
  }
  if (query.length > MAX_QUERY_LENGTH) {
    return Response.json(
      { error: `The q query parameter must be ${MAX_QUERY_LENGTH} characters or fewer.` },
      { status: 400 },
    );
  }

  const limit = parseLimit(request.nextUrl.searchParams.get('limit'));
  const matches = await docsSearch.search(query);
  const seenUrls = new Set<string>();
  const results = [];

  for (const match of matches) {
    if (results.length >= limit) break;
    if (seenUrls.has(match.url)) continue;

    const [htmlPath, hash] = match.url.split('#', 2);
    const resolved = source.getPageByHref(htmlPath);
    if (!resolved) continue;

    const page = resolved.page;
    const markdownPath = getMarkdownPath(page);
    const anchoredMarkdownPath = hash ? `${markdownPath}#${hash}` : markdownPath;

    seenUrls.add(match.url);
    results.push({
      title: page.data.title ?? match.content,
      section: match.type === 'page' ? null : match.type === 'heading' ? match.content : null,
      summary: page.data.description ?? null,
      excerpt: match.type === 'text' ? match.content : page.data.description ?? null,
      type: match.type,
      html_url: match.url,
      markdown_url: anchoredMarkdownPath,
    });
  }

  return Response.json(
    { query, count: results.length, results },
    {
      headers: {
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=3600',
      },
    },
  );
}
