type MarkdownPage = {
  url: string;
  file: {
    path: string;
  };
};

/**
 * Return the public Markdown companion URL for a Fumadocs page.
 * Directory/index pages follow the llms.txt convention and use index.html.md.
 */
export function getMarkdownPath(page: MarkdownPage): string {
  const sourceName = page.file.path.split('/').at(-1) ?? '';
  const isIndex = /^index\.mdx?$/i.test(sourceName);
  const url = page.url === '/' ? '' : page.url.replace(/\/$/, '');

  return isIndex ? `${url}/index.html.md` : `${url}.md`;
}
