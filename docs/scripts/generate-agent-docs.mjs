import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import matter from 'gray-matter';
import remarkGfm from 'remark-gfm';
import remarkMdx from 'remark-mdx';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import { unified } from 'unified';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.resolve(SCRIPT_DIR, '..');
const DEFAULT_CONTENT_ROOT = path.join(DOCS_ROOT, 'content', 'docs');
const DEFAULT_PUBLIC_ROOT = path.join(DOCS_ROOT, 'public');
const DEFAULT_SITE_URL = 'https://ml.kanaries.net';

const MARKDOWN_EXTENSIONS = new Set(['.md', '.mdx']);

function toPosix(value) {
  return value.split(path.sep).join('/');
}

function normalizeSiteUrl(value) {
  return (value || DEFAULT_SITE_URL).replace(/\/$/, '');
}

function markdownPathFor(sourceRelative, htmlPath) {
  const sourceName = path.posix.basename(sourceRelative);
  const isIndex = /^index\.mdx?$/i.test(sourceName);
  const normalizedHtmlPath = htmlPath === '/' ? '' : htmlPath.replace(/\/$/, '');

  return isIndex ? `${normalizedHtmlPath}/index.html.md` : `${normalizedHtmlPath}.md`;
}

async function walkMarkdownFiles(directory) {
  const result = [];
  const entries = await fs.readdir(directory, { withFileTypes: true });

  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      result.push(...await walkMarkdownFiles(absolutePath));
    } else if (MARKDOWN_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      result.push(absolutePath);
    }
  }

  return result;
}

async function readJsonIfPresent(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    if (error && error.code === 'ENOENT') return null;
    throw error;
  }
}

function sourceRelativeToSlugs(sourceRelative) {
  const withoutExtension = sourceRelative.replace(/\.mdx?$/i, '');
  const parts = withoutExtension.split('/');
  if (parts.at(-1)?.toLowerCase() === 'index') parts.pop();
  return parts;
}

function getAttribute(node, name, sourceRelative) {
  const attribute = node.attributes.find(
    (candidate) => candidate.type === 'mdxJsxAttribute' && candidate.name === name,
  );
  if (!attribute) {
    throw new Error(`${sourceRelative}: <${node.name}> is missing the ${name} property.`);
  }
  if (typeof attribute.value === 'string') return attribute.value;
  if (!attribute.value || attribute.value.type !== 'mdxJsxAttributeValueExpression') {
    throw new Error(`${sourceRelative}: <${node.name}> has an unsupported ${name} property.`);
  }

  const expression = attribute.value.data?.estree?.body?.[0]?.expression;
  if (expression?.type === 'Literal' && typeof expression.value === 'string') {
    return expression.value;
  }
  if (
    expression?.type === 'TaggedTemplateExpression'
    && expression.tag?.type === 'MemberExpression'
    && expression.tag.object?.name === 'String'
    && expression.tag.property?.name === 'raw'
    && expression.quasi?.expressions?.length === 0
    && expression.quasi?.quasis?.length === 1
  ) {
    return expression.quasi.quasis[0].value.raw;
  }

  throw new Error(
    `${sourceRelative}: <${node.name}> must use a string literal or String.raw template for ${name}.`,
  );
}

function text(value) {
  return { type: 'text', value };
}

function paragraph(value) {
  return { type: 'paragraph', children: [text(value)] };
}

function componentToMarkdown(node, record, siteUrl) {
  if (node.name === 'AlgorithmComparison') {
    const title = getAttribute(node, 'title', record.sourceRelative);
    const pythonCode = getAttribute(node, 'pythonCode', record.sourceRelative);
    const tsCode = getAttribute(node, 'tsCode', record.sourceRelative);

    return [
      { type: 'heading', depth: 3, children: [text(`${title}: Python and JavaScript / TypeScript`)] },
      paragraph('The Python example uses scikit-learn; the TypeScript example uses @kanaries/ml in browser or Node.js runtimes.'),
      { type: 'heading', depth: 4, children: [text('Python (scikit-learn)')] },
      { type: 'code', lang: 'python', value: pythonCode },
      { type: 'heading', depth: 4, children: [text('JavaScript / TypeScript (@kanaries/ml)')] },
      { type: 'code', lang: 'ts', value: tsCode },
    ];
  }

  const interactiveDescriptions = {
    ClusteringComparison: 'The HTML version includes an interactive comparison of clustering algorithms. The agent-readable algorithm links and selection guidance continue below.',
    LogisticRegressionDemo: 'The HTML version includes an interactive logistic regression visualization. The complete runnable example and API details are included in this Markdown page.',
  };
  const description = interactiveDescriptions[node.name];
  if (description) {
    return [{
      type: 'blockquote',
      children: [{
        type: 'paragraph',
        children: [
          text(description),
          text(' '),
          { type: 'link', url: `${siteUrl}${record.htmlPath}`, children: [text('Open the HTML page')] },
          text('.'),
        ],
      }],
    }];
  }

  throw new Error(
    `${record.sourceRelative}: no agent Markdown renderer is registered for <${node.name}>.`,
  );
}

function transformMdxNodes(children, record, siteUrl) {
  return children.flatMap((node) => {
    if (node.type === 'mdxjsEsm') return [];
    if (node.type === 'mdxJsxFlowElement') {
      return componentToMarkdown(node, record, siteUrl);
    }
    if (
      node.type === 'mdxJsxTextElement'
      || node.type === 'mdxFlowExpression'
      || node.type === 'mdxTextExpression'
    ) {
      throw new Error(
        `${record.sourceRelative}: unsupported MDX node ${node.type}${node.name ? ` <${node.name}>` : ''}.`,
      );
    }
    if (Array.isArray(node.children)) {
      node.children = transformMdxNodes(node.children, record, siteUrl);
    }
    return [node];
  });
}

function splitLink(url) {
  const hashIndex = url.indexOf('#');
  if (hashIndex === -1) return { pathname: url, hash: '' };
  return { pathname: url.slice(0, hashIndex), hash: url.slice(hashIndex) };
}

function findRelativePage(record, pathname, recordsBySource) {
  const decodedPath = decodeURIComponent(pathname).replace(/[?#].*$/, '');
  const currentDirectory = path.posix.dirname(record.sourceRelative);
  const resolved = path.posix.normalize(path.posix.join(currentDirectory, decodedPath));
  const extension = path.posix.extname(resolved).toLowerCase();
  const candidates = MARKDOWN_EXTENSIONS.has(extension)
    ? [resolved]
    : [`${resolved}.md`, `${resolved}.mdx`, `${resolved}/index.md`, `${resolved}/index.mdx`];

  for (const candidate of candidates) {
    const page = recordsBySource.get(candidate);
    if (page) return page;
  }
  return null;
}

function rewriteInternalLinks(tree, record, recordsBySource, recordsByHtmlPath) {
  const visitNode = (node) => {
    if (node.type === 'link' || node.type === 'definition') {
      const url = node.url;
      if (
        typeof url === 'string'
        && url
        && !url.startsWith('#')
        && !/^[a-z][a-z\d+.-]*:/i.test(url)
        && !url.startsWith('//')
      ) {
        const { pathname, hash } = splitLink(url);
        let target = null;
        if (pathname.startsWith('/docs')) {
          target = recordsByHtmlPath.get(pathname.replace(/\/$/, ''));
        } else if (!pathname.startsWith('/')) {
          target = findRelativePage(record, pathname, recordsBySource);
        }
        if (target) node.url = `${target.markdownPath}${hash}`;
      }
    }

    if (Array.isArray(node.children)) {
      for (const child of node.children) visitNode(child);
    }
  };

  visitNode(tree);
}

function yamlString(value) {
  return JSON.stringify(value);
}

function serializeAgentMarkdown(record, markdownBody, siteUrl) {
  const metadata = [
    '---',
    `title: ${yamlString(record.title)}`,
    `description: ${yamlString(record.description)}`,
    `canonical_url: ${yamlString(`${siteUrl}${record.htmlPath}`)}`,
    `markdown_url: ${yamlString(`${siteUrl}${record.markdownPath}`)}`,
    '---',
    '',
  ].join('\n');

  return `${metadata}${markdownBody.trim()}\n`;
}

async function renderPage(record, catalog, siteUrl) {
  const processor = unified()
    .use(remarkParse)
    .use(remarkMdx)
    .use(remarkGfm)
    .use(remarkStringify, {
      bullet: '-',
      fences: true,
      listItemIndent: 'one',
    });
  const tree = processor.parse(record.body);
  tree.children = transformMdxNodes(tree.children, record, siteUrl);
  rewriteInternalLinks(tree, record, catalog.bySource, catalog.byHtmlPath);

  if (!tree.children.some((node) => node.type === 'heading' && node.depth === 1)) {
    tree.children.unshift({ type: 'heading', depth: 1, children: [text(record.title)] });
  }

  return serializeAgentMarkdown(record, processor.stringify(tree), siteUrl);
}

async function buildCatalog(contentRoot) {
  const files = await walkMarkdownFiles(contentRoot);
  const records = [];

  for (const absolutePath of files) {
    const sourceRelative = toPosix(path.relative(contentRoot, absolutePath));
    const raw = await fs.readFile(absolutePath, 'utf8');
    const parsed = matter(raw);
    if (typeof parsed.data.title !== 'string' || !parsed.data.title.trim()) {
      throw new Error(`${sourceRelative}: frontmatter title is required.`);
    }
    if (typeof parsed.data.description !== 'string' || !parsed.data.description.trim()) {
      throw new Error(`${sourceRelative}: frontmatter description is required.`);
    }

    const slugs = sourceRelativeToSlugs(sourceRelative);
    const htmlPath = slugs.length === 0 ? '/docs' : `/docs/${slugs.join('/')}`;
    const markdownPath = markdownPathFor(sourceRelative, htmlPath);

    records.push({
      sourceAbsolute: absolutePath,
      sourceRelative,
      sourceBaseName: path.posix.basename(sourceRelative).replace(/\.mdx?$/i, ''),
      slugs,
      htmlPath,
      markdownPath,
      outputRelative: markdownPath.replace(/^\//, ''),
      title: parsed.data.title.trim(),
      description: parsed.data.description.trim().replace(/\s+/g, ' '),
      body: parsed.content,
    });
  }

  return {
    records,
    bySource: new Map(records.map((record) => [record.sourceRelative, record])),
    byHtmlPath: new Map(records.map((record) => [record.htmlPath, record])),
  };
}

function pageOrder(record, metaPages) {
  if (record.sourceBaseName === 'index') return -1;
  const index = metaPages.indexOf(record.sourceBaseName);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function llmsListItem(record, siteUrl) {
  return `- [${record.title}](${siteUrl}${record.markdownPath}): ${record.description}`;
}

async function generateLlms(catalog, contentRoot, siteUrl) {
  const apiMeta = await readJsonIfPresent(path.join(contentRoot, 'apis', 'meta.json'));
  const moduleOrder = (apiMeta?.pages ?? []).filter((entry) => entry !== 'index');
  const startPages = catalog.records.filter(
    (record) => record.slugs.length === 0 || (record.slugs.length === 1 && record.slugs[0] === 'apis'),
  );
  const moduleGroups = new Map();
  const optionalPages = [];

  for (const record of catalog.records) {
    if (startPages.includes(record)) continue;
    if (record.slugs[0] === 'apis' && record.slugs[1]) {
      const key = record.slugs[1];
      if (!moduleGroups.has(key)) moduleGroups.set(key, []);
      moduleGroups.get(key).push(record);
    } else {
      optionalPages.push(record);
    }
  }

  const lines = [
    '# @kanaries/ml Documentation',
    '',
    '> @kanaries/ml is a scikit-learn-style machine learning library for JavaScript and TypeScript that runs in browser and Node.js environments.',
    '',
    'Use the Markdown links below for retrieval and reading. HTML pages are the canonical human-facing documentation. The API follows familiar `fit`, `predict`, and transformation patterns for classical machine learning.',
    '',
    'Package: https://www.npmjs.com/package/@kanaries/ml',
    'Repository: https://github.com/Kanaries/ml',
    `Human documentation: ${siteUrl}/docs`,
    `Agent search: ${siteUrl}/api/agent/search?q={query}&limit={1-20}`,
    '',
    '## Start here',
    '',
    ...startPages.sort((a, b) => a.slugs.length - b.slugs.length).map((record) => llmsListItem(record, siteUrl)),
  ];

  const orderedModules = [...moduleGroups.keys()].sort((a, b) => {
    const aIndex = moduleOrder.indexOf(a);
    const bIndex = moduleOrder.indexOf(b);
    if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  for (const moduleName of orderedModules) {
    const moduleMeta = await readJsonIfPresent(path.join(contentRoot, 'apis', moduleName, 'meta.json'));
    const title = moduleMeta?.title ?? moduleName;
    const metaPages = moduleMeta?.pages ?? [];
    const records = moduleGroups.get(moduleName).sort((a, b) => {
      const order = pageOrder(a, metaPages) - pageOrder(b, metaPages);
      return order || a.title.localeCompare(b.title);
    });
    lines.push('', `## ${title}`, '', ...records.map((record) => llmsListItem(record, siteUrl)));
  }

  if (optionalPages.length > 0) {
    lines.push(
      '',
      '## Optional',
      '',
      ...optionalPages.sort((a, b) => a.htmlPath.localeCompare(b.htmlPath)).map((record) => llmsListItem(record, siteUrl)),
    );
  }

  return `${lines.join('\n')}\n`;
}

export async function generateAgentDocs(options = {}) {
  const contentRoot = path.resolve(options.contentRoot ?? DEFAULT_CONTENT_ROOT);
  const publicRoot = path.resolve(options.publicRoot ?? DEFAULT_PUBLIC_ROOT);
  const siteUrl = normalizeSiteUrl(options.siteUrl ?? process.env.NEXT_PUBLIC_SITE_URL);
  const catalog = await buildCatalog(contentRoot);
  const generatedDocsRoot = path.join(publicRoot, 'docs');

  await fs.rm(generatedDocsRoot, { recursive: true, force: true });

  for (const record of catalog.records) {
    const outputPath = path.join(publicRoot, record.outputRelative);
    const markdown = await renderPage(record, catalog, siteUrl);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, markdown, 'utf8');
  }

  const llms = await generateLlms(catalog, contentRoot, siteUrl);
  await fs.mkdir(publicRoot, { recursive: true });
  await Promise.all([
    fs.writeFile(path.join(publicRoot, 'llms.txt'), llms, 'utf8'),
    fs.writeFile(path.join(publicRoot, 'llm.txt'), llms, 'utf8'),
  ]);

  return {
    pageCount: catalog.records.length,
    pages: catalog.records.map(({ body: _body, sourceAbsolute: _sourceAbsolute, ...record }) => record),
    llms,
  };
}

const invokedPath = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : '';
if (invokedPath === import.meta.url) {
  generateAgentDocs()
    .then(({ pageCount }) => {
      process.stdout.write(`Generated ${pageCount} agent-readable documentation pages.\n`);
    })
    .catch((error) => {
      process.stderr.write(`${error.stack ?? error}\n`);
      process.exitCode = 1;
    });
}
