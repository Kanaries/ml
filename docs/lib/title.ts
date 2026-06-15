export function getDisplayTitle(title: string | undefined): string {
  if (!title) return '';

  return title
    .replace(/\s+with\s+@kanaries\/ml\s*$/i, '')
    .replace(/\s+in\s+(?:JavaScript|TypeScript)(?:\s+(?:and|or)\s+(?:JavaScript|TypeScript))*\s*$/i, '')
    .replace(/\s+JavaScript\s+implementation\s*$/i, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

export function getTextFromNode(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return String(value);

  if (Array.isArray(value)) {
    const text = value.map((item) => getTextFromNode(item)).filter(Boolean).join('');
    return text || undefined;
  }

  if (value && typeof value === 'object' && 'props' in value) {
    return getTextFromNode((value as { props?: { children?: unknown } }).props?.children);
  }

  return undefined;
}

export function getDisplayTitleNode<T>(value: T): T | string {
  const text = getTextFromNode(value);
  if (!text) return value;

  return getDisplayTitle(text) || text;
}
