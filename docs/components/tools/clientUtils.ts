const DEFAULT_SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export function downloadText(filename: string, content: string, type = 'text/csv;charset=utf-8') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadSvgAsPng(svg: SVGSVGElement | null, filename: string) {
  if (!svg) return;
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const sourceElements = [svg, ...Array.from(svg.querySelectorAll('*'))];
  const clonedElements = [clone, ...Array.from(clone.querySelectorAll('*'))];
  sourceElements.forEach((sourceElement, index) => {
    const targetElement = clonedElements[index] as SVGElement | undefined;
    if (!targetElement) return;
    const computed = getComputedStyle(sourceElement);
    const properties = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray', 'font-family', 'font-size', 'font-weight'];
    const inlined = properties
      .map((property) => `${property}:${computed.getPropertyValue(property)}`)
      .join(';');
    targetElement.setAttribute('style', inlined);
  });
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const viewBox = svg.viewBox.baseVal;
  const width = Math.max(600, viewBox.width * 2);
  const height = Math.max(400, viewBox.height * 2);
  const source = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.fillStyle = '#fffdf8';
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    URL.revokeObjectURL(url);
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };
  image.src = url;
}

const MAX_URL_STATE_CHARS = 3500;
const HASH_PREFIX = '#s=';

function toBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  const b64 = typeof btoa === 'function' ? btoa(binary) : Buffer.from(bytes).toString('base64');
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(encoded: string): Uint8Array {
  const padded = encoded.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(encoded.length / 4) * 4, '=');
  if (typeof atob === 'function') {
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }
  return new Uint8Array(Buffer.from(padded, 'base64'));
}

export function encodeToolState(state: Record<string, unknown>): string | null {
  try {
    const json = JSON.stringify(state);
    const bytes = new TextEncoder().encode(json);
    const encoded = toBase64Url(bytes);
    return encoded.length <= MAX_URL_STATE_CHARS ? encoded : null;
  } catch {
    return null;
  }
}

export function decodeToolState<T extends Record<string, unknown>>(encoded: string): T | null {
  try {
    if (!encoded || encoded.length > MAX_URL_STATE_CHARS * 2) return null;
    const bytes = fromBase64Url(encoded);
    const parsed = JSON.parse(new TextDecoder().decode(bytes));
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as T) : null;
  } catch {
    return null;
  }
}

export function readUrlToolState<T extends Record<string, unknown>>(): T | null {
  if (typeof window === 'undefined') return null;
  const hash = window.location.hash;
  if (!hash.startsWith(HASH_PREFIX)) return null;
  return decodeToolState<T>(hash.slice(HASH_PREFIX.length));
}

export function writeUrlToolState(state: Record<string, unknown>): void {
  if (typeof window === 'undefined') return;
  const encoded = encodeToolState(state);
  const nextUrl = encoded
    ? `${window.location.pathname}${window.location.search}${HASH_PREFIX}${encoded}`
    : `${window.location.pathname}${window.location.search}`;
  window.history.replaceState(null, '', nextUrl);
}

export function buildEmbedSnippet(pathname: string, title: string, hash = '', siteUrl = DEFAULT_SITE_URL): string {
  const safeHash = hash.startsWith(HASH_PREFIX) ? hash : '';
  const src = `${siteUrl}${pathname}?embed=1${safeHash}`;
  const escapedTitle = title.replace(/"/g, '&quot;');
  return `<iframe src="${src}" title="${escapedTitle}" width="100%" height="760" style="border:1px solid #e5e7eb;border-radius:12px;" loading="lazy"></iframe>`;
}
