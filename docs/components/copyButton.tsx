'use client';

import { useCallback, useState } from 'react';

export function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = useCallback(() => {
    navigator.clipboard?.writeText(value).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      },
      () => {},
    );
  }, [value]);

  return (
    <button
      type="button"
      onClick={onCopy}
      aria-label={copied ? 'Copied to clipboard' : 'Copy install command'}
      style={{
        fontFamily: 'var(--font-jbmono), ui-monospace, monospace',
        fontSize: '0.66rem',
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        padding: '0.4rem 0.65rem',
        border: '1px solid currentColor',
        color: copied ? 'var(--accent)' : 'var(--muted)',
        background: 'transparent',
        cursor: 'pointer',
        transition: 'color 0.2s ease',
        whiteSpace: 'nowrap',
      }}
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}
