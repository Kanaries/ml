'use client';

import { useEffect, useState } from 'react';
import styles from './toolPage.module.css';
import { buildEmbedSnippet } from './clientUtils';

const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

type ToolShareBarProps = {
  name: string;
  pathname: string;
  activityLabel: string;
};

export function ToolShareBar({ name, pathname, activityLabel }: ToolShareBarProps) {
  const [isEmbed, setIsEmbed] = useState(false);
  const [showEmbed, setShowEmbed] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [embedCopied, setEmbedCopied] = useState(false);
  const [hash, setHash] = useState('');

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const embedded = params.get('embed') === '1' || params.get('embed') === 'true';
    setIsEmbed(embedded);
    setHash(window.location.hash);

    if (embedded) {
      document.documentElement.dataset.toolEmbed = 'true';
    } else {
      delete document.documentElement.dataset.toolEmbed;
    }

    const syncHash = () => setHash(window.location.hash);
    window.addEventListener('hashchange', syncHash);
    return () => {
      window.removeEventListener('hashchange', syncHash);
      delete document.documentElement.dataset.toolEmbed;
    };
  }, []);

  const copyShareUrl = async () => {
    const currentUrl = window.location.href;
    await navigator.clipboard.writeText(currentUrl);
    setHash(window.location.hash);
    setShareCopied(true);
    window.setTimeout(() => setShareCopied(false), 1800);
  };

  const toggleEmbed = () => {
    setHash(window.location.hash);
    setShowEmbed((prev) => !prev);
  };

  const iframeCode = buildEmbedSnippet(pathname, name, hash);

  const copyEmbedCode = async () => {
    const fresh = buildEmbedSnippet(pathname, name, window.location.hash);
    await navigator.clipboard.writeText(fresh);
    setEmbedCopied(true);
    window.setTimeout(() => setEmbedCopied(false), 1800);
  };

  return (
    <>
      <div className={styles.toolHeader}>
        <span className={styles.live}>{activityLabel}</span>
        <div className={styles.toolHeaderActions}>
          {isEmbed ? (
            <a
              className={styles.embedAttribution}
              href={`${SITE_URL}${pathname}${hash}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Powered by @kanaries/ml · Open full tool ↗
            </a>
          ) : (
            <>
              <span className={styles.poweredBy}>Powered by @kanaries/ml</span>
              <button
                type="button"
                className={styles.headerActionBtn}
                onClick={copyShareUrl}
                title="Copy a direct link preserving current calculator state"
              >
                {shareCopied ? 'Link copied' : 'Share link'}
              </button>
              <button
                type="button"
                className={styles.headerActionBtn}
                onClick={toggleEmbed}
                aria-expanded={showEmbed}
                title="Embed this interactive tool on your blog, docs, or course page"
              >
                {showEmbed ? 'Hide embed' : 'Embed </>'}
              </button>
            </>
          )}
        </div>
      </div>

      {!isEmbed && showEmbed && (
        <div className={styles.embedDrawer}>
          <div className={styles.embedDrawerHead}>
            <span>Embed this calculator on your site (includes current state &amp; zero-upload browser execution):</span>
            <button type="button" className={styles.headerActionBtn} onClick={copyEmbedCode}>
              {embedCopied ? 'Copied iframe' : 'Copy iframe'}
            </button>
          </div>
          <pre className={styles.embedCode}>
            <code>{iframeCode}</code>
          </pre>
        </div>
      )}
    </>
  );
}
