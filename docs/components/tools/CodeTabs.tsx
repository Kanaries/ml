'use client';

import { useState } from 'react';
import styles from './interactive.module.css';

type CodeTabsProps = {
  javascript: string;
  python: string;
};

export function CodeTabs({ javascript, python }: CodeTabsProps) {
  const [language, setLanguage] = useState<'javascript' | 'python'>('javascript');
  const [copied, setCopied] = useState(false);
  const code = language === 'javascript' ? javascript : python;

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className={styles.codeBox}>
      <div className={styles.codeTabs}>
        <button
          className={`${styles.tab} ${language === 'javascript' ? styles.tabActive : ''}`}
          onClick={() => setLanguage('javascript')}
          type="button"
        >
          JavaScript
        </button>
        <button
          className={`${styles.tab} ${language === 'python' ? styles.tabActive : ''}`}
          onClick={() => setLanguage('python')}
          type="button"
        >
          Python
        </button>
        <button className={`${styles.buttonSecondary} ${styles.codeCopy}`} onClick={copy} type="button">
          {copied ? 'Copied' : 'Copy code'}
        </button>
      </div>
      <pre className={styles.codePre}><code>{code}</code></pre>
    </div>
  );
}
