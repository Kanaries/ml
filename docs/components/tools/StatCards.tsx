import styles from './interactive.module.css';

export type StatItem = {
  label: string;
  value: number | string;
  digits?: number;
};

export function formatMetric(value: number, digits = 3) {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(digits).replace(/\.0+$/, '');
}

export function StatCards({ items }: { items: StatItem[] }) {
  return (
    <div className={styles.statGrid}>
      {items.map((item) => (
        <div className={styles.stat} key={item.label} title={item.label}>
          <div className={styles.statLabel}>{item.label}</div>
          <div className={styles.statValue}>
            {typeof item.value === 'number' ? formatMetric(item.value, item.digits) : item.value}
          </div>
        </div>
      ))}
    </div>
  );
}
