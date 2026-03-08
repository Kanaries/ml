type AlgorithmComparisonProps = {
  title: string;
  pythonCode: string;
  tsCode: string;
};

function CodePane({
  label,
  subtitle,
  code,
}: {
  label: string;
  subtitle: string;
  code: string;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-fd-border bg-fd-card">
      <div className="border-b border-fd-border bg-fd-muted/60 px-4 py-3">
        <div className="text-sm font-semibold">{label}</div>
        <div className="text-xs text-fd-muted-foreground">{subtitle}</div>
      </div>
      <pre className="overflow-x-auto p-4 text-[13px] leading-6">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function AlgorithmComparison({
  title,
  pythonCode,
  tsCode,
}: AlgorithmComparisonProps) {
  return (
    <section className="mb-10 rounded-2xl border border-fd-border bg-fd-muted/30 p-6">
      <h2 className="text-xl font-semibold">{title} in Python vs JavaScript / TypeScript</h2>
      <p className="mt-2 text-sm text-fd-muted-foreground">
        If you searched for "{title} in JavaScript" or "{title} in TypeScript", this section maps the familiar
        scikit-learn call to the equivalent <code>@kanaries/ml</code> usage for browser and Node.js runtimes.
      </p>
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <CodePane label="Python" subtitle="scikit-learn" code={pythonCode} />
        <CodePane label="JavaScript / TypeScript" subtitle="@kanaries/ml" code={tsCode} />
      </div>
    </section>
  );
}
