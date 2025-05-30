import Link from 'next/link';

const FEATURES = [
  {
    title: 'Familiar API',
    description:
      'Scikit–learn inspired interface designed for TypeScript and JavaScript.',
  },
  {
    title: 'Runs Anywhere',
    description: 'Use in browsers and Node.js with zero native dependencies.',
  },
  {
    title: 'Comprehensive Algorithms',
    description:
      'Includes classification, clustering, dimensionality reduction and more.',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-16 py-20 text-center">
      <section className="space-y-6">
        <h1 className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-5xl font-extrabold text-transparent">
          @kanaries/ml
        </h1>
        <p className="max-w-xl text-lg text-fd-muted-foreground">
          A fast and friendly machine learning toolkit for modern web
          development.
        </p>
        <Link
          href="/docs"
          className="inline-block rounded-md bg-fd-foreground px-6 py-3 font-semibold text-white hover:opacity-90"
        >
          Get Started
        </Link>
      </section>
      <section className="grid max-w-4xl grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((f) => (
          <div
            key={f.title}
            className="rounded-lg border border-fd-border bg-fd-background p-6 text-left"
          >
            <h3 className="mb-2 text-lg font-semibold">{f.title}</h3>
            <p className="text-sm text-fd-muted-foreground">{f.description}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
