---
title: asyncMode
description: Run a synchronous function in a worker
---

# Utils.asyncMode

```ts
asyncMode<P extends any[], R>(fn: (...args: P) => R): (...args: P) => Promise<R>
```

This helper wraps a synchronous function so it runs in a Web Worker in the browser or a worker thread in Node.js.

```ts
const heavy = (x: number) => x * x;
const runAsync = asyncMode(heavy);
const result = await runAsync(5);
```

This function is useful for CPU intensive operations.

## Practical guide: asyncMode in JavaScript and TypeScript

asyncMode helps run heavy ML computation without blocking the browser UI or event-loop-sensitive Node.js services.

### When to use asyncMode
- You need responsive user interfaces during model training or inference.
- CPU-heavy operations should run in worker-like or deferred execution paths.
- You want predictable runtime behavior in interactive apps.

### Implementation workflow
1. Wrap model operations with asyncMode-compatible execution flow.
2. Move expensive workloads off the main thread when possible.
3. Profile latency and adjust batching/chunking for smooth UX.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `asyncMode JavaScript`
- `asyncMode TypeScript`
- `asyncMode browser machine learning`
- `@kanaries/ml asyncMode`

