---
title: asyncMode in JavaScript with @kanaries/ml
description: Learn what asyncMode does, when to use it, and how to run synchronous machine learning work in JavaScript without blocking browser or Node.js execution.
---

# asyncMode in JavaScript

## Helper overview

`asyncMode` is a utility that wraps a synchronous function so it runs in a worker-like execution path. In the browser that means a Web Worker, and in Node.js it means a worker thread.

This helper is especially useful when:

- model training or inference is CPU-intensive
- you need to keep browser interfaces responsive during ML work
- event-loop-sensitive Node.js services should avoid blocking synchronous computation

## JavaScript implementation

`@kanaries/ml` provides `asyncMode` as a JavaScript and TypeScript helper for moving heavy synchronous work off the main execution path. This is particularly useful when ML logic already lives in a JS application but should not freeze the UI, delay input handling, or block other latency-sensitive work.

If someone searches for "run machine learning in a Web Worker" or "async ML execution in JavaScript", this page should make it clear that `asyncMode` is the relevant integration helper.

## Quick start

```ts
import { utils } from '@kanaries/ml';

const heavy = (x: number) => x * x;
const runAsync = utils.asyncMode(heavy);

const result = await runAsync(5);
console.log(result);
```

## Detailed API reference

```ts
asyncMode<P extends any[], R>(fn: (...args: P) => R): (...args: P) => Promise<R>
```

`asyncMode` takes a synchronous function and returns an async wrapper that executes the work off the main thread when supported by the runtime.

### Usage notes

- Wrap CPU-heavy functions rather than tiny helper functions.
- Use this helper for responsiveness, not as a substitute for model-level optimization.
- Profile long-running workloads and consider batching when jobs are still too large for smooth UX.

### JavaScript deployment notes

- In browser products, this is one of the simplest ways to keep training or inference from freezing the UI.
- In Node.js services, it helps isolate expensive CPU work from the main event loop.
- Pair it with model pages in this documentation when you need a production-friendly execution path.
