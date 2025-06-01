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
