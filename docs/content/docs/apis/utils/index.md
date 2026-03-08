---
title: Machine Learning Utilities in JavaScript with @kanaries/ml
description: Explore machine learning workflow utilities in JavaScript and TypeScript with @kanaries/ml, including async execution helpers for browser and Node.js applications.
---

# Machine Learning Utilities in JavaScript

## Module overview

The Utils module contains workflow helpers that make machine learning code easier to integrate into real JavaScript applications. These utilities are useful when the challenge is not choosing a model, but making model execution behave well inside browser or Node.js environments.

This module is a strong fit when:

- you need non-blocking execution around training or inference
- application responsiveness matters as much as model quality
- you want ML helpers that work naturally with the rest of a JS stack

## JavaScript implementation

`@kanaries/ml` provides utility helpers in JavaScript and TypeScript so teams can shape execution behavior, integration patterns, and developer ergonomics without leaving the main application runtime. These helpers are especially useful in UI-heavy products and services where ML work should not block interaction or event-loop responsiveness.

If someone searches for "machine learning utilities in JavaScript" or "async ML execution in TypeScript", this module is the right entry point.

## Quick navigation

- [asyncMode](asyncMode): run synchronous functions in a worker-like async execution path

## Detailed module guide

### How to use this module

1. Start with [asyncMode](asyncMode) when training or inference should not block the main browser thread or Node.js event loop.
2. Pair utilities with model modules so application behavior stays predictable under load.
3. Treat utility helpers as part of production readiness, not just developer convenience.

### JavaScript deployment notes

- Utilities matter most when ML code runs inside interactive or latency-sensitive product surfaces.
- Use them to keep heavy work off the main thread when responsiveness is part of the user experience.
- Combine them with the rest of the API catalog to build production-friendly JS ML workflows.
