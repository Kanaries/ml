This is a library for machine learning for frontend, it provides sklearn-like API for machine learning, and also some linear algebra functions.

You can test the library with the following command:
```bash
npm run test
```

`examples` contains some visual examples of algorithms, usually to show the effect of algorithms.

The library is designed to be used in both browser and node.js.
The library uses yarn as package manager.

We also have sklearn and numpy installed, so you can use python library to generate output for testing the js library.

## API doc writing guidance

When writing or revising API documentation pages for algorithms, do not use a bare algorithm name as the full title or as the only framing. The page should be written to compete on searches for the JavaScript implementation, not just the generic algorithm name that is already dominated by Python content.

- Titles and descriptions should explicitly emphasize the JavaScript or TypeScript implementation, for example using patterns like `X in JavaScript`, `X in TypeScript`, or `X JavaScript implementation`.
- The description should explain both what the algorithm does and why someone would use the `@kanaries/ml` implementation in browser or Node.js environments.
- Avoid opening the page with raw API signatures or code comparisons only. The page should first establish context and value for the reader.

Preferred structure for each algorithm page:

1. Algorithm overview
   - Explain what the algorithm is.
   - Explain what kinds of problems it solves and what its main characteristics are.
2. JavaScript implementation
   - Explain that this page covers the JavaScript implementation.
   - Emphasize that `@kanaries/ml` lets users run this algorithm in browser or Node.js with a JavaScript API.
3. Quick start example
   - Show a fast usage example.
   - Include a side-by-side comparison when helpful, especially Python vs JavaScript/TypeScript.
4. Detailed API reference
   - Document constructor options, methods, parameters, return values, and behavior details.

Style constraints:

- The article should read like a useful guide, not a dry symbol dump.
- Background and problem framing should come before low-level API details.
- Comparison blocks are useful, but they should appear after the reader understands what the algorithm is and why the JavaScript implementation matters.

