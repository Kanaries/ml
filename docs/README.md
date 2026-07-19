# docs

This is a Next.js application generated with
[Create Fumadocs](https://github.com/fuma-nama/fumadocs).

Run development server:

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
```

Open http://localhost:3000 with your browser to see the result.

## Explore

In the project, you can see:

- `lib/source.ts`: Code for content source adapter, [`loader()`](https://fumadocs.dev/docs/headless/source-api) provides the interface to access your content.
- `app/layout.config.tsx`: Shared options for layouts, optional but preferred to keep.

| Route                     | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `app/(home)`              | The route group for your landing page and other pages. |
| `app/docs`                | The documentation layout and pages.                    |
| `app/api/search/route.ts` | The Route Handler for search.                          |

### Fumadocs MDX

A `source.config.ts` config file has been included, you can customise different options like frontmatter schema.

Read the [Introduction](https://fumadocs.dev/docs/mdx) for further details.

## Agent-readable documentation

The files in `content/docs` are also the source of truth for the agent-facing
documentation surface:

- `/llms.txt` contains the complete documentation hierarchy and links to the
  Markdown version of every page. `/llm.txt` is a compatibility alias with the
  same generated content.
- Documentation pages expose a Markdown companion URL. Leaf pages append
  `.md`; directory/index pages use `index.html.md`.
- `/api/agent/search?q=...&limit=...` returns structured results with both HTML
  and Markdown URLs.

Run `yarn agent:generate` to refresh these files locally. The command also runs
before `yarn dev` and `yarn build`. If a new MDX component contains meaningful
content, add an agent Markdown renderer in `scripts/generate-agent-docs.mjs`;
generation intentionally fails for unsupported components so content cannot be
silently omitted.

## Learn More

To learn more about Next.js and Fumadocs, take a look at the following
resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js
  features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [Fumadocs](https://fumadocs.vercel.app) - learn about Fumadocs
