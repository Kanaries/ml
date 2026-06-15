import { docs } from '@/.source';
import { loader } from 'fumadocs-core/source';
import { getDisplayTitle } from '@/lib/title';

// See https://fumadocs.vercel.app/docs/headless/source-api for more info
export const source = loader({
  // it assigns a URL to your pages
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
  pageTree: {
    attachFile(node, file) {
      if (!file) return node;

      return {
        ...node,
        name: getDisplayTitle(file.data.data.title) || node.name,
      };
    },
  },
});
