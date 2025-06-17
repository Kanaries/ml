import { defineConfig } from 'vite';
import { resolve } from 'path';
import { readFileSync } from 'fs';

// Read package.json to get dependencies
const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

export default defineConfig({
  // Library build configuration
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'KanariesML',
      formats: ['es', 'cjs'],
      fileName: (format) => `index.${format === 'es' ? 'mjs' : 'js'}`,
    },
    outDir: 'build',
    emptyOutDir: true,
    rollupOptions: {
      // Externalize Node.js built-in modules and dependencies
      external: [
        'worker_threads',
        'fs',
        'path',
        'os',
        'crypto',
        'util',
        'stream',
        'events',
        'buffer',
        'url',
        'querystring',
        'http',
        'https',
        'zlib',
        'child_process',
        'cluster',
        'dgram',
        'dns',
        'net',
        'readline',
        'repl',
        'tls',
        'tty',
        'v8',
        'vm',
        'assert',
        ...Object.keys(pkg.dependencies || {}),
        ...Object.keys(pkg.peerDependencies || {}),
      ],
      output: {
        // Provide global variables for externalized deps in UMD build
        globals: {
          'worker_threads': 'worker_threads',
        },
      },
    },
  },
  // Example development configuration
  root: process.env.NODE_ENV === 'development' ? 'examples' : undefined,
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  server: {
    port: 9000,
  },
}); 