import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
    root: __dirname,
    base: './',
    server: { port: 9000 },
    build: {
        outDir: 'build',
        rollupOptions: {
            input: [resolve(__dirname, 'index.html'), resolve(__dirname, 'e-means.html')],
        },
    },
});
