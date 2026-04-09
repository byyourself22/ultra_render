import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import path from 'path';

export default defineConfig({
  plugins: [wasm()],
  server: {
    port: 3000,
    fs: {
      // Allow serving files from the parent dir (for pkg/)
      allow: [
        path.resolve(__dirname, '..'),
      ],
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
});
