import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Vite configuration for the REE Digital Twin frontend.
 *
 * In development (`npm run dev`):
 *   • The React dev server runs on http://localhost:5173
 *   • API calls to /api/* and /stream/* are proxied to the Flask server
 *     on http://localhost:5000 so CORS is not an issue.
 *   • GLSL shader files are imported as raw strings via ?raw query.
 *
 * In production (`npm run build`):
 *   • Output goes to ../static/dist/ so Flask's static_folder can serve it.
 *   • Set base to '/static/dist/' to match Flask's URL prefix.
 *
 * Performance notes (addressing the "frame rate" concern):
 *   • Heavy spectral math (SAM, PCA) runs in a Web Worker
 *     (src/workers/spectralWorker.js) — never on the main thread.
 *   • OGI / Schlieren filters run as WebGL2 fragment shaders
 *     (static/glsl/ogi.frag, schlieren.frag) — fully on the GPU.
 *   • React rendering is batched via useTransition for the element grid.
 */
export default defineConfig({
  plugins: [react()],
  base: './',

  // Proxy API calls to the Flask element_server.py
  server: {
    proxy: {
      '/api':    { target: 'http://localhost:5000', changeOrigin: true },
      '/stream': { target: 'http://localhost:5000', changeOrigin: true,
                   ws: false /* SSE, not WS */ },
      '/static': { target: 'http://localhost:5000', changeOrigin: true },
    },
  },

  // Build output goes into Flask's static folder
  build: {
    outDir:      '../static/dist',
    emptyOutDir: true,
  },

  // Allow importing GLSL files as strings
  assetsInclude: ['**/*.vert', '**/*.frag', '**/*.glsl'],
});
