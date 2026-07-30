import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: 'ui',
  base: './', // Electron loads dist/ui/index.html via file://, where absolute asset
  // paths (Vite's default) resolve to the filesystem root, not this directory.
  plugins: [react()],
  build: {
    outDir: '../dist/ui',
    emptyOutDir: true,
  },
});
