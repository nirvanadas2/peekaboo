import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 1000,
  },
  optimizeDeps: {
    include: ["three", "gsap", "lenis"],
  },
  server: {
    proxy: {
      // peekaboo/api.py, run separately: uvicorn peekaboo.api:app --reload
      // (PHASE7.md). Proxied rather than called cross-origin so the
      // frontend's fetch calls stay same-origin in dev, with no CORS
      // config needed on the Python side.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
