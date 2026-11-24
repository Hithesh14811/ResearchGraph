import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const API_TARGET = process.env.VITE_PROXY_TARGET ?? "http://localhost:8000";

// Stamps the built index.html with the commit it was built from, so the deploy pipeline can
// confirm which build the hosted dashboard serves. GIT_COMMIT comes from the Docker build;
// Render sets RENDER_GIT_COMMIT for static-site builds.
const COMMIT = process.env.GIT_COMMIT || process.env.RENDER_GIT_COMMIT || "";

const revisionMeta = (): Plugin => ({
  name: "revision-meta",
  transformIndexHtml: () =>
    COMMIT ? [{ tag: "meta", attrs: { name: "revision", content: COMMIT }, injectTo: "head" }] : [],
});

// The dashboard talks to the API under /api; in development Vite proxies that prefix to
// the FastAPI server (in Docker, nginx does the same), so the browser never needs CORS.
export default defineConfig({
  plugins: [react(), tailwindcss(), revisionMeta()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // FastAPI's Swagger UI (/api/docs) fetches its schema from this absolute path.
      "/openapi.json": { target: API_TARGET, changeOrigin: true },
    },
  },
});
