import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const API_TARGET = process.env.VITE_PROXY_TARGET ?? "http://localhost:8000";

// The dashboard talks to the API under /api; in development Vite proxies that prefix to
// the FastAPI server (in Docker, nginx does the same), so the browser never needs CORS.
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
