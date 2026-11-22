/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 5180, strictPort: true },
  preview: { port: 5180, strictPort: true },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 800 },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
