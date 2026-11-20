// ==========================================================
// web/vite.config.ts
// ==========================================================
// Build configuration. Relative asset paths let the site run from any host
// or subpath.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ------------------------------------------
// Config
// ------------------------------------------
// React plugin and relative base path.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
