// ==========================================================
// web/src/main.tsx
// ==========================================================
// Entry point: mounts the app in strict mode with the global styles.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// ------------------------------------------
// Mount
// ------------------------------------------
// Renders the app into #root.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
