// ==========================================================
// web/src/theme.ts
// ==========================================================
// Light and dark theme. The no-flash script in index.html is a copy of
// storedChoice() and applyTheme(); keep them in step.

// ------------------------------------------
// THEME_KEY
// ------------------------------------------
// Local-storage key of the visitor's choice.
export const THEME_KEY = "mts-theme";

// ------------------------------------------
// Theme
// ------------------------------------------
// The two themes.
export type Theme = "light" | "dark";

// ------------------------------------------
// storedChoice
// ------------------------------------------
// The saved choice, or "system" when there is none or storage is blocked.
export function storedChoice(): Theme | "system" {
  try {
    const v = localStorage.getItem(THEME_KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    // Storage blocked: follow the system.
  }
  return "system";
}

// ------------------------------------------
// systemTheme
// ------------------------------------------
// The operating system's preference.
export const systemTheme = (): Theme => (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

// ------------------------------------------
// applyTheme
// ------------------------------------------
// Sets the theme on <html> and matches the browser's theme colour.
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  const paper = getComputedStyle(root).getPropertyValue("--canvas").trim();
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", paper);
}
