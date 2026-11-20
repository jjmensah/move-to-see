// ==========================================================
// web/src/App.tsx
// ==========================================================
// The page. Sections are added one at a time as they are built.

import ThemeToggle from "./ThemeToggle";

// ------------------------------------------
// App
// ------------------------------------------
// Top bar with the brand and the theme switch.
export default function App() {
  return (
    <div className="bar">
      <header className="wrap topbar">
        <a className="brand" href="#replay">
          <img src="./icon.svg" alt="" />
          Move to See
        </a>
        <ThemeToggle />
      </header>
    </div>
  );
}
