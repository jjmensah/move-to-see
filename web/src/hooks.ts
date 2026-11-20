// ==========================================================
// web/src/hooks.ts
// ==========================================================
// Small page-level hooks shared by the layout.

import { useEffect, useState } from "react";

// ------------------------------------------
// useScrolled
// ------------------------------------------
// True once the page has scrolled past `offset` pixels; the navigation turns
// to glass from then on.
export function useScrolled(offset = 24): boolean {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const update = () => setScrolled(window.scrollY > offset);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, [offset]);
  return scrolled;
}
