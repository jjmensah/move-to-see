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

// ------------------------------------------
// useReveal
// ------------------------------------------
// Adds `in` to every `.reveal` element once it enters the viewport, so blocks
// rise into place as the visitor scrolls. Runs again when `ready` changes,
// because sections appear only after the data has loaded.
export function useReveal(ready: boolean) {
  useEffect(() => {
    if (!ready) return;
    const items = document.querySelectorAll(".reveal:not(.in)");
    if (!("IntersectionObserver" in window)) {
      items.forEach(el => el.classList.add("in"));
      return;
    }
    const observer = new IntersectionObserver(
      entries => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          e.target.classList.add("in");
          observer.unobserve(e.target);
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    items.forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, [ready]);
}
