// ==========================================================
// web/scripts/smoke.mjs
// ==========================================================
// End-to-end smoke test of the built site in headless Chrome. Serves dist/,
// opens the page at phone and desktop widths in both themes, exercises every
// section, and fails on console errors, failed requests or horizontal overflow.
//
//   npm run build && npm run smoke [-- <screenshot dir>]

import { chromium } from "playwright-core";
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";

// ------------------------------------------
// Setup
// ------------------------------------------
// Optional screenshot folder, the preview server and the failure list.
const shots = process.argv[2];
if (shots) mkdirSync(shots, { recursive: true });
const port = 4317;
const server = spawn("npx", ["vite", "preview", "--port", String(port), "--strictPort"], { stdio: "ignore" });
const url = `http://localhost:${port}/`;
const fail = [];
const check = (ok, msg) => {
  if (!ok) fail.push(msg);
};

// ------------------------------------------
// Run
// ------------------------------------------
// Waits for the server, then checks each width and theme in turn.
try {
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(url)).ok) break;
    } catch {
      // The server is not up yet.
    }
    await new Promise(r => setTimeout(r, 200));
  }
  const browser = await chromium.launch({ channel: "chrome" });
  for (const [w, h] of [
    [390, 844],
    [1280, 900],
  ]) {
    for (const scheme of ["light", "dark"]) {
      const tag = `${w}-${scheme}`;
      const page = await browser.newPage({ viewport: { width: w, height: h }, colorScheme: scheme });
      page.on("console", m => {
        if (m.type() === "error") fail.push(`${tag} console: ${m.text()}`);
      });
      page.on("pageerror", e => fail.push(`${tag} page error: ${e.message}`));
      page.on("requestfailed", r => fail.push(`${tag} request failed: ${r.url()}`));
      page.on("response", r => {
        if (r.status() >= 400) fail.push(`${tag} HTTP ${r.status()}: ${r.url()}`);
      });
      await page.goto(url, { waitUntil: "networkidle" });
      check(
        (await page.evaluate(() => document.documentElement.dataset.theme)) === scheme,
        `${tag}: theme not resolved from the system`,
      );
      await page.getByRole("img", { name: /Top-down replay of trial 3/ }).waitFor();
      // Play the trial to the end.
      await page.getByRole("button", { name: "Play" }).click();
      await page
        .getByText(/Within 0.10 m from|Not localized within 30 s/)
        .first()
        .waitFor({ timeout: 20000 });
      await page.locator('input[aria-label="Time step"]').fill("60");
      check(
        (await page.getByText(/Within 0.10 m from/).count()) >= 1,
        `${tag}: no controller localized in fog trial 3`,
      );
      // Change condition and trial.
      await page
        .getByRole("group", { name: "Replay condition" })
        .getByRole("button", { name: /Sunlight/ })
        .click();
      await page.getByRole("img", { name: /Top-down replay of trial 3 in Sunlight/ }).waitFor();
      check(new URL(page.url()).searchParams.get("condition") === "C3", `${tag}: URL does not record the condition`);
      // Sensor presets move r50.
      await page.locator("#sensor").scrollIntoViewIfNeeded();
      const r50 = page.locator("#sensor .stat b").first();
      const sensorGroup = page.getByRole("group", { name: "Tested condition" });
      await sensorGroup.getByRole("button", { name: /White target/ }).click();
      const white = await r50.textContent();
      await sensorGroup.getByRole("button", { name: /Fog/ }).click();
      const fog = await r50.textContent();
      check(white !== fog && /m/.test(fog ?? ""), `${tag}: sensor r50 did not change (${white} / ${fog})`);
      // Decision steps.
      await page.locator("#decision").scrollIntoViewIfNeeded();
      // Picking a viewpoint and then changing the step must not break the page.
      await page.locator("#decision details summary").click();
      await page.locator("#decision details").getByRole("button").last().click();
      await page.getByRole("button", { name: "t = 10 s" }).click();
      await page.getByRole("button", { name: "t = 5 s" }).click();
      check((await page.locator("#decision table tbody tr").count()) >= 5, `${tag}: decision table is empty`);
      if (shots) await page.screenshot({ path: `${shots}/${tag}.png`, fullPage: true });
      // Results → replay.
      await page.locator("#results").scrollIntoViewIfNeeded();
      await page
        .locator("#results")
        .getByRole("button", { name: /^Trial 5:/ })
        .focus();
      await page.keyboard.press("Enter");
      const open = page.getByRole("button", { name: "Open in the replay" });
      if (await open.count()) {
        await open.click();
        await page.waitForTimeout(400);
        check(
          (await page.evaluate(() => window.location.hash)) === "#replay",
          `${tag}: results did not open the replay`,
        );
        await page.getByRole("img", { name: /Top-down replay of trial 5 in Clear/ }).waitFor();
      } else fail.push(`${tag}: no "open in the replay" button`);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      check(overflow <= 0, `${tag}: horizontal overflow of ${overflow}px`);
      await page.close();
    }
  }
  await browser.close();
} finally {
  server.kill();
}

// ------------------------------------------
// Report
// ------------------------------------------
// Lists every failure, or confirms the run passed.
if (fail.length) {
  console.error(`FAIL\n${fail.join("\n")}`);
  process.exit(1);
}
console.log(
  "PASS: 390 px and 1280 px, light and dark: replay, sensor, decision and results all respond; no errors or overflow.",
);
