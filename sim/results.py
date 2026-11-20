# ==========================================================
# sim/results.py
# ==========================================================
# Generate ``docs/verification/results.md`` from ``sim/out/<ID>/metrics.json`` and ``run.json``.
#
#     python -m sim.results
#
# Every registered criterion is reported with the verdict the experiment wrote (supported / NOT supported);
# nothing is re-evaluated here. Missing experiments are listed as not run.

from __future__ import annotations

import json

from sim.run import ROOT, load_registry

OUT = ROOT / "docs" / "verification" / "results.md"
COND = {"C1": "clear", "C2": "fog (V = 20 m)", "C3": "sunlight (1 sun)", "C4": "white target, clear"}
CTRL = {"info": "information-driven", "blind": "physics-blind", "fixed": "fixed route"}


# ------------------------------------------
# _mm
# ------------------------------------------
# Formats a length in mm below 0.1 m and in m above.
def _mm(x: float) -> str:
    return f"{1e3 * x:.1f} mm" if x < 0.0995 else f"{x:.2f} m"


# ------------------------------------------
# _f01
# ------------------------------------------
# Summary lines for F01.
def _f01(m: dict) -> list[str]:
    lo, hi = m["std_ratio_range"]
    out = [f"- {m['n_cells']} cells (3 environments × 2 reflectivities × 2 elevations × 5 ranges), "
           f"{m['cells'][0]['n']} scans each; largest |simulated crossing rate − P_D| = {m['max_abs_crossing_error']:.4f}.",
           f"- {m['n_high_snr']} high-SNR cells: unwrap-free range standard deviation / σ_R between {lo:.3f} and {hi:.3f}.",
           f"- Traced power vs hand-computed E7–E8 power: largest relative difference {m['max_power_rel_error_vs_hand']:.1e}."]
    for e in m["no_return"]:
        out.append(f"- No-return beams, {COND[e['condition']]}: {e['reports']} reports in {e['n']:,} beams "
                   f"(rate {e['rate']:.2e}; predicted {e['predicted']:.2e}; 99 % Wilson interval "
                   f"{e['wilson99'][0]:.2e}–{e['wilson99'][1]:.2e}).")
    gaps = m["near_threshold_report_gap"]
    if gaps:
        worst = max(gaps, key=lambda g: abs(g["predicted"] - g["simulated"]))
        out.append(f"- Reported, not gated — near-threshold reporting probability P_D·M vs simulated report rate: largest "
                   f"gap {abs(worst['predicted'] - worst['simulated']):.3f} ({worst['material']}, "
                   f"{COND[worst['condition']]}, {worst['range_m']:g} m; predicted {worst['predicted']:.3f}, simulated "
                   f"{worst['simulated']:.3f}) over {len(gaps)} cells with 0.02 < P_D < 0.98.")
    return out


# ------------------------------------------
# _f02
# ------------------------------------------
# Volume table and summary lines for F02.
def _f02(m: dict) -> list[str]:
    out = ["| pose (x, y, yaw) | probe | clear | fog | sunlight | line of sight | geometric |", "|---|---|---|---|---|---|---|"]
    worst = 0.0
    for row in m["volumes"]:
        for probe in ("black", "white"):
            c = [row["cases"][f"{probe}/{k}"] for k in ("C1", "C2", "C3")]
            worst = max(worst, *(x["rel_diff"] for x in c))
            out.append(f"| ({row['pose'][0]:g}, {row['pose'][1]:g}, {row['pose'][2]:.2f}) | {probe} | "
                       + " | ".join(f"{x['effective_m3']:.1f}" for x in c)
                       + f" | {c[0]['los_m3']:.1f} | {row['geometric_m3']:.1f} |")
    out += ["", "Effective FOV volumes in m³ (Monte Carlo). Largest relative difference from the independent direction "
            f"quadrature (face-plane reference): {100 * worst:.2f} %"
            + (f"; with the original scene-based reference: {100 * m['max_rel_diff_scene_reference']:.2f} %; half-sample "
               f"change ≤ {100 * m['max_half_sample_rel_change']:.2f} %, quadrature refinement ≤ "
               f"{100 * m['max_quadrature_refinement_rel_change']:.3f} %." if "max_rel_diff_scene_reference" in m else "."), ""]
    for v in m["velocity"]:
        out.append(f"- Robot speed {v['speed']:g} m/s: max |traced − analytic moving-origin range| = "
                   f"{v['max_abs_range_error_m']:.1e} m; max scan skew {1e3 * v['max_skew_m']:.1f} mm; speed recovered "
                   f"from the skew {v['fitted_speed']:.6f} m/s.")
    return out


# ------------------------------------------
# _f03
# ------------------------------------------
# Coverage table for F03.
def _f03(m: dict) -> list[str]:
    out = [f"{m['n_trials']} fixed-route trials per condition. Coverage = fraction of trials with NEES ≤ χ²₃(0.90) "
           f"= {m['chi2_3_q90']:.2f} (nominal 0.90).", "",
           "| condition | scans | coverage | median NEES / χ²₃ median | median error | median posterior std (x, y, z) |",
           "|---|---|---|---|---|---|"]
    for c, cond in m["conditions"].items():
        for k, v in cond.items():
            sd = ", ".join(_mm(s) for s in v["median_posterior_std_m"])
            out.append(f"| {COND[c]} | {k} | {v['coverage_90']:.2f} | {v['nees_over_chi2_median']:.2f} | "
                       f"{_mm(v['median_error_m'])} | {sd} |")
    return out


# ------------------------------------------
# _f04
# ------------------------------------------
# Summary lines for F04.
def _f04(m: dict) -> list[str]:
    out = []
    for key in ("n_cases", "n_violations", "n_cases_below_tolerance", "max_excess_in_se", "median_spearman",
                "median_regret_nat", "max_regret_nat"):
        if key in m:
            out.append(f"- {key.replace('_', ' ')}: {m[key]:.4g}" if isinstance(m[key], float) else
                       f"- {key.replace('_', ' ')}: {m[key]}")
    return out or ["- see `metrics.json`"]


# ------------------------------------------
# _f05
# ------------------------------------------
# Result and hypothesis tables for F05.
def _f05(m: dict) -> list[str]:
    out = [f"{m['n_trials']} paired trials per condition, {m['steps']} scans ({m['trial_length_s']:g} s). Time score = "
           "time to reach and hold 0.10 m accuracy for 5 scans, restricted to the trial length.", "",
           "| condition | controller | localized | median time score [s] (95 % CI) | median final error | path [m] | "
           "no target return |", "|---|---|---|---|---|---|---|"]
    for c, d in m["summary"].items():
        for ctrl, v in d.items():
            ci = v["time_score_ci95"]
            out.append(f"| {COND[c]} | {CTRL[ctrl]} | {100 * v['success_rate']:.0f} % | {v['median_time_score_s']:.1f} "
                       f"({ci[0]:.1f}–{ci[1]:.1f}) | {_mm(v['median_final_error_m'])} | {v['mean_path_length_m']:.1f} | "
                       f"{100 * v['no_target_return_fraction']:.0f} % |")
    out += ["", "| hypothesis | condition | information-driven vs | median paired difference [s] (95 % CI) | "
            "faster / tie / slower | p (raw) | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for c in m["comparisons"]:
        ci = c["difference_ci95"]
        out.append(f"| {c['hypothesis']} | {COND[c['condition']]} | {CTRL[c['against']]} | "
                   f"{c['median_paired_difference_s']:.1f} ({ci[0]:.1f} to {ci[1]:.1f}) | {c['info_faster']} / {c['ties']} / "
                   f"{c['info_slower']} | {c['p_raw']:.2e} | {c['p_holm']:.2e} |")
    k = m["control_C4"]
    out += ["", f"Control (white target, clear): median paired difference information-driven − physics-blind = "
            f"{k['median_paired_difference_s']:.2f} s, 95 % CI {k['difference_ci95'][0]:.2f} to {k['difference_ci95'][1]:.2f} s "
            f"(equivalence margin ±{k['margin_s']:g} s).",
            f"Fallback steps over all trials: {sum(v['fallback_steps'] for d in m['summary'].values() for v in d.values())}; "
            f"infeasible steps: {sum(v['infeasible_steps'] for d in m['summary'].values() for v in d.values())}."]
    return out


# ------------------------------------------
# _f06
# ------------------------------------------
# Ablation table for F06.
def _f06(m: dict) -> list[str]:
    out = [f"Exploratory, {m['n_paired_trials']} paired trials per variant in {COND[m['condition']]}; no gate.", "",
           "| variant | controller | localized | median time score [s] | path [m] | time in rough-patch area [s] |",
           "|---|---|---|---|---|---|"]
    for name, d in m["summary"].items():
        for ctrl, v in d.items():
            patch = f"{v['mean_patch_time_s']:.2f}" if "mean_patch_time_s" in v else "–"
            out.append(f"| {name} | {CTRL[ctrl]} | {100 * v['success_rate']:.0f} % | {v['median_time_score_s']:.1f} | "
                       f"{v['mean_path_length_m']:.1f} | {patch} |")
    out += ["", "The baseline is M = 24, λ_t = 0.05 nat/s with the rough patch. 'rough off' removes the patch (the last "
            "column then measures time spent in the area where the patch would be). 'dark background' makes walls, "
            "crates and ground black, removing most negative information."]
    return out


# ------------------------------------------
# _m01
# ------------------------------------------
# Summary lines for M01.
def _m01(m: dict) -> list[str]:
    out = [f"Trial {m['trial']} in {COND[m['condition']]}; files: " + ", ".join(f"`{f}`" for f in m["files"]) + "."]
    for ctrl, v in m["controllers"].items():
        out.append(f"- {CTRL[ctrl]}: final error {_mm(v['final_error_m'])}, time score {v['time_score_s']:.1f} s, "
                   f"path {v['path_length_m']:.1f} m.")
    return out


# ------------------------------------------
# _dirty
# ------------------------------------------
# Notes uncommitted changes recorded at the start of a run.
def _dirty(r: dict) -> str:
    if not r.get("git_dirty"):
        return ""
    files = r.get("git_modified") or ["src/ or sim/"]
    return " (uncommitted modifications at start: " + ", ".join(f"`{f}`" for f in files) + ")"


DETAIL = {"F01": _f01, "F02": _f02, "F03": _f03, "F04": _f04, "F05": _f05, "F06": _f06, "M01": _m01}


# ------------------------------------------
# build
# ------------------------------------------
# Builds docs/verification/results.md from every experiment's metrics and run record.
def build() -> str:
    lines = ["# Move to See — verification results", "",
             "Generated by `python -m sim.results` from `sim/out/*/metrics.json`; do not edit by hand. Hypotheses and "
             "pass criteria are those registered in `sim/registry.yaml` before each experiment's first full run "
             "(`docs/verification/vv-plan.md`). A verdict is reported as registered, whatever it is.", ""]
    summary = ["| ID | experiment | registered criteria | verdict |", "|---|---|---|---|"]
    body = []
    for e in load_registry():
        d = ROOT / "sim" / "out" / e["id"]
        if not (d / "metrics.json").exists():
            summary.append(f"| {e['id']} | {e['title']} | {len(e['criteria'])} | not run |")
            continue
        m = json.loads((d / "metrics.json").read_text())
        r = json.loads((d / "run.json").read_text())
        v = m.get("verdicts", {})
        missing = set(e["criteria"]) - set(v)
        if e["exploratory"]:
            verdict = "exploratory (no gate)"
        elif missing:
            verdict = f"INCOMPLETE (no verdict for {sorted(missing)})"
        else:
            n_ok = sum(bool(v[k]) for k in e["criteria"])
            verdict = "supported" if n_ok == len(e["criteria"]) else f"**NOT supported** ({n_ok}/{len(e['criteria'])})"
        summary.append(f"| {e['id']} | {e['title']} | {len(e['criteria'])} | {verdict} |")
        body += [f"## {e['id']} — {e['title']}", "", f"*Hypothesis.* {e['hypothesis']}", ""]
        for k, text in e["criteria"].items():
            mark = "supported" if v.get(k) else "**NOT supported**"
            body.append(f"- `{k}` — {text}: **{mark}**" if v.get(k) else f"- `{k}` — {text}: {mark}")
        body += ["", *DETAIL.get(e["id"], lambda m: ["- see `metrics.json`"])(m), "",
                 f"<sub>seed {r['seed']}, revision `{r['git_sha'][:10]}`{_dirty(r)}, sources "
                 f"`{r.get('source_sha256', 'n/a')[:10]}`{'' if r.get('source_unchanged_during_run', True) else ' (changed during the run)'}, "
                 f"{r['timestamp'][:19]}Z, wall time {r['wall_time_s'] / 60:.1f} min, Python {r['versions']['python']}, "
                 f"NumPy {r['versions']['numpy']}, SciPy {r['versions']['scipy']}.</sub>", ""]
    return "\n".join(lines + summary + [""] + body)


# ------------------------------------------
# main
# ------------------------------------------
# Writes docs/verification/results.md.
def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
