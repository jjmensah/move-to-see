<p align="center">
  <img src="docs/figures/readme/concept.png" alt="Concept sketch: a rover with a 3D LiDAR, crates hiding a dark target, fog and sunlight, a fixed route and an information-driven route" width="820">
</p>

<h1 align="center">Move to See</h1>

<p align="center">
  <strong>Physics-informed active perception: a robot that drives to where its LiDAR will actually see the target.</strong>
</p>

<p align="center">
  <a href="https://robot.josephjuniormensah.com"><img src="https://img.shields.io/badge/Interactive_site-robot.josephjuniormensah.com-2ea44f" alt="Interactive site"></a>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557c" alt="Matplotlib">
  <img src="https://img.shields.io/badge/tests-666_passing-2ea44f" alt="666 tests passing">
  <img src="https://img.shields.io/badge/React-TypeScript-3178C6?logo=typescript&logoColor=white" alt="React and TypeScript">
</p>

---

> **Try it:** the [interactive site](https://robot.josephjuniormensah.com) replays all 480 recorded trials, lets you
> change the target, the fog and the sunlight to see what the sensor model predicts, and steps through the planner's
> real choices. Its source is in [`web/`](web/).

A small rover carries an amplitude-modulated continuous-wave (AMCW) LiDAR and must find a **dark box hidden behind
crates**, in clear air, in fog and in bright sunlight. A dark target returns little light, fog attenuates it, and sunlight
buries it in noise, so a target in plain view can still be invisible to the sensor. The question this project asks is
the one from its proposal: **can knowing when a LiDAR fails help a robot localize better?**

The robot keeps a particle belief over where the target is. Every half second it predicts, with a physics-based model
of its sensor, how much each candidate viewpoint would teach it about the target, subtracts a cost for the time it takes to
drive there, and heads for the best one. It is compared trial by trial with a **fixed inspection route** and with the
**same planner using a physics-blind sensor model**. Every experiment was registered, with its hypothesis, sample size,
seed and pass criteria, before its first full run; the few later changes are documented.

This is my final project for ECE 4240 (Robot Perception) at Cornell. It builds on my midterm, a waveform-level AMCW
LiDAR simulator, which it extends to 3D and then uses for inference and planning.

## Highlights

- **Faster in every condition.** With a black target the physics-aware planner localizes it in a median **7.0 s** in
  clear air, **9.0 s** in fog and **11.5 s** in sunlight, against 9.5, 17.0 and 18.0 s for the fixed route.
- **The physics matters most when the sensor struggles.** The same planner with a physics-blind sensor model localizes
  the target in **0 of 80** fog and sunlight trials; with the physics it localizes 79 of 80.
- **A clean control.** With a white target in clear air, which the sensor detects easily, the median paired
  difference between the two planners is 0 s (95 % interval 0 to 0 s), inside the registered ±2 s margin; 30 of 40
  pairs tie exactly. That points to the sensor model, not the planner, as the source of the gain.
- **Registered before it was run.** Five one-sided paired tests, all with Holm-adjusted p ≤ 7.2 × 10⁻⁴; every verdict
  and the four documented deviations are published.
- **Verified piece by piece.** 666 fast tests, including oracle suites with independently computed reference values,
  plus registered experiments that check the sensor model, the field of view, the filter and the information bound on
  their own.

---

<p align="center">Joseph Junior Mensah · <a href="https://josephjuniormensah.com">josephjuniormensah.com</a></p>
