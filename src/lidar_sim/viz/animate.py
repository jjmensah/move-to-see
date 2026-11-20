# ==========================================================
# src/lidar_sim/viz/animate.py
# ==========================================================
# Frame-by-frame animation writers (MP4 via the bundled imageio-ffmpeg binary, GIF via Pillow).
#
# Frames are rendered with ordinary matplotlib figures (``viz.plots`` style) and rasterized, so every frame
# obeys the same figure rules as the report figures.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


# ------------------------------------------
# figure_to_rgb
# ------------------------------------------
# Rasterize a matplotlib figure to an (H, W, 3) uint8 array.
def figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# ------------------------------------------
# write_mp4
# ------------------------------------------
# Encode RGB frames to H.264 MP4 (yuv420p, even dimensions enforced by padding).
def write_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 20) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                            ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            h, wd = f.shape[:2]
            w.append_data(np.pad(f, ((0, h % 2), (0, wd % 2), (0, 0)), mode="edge"))
    return path


# ------------------------------------------
# write_gif
# ------------------------------------------
# Write an optimized looping GIF (downscaled; every frame shown for 1/fps s).
def write_gif(frames: list[np.ndarray], path: str | Path, fps: int = 10, scale: float = 0.6) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    for f in frames:
        im = Image.fromarray(f)
        if scale != 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path
