"""Animated convergence visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from slsn.localization import Localizer
from slsn.network import SensorNetwork


def animate_localization(network: SensorNetwork,
                         localizer: Localizer,
                         save_path: Optional[Path | str] = None,
                         fps: int = 4):
    from matplotlib.collections import LineCollection

    history = localizer.history
    true = network.true_positions
    true_c = true - true.mean(axis=0)

    # Align every frame to the (centered) ground truth
    aligned_frames = []
    for est in history:
        est_c = est - est.mean(axis=0)
        U, _, Vt = np.linalg.svd(est_c.T @ true_c)
        R = U @ Vt
        aligned_frames.append(est_c @ R)

    edges = list(network.edges.keys())
    n_frames = len(aligned_frames)
    stress = localizer.stress_history

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.scatter(true_c[:, 0], true_c[:, 1], c="steelblue", s=55, alpha=0.35,
               marker="o", edgecolors="k", linewidths=0.5, label="True", zorder=3)

    est_scat = ax.scatter([], [], c="crimson", s=70, marker="x",
                          label="Estimated", zorder=5, linewidths=1.8)

    # ---- Use a single LineCollection for all edges ----
    edge_collection = LineCollection([], colors="gray", alpha=0.25,
                                     linewidths=0.6, zorder=2)
    ax.add_collection(edge_collection)

    # Mark anchors (their indices don't change across frames)
    anchor_handles = []
    for k, a in enumerate(localizer.anchors):
        h = ax.scatter([], [], c="red", s=320, marker="*", zorder=6,
                       edgecolors="k", linewidths=0.6,
                       label="Anchors" if k == 0 else None)
        anchor_handles.append(h)

    # Fixed bounds across all frames
    all_pts = np.vstack([true_c] + aligned_frames)
    pad = 0.05 * (all_pts.max() - all_pts.min())
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def init():
        est_scat.set_offsets(np.empty((0, 2)))
        edge_collection.set_segments([])
        for h in anchor_handles:
            h.set_offsets(np.empty((0, 2)))
        return [est_scat, title, edge_collection, *anchor_handles]

    def update(frame):
        est = aligned_frames[frame]
        est_scat.set_offsets(est)
        # build segments from current estimates
        segs = [(est[i], est[j]) for (i, j) in edges]
        edge_collection.set_segments(segs)
        for h, a in zip(anchor_handles, localizer.anchors):
            h.set_offsets(est[a:a + 1])
        s = stress[frame] if frame < len(stress) else stress[-1]
        title.set_text(f"Iteration {frame} / {n_frames - 1}    "
                       f"stress = {s:.3f}")
        return [est_scat, title, edge_collection, *anchor_handles]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=True, interval=1000 // fps, repeat=True)

    if save_path:
        ani.save(save_path, writer="ffmpeg", fps=fps)
    return ani
