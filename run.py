
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from slsn import SensorNetwork, Localizer
from slsn.graph import is_connected
from visualizers import (
    plot_true_vs_estimated,
    plot_connectivity,
    plot_error_analysis,
    animate_localization,
)


def main() -> None:
    np.random.seed(42)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("\ndeploy sensors...")
    net = SensorNetwork(
        n_nodes=60,
        space_size=100.0,     # 100×100
        radio_range=30.0,
        noise_sigma=0.3,
        link_dropout=0.15,
        seed=42,
    )
    print(f"    {net.n_nodes} nodes")
    print(f"    {len(net.edges)} measured links")
    print(f"    Graph connected: {is_connected(net)}")

    print("\nlocalization start...")
    loc = Localizer(net)
    loc.run(max_iter=20, refine_iterations=100, refine_lr=0.050)
    print(f"{loc.anchors} anchors")
    n_loc = sum(loc.localized)
    print(f"{n_loc} / {net.n_nodes} nodes")

    errors = loc.per_node_errors()
    print("\nerror report vs gt:")
    print(f"    Mean error : {errors.mean():.3f} m")
    print(f"    Std  error : {errors.std():.3f} m")
    print(f"    Max  error : {errors.max():.3f} m")
    print(f"    Median err : {np.median(errors):.3f} m")
    print(f"    Final stress: {loc.compute_stress():.4f}")

    print("\nvisualize...")
    plot_true_vs_estimated(net, loc, save_path=out_dir / "true_v_est.png")
    plot_connectivity(net, loc, save_path=out_dir / "connectivity.png")
    plot_error_analysis(net, loc, save_path=out_dir / "error_analysis.png")
    animate_localization(net, loc, save_path=out_dir / "localization.mp4", fps=5)
    print(f"    saved to: {out_dir}")


if __name__ == "__main__":
    main()
