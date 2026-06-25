"""Core localization: anchor selection, trilateration, stress refinement."""
from __future__ import annotations

import numpy as np

from .graph import shortest_path_distances
from .network import SensorNetwork


class Localizer:
    """Reconstruct node positions from noisy inter-node distances.

    Pipeline
    --------
    1. ``select_anchors()``     — pick 3 well-spaced, directly-connected nodes
    2. ``place_anchors()``      — fix them at a canonical reference frame
    3. ``localize_iterative()`` — trilaterate every node with ≥3 known neighbors
    4. ``refine_gradient_descent()`` — minimize stress (sum of squared edge errors)

    Convenience: ``run()`` does all four steps.
    """

    def __init__(self, network: SensorNetwork):
        self.network = network
        self.est_positions = np.zeros((network.n_nodes, 2))
        self.anchors: list[int] = []
        self.localized = [False] * network.n_nodes
        self.history: list[np.ndarray] = [self.est_positions.copy()]
        self.stress_history: list[float] = [self.compute_stress()]

    # ==================================================================
    # Public convenience
    # ==================================================================
    def run(self,
            max_iter: int = 30,
            refine_iterations: int = 500,
            refine_lr: float = 0.005,
            snapshot_every: int = 1) -> None:
        """Full localization pipeline."""
        if not self.anchors:
            self.select_anchors()
        if not any(self.localized):
            self.place_anchors()
        self.localize_iterative(max_iter=max_iter)
        self.refine_gradient_descent(
            iterations=refine_iterations,
            lr=refine_lr,
            snapshot_every=snapshot_every)

    # ==================================================================
    # Anchor selection
    # ==================================================================
    def select_anchors(self) -> list[int]:
        """Pick three anchors forming a well-conditioned triangle.

        Prefer directly-connected nodes so anchor pairwise distances
        have low noise (no multi-hop compounding).
        """
        n = self.network.n_nodes
        # a0: highest-degree node
        degrees = [self.network.degree(i) for i in range(n)]
        a0 = int(np.argmax(degrees))

        # a1: farthest direct neighbor of a0
        nbrs0 = self.network.neighbors(a0)
        if not nbrs0:
            raise RuntimeError("Anchor a0 has no neighbors; graph too sparse.")
        a1 = max(nbrs0, key=lambda j: self.network.measured_distance(a0, j))

        # a2: common neighbor of a0 and a1 maximizing triangle area
        nbrs1 = set(self.network.neighbors(a1))
        candidates = [j for j in nbrs0 if j in nbrs1 and j != a1]
        if not candidates:
            # Fallback: any other node, distances via shortest path
            candidates = [j for j in range(n) if j != a0 and j != a1]

        d01 = self._estimate_distance(a0, a1)
        best, best_area = None, -1.0
        for j in candidates:
            d0j = self._estimate_distance(a0, j)
            d1j = self._estimate_distance(a1, j)
            s = (d01 + d0j + d1j) / 2.0
            area_sq = s * (s - d01) * (s - d0j) * (s - d1j)
            if area_sq > best_area:
                best_area = area_sq
                best = j
        if best is None:
            raise RuntimeError("Could not find a valid third anchor.")
        self.anchors = [a0, a1, best]
        return self.anchors

    # ==================================================================
    # Distance estimation
    # ==================================================================
    def _estimate_distance(self, i: int, j: int) -> float:
        """Measured distance if available, else shortest-path estimate."""
        d = self.network.measured_distance(i, j)
        if d is not None:
            return d
        sp = shortest_path_distances(self.network)
        return float(sp[i, j])

    # ==================================================================
    # Anchor placement (canonical frame)
    # ==================================================================
    def place_anchors(self) -> None:
        """Place anchors: a0 at origin, a1 on +x axis, a2 in +y half-plane."""
        a0, a1, a2 = self.anchors
        d01 = self._estimate_distance(a0, a1)
        d02 = self._estimate_distance(a0, a2)
        d12 = self._estimate_distance(a1, a2)
        self.est_positions[a0] = [0.0, 0.0]
        self.est_positions[a1] = [d01, 0.0]
        # Trilateration of a2 from a0, a1
        x2 = (d01 ** 2 + d02 ** 2 - d12 ** 2) / (2.0 * d01)
        y2_sq = d02 ** 2 - x2 ** 2
        y2 = np.sqrt(max(y2_sq, 0.0))
        self.est_positions[a2] = [x2, y2]
        for a in self.anchors:
            self.localized[a] = True
        self._snapshot()

    # ==================================================================
    # Trilateration
    # ==================================================================
    @staticmethod
    def trilaterate(anchor_positions: np.ndarray,
                    distances: np.ndarray) -> np.ndarray:
        """Least-squares trilateration.

        For k ≥ 3 anchors p_i at noisy distances d_i to target x:

            ‖x − p_i‖² = d_i²

        Subtract equation for i=0 from the rest to obtain a linear system
        in x, solved via least squares.
        """
        p0 = anchor_positions[0]
        d0 = distances[0]
        A = np.empty((len(anchor_positions) - 1, p0.shape[0]))
        b = np.empty(len(anchor_positions) - 1)
        for i in range(1, len(anchor_positions)):
            pi = anchor_positions[i]
            di = distances[i]
            A[i - 1] = 2.0 * (pi - p0)
            b[i - 1] = d0 ** 2 - di ** 2 + np.dot(pi, pi) - np.dot(p0, p0)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    # ==================================================================
    # Iterative trilateration sweep
    # ==================================================================
    def localize_iterative(self, max_iter: int = 30) -> None:
        """Trilaterate every node with ≥3 localized neighbors; repeat."""
        n = self.network.n_nodes
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                if self.localized[i]:
                    continue
                avail_pos, avail_d = [], []
                for j in self.network.neighbors(i):
                    d = self.network.measured_distance(i, j)
                    if d is not None and self.localized[j]:
                        avail_pos.append(self.est_positions[j])
                        avail_d.append(d)
                if len(avail_pos) >= 3:
                    self.est_positions[i] = self.trilaterate(
                        np.array(avail_pos), np.array(avail_d))
                    self.localized[i] = True
                    changed = True
            self._snapshot()
            if not changed:
                break

    # ==================================================================
    # Stress refinement (gradient descent)
    # ==================================================================
    def compute_stress(self) -> float:
        """Mean squared edge-length error."""
        total, count = 0.0, 0
        for (i, j), d_meas in self.network.edges.items():
            d_est = float(np.linalg.norm(
                self.est_positions[i] - self.est_positions[j]))
            total += (d_est - d_meas) ** 2
            count += 1
        return total / max(count, 1)

    def refine_gradient_descent(self,
                                iterations: int = 500,
                                lr: float = 0.005,
                                snapshot_every: int = 1) -> None:
        """Gradient descent on the stress function. Anchors are fixed.

        Stress:   E = Σ (‖x_i − x_j‖ − d_ij)²
        Gradient: ∂E/∂x_i = 2 Σ_j (‖x_i−x_j‖ − d_ij) · (x_i − x_j) / ‖x_i−x_j‖
        """
        n = self.network.n_nodes
        edges = list(self.network.edges.items())
        anchor_set = set(self.anchors)
        free_idx = [i for i in range(n) if i not in anchor_set]

        for it in range(iterations):
            grad = np.zeros_like(self.est_positions)
            for (i, j), d_meas in edges:
                diff = self.est_positions[i] - self.est_positions[j]
                d_est = float(np.linalg.norm(diff)) + 1e-9
                err = d_est - d_meas
                g = err * diff / d_est
                grad[i] += g
                grad[j] -= g
            for i in free_idx:
                self.est_positions[i] -= lr * grad[i]
            if it % snapshot_every == 0 or it == iterations - 1:
                self._snapshot()

    # ==================================================================
    # Evaluation (uses ground truth ONLY for reporting, not for solving)
    # ==================================================================
    def align_to_ground_truth(self):
        """Procrustes-align estimates to ground truth for visualization.

        Returns (est_aligned, true_centered). The alignment finds the
        optimal rotation/reflection + translation that maps estimates
        onto ground truth. The localization itself never uses this.
        """
        true = self.network.true_positions
        est = self.est_positions.copy()
        true_c = true - true.mean(axis=0)
        est_c = est - est.mean(axis=0)
        # Optimal orthogonal transformation: R = U V^T  where  est_c^T true_c = U Σ V^T
        U, _, Vt = np.linalg.svd(est_c.T @ true_c)
        R = U @ Vt
        est_aligned = est_c @ R
        return est_aligned, true_c

    def per_node_errors(self) -> np.ndarray:
        """Euclidean error per node, after Procrustes alignment."""
        est_aligned, true_c = self.align_to_ground_truth()
        return np.linalg.norm(est_aligned - true_c, axis=1)

    # ==================================================================
    # Internal: snapshot for animation
    # ==================================================================
    def _snapshot(self) -> None:
        self.history.append(self.est_positions.copy())
        self.stress_history.append(self.compute_stress())
