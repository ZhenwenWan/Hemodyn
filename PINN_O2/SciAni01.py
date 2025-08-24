# SciAni01.py — Scientific animation of microvascular oxygen using PINN04.py outputs
# Fixes vs SciAni00.py:
#  • Force PyTorch backend (must match .pt checkpoint)
#  • Resolve checkpoint path relative to run_meta.json
#  • Load WEIGHTS ONLY from checkpoint (do NOT load optimizer state) to avoid
#    "parameter group size mismatch" when external trainable variables were used in training.

import os
os.environ.setdefault("DDE_BACKEND", "pytorch")  # set BEFORE importing deepxde

import argparse
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="checkpoints/run_meta.json",
                   help="Path to run_meta.json from PINN04.py")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--cap_h_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()

# ---------------------- I/O helpers ----------------------
def load_meta(meta_path):
    meta_path = os.path.abspath(meta_path)
    meta_dir = os.path.dirname(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ckpt = meta.get("checkpoint_path", "")
    if not ckpt:
        raise RuntimeError("run_meta.json missing 'checkpoint_path'.")
    if not os.path.isabs(ckpt):
        ckpt = os.path.normpath(os.path.join(meta_dir, ckpt))
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    tp = meta.get("true_params", {})
    return ckpt, tp, meta

def build_model_and_load_weights(true_params, checkpoint_path):
    import deepxde as dde
    from deepxde.backend import backend as F
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch not available for DeepXDE backend 'pytorch'.") from e

    # Sanity: ensure PyTorch backend
    if dde.backend.backend_name != "pytorch":
        raise RuntimeError(f"DeepXDE backend is {dde.backend.backend_name!r}, expected 'pytorch'.")

    # Defaults if keys missing
    L = float(true_params.get("L", 1.0))
    H = float(true_params.get("H", 0.2))
    T = float(true_params.get("T", 2.0))

    # Minimal geometry/timebox & dummy PDE (no training, just to create a Model)
    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_zero(X, Y):
        z = F.zeros_like(X[:, :1])
        return [z, z]  # two outputs (Oc, Ot)

    data = dde.data.TimePDE(geomtime, pde_zero, [], num_domain=1, num_boundary=0,
                            num_initial=0, train_distribution="uniform")
    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)  # create a no-op optimizer; we won't load its state

    # ---- Load WEIGHTS ONLY (avoid optimizer restore mismatch) ----
    try:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    except TypeError:
        # for older torch without weights_only kw or strict typing
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    # DeepXDE's PyTorch save() stores dict with keys "model_state_dict" and "optimizer_state_dict"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        # Fallback: assume the checkpoint is directly a state_dict
        state = ckpt
    model.net.load_state_dict(state)  # weights only; DO NOT touch model.opt

    return model, (L, H, T)

# ---------------------- Field evaluators ----------------------
def predict_fields(model, L, H, nx, ny, t_scalar):
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    Xcap = np.column_stack([x, np.zeros_like(x), np.full_like(x, t_scalar)])
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xmat = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, t_scalar)])
    out_cap = model.predict(Xcap)           # (nx, 2) -> [Oc, Ot]
    out_tis = model.predict(Xmat)[:, 1]     # take Ot
    Oc_line = out_cap[:, 0]
    Ot_grid = out_tis.reshape(ny, nx)
    return x, y, Oc_line, Ot_grid

# ---------------------- Particle systems ----------------------
class RBCStream:
    def __init__(self, n, L, cap_h, u, fps, seed=0):
        rng = np.random.default_rng(seed)
        self.L, self.cap_h, self.u = L, cap_h, u
        self.dt = 1.0 / fps
        self.x = rng.uniform(0, L, n)
        self.y = rng.uniform(0.15 * cap_h, 0.85 * cap_h, n)
        self.oxy = rng.uniform(0.7, 1.0, n)     # oxy fraction
        self.sizes = rng.uniform(15, 35, n)
        self.jitter = 0.05 * cap_h

    def step(self, acid_fun):
        self.x += self.u * self.dt
        self.x[self.x > self.L] -= self.L
        self.y += np.random.normal(0.0, self.jitter, size=self.y.shape) * self.dt
        self.y = np.clip(self.y, 0.05 * self.cap_h, 0.95 * self.cap_h)
        acid = acid_fun(self.x)
        k_bohr = 1.5  # s^-1 (visual)
        p = 1.0 - np.exp(-k_bohr * acid * self.dt)
        self.oxy = np.maximum(0.05, self.oxy * (1.0 - 0.6 * p))

    def colors(self):
        r = self.oxy
        return np.stack([r, 0.1 * (1 - r), 1 - r, np.full_like(r, 0.9)], axis=1)

class TissueO2:
    def __init__(self, L, H, cap_h, fps, seed=0):
        self.L, self.H, self.cap_h = L, H, cap_h
        self.dt = 1.0 / fps
        self.xy = np.empty((0, 2), dtype=float)
        self.alive = np.empty((0,), dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.sigma = math.sqrt(2e-3)  # mm / sqrt(s)
        self.lambda_metab = 0.8       # s^-1
        self.depots = self.rng.uniform([0.1*L, 0.2*H], [0.9*L, 0.95*H], size=(25, 2))
        self.depot_r = 0.02 * H
        self.depot_hit = np.zeros(self.depots.shape[0], dtype=int)

    def inject_from_interface(self, n_new, where_x):
        if n_new <= 0:
            return
        y0 = self.cap_h + self.rng.uniform(0.0, 0.02 * self.H, size=n_new)
        x0 = where_x + self.rng.uniform(-0.01 * self.L, 0.01 * self.L, size=n_new)
        x0 = np.clip(x0, 0.0, self.L)
        pts = np.stack([x0, y0], axis=1)
        self.xy = np.vstack([self.xy, pts])
        self.alive = np.concatenate([self.alive, np.ones(n_new, dtype=bool)])

    def step(self):
        if self.xy.size == 0:
            return
        disp = self.rng.normal(0.0, self.sigma * math.sqrt(self.dt), size=self.xy.shape)
        self.xy[self.alive] += disp[self.alive]
        self.xy[:, 0] = np.clip(self.xy[:, 0], 0.0, self.L)
        self.xy[:, 1] = np.clip(self.xy[:, 1], self.cap_h + 1e-6, self.H)
        p_die = 1.0 - np.exp(-self.lambda_metab * self.dt)
        dies = self.rng.random(self.alive.shape) < p_die
        cap = np.zeros_like(self.alive)
        if self.xy.size:
            for i, c in enumerate(self.depots):
                d2 = np.sum((self.xy - c) ** 2, axis=1)
                hit = (d2 < self.depot_r ** 2) & self.alive
                if np.any(hit):
                    self.depot_hit[i] += int(np.sum(hit))
                    cap = cap | hit
        self.alive = self.alive & (~dies) & (~cap)

    def scatter_data(self):
        return self.xy[self.alive, :]

# ---------------------- Animation ----------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    ckpt_path, tp, meta = load_meta(args.meta)
    try:
        model, (L, H, T) = build_model_and_load_weights(tp, ckpt_path)
    except Exception as e:
        print(f"[FATAL] Could not load PINN weights: {e}")
        sys.exit(1)

    cap_h = max(1e-3, args.cap_h_frac * H)
    ts = np.linspace(0.0, T, args.frames)

    u = float(tp.get("u", 1.0))  # mm/s
    rbc = RBCStream(n=80, L=L, cap_h=cap_h, u=u, fps=args.fps, seed=args.seed)
    tis = TissueO2(L=L, H=H, cap_h=cap_h, fps=args.fps, seed=args.seed + 1)

    plt.rcParams["figure.figsize"] = (10, 7)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])
    ax_cap = fig.add_subplot(gs[0, 0])
    ax_tis = fig.add_subplot(gs[1, 0])

    ax_cap.set_xlim(0, L)
    ax_cap.set_ylim(0, cap_h)
    ax_cap.set_xlabel("x [mm]")
    ax_cap.set_ylabel("capillary y")
    scat_rbc = ax_cap.scatter([], [], s=[], c=[], edgecolors="none")

    ax_tis.axhline(cap_h, color="k", linewidth=1)
    ax_tis.set_xlim(0, L)
    ax_tis.set_ylim(0, H)
    ax_tis.set_xlabel("x [mm]")
    ax_tis.set_ylabel("y [mm]")

    im = ax_tis.imshow(
        np.zeros((args.ny, args.nx)),
        extent=[0, L, 0, H],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        vmin=0.0,
        vmax=0.15,
    )
    cb = fig.colorbar(im, ax=ax_tis, fraction=0.03, pad=0.02)
    cb.set_label("Ot [arb]")
    scat_o2 = ax_tis.scatter([], [], s=6, c="white", alpha=0.7, edgecolors="none")

    def make_acid_fun(Ot_grid, y):
        idx_row = min(max(1, int(0.02 * len(y))), len(y) - 1)
        row = Ot_grid[idx_row, :]
        rmin, rmax = np.percentile(row, 5), np.percentile(row, 95)
        norm = np.clip((row - rmin) / (rmax - rmin + 1e-9), 0, 1)
        acid = 1.0 - norm
        x_axis = np.linspace(0, L, len(row))
        def acid_fun(xs):
            return np.interp(xs, x_axis, acid)
        return acid_fun

    def update(i):
        t = ts[i]
        x, y, Oc_line, Ot_grid = predict_fields(model, L, H, args.nx, args.ny, t)
        Ot_noisy = Ot_grid + 0.02 * (np.std(Ot_grid) + 1e-12) * np.random.randn(*Ot_grid.shape)
        im.set_data(Ot_noisy)
        im.set_clim(vmin=max(0.0, float(np.min(Ot_grid))), vmax=float(np.max(Ot_grid)) + 1e-6)

        acid_fun = make_acid_fun(Ot_grid, y)
        rbc.step(acid_fun)
        scat_rbc.set_offsets(np.c_[rbc.x, rbc.y])
        scat_rbc.set_sizes(rbc.sizes)
        scat_rbc.set_facecolors(rbc.colors())

        bins = 12
        counts, edges = np.histogram(rbc.x, bins=bins, range=(0, L))
        idx = np.clip(np.digitize(rbc.x, edges) - 1, 0, bins - 1)
        deoxy_mean = np.zeros(bins)
        for b in range(bins):
            m = (idx == b)
            if np.any(m):
                deoxy_mean[b] = np.mean(1.0 - rbc.oxy[m])
        spawn_base = 6
        for b in range(bins):
            n_new = np.random.poisson(spawn_base * (0.3 + deoxy_mean[b]))
            x_mid = 0.5 * (edges[b] + edges[b + 1])
            tis.inject_from_interface(n_new, x_mid)

        tis.step()
        scat_o2.set_offsets(tis.scatter_data())

        ax_cap.set_title(f"Capillary RBCs (t={t:.2f} s)")
        ax_tis.set_title("Tissue O$_2$ (PINN prediction + noise)")
        return (im, scat_rbc, scat_o2)

    ani = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()

