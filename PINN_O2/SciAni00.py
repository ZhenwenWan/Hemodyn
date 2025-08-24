# animate_microvascular_oxygen.py
"""
Scientific animation of microvascular oxygen delivery using a trained PINN (PINN04.py).

Features
- Two 2D panels: (top) capillary strip; (bottom) tissue slab.
- RBC particles advect along capillary; oxy/deoxy state shifts via a Bohr-effect proxy (low pH ~ low Ot near interface).
- Only O2 molecules cross into tissue; they diffuse (random walk) and are metabolized by Poisson thinning.
- A fraction of tissue O2 gets captured by myoglobin "depots".

Inputs
- Expects 'run_meta.json' (written by PINN04.py) with:
    { "checkpoint_path": ".../oxygen_pinn-<N>.pt", "true_params": {...}, ... }
- Restores the DeepXDE model and predicts Oc(x,t), Ot(x,y,t) on-the-fly.

Usage
    python animate_microvascular_oxygen.py --meta checkpoints/run_meta.json \
        --nx 128 --ny 48 --frames 200 --fps 25 --cap_h_frac 0.15

Notes
- Physiological fidelity: hemoglobin remains inside RBCs; oxygen diffuses into tissue and can bind myoglobin.
- If DeepXDE is missing or restore fails, the script exits with a helpful message.

Refs (concepts only; see chat for links)
- Bohr effect (low pH -> ↓ Hb–O2 affinity)
- O2 diffusion from capillary to tissue
- Myoglobin stores/facilitates O2 in muscle
- Poisson process for random reaction events
"""
# --- FORCE PYTORCH BACKEND *BEFORE* importing deepxde ---
import os
os.environ["DDE_BACKEND"] = "pytorch"   # override any prior setting

# (optional but helpful) make sure the selected backend is really PyTorch
import deepxde as dde
if dde.backend.backend_name != "pytorch":
    raise RuntimeError(
        f"DeepXDE backend is {dde.backend.backend_name!r}, but the checkpoint is PyTorch (.pt). "
        "Launch as:  (Windows CMD)  set DDE_BACKEND=pytorch && python SciAni00.py --meta checkpoints\\run_meta.json  "
        "or fix ~/.deepxde/config.json to 'pytorch'."
    )

import argparse
import json
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="checkpoints/run_meta.json", help="Path to run_meta.json from PINN04.py")
    p.add_argument("--nx", type=int, default=128, help="grid points in x")
    p.add_argument("--ny", type=int, default=48, help="grid points in y (tissue)")
    p.add_argument("--frames", type=int, default=200, help="animation frames (time samples)")
    p.add_argument("--fps", type=int, default=25, help="frames per second")
    p.add_argument("--cap_h_frac", type=float, default=0.15, help="capillary strip height as fraction of H")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


# ----------------------
# DeepXDE model restore
# ----------------------
def load_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if "checkpoint_path" not in meta:
        raise RuntimeError("run_meta.json missing 'checkpoint_path'.")
    tp = meta.get("true_params", {})
    return meta["checkpoint_path"], tp, meta

def build_model_for_restore(true_params, checkpoint_path):
    """
    Rebuild a minimal DeepXDE model with the same net architecture used in PINN04.py:
    net = FNN([3] + [64]*4 + [2], 'tanh', 'Glorot uniform')
    Geometry/timebox: Rectangle([0,0]->[L,H]) x Time[0,T]
    We attach a zero-PDE dataset (1 dummy point) just to make Model() happy.
    """
    try:
        import deepxde as dde
        from deepxde.backend import backend as F
    except Exception as e:
        raise RuntimeError("DeepXDE not available in this Python env.") from e

    # Params (defaults if keys missing)
    L = float(true_params.get("L", 1.0))
    H = float(true_params.get("H", 0.2))
    T = float(true_params.get("T", 2.0))

    # Geometry/time
    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Dummy PDE and data (no training will occur)
    def pde_zero(X, Y):
        # returns two zeros to match output dim=2
        z = F.zeros_like(X[:, :1])
        return [z, z]

    data = dde.data.TimePDE(
        geomtime, pde_zero, [], num_domain=1, num_boundary=0, num_initial=0, train_distribution="uniform"
    )
    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # Compile with a no-op optimizer just to satisfy internal state
    model.compile("adam", lr=1e-3)

    # Restore weights
    model.restore(checkpoint_path, device="cpu")
    return model, (L, H, T)


# ----------------------
# Field evaluators
# ----------------------
def predict_fields(model, L, H, nx, ny, t_scalar):
    """
    Return:
      x (nx,), y (ny,), Oc_line (nx,), Ot_grid (ny, nx)
    """
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    # Capillary line at y=0
    Xcap = np.column_stack([x, np.zeros_like(x), np.full_like(x, t_scalar)])
    # Tissue grid
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xmat = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, t_scalar)])

    out_cap = model.predict(Xcap)           # shape (nx, 2) -> [Oc, Ot]
    out_tis = model.predict(Xmat)[:, 1]     # take Ot
    Oc_line = out_cap[:, 0]
    Ot_grid = out_tis.reshape(ny, nx)
    return x, y, Oc_line, Ot_grid


# ----------------------
# RBC & O2 particle system (capillary/tissue)
# ----------------------
class RBCStream:
    def __init__(self, n, L, cap_h, u, fps, seed=0):
        rng = np.random.default_rng(seed)
        self.L = L
        self.cap_h = cap_h
        self.u = u  # mm/s
        self.dt = 1.0 / fps
        self.x = rng.uniform(0, L, n)
        self.y = rng.uniform(0.15 * cap_h, 0.85 * cap_h, n)
        # oxy fraction [0,1]
        self.oxy = rng.uniform(0.7, 1.0, n)
        # visual sizes
        self.sizes = rng.uniform(15, 35, n)
        # lateral jitter scale
        self.jitter = 0.05 * cap_h

    def step(self, acid_fun):
        # advection
        self.x += self.u * self.dt
        # wrap around
        self.x[self.x > self.L] -= self.L
        # small transverse random walk
        self.y += np.random.normal(0.0, self.jitter, size=self.y.shape) * self.dt
        self.y = np.clip(self.y, 0.05 * self.cap_h, 0.95 * self.cap_h)
        # Bohr effect proxy: convert oxy->deoxy faster where 'acid' is high
        acid = acid_fun(self.x)
        k_bohr = 1.5   # s^-1 max (tunable)
        p = 1.0 - np.exp(-k_bohr * acid * self.dt)
        # decrease oxy fraction, floor at ~0.05
        self.oxy = np.maximum(0.05, self.oxy * (1.0 - 0.6 * p))

    def colors(self):
        # blend red(oxy) -> blue(deoxy)
        r = self.oxy
        # RGBA: red to blue interpolation
        return np.stack([r, 0.1 * (1 - r), 1 - r, np.full_like(r, 0.9)], axis=1)


class TissueO2:
    def __init__(self, L, H, cap_h, fps, seed=0):
        self.L, self.H, self.cap_h = L, H, cap_h
        self.dt = 1.0 / fps
        self.xy = np.empty((0, 2), dtype=float)  # particle positions
        self.alive = np.empty((0,), dtype=bool)
        self.rng = np.random.default_rng(seed)
        # diffusion step (std dev per sqrt(s)): choose D~1e-3 mm^2/s
        self.sigma = math.sqrt(2e-3)  # mm / sqrt(s)
        # Poisson consumption rate [s^-1]
        self.lambda_metab = 0.8
        # myoglobin depots
        self.depots = self.rng.uniform([0.1*L, 0.2*H], [0.9*L, 0.95*H], size=(25, 2))
        self.depot_r = 0.02 * H
        self.depot_hit = np.zeros(self.depots.shape[0], dtype=int)

    def inject_from_interface(self, n_new, where_x):
        if n_new <= 0:
            return
        # spawn just below the interface
        y0 = self.cap_h + self.rng.uniform(0.0, 0.02 * self.H, size=n_new)
        x0 = where_x + self.rng.uniform(-0.01 * self.L, 0.01 * self.L, size=n_new)
        x0 = np.clip(x0, 0.0, self.L)
        pts = np.stack([x0, y0], axis=1)
        self.xy = np.vstack([self.xy, pts])
        self.alive = np.concatenate([self.alive, np.ones(n_new, dtype=bool)])

    def step(self):
        if self.xy.size == 0:
            return
        # random walk
        disp = self.rng.normal(0.0, self.sigma * math.sqrt(self.dt), size=self.xy.shape)
        self.xy[self.alive] += disp[self.alive]
        # keep inside tissue region
        self.xy[:, 0] = np.clip(self.xy[:, 0], 0.0, self.L)
        self.xy[:, 1] = np.clip(self.xy[:, 1], self.cap_h + 1e-6, self.H)
        # Poisson thinning for metabolism
        p_die = 1.0 - np.exp(-self.lambda_metab * self.dt)
        dies = self.rng.random(self.alive.shape) < p_die
        # myoglobin capture
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


# ----------------------
# Animation
# ----------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load meta + model
    checkpoint_path, tp, meta = load_meta(args.meta)
    try:
        model, (L, H, T) = build_model_for_restore(tp, checkpoint_path)
    except Exception as e:
        print(f"[FATAL] Could not restore PINN model: {e}")
        sys.exit(1)

    cap_h = max(1e-3, args.cap_h_frac * H)  # capillary strip height

    # Time samples
    frames = args.frames
    ts = np.linspace(0.0, T, frames)
    dt = ts[1] - ts[0]

    # RBC and tissue O2 systems
    u = float(tp.get("u", 1.0))  # mm/s
    rbc = RBCStream(n=80, L=L, cap_h=cap_h, u=u, fps=args.fps, seed=args.seed)
    tis = TissueO2(L=L, H=H, cap_h=cap_h, fps=args.fps, seed=args.seed + 1)

    # Figure & artists
    plt.rcParams["figure.figsize"] = (10, 7)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])
    ax_cap = fig.add_subplot(gs[0, 0])
    ax_tis = fig.add_subplot(gs[1, 0])

    # Capillary setup
    ax_cap.set_xlim(0, L)
    ax_cap.set_ylim(0, cap_h)
    ax_cap.set_xlabel("x [mm]")
    ax_cap.set_ylabel("capillary y")
    ax_cap.set_title("Capillary: RBC flow (oxy red → deoxy blue)")

    scat_rbc = ax_cap.scatter([], [], s=[], c=[], edgecolors="none")
    # Draw interface line
    ax_tis.axhline(cap_h, color="k", linewidth=1)

    # Tissue setup
    ax_tis.set_xlim(0, L)
    ax_tis.set_ylim(0, H)
    ax_tis.set_xlabel("x [mm]")
    ax_tis.set_ylabel("y [mm]")
    ax_tis.set_title("Tissue: O$_2$ concentration & diffusing molecules")

    # init heatmap
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

    # tissue O2 scatter
    scat_o2 = ax_tis.scatter([], [], s=6, c="white", alpha=0.7, edgecolors="none")

    # myoglobin depots
    depots_plot = ax_tis.scatter(tis.depots[:, 0], tis.depots[:, 1], s=40, c="#66c2a5", marker="s", alpha=0.8, label="myoglobin")
    ax_tis.legend(loc="upper right")

    # Acid proxy along interface (for Bohr effect): based on low Ot near y≈cap_h
    def make_acid_fun(Ot_grid, y):
        # take first row inside tissue
        row = Ot_grid[max(1, int(0.02 * len(y))) , :]  # small offset from interface
        # normalize to [0,1] and invert
        rmin, rmax = np.percentile(row, 5), np.percentile(row, 95)
        norm = np.clip((row - rmin) / (rmax - rmin + 1e-9), 0, 1)
        acid = 1.0 - norm
        x_axis = np.linspace(0, L, len(row))
        def acid_fun(xs):
            # linear interp
            return np.interp(xs, x_axis, acid)
        return acid_fun

    # Animation update
    def update(i):
        t = ts[i]
        x, y, Oc_line, Ot_grid = predict_fields(model, L, H, args.nx, args.ny, t)

        # Blend small visual noise into heatmap
        Ot_noisy = Ot_grid + 0.02 * np.std(Ot_grid) * np.random.randn(*Ot_grid.shape)

        # Update tissue heatmap
        im.set_data(Ot_noisy)
        im.set_clim(vmin=max(0.0, np.min(Ot_grid)), vmax=np.max(Ot_grid) + 1e-6)

        # Acid proxy
        acid_fun = make_acid_fun(Ot_grid, y)

        # RBC dynamics
        rbc.step(acid_fun)
        colors = rbc.colors()
        scat_rbc.set_offsets(np.c_[rbc.x, rbc.y])
        scat_rbc.set_sizes(rbc.sizes)
        scat_rbc.set_facecolors(colors)

        # Inject O2 into tissue proportionally to local deoxy near interface
        # Sample a subset of RBCs near x bins
        bins = 12
        counts, edges = np.histogram(rbc.x, bins=bins, range=(0, L))
        deoxy_mean = np.zeros(bins)
        # compute mean deoxy per bin
        idx = np.digitize(rbc.x, edges) - 1
        for b in range(bins):
            mask = idx == b
            if np.any(mask):
                deoxy_mean[b] = np.mean(1.0 - rbc.oxy[mask])
        spawn_base = 6  # tune
        for b in range(bins):
            n_new = np.random.poisson(spawn_base * (0.3 + deoxy_mean[b]))
            x_mid = 0.5 * (edges[b] + edges[b+1])
            tis.inject_from_interface(n_new, x_mid)

        # Tissue O2 dynamics
        tis.step()
        xy = tis.scatter_data()
        scat_o2.set_offsets(xy)

        # Update myoglobin marker sizes by hits
        sizes = 40 + 2.0 * tis.depot_hit
        depots_plot.set_sizes(sizes)

        ax_cap.set_title(f"Capillary: RBC flow  (t = {t:.2f} s)")
        ax_tis.set_title("Tissue: O$_2$ concentration & diffusing molecules")
        return (im, scat_rbc, scat_o2, depots_plot)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/args.fps, blit=False, repeat=True)
    # Save optional MP4 if ffmpeg available:
    # ani.save("oxygen_microvasc.mp4", fps=args.fps, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

