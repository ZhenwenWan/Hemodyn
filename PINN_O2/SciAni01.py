# SciAni01.py — Scientific animation of microvascular oxygen using PINN04.py outputs
# Notes:
#  • Forces the PyTorch backend (must match .pt checkpoints).
#  • Resolves checkpoint paths robustly (avoids duplicated "checkpoints/checkpoints").
#  • Loads weights only (not optimizer) for inference.
#  • Every function/class has a docstring explaining purpose, args, returns, and behavior.

import os
os.environ.setdefault("DDE_BACKEND", "pytorch")  # must be set BEFORE importing deepxde

import argparse
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed options:
        - meta: path to run_meta.json produced by PINN04.py
        - nx, ny: grid resolution for x and y
        - frames: number of animation frames across [0, T]
        - fps: frames per second
        - cap_h_frac: capillary strip height as fraction of tissue height H
        - seed: RNG seed for reproducibility

    Notes
    -----
    Docstrings follow PEP 257/Google guidance for clarity. :contentReference[oaicite:1]{index=1}
    """
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


def _resolve_checkpoint(meta_path: str, ckpt_field: str) -> str:
    """Resolve a checkpoint path relative to `run_meta.json`, avoiding duplicates.

    Parameters
    ----------
    meta_path : str
        Absolute or relative path to run_meta.json.
    ckpt_field : str
        The raw 'checkpoint_path' value stored in the meta JSON.

    Returns
    -------
    str
        Absolute path to an existing checkpoint file.

    Raises
    ------
    FileNotFoundError
        If no candidate path exists.

    Notes
    -----
    Handles cases where meta is saved under `checkpoints/` and `checkpoint_path`
    also begins with `checkpoints/`, preventing `.../checkpoints/checkpoints/...`.
    """
    meta_path = os.path.abspath(meta_path)
    meta_dir = os.path.dirname(meta_path)
    proj_root = os.path.dirname(meta_dir)

    # If absolute and exists, use it directly.
    if os.path.isabs(ckpt_field) and os.path.exists(ckpt_field):
        return ckpt_field

    # Candidate list (first existing wins).
    candidates = [
        os.path.normpath(os.path.join(meta_dir, ckpt_field)),
        os.path.normpath(os.path.join(proj_root, ckpt_field)),
        os.path.normpath(os.path.join(meta_dir, os.path.basename(ckpt_field))),
        os.path.normpath(os.path.join(proj_root, os.path.basename(ckpt_field))),
    ]

    # Collapse accidental ".../checkpoints/checkpoints/..." duplication.
    for c in list(candidates):
        candidates.append(
            c.replace(os.sep + "checkpoints" + os.sep + "checkpoints" + os.sep,
                      os.sep + "checkpoints" + os.sep)
        )

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n  " + "\n  ".join(candidates)
        + "\n(Hint: keep checkpoint_path either absolute, or relative to the project root "
          "OR to the directory of run_meta.json)."
    )


def load_meta(meta_path: str):
    """Load `run_meta.json` and resolve the stored checkpoint path.

    Parameters
    ----------
    meta_path : str
        Path to run_meta.json.

    Returns
    -------
    tuple
        (checkpoint_abs_path: str, true_params: dict, full_meta: dict)

    Raises
    ------
    RuntimeError
        If 'checkpoint_path' is missing in the meta.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ckpt_field = meta.get("checkpoint_path", "")
    if not ckpt_field:
        raise RuntimeError("run_meta.json missing 'checkpoint_path'.")
    ckpt = _resolve_checkpoint(meta_path, ckpt_field)
    tp = meta.get("true_params", {})
    return ckpt, tp, meta


def build_model_and_load_weights(true_params: dict, checkpoint_path: str):
    """Recreate the DeepXDE model skeleton and load weights from a PyTorch checkpoint.

    Parameters
    ----------
    true_params : dict
        Dictionary with keys like L, H, T (domain sizes/time).
    checkpoint_path : str
        Absolute path to a `.pt` file saved by PINN04.py.

    Returns
    -------
    tuple
        (model: dde.Model, (L: float, H: float, T: float))

    Raises
    ------
    RuntimeError
        If PyTorch is unavailable or the backend is not 'pytorch'.

    Notes
    -----
    Weights-only restore: create the same net shape, then call `load_state_dict`
    with the model portion of the checkpoint (ignore optimizer). This is the
    recommended inference pattern in PyTorch. :contentReference[oaicite:2]{index=2}
    """
    import deepxde as dde
    from deepxde.backend import backend as F
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch not available for DeepXDE backend 'pytorch'.") from e

    if dde.backend.backend_name != "pytorch":
        raise RuntimeError(f"DeepXDE backend is {dde.backend.backend_name!r}, expected 'pytorch'.")

    # Domain defaults (if absent in meta)
    L = float(true_params.get("L", 1.0))
    H = float(true_params.get("H", 0.2))
    T = float(true_params.get("T", 2.0))

    # Minimal geometry/timebox & dummy PDE (no training; needed to instantiate Model)
    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_zero(X, Y):
        """Dummy PDE returning zeros for (Oc, Ot); used only to satisfy Model API."""
        z = F.zeros_like(X[:, :1])
        return [z, z]  # two outputs (Oc, Ot)

    data = dde.data.TimePDE(
        geomtime, pde_zero, [], num_domain=1, num_boundary=0, num_initial=0,
        train_distribution="uniform"
    )
    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)  # create a no-op optimizer; we won't load its state

    # ---- Load WEIGHTS ONLY (avoid optimizer restore issues) ----
    try:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    # DeepXDE (PyTorch) typically saves {"model_state_dict": ..., "optimizer_state_dict": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt  # fallback: assume raw state_dict

    model.net.load_state_dict(state)  # weights only
    return model, (L, H, T)


def predict_fields(model, L: float, H: float, nx: int, ny: int, t_scalar: float):
    """Evaluate Oc(x, t) along the capillary and Ot(x, y, t) in tissue at time t.

    Parameters
    ----------
    model : dde.Model
        Restored DeepXDE model (PyTorch backend).
    L, H : float
        Domain sizes in x (capillary/tissue length) and y (tissue height).
    nx, ny : int
        Grid resolution for x and y.
    t_scalar : float
        Time at which to evaluate the fields.

    Returns
    -------
    tuple
        (x: (nx,), y: (ny,), Oc_line: (nx,), Ot_grid: (ny, nx))

    Notes
    -----
    - Capillary slice is sampled at y=0 for Oc(x, t).
    - Tissue grid spans full [0, L] × [0, H] for Ot(x, y, t).
    """
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)

    # Capillary line at interface y=0
    Xcap = np.column_stack([x, np.zeros_like(x), np.full_like(x, t_scalar)])

    # Tissue grid
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xmat = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, t_scalar)])

    out_cap = model.predict(Xcap)           # (nx, 2) -> [Oc, Ot]
    out_tis = model.predict(Xmat)[:, 1]     # take Ot only

    Oc_line = out_cap[:, 0]
    Ot_grid = out_tis.reshape(ny, nx)
    return x, y, Oc_line, Ot_grid


def make_acid_fun(Ot_grid: np.ndarray, y: np.ndarray, L: float):
    """Create a Bohr-effect proxy along the interface from the tissue field.

    Parameters
    ----------
    Ot_grid : ndarray, shape (ny, nx)
        Tissue O2 field at the current time.
    y : ndarray, shape (ny,)
        y-coordinates used to build `Ot_grid`.
    L : float
        Domain length in x.

    Returns
    -------
    callable
        Function acid_fun(xs) -> array in [0,1], higher where Ot near interface is low.

    Notes
    -----
    We sample a row just inside the tissue (slightly above y=0), normalize it to [0,1],
    and invert as a heuristic proxy for acidity: lower Ot ⇒ higher acidity.
    """
    idx_row = min(max(1, int(0.02 * len(y))), len(y) - 1)  # small offset from interface
    row = Ot_grid[idx_row, :]
    rmin, rmax = np.percentile(row, 5), np.percentile(row, 95)
    norm = np.clip((row - rmin) / (rmax - rmin + 1e-9), 0, 1)
    acid = 1.0 - norm
    x_axis = np.linspace(0, L, len(row))

    def acid_fun(xs):
        return np.interp(xs, x_axis, acid)

    return acid_fun


class RBCStream:
    """Simple RBC particle system flowing along the capillary.

    Attributes
    ----------
    L : float
        Domain length.
    cap_h : float
        Capillary strip height (for plotting and jitter bounds).
    u : float
        Axial speed (mm/s).
    dt : float
        Simulation time step (1/fps).
    x, y : ndarray
        Particle positions.
    oxy : ndarray
        Oxyhemoglobin fraction in [0,1].
    sizes : ndarray
        Marker sizes for plotting.
    jitter : float
        Lateral random-walk magnitude (visual).

    Notes
    -----
    The Bohr effect is approximated via `acid_fun`, which increases oxy→deoxy
    conversion where acidity is high (lower Ot near interface).
    """

    def __init__(self, n, L, cap_h, u, fps, seed=0):
        """Initialize particles with uniform x, bounded y, and random oxy-fractions."""
        rng = np.random.default_rng(seed)
        self.L, self.cap_h, self.u = L, cap_h, u
        self.dt = 1.0 / fps
        self.x = rng.uniform(0, L, n)
        self.y = rng.uniform(0.15 * cap_h, 0.85 * cap_h, n)
        self.oxy = rng.uniform(0.7, 1.0, n)     # oxy fraction
        self.sizes = rng.uniform(15, 35, n)
        self.jitter = 0.05 * cap_h

    def step(self, acid_fun):
        """Advance RBCs by advection + lateral jitter and update oxy-fraction.

        Parameters
        ----------
        acid_fun : callable
            Function acid_fun(x) in [0,1]; higher values cause faster oxy→deoxy.
        """
        # Axial advection with wrap-around
        self.x += self.u * self.dt
        self.x[self.x > self.L] -= self.L

        # Small transverse random walk, clipped within capillary
        self.y += np.random.normal(0.0, self.jitter, size=self.y.shape) * self.dt
        self.y = np.clip(self.y, 0.05 * self.cap_h, 0.95 * self.cap_h)

        # Bohr-effect proxy: probabilistic oxy→deoxy conversion
        acid = acid_fun(self.x)
        k_bohr = 1.5  # s^-1 (visual parameter)
        p = 1.0 - np.exp(-k_bohr * acid * self.dt)
        self.oxy = np.maximum(0.05, self.oxy * (1.0 - 0.6 * p))

    def colors(self):
        """Return RGBA colors blending red (oxy) to blue (deoxy) per particle."""
        r = self.oxy
        return np.stack([r, 0.1 * (1 - r), 1 - r, np.full_like(r, 0.9)], axis=1)


class TissueO2:
    """Tissue oxygen particle system with diffusion, metabolism, and myoglobin capture.

    Behavior
    --------
    - New O2 particles are injected just below the capillary interface.
    - Particles undergo a 2D random walk (diffusion).
    - Poisson thinning removes particles (metabolism) at rate `lambda_metab`.
    - Particles entering "myoglobin depots" are captured (stored).

    Parameters
    ----------
    L, H : float
        Domain sizes.
    cap_h : float
        Interface height separating capillary strip from tissue.
    fps : int
        Frames per second; sets the simulation time step dt = 1/fps.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(self, L, H, cap_h, fps, seed=0):
        self.L, self.H, self.cap_h = L, H, cap_h
        self.dt = 1.0 / fps
        self.xy = np.empty((0, 2), dtype=float)   # particle positions
        self.alive = np.empty((0,), dtype=bool)   # boolean mask
        self.rng = np.random.default_rng(seed)
        self.sigma = math.sqrt(2e-3)  # mm / sqrt(s) — visual diffusion scale
        self.lambda_metab = 0.8       # s^-1 — metabolism rate (visual)
        self.depots = self.rng.uniform([0.1*L, 0.2*H], [0.9*L, 0.95*H], size=(25, 2))
        self.depot_r = 0.02 * H
        self.depot_hit = np.zeros(self.depots.shape[0], dtype=int)

    def inject_from_interface(self, n_new: int, where_x: float):
        """Inject `n_new` O2 particles slightly below the interface near `where_x`."""
        if n_new <= 0:
            return
        y0 = self.cap_h + self.rng.uniform(0.0, 0.02 * self.H, size=n_new)
        x0 = where_x + self.rng.uniform(-0.01 * self.L, 0.01 * self.L, size=n_new)
        x0 = np.clip(x0, 0.0, self.L)
        pts = np.stack([x0, y0], axis=1)
        self.xy = np.vstack([self.xy, pts])
        self.alive = np.concatenate([self.alive, np.ones(n_new, dtype=bool)])

    def step(self):
        """Advance all particles by diffusion; apply metabolism and myoglobin capture."""
        if self.xy.size == 0:
            return
        # Random walk (diffusion)
        disp = self.rng.normal(0.0, self.sigma * math.sqrt(self.dt), size=self.xy.shape)
        self.xy[self.alive] += disp[self.alive]
        # Keep inside tissue region
        self.xy[:, 0] = np.clip(self.xy[:, 0], 0.0, self.L)
        self.xy[:, 1] = np.clip(self.xy[:, 1], self.cap_h + 1e-6, self.H)
        # Poisson thinning for metabolism
        p_die = 1.0 - np.exp(-self.lambda_metab * self.dt)
        dies = self.rng.random(self.alive.shape) < p_die
        # Myoglobin capture in depot disks
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
        """Return positions of currently alive O2 particles for plotting."""
        return self.xy[self.alive, :]


def main():
    """Run the animation.

    Flow
    ----
    1) Load meta and resolve checkpoint path.
    2) Rebuild the PINN structure and load weights (PyTorch state_dict).
    3) Create RBC and tissue particle systems.
    4) Animate tissue field (heatmap) + particles with matplotlib.
    """
    args = parse_args()
    np.random.seed(args.seed)

    # 1) Resolve checkpoint from meta
    try:
        ckpt_path, tp, meta = load_meta(args.meta)
    except Exception as e:
        print(f"[FATAL] Could not resolve checkpoint: {e}")
        sys.exit(1)

    # 2) Build model and load weights
    try:
        model, (L, H, T) = build_model_and_load_weights(tp, ckpt_path)
    except Exception as e:
        print(f"[FATAL] Could not load PINN weights: {e}")
        sys.exit(1)

    # Capillary strip height (visual)
    cap_h = max(1e-3, args.cap_h_frac * H)

    # Time samples across [0, T]
    ts = np.linspace(0.0, T, args.frames)

    # Particle systems
    u = float(tp.get("u", 1.0))  # mm/s axial speed
    rbc = RBCStream(n=80, L=L, cap_h=cap_h, u=u, fps=args.fps, seed=args.seed)
    tis = TissueO2(L=L, H=H, cap_h=cap_h, fps=args.fps, seed=args.seed + 1)

    # Figure layout
    plt.rcParams["figure.figsize"] = (10, 7)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])
    ax_cap = fig.add_subplot(gs[0, 0])
    ax_tis = fig.add_subplot(gs[1, 0])

    # Capillary panel
    ax_cap.set_xlim(0, L)
    ax_cap.set_ylim(0, cap_h)
    ax_cap.set_xlabel("x [mm]")
    ax_cap.set_ylabel("capillary y")
    scat_rbc = ax_cap.scatter([], [], s=[], c=[], edgecolors="none")

    # Tissue panel
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

    def update(i):
        """Matplotlib animation callback: advance simulation one frame."""
        t = ts[i]

        # Predict fields at time t from the PINN
        x, y, Oc_line, Ot_grid = predict_fields(model, L, H, args.nx, args.ny, t)

        # Visual noise for the heatmap (cosmetic only)
        Ot_noisy = Ot_grid + 0.02 * (np.std(Ot_grid) + 1e-12) * np.random.randn(*Ot_grid.shape)
        im.set_data(Ot_noisy)
        im.set_clim(vmin=max(0.0, float(np.min(Ot_grid))), vmax=float(np.max(Ot_grid)) + 1e-6)

        # Bohr-effect proxy based on low Ot near interface
        acid_fun = make_acid_fun(Ot_grid, y, L)

        # RBC dynamics (capillary)
        rbc.step(acid_fun)
        scat_rbc.set_offsets(np.c_[rbc.x, rbc.y])
        scat_rbc.set_sizes(rbc.sizes)
        scat_rbc.set_facecolors(rbc.colors())

        # Spawn tissue O2 proportional to local deoxy level along x
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

        # Tissue dynamics
        tis.step()
        scat_o2.set_offsets(tis.scatter_data())

        ax_cap.set_title(f"Capillary RBCs (t={t:.2f} s)")
        ax_tis.set_title("Tissue O$_2$ (PINN prediction + noise)")
        return (im, scat_rbc, scat_o2)

    ani = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps,
                        blit=False, repeat=True)
    plt.show()


if __name__ == "__main__":
    main()

