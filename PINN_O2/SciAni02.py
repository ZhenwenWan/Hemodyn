# SciAni.py — High-fidelity physiological oxygen dynamics visualization
# Uses PINN outputs already available (capillary Oc 1D line, tissue Ot 2D grid)
# and synthesizes the missing pieces (CO fields, carboxyHb, myoglobin sites,
# textures, and 2D vessel rendering).
#
# Usage (typical):
#   python SciAni.py --meta checkpoints/run_meta.json --frames 300 --fps 25
#
# Notes:
# - Top 1/3 of the domain is microvasculature; bottom 2/3 is tissue.
# - RBCs: red outer circle (white→red by O2 saturation), blue inner circle
#   (white→blue by carboxyHb = 1 - O2 saturation), inner center randomly offset
#   up to 2/3 of outer radius. (Matplotlib Circle patches, layered above images.)
# - Dissolved O2 = green spots; dissolved CO = black spots; both diffuse in
#   microvasculature and tissue. Myoglobins = fixed red circles in tissue, fill
#   (white→red) follows local Ot.
#
# Implementation details leveraged from Matplotlib docs:
#   - patches.Circle for the nested RBC & myoglobin icons
#   - FuncAnimation for the animation loop
#   - imshow layering + zorder for textures/fields ordering
#
# (Citations: see short notes at bottom.)

import os
os.environ.setdefault("DDE_BACKEND", "pytorch")  # must match .pt checkpoints

import argparse
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="checkpoints/run_meta.json",
                   help="Path to run_meta.json produced by training")
    p.add_argument("--nx", type=int, default=160, help="grid x for fields")
    p.add_argument("--ny", type=int, default=96, help="grid y for fields")
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    # Visual entity counts
    p.add_argument("--rbc", type=int, default=90)
    p.add_argument("--o2_spots", type=int, default=900)
    p.add_argument("--co_spots", type=int, default=700)
    p.add_argument("--myoglobins", type=int, default=36)
    # Vessel/tissue split (top fraction is microvasculature)
    p.add_argument("--cap_frac", type=float, default=1.0/3.0)
    return p.parse_args()

# ---------------------- Meta & ckpt I/O ----------------------
def _resolve_checkpoint(meta_path, ckpt_field):
    """Return a valid absolute checkpoint path, fixing common path issues."""
    meta_path = os.path.abspath(meta_path)
    meta_dir  = os.path.dirname(meta_path)
    proj_root = os.path.dirname(meta_dir)

    if os.path.isabs(ckpt_field) and os.path.exists(ckpt_field):
        return ckpt_field

    candidates = [
        os.path.normpath(os.path.join(meta_dir, ckpt_field)),
        os.path.normpath(os.path.join(proj_root, ckpt_field)),
        os.path.normpath(os.path.join(meta_dir, os.path.basename(ckpt_field))),
        os.path.normpath(os.path.join(proj_root, os.path.basename(ckpt_field))),
    ]
    for c in list(candidates):
        candidates.append(c.replace(os.sep + "checkpoints" + os.sep + "checkpoints" + os.sep,
                                    os.sep + "checkpoints" + os.sep))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError("Checkpoint not found. Tried:\n  " + "\n  ".join(candidates))

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ckpt_field = meta.get("checkpoint_path", "")
    if not ckpt_field:
        raise RuntimeError("run_meta.json missing 'checkpoint_path'.")
    ckpt = _resolve_checkpoint(meta_path, ckpt_field)
    tp = meta.get("true_params", {})
    return ckpt, tp, meta

# ---------------------- Model load (weights-only) ----------------------
def build_model_and_load_weights(true_params, checkpoint_path):
    import deepxde as dde
    from deepxde.backend import backend as F  # noqa: F401
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch not available for DeepXDE backend 'pytorch'.") from e

    if dde.backend.backend_name != "pytorch":
        raise RuntimeError(f"DeepXDE backend is {dde.backend.backend_name!r}, expected 'pytorch'.")

    L = float(true_params.get("L", 1.0))
    H = float(true_params.get("H", 0.3))
    T = float(true_params.get("T", 2.0))

    # Build a minimal model (no training) so we can load the net's weights
    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_zero(X, Y):
        # Two outputs (Oc, Ot) — we only need the network structure, not residuals.
        return [Y[:, :1]*0.0, Y[:, :1]*0.0]

    data = dde.data.TimePDE(geomtime, pde_zero, [], num_domain=1, num_boundary=0,
                            num_initial=0, train_distribution="uniform")
    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    # Load WEIGHTS ONLY
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.net.load_state_dict(state)
    return model, (L, H, T)

# ---------------------- Field evaluators ----------------------
def predict_fields(model, L, H, nx, ny, t_scalar):
    """Return 1D vessel line Oc(x,t) and 2D tissue grid Ot(x,y,t)."""
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    Xcap = np.column_stack([x, np.zeros_like(x), np.full_like(x, t_scalar)])
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xmat = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, t_scalar)])
    out_cap = model.predict(Xcap)          # (nx, 2) -> [Oc, Ot]
    out_tis = model.predict(Xmat)[:, 1]    # Ot component
    Oc_line = out_cap[:, 0]
    Ot_grid = out_tis.reshape(ny, nx)
    return x, y, Oc_line, Ot_grid

def normalize01(arr, eps=1e-12):
    a = np.asarray(arr)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    return np.clip((a - lo) / (hi - lo + eps), 0.0, 1.0)

# ---------------------- Background textures ----------------------
def make_textures(nx, ny, L, H, cap_h, rng):
    """Return a single (ny,nx) texture array with distinct top/bottom looks."""
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Tissue: soft striations + low-frequency noise
    tex_tis = 0.55 + 0.10*np.sin(3*np.pi*X/L)*np.sin(12*np.pi*(Y/(H-cap_h+1e-9)))
    tex_tis += 0.06 * rng.standard_normal((ny, nx))
    tex_tis = normalize01(tex_tis)

    # Microvasculature: faint wavy/bubbly pattern
    Ycap = (Y - (H - cap_h)) / (cap_h + 1e-9)  # 0 at interface, 1 at top
    tex_cap = 0.60 + 0.12*np.sin(10*np.pi*X/L) + 0.10*np.sin(16*np.pi*Ycap.clip(0,1))
    tex_cap += 0.05 * rng.standard_normal((ny, nx))
    tex_cap = normalize01(tex_cap)

    texture = tex_tis.copy()
    texture[Y >= (H - cap_h)] = tex_cap[Y >= (H - cap_h)]
    return texture

# ---------------------- Entities: RBCs, gases, myoglobin ----------------------
class RBCStream:
    """RBC kinematics + physiology placeholders; values updated per frame."""
    def __init__(self, n, L, H, cap_h, u, fps, seed=0):
        self.rng = np.random.default_rng(seed)
        self.L, self.H, self.cap_h = L, H, cap_h
        self.y_min, self.y_max = H - cap_h, H
        self.dt = 1.0 / fps
        self.u = float(u)

        self.x = self.rng.uniform(0, L, n)
        self.y = self.rng.uniform(self.y_min + 0.08*cap_h, self.y_max - 0.08*cap_h, n)
        self.oxy = self.rng.uniform(0.65, 0.95, n)  # relative O2 saturation placeholder
        self.deform = self.rng.uniform(0.012*cap_h, 0.024*cap_h, n)  # outer radii in data units
        # Inner circle offset: up to 2/3 of outer radius
        theta = self.rng.uniform(0, 2*np.pi, n)
        rho   = self.rng.uniform(0, 2/3, n) * self.deform
        self.offset = np.stack([rho*np.cos(theta), rho*np.sin(theta)], axis=1)
        self.jitter = 0.12 * cap_h

    def step(self, acid_fun, oc_sampler):
        """Advect + jitter; update oxy by (i) Bohr proxy (acid) and (ii) Oc line."""
        # Kinematics
        self.x += self.u * self.dt
        self.x[self.x > self.L] -= self.L
        self.y += self.rng.normal(0.0, self.jitter, size=self.y.shape) * self.dt
        self.y = np.clip(self.y, self.y_min + 0.04*self.cap_h, self.y_max - 0.04*self.cap_h)

        # Physiology (visual model)
        acid = acid_fun(self.x)                      # higher acid -> unload O2
        oc_norm = oc_sampler(self.x)                 # normalized vessel O2 at RBC x
        # Move saturation toward oc_norm while down-weighting in acidic zones
        target = oc_norm * (1.0 - 0.45*acid)
        self.oxy += 0.65 * (target - self.oxy) * self.dt
        self.oxy = np.clip(self.oxy, 0.02, 0.98)

class RBCRenderer:
    """Manages nested Circle patches for RBCs."""
    def __init__(self, ax, rbc: RBCStream, z=4, alpha=0.95):
        self.ax = ax
        self.rbc = rbc
        self.outer = []
        self.inner = []
        for i in range(rbc.x.size):
            r_out = rbc.deform[i]
            r_in  = r_out / 3.0
            c_out = Circle((rbc.x[i], rbc.y[i]), r_out, facecolor=(1, 0.6, 0.6),
                           edgecolor="none", alpha=alpha, zorder=z)
            # inner center offset relative to outer
            cx, cy = rbc.x[i] + rbc.offset[i, 0], rbc.y[i] + rbc.offset[i, 1]
            c_in  = Circle((cx, cy), r_in, facecolor=(0.6, 0.6, 1.0),
                           edgecolor="none", alpha=alpha, zorder=z+0.1)
            ax.add_patch(c_out)
            ax.add_patch(c_in)
            self.outer.append(c_out)
            self.inner.append(c_in)

    @staticmethod
    def _color_white_to_red(s):   # s in [0,1]
        return (1.0, 1.0 - s, 1.0 - s)
    @staticmethod
    def _color_white_to_blue(s):  # s in [0,1]
        return (1.0 - s, 1.0 - s, 1.0)

    def update(self):
        rbc = self.rbc
        for i in range(rbc.x.size):
            r_out = rbc.deform[i]
            r_in  = r_out / 3.0
            # colors: outer by oxy; inner by carboxy (1 - oxy)
            oxy = float(rbc.oxy[i])
            car = float(1.0 - oxy)   # per spec: carboxyHb = 1 - pO2
            self.outer[i].set_center((rbc.x[i], rbc.y[i]))
            self.outer[i].set_radius(r_out)
            self.outer[i].set_facecolor(self._color_white_to_red(oxy))
            # inner offset stays relative to outer center
            cx = rbc.x[i] + rbc.offset[i, 0]
            cy = rbc.y[i] + rbc.offset[i, 1]
            self.inner[i].set_center((cx, cy))
            self.inner[i].set_radius(r_in)
            self.inner[i].set_facecolor(self._color_white_to_blue(car))

class GasSpots:
    """Diffusing small molecules (O2 green / CO black), with simple reflecting boundaries."""
    def __init__(self, n, L, H, fps, color="green", size=6, seed=0):
        self.rng = np.random.default_rng(seed)
        self.L, self.H = L, H
        self.dt = 1.0 / fps
        self.sigma = 0.018 * max(L, H)  # visual diffusion speed
        self.xy = np.column_stack([
            self.rng.uniform(0, L, n),
            self.rng.uniform(0, H, n),
        ])
        self.sizes = np.full(n, size, dtype=float)
        self.color = color
        self._coll = None  # PathCollection

    def add_to_axes(self, ax, z=3, alpha=0.75):
        self._coll = ax.scatter(self.xy[:,0], self.xy[:,1], s=self.sizes,
                                c=self.color, edgecolors="none", alpha=alpha, zorder=z)
        return self._coll

    def step(self):
        if self.xy.size == 0:
            return
        disp = self.rng.normal(0.0, self.sigma * math.sqrt(self.dt), size=self.xy.shape)
        self.xy += disp
        # reflect at boundaries
        for d, lo, hi in [(0, 0.0, self.L), (1, 0.0, self.H)]:
            low = self.xy[:, d] < lo
            high = self.xy[:, d] > hi
            self.xy[low, d]  = 2*lo - self.xy[low, d]
            self.xy[high, d] = 2*hi - self.xy[high, d]
            self.xy[:, d] = np.clip(self.xy[:, d], lo, hi)

    def update_artist(self):
        if self._coll is not None:
            self._coll.set_offsets(self.xy)

class MyoglobinSites:
    """Fixed myoglobin depots in tissue; color follows local Ot (white→red)."""
    def __init__(self, n, L, H, cap_h, seed=0):
        self.rng = np.random.default_rng(seed)
        self.L, self.H, self.cap_h = L, H, cap_h
        self.y_max = H - cap_h
        self.xy = self.rng.uniform([0.08*L, 0.10*self.y_max],
                                   [0.92*L, 0.95*self.y_max], size=(n, 2))
        self.r = 0.02 * self.y_max
        self.patches = []

    @staticmethod
    def _color_white_to_red(s):
        return (1.0, 1.0 - s, 1.0 - s)

    def add_to_axes(self, ax, z=4, alpha=0.9):
        for i in range(self.xy.shape[0]):
            c = Circle(tuple(self.xy[i]), radius=self.r,
                       facecolor=(0.9, 0.9, 0.9), edgecolor="none",
                       alpha=alpha, zorder=z)
            ax.add_patch(c)
            self.patches.append(c)
        return self.patches

    def update(self, Ot_grid, x_grid, y_grid):
        # Nearest-neighbor sample of Ot at each depot center
        nx, ny = x_grid.size, y_grid.size
        for i, c in enumerate(self.patches):
            cx, cy = self.xy[i]
            ix = int(np.clip(np.searchsorted(x_grid, cx) - 1, 0, nx-1))
            iy = int(np.clip(np.searchsorted(y_grid, cy) - 1, 0, ny-1))
            s = float(normalize01(Ot_grid)[iy, ix])
            c.set_facecolor(self._color_white_to_red(s))

# ---------------------- Acid / CO synthesis helpers ----------------------
def make_acid_fun_from_Ot(Ot_grid, y, H, cap_h, L):
    """Build a 1D 'acid' profile along x using tissue Ot just below the interface."""
    y_int = H - cap_h
    row = np.clip(np.searchsorted(y, y_int) - 1, 1, len(y)-2)
    line = Ot_grid[row, :]
    acid = 1.0 - normalize01(line)  # low Ot -> high 'acid' (Bohr proxy)
    x_axis = np.linspace(0, L, line.size)
    def acid_fun(xs):
        return np.interp(xs, x_axis, acid)
    return acid_fun

def make_oc_sampler(Oc_line, L):
    oc_norm = normalize01(Oc_line)
    x_axis = np.linspace(0, L, oc_norm.size)
    def sampler(xs):
        return np.interp(xs, x_axis, oc_norm)
    return sampler

def synthesize_CO_fields(Oc_line, Ot_grid):
    """Return (CO_line, CO_grid) in [0,1] as complements (visual placeholder)."""
    CO_line = 1.0 - normalize01(Oc_line)
    CO_grid = 1.0 - normalize01(Ot_grid)
    return CO_line, CO_grid

# ---------------------- Animation ----------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Load model
    try:
        ckpt_path, tp, meta = load_meta(args.meta)
    except Exception as e:
        print(f"[FATAL] Could not resolve checkpoint: {e}")
        sys.exit(1)
    try:
        model, (L, H, T) = build_model_and_load_weights(tp, ckpt_path)
    except Exception as e:
        print(f"[FATAL] Could not load PINN weights: {e}")
        sys.exit(1)

    cap_h = max(1e-3, float(args.cap_frac) * H)   # top band height (microvasculature)
    y_interface = H - cap_h

    # --- Figure & axes (single 2D domain)
    plt.rcParams["figure.figsize"] = (10.5, 7.0)
    fig, ax = plt.subplots()
    ax.set_xlim(0, L); ax.set_ylim(0, H)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

    # --- Background textures (static)
    tex = make_textures(args.nx, args.ny, L, H, cap_h, rng)
    im_tex = ax.imshow(tex, extent=[0, L, 0, H], origin="lower",
                       cmap="Greys", interpolation="bilinear",
                       alpha=0.28, zorder=0)

    # Interface line
    ax.axhline(y_interface, color="k", linewidth=1.0, zorder=2)

    # --- Dynamic field overlay: combine vessel Oc (extruded to top band) + tissue Ot
    field_img = ax.imshow(np.zeros((args.ny, args.nx)), extent=[0, L, 0, H],
                          origin="lower", aspect="auto", interpolation="bilinear",
                          vmin=0.0, vmax=1.0, cmap="viridis", alpha=0.65, zorder=1)
    cbar = fig.colorbar(field_img, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative O$_2$ (vessel+ tissue)")

    # --- Entities
    u = float(tp.get("u", 1.0))
    rbc = RBCStream(n=args.rbc, L=L, H=H, cap_h=cap_h, u=u, fps=args.fps, seed=args.seed)
    rbc_art = RBCRenderer(ax, rbc, z=4, alpha=0.96)

    o2 = GasSpots(args.o2_spots, L, H, args.fps, color="lime", size=7, seed=args.seed+1)
    co = GasSpots(args.co_spots, L, H, args.fps, color="black", size=6, seed=args.seed+2)
    o2_coll = o2.add_to_axes(ax, z=3, alpha=0.75)
    co_coll = co.add_to_axes(ax, z=3, alpha=0.70)

    myo = MyoglobinSites(args.myoglobins, L, H, cap_h, seed=args.seed+3)
    myo_patches = myo.add_to_axes(ax, z=4, alpha=0.92)

    # --- Time grid
    ts = np.linspace(0.0, T, args.frames)
    x_grid = np.linspace(0.0, L, args.nx)
    y_grid = np.linspace(0.0, H, args.ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="xy")
    top_mask = Yg >= y_interface

    # --- Per-frame update
    def update(i):
        t = float(ts[i])

        # 1) Get model fields
        x, y, Oc_line, Ot_grid = predict_fields(model, L, H, args.nx, args.ny, t)

        # 2) Build combined display field (relative O2):
        #    - vessel region (top band): extrude normalized Oc_line over the band
        #    - tissue region: normalized Ot_grid
        Oc_norm = normalize01(Oc_line)
        Ot_norm = normalize01(Ot_grid)
        field = Ot_norm.copy()
        #field[top_mask] = np.repeat(Oc_norm[np.newaxis, :], top_mask.sum(axis=0)[0], axis=0)
        iy0 = int(np.searchsorted(y_grid, y_interface, side="left"))
        field[iy0:, :] = Oc_norm  # broadcasts across the top rows

        # optional: gentle noise for visual depth (doesn't alter dynamics)
        field += 0.02 * (np.std(field) + 1e-9) * rng.standard_normal(field.shape)
        field = np.clip(field, 0.0, 1.0)

        field_img.set_data(field)
        field_img.set_clim(0.0, 1.0)

        # 3) Update Bohr-proxy & Oc sampler; step RBCs then recolor
        acid_fun = make_acid_fun_from_Ot(Ot_grid, y, H, cap_h, L)
        oc_smpl = make_oc_sampler(Oc_line, L)
        rbc.step(acid_fun, oc_smpl)
        rbc_art.update()

        # 4) Synthesize CO fields (for molecules; simple complement)
        CO_line, CO_grid = synthesize_CO_fields(Oc_line, Ot_grid)

        # 5) Diffusing molecules (both regions)
        o2.step()
        co.step()
        o2.update_artist()
        co.update_artist()

        # 6) Myoglobin sites color by local Ot
        myo.update(Ot_grid, x_grid, y_grid)

        ax.set_title(f"Oxygen dynamics — t={t:.2f} s   (top: microvasculature, bottom: tissue)")
        return (field_img, o2_coll, co_coll, *myo_patches, *rbc_art.outer, *rbc_art.inner)

    ani = FuncAnimation(fig, update, frames=args.frames, interval=1000/args.fps,
                        blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()

