
"""
SciAni16.py
-------------------
Blitting-safe fix for time_text on Matplotlib:
- Use axes-level text (ax.text) so artist has a valid Axes (prevents NoneType._get_view crash).
- Update label EVERY frame; return it from init() and update().
- Re-cache blit backgrounds on resize/draw.
Also keeps the precompute-first pipeline for smooth playback.
"""

import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import argparse
import json
import math
import sys
from typing import Callable, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
from matplotlib.patches import Circle
import imageio_ffmpeg

matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, default="checkpoints/run_meta.json")
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--ny", type=int, default=72)
    p.add_argument("--frames", type=int, default=480)   # default aligned with prior run
    p.add_argument("--fps", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rbc", type=int, default=20)
    p.add_argument("--myoglobins", type=int, default=20)
    p.add_argument("--o2_spots", type=int, default=80)
    p.add_argument("--co_spots", type=int, default=80)
    p.add_argument("--cap_frac", type=float, default=1.0/3.0)
    p.add_argument("--vy_factor", type=float, default=0.01)
    p.add_argument("--vy_tau", type=float, default=0.9)
    p.add_argument("--sim_dt", type=float, default=0.01)
    p.add_argument("--sim_rate", type=float, default=1.0)
    # Perf toggles (fields are precomputed; stride kept for compatibility)
    p.add_argument("--field_stride", type=int, default=4,
                   help="Ignored during playback (fields are precomputed).")
    p.add_argument("--title_stride", type=int, default=8,
                   help="Kept for CLI compatibility; we update text every frame.")
    p.add_argument("--no_halos", action="store_true",
                   help="Disable glow halos for RBCs and myoglobins")
    p.add_argument("--blit", action="store_true",
                   help="Enable Matplotlib blitting")
    # Flashing controls
    p.add_argument("--flash", action="store_false",
                   help="Enable pulsing brightness for all spots")
    p.add_argument("--flash_freq", type=float, default=1.2,
                   help="Pulse frequency (Hz)")
    p.add_argument("--flash_depth", type=float, default=0.35,
                   help="Pulse amplitude (0..1)")
    p.add_argument("--flash_bias", type=float, default=0.75,
                   help="Baseline brightness (0..1)")
    # Saving
    p.add_argument("--save", action="store_true",
                   help="Enable saving into a mp4 file named saveFile")
    p.add_argument("--saveFile", type=str, default="out.mp4",
                   help="Output MP4 filename. Use empty string to disable saving.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--bitrate", type=int, default=3000)
    p.add_argument("--codec", type=str, default="libx264")
    return p.parse_args()


# ============================================================================
# Checkpoint / meta handling
# ============================================================================

def _resolve_checkpoint(meta_path: str, ckpt_field: str) -> str:
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
        candidates.append(
            c.replace(os.sep + "checkpoints" + os.sep + "checkpoints" + os.sep,
                      os.sep + "checkpoints" + os.sep)
        )

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    tried = "\n  ".join(candidates)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n  {tried}")


def load_meta(meta_path: str) -> Tuple[str, dict, dict]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ckpt_field = meta.get("checkpoint_path", "")
    if not ckpt_field:
        raise RuntimeError("run_meta.json missing 'checkpoint_path'.")
    ckpt = _resolve_checkpoint(meta_path, ckpt_field)
    true_params = meta.get("true_params", {})
    return ckpt, true_params, meta


# ============================================================================
# Model construction & weight loading (weights-only)
# ============================================================================

def build_model_and_load_weights(true_params: dict, checkpoint_path: str):
    import deepxde as dde
    try:
        import torch
        torch.set_grad_enabled(False)  # inference only
    except Exception as e:
        raise RuntimeError("PyTorch not available for DeepXDE backend 'pytorch'.") from e

    if dde.backend.backend_name != "pytorch":
        raise RuntimeError(f"DeepXDE backend is {dde.backend.backend_name!r}, expected 'pytorch'.")

    L = float(true_params.get("L", 1.0))
    H = float(true_params.get("H", 0.3))
    T = float(true_params.get("T", 2.0))

    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_zero(X, Y):
        return [Y[:, :1] * 0.0, Y[:, :1] * 0.0]

    data = dde.data.TimePDE(
        geomtime, pde_zero, [],
        num_domain=1, num_boundary=0, num_initial=0,
        train_distribution="uniform"
    )
    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.net.load_state_dict(state)
    return model, (L, H, T)


# ============================================================================
# Utilities & colors
# ============================================================================

def normalize01(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(arr)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    return np.clip((a - lo) / (hi - lo + eps), 0.0, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(c0: Sequence[float], c1: Sequence[float], s: float) -> Tuple[float, float, float]:
    s = float(np.clip(s, 0.0, 1.0))
    return (_lerp(c0[0], c1[0], s), _lerp(c0[1], c1[1], s), _lerp(c0[2], c1[2], s))


def _mix_to_white(rgb: Sequence[float], k: float) -> np.ndarray:
    return (1 - k) * np.array(rgb) + k * np.array((1, 1, 1))


_RED_LO,  _RED_HI  = (1.00, 0.82, 0.82), (0.65, 0.05, 0.05)
_BLUE_LO, _BLUE_HI = (0.82, 0.90, 1.00), (0.05, 0.10, 0.55)
_MEAT_BASE      = (0.93, 0.63, 0.66)
_EGGPLANT_BASE  = (0.35, 0.16, 0.36)


def make_textures(nx: int, ny: int, L: float, H: float, cap_h: float, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    tis = np.zeros((ny, nx, 3), dtype=float)
    for k, basec in enumerate(_MEAT_BASE):
        wav = 0.06 * np.sin(2.5 * np.pi * X / L) * np.sin(5.5 * np.pi * (Y / (H - cap_h + 1e-9)))
        noise = 0.05 * rng.standard_normal((ny, nx))
        tis[..., k] = np.clip(basec + wav + noise, 0.0, 1.0)

    egg = np.zeros((ny, nx, 3), dtype=float)
    for k, basec in enumerate(_EGGPLANT_BASE):
        Ycap = ((Y - (H - cap_h)) / (cap_h + 1e-9)).clip(0, 1)
        wav = 0.08 * np.sin(10 * np.pi * X / L) + 0.05 * np.sin(16 * np.pi * Ycap)
        noise = 0.05 * rng.standard_normal((ny, nx))
        egg[..., k] = np.clip(basec + wav + noise, 0.0, 1.0)

    texture = tis.copy()
    mask = Y >= (H - cap_h)
    texture[mask] = egg[mask]
    return np.clip(texture * 0.85, 0.0, 1.0)


# ============================================================================
# Field evaluator
# ============================================================================

class FieldsEvaluator:
    def __init__(self, model, L: float, H: float, nx: int, ny: int) -> None:
        self.model = model
        self.L, self.H = float(L), float(H)
        self.nx, self.ny = int(nx), int(ny)

        self.x = np.linspace(0.0, L, nx)
        self.y = np.linspace(0.0, H, ny)

        self.Xcap = np.column_stack([self.x, np.zeros_like(self.x), np.zeros_like(self.x)])
        Xg, Yg = np.meshgrid(self.x, self.y, indexing="xy")
        self.Xmat = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)])

    def predict(self, t_scalar: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Xcap = self.Xcap
        Xmat = self.Xmat
        Xcap[:, 2] = t_scalar
        Xmat[:, 2] = t_scalar
        out_cap = self.model.predict(Xcap)
        out_tis = self.model.predict(Xmat)[:, 1]
        Oc_line = out_cap[:, 0]
        Ot_grid = out_tis.reshape(self.ny, self.nx)
        return self.x, self.y, Oc_line, Ot_grid


# ============================================================================
# Entities (RBC stream, renderer, gases, myoglobin)
# ============================================================================

class RBCStream:
    """Kinematics + per-cell properties with motion integrator."""
    def __init__(
        self,
        n: int,
        L: float,
        H: float,
        cap_h: float,
        u: float,
        sim_dt: float,
        seed: int = 0,
        vy_factor: float = 0.2,
        vy_tau: float = 0.9,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.L, self.H, self.cap_h = L, H, cap_h
        self.y_min, self.y_max = H - cap_h, H
        self.dt = float(sim_dt)
        self.u = float(u)

        self.x = self.rng.uniform(0, L, n)
        self.y = self.rng.uniform(self.y_min + 0.06 * cap_h, self.y_max - 0.06 * cap_h, n)

        self.vy_factor = float(vy_factor)
        self.vy_tau = float(vy_tau)
        self.r_vert = np.zeros_like(self.y)

        self.oxy = self.rng.uniform(0.70, 0.95, n)
        # Keep modest RBC radius (adjust to match your SciAni14 visuals if needed)
        self.r_out = self.rng.uniform(0.08 * cap_h, 0.08 * cap_h, n)
        theta = self.rng.uniform(0, 2 * np.pi, n)
        rho = self.rng.uniform(0, 2 / 3, n) * self.r_out
        self.offset = np.stack([rho * np.cos(theta), rho * np.sin(theta)], axis=1)

    def _update_vertical_driver(self) -> None:
        alpha = math.exp(-self.dt / max(1e-6, self.vy_tau))
        self.r_vert = alpha * self.r_vert + math.sqrt(max(0.0, 1.0 - alpha * alpha)) *                       self.rng.standard_normal(self.r_vert.shape)

    def step(self, acid_fun: Callable[[np.ndarray], np.ndarray],
             oc_sampler: Callable[[np.ndarray], np.ndarray],
             u_fun: Callable[[np.ndarray], np.ndarray] = None) -> None:
        if u_fun is None:
            u_local = np.full_like(self.x, self.u, dtype=float)
        else:
            u_local = np.asarray(u_fun(self.x), dtype=float)

        self.x += u_local * self.dt
        self.x[self.x > self.L] -= self.L

        self._update_vertical_driver()
        v_y = self.vy_factor * u_local * self.r_vert
        self.y += v_y * self.dt
        self.y = np.clip(self.y, self.y_min + 0.03*self.cap_h, self.y_max - 0.03*self.cap_h)

        acid = acid_fun(self.x)
        oc_norm = oc_sampler(self.x)
        target = oc_norm * (1.0 - 0.50 * acid)
        self.oxy += 0.80 * (target - self.oxy) * self.dt
        self.oxy = np.clip(self.oxy, 0.02, 0.98)


class RBCRenderer:
    def __init__(
        self,
        ax,
        rbc: RBCStream,
        draw_halos: bool,
        z: float = 4,
        alpha: float = 0.96,
        flash: bool = False,
        freq: float = 1.2,
        depth: float = 0.35,
        bias: float = 0.75,
        seed: int = 0,
    ) -> None:
        self.ax = ax
        self.rbc = rbc
        self.draw_halos = draw_halos
        self.flash = flash
        self.freq = float(freq)
        self.depth = float(depth)
        self.bias = float(bias)

        self.outer: List[Circle] = []
        self.inner: List[Circle] = []
        self.halos: List[Circle] = []

        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * np.pi, rbc.x.size)

        for i in range(rbc.x.size):
            r_out = rbc.r_out[i]
            r_in = r_out / 3.0

            if draw_halos:
                halo = Circle(
                    (rbc.x[i], rbc.y[i]),
                    1.6 * r_out,
                    facecolor=(1, 0.6, 0.6),
                    edgecolor="none",
                    alpha=0.18,
                    zorder=z - 0.2,
                    animated=True,
                )
                ax.add_patch(halo)
                self.halos.append(halo)

            c_out = Circle(
                (rbc.x[i], rbc.y[i]),
                r_out,
                facecolor=(1, 0.82, 0.82),
                edgecolor="none",
                alpha=alpha,
                zorder=z,
                animated=True,
            )
            cx = rbc.x[i] + rbc.offset[i, 0]
            cy = rbc.y[i] + rbc.offset[i, 1]
            c_in = Circle(
                (cx, cy),
                r_in,
                facecolor=(0.82, 0.90, 1.0),
                edgecolor="none",
                alpha=alpha,
                zorder=z + 0.1,
                animated=True,
            )
            ax.add_patch(c_out)
            ax.add_patch(c_in)
            self.outer.append(c_out)
            self.inner.append(c_in)

    @staticmethod
    def _red_color(s: float) -> Tuple[float, float, float]:
        return _lerp3(_RED_LO, _RED_HI, s)

    @staticmethod
    def _blue_color(s: float) -> Tuple[float, float, float]:
        return _lerp3(_BLUE_LO, _BLUE_HI, s)

    def update(self, t: float) -> None:
        rbc = self.rbc
        if self.flash:
            B = np.clip(self.bias + self.depth * np.sin(2 * np.pi * self.freq * t + self.phase), 0.0, 1.0)
        else:
            B = np.ones_like(rbc.x)

        for i in range(rbc.x.size):
            r_out = rbc.r_out[i]
            r_in = r_out / 3.0
            oxy = float(rbc.oxy[i])
            car = 1.0 - oxy
            base_red = self._red_color(oxy)
            base_blue = self._blue_color(car)

            col_out = _mix_to_white(base_red, 1.0 - B[i])
            col_in  = _mix_to_white(base_blue, 1.0 - B[i])

            if self.draw_halos:
                self.halos[i].set_center((rbc.x[i], rbc.y[i]))
                self.halos[i].set_radius(1.6 * r_out)
                halo_col = _mix_to_white(self._red_color(oxy), 1.0 - B[i])
                self.halos[i].set_facecolor(_lerp3((1, 1, 1), tuple(halo_col), 0.55))
                self.halos[i].set_alpha(0.12 + 0.10 * B[i])

            self.outer[i].set_center((rbc.x[i], rbc.y[i]))
            self.outer[i].set_radius(r_out)
            self.outer[i].set_facecolor(tuple(col_out))

            cx = rbc.x[i] + rbc.offset[i, 0]
            cy = rbc.y[i] + rbc.offset[i, 1]
            self.inner[i].set_center((cx, cy))
            self.inner[i].set_radius(r_in)
            self.inner[i].set_facecolor(tuple(col_in))


class GasSpots:
    def __init__(
        self,
        n: int,
        L: float,
        H: float,
        fps: int,
        color: str = "green",
        size: float = 6,
        seed: int = 0,
        flash: bool = False,
        freq: float = 1.2,
        depth: float = 0.35,
        bias: float = 0.75,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.L, self.H = L, H
        self.dt = 1.0 / max(1, fps)
        self.sigma = 0.018 * max(L, H)

        self.xy = np.column_stack([
            self.rng.uniform(0, L, n),
            self.rng.uniform(0, H, n),
        ])
        self.sizes = np.full(n, size, dtype=float)

        self.flash = flash
        self.freq = float(freq)
        self.depth = float(depth)
        self.bias = float(bias)
        self.phase = self.rng.uniform(0, 2 * np.pi, n)

        if color == "lime":
            self.base_rgb = np.array((0.0, 1.0, 0.0))
        elif color == "black":
            self.base_rgb = np.array((0.0, 0.0, 0.0))
        else:
            self.base_rgb = np.array((0.5, 0.5, 0.5))

        self._coll = None
        self._colors = None

    def add_to_axes(self, ax, z: float = 3, alpha: float = 0.75):
        n = self.xy.shape[0]
        self._colors = np.tile(np.append(self.base_rgb, [alpha]), (n, 1))
        self._coll = ax.scatter(
            self.xy[:, 0],
            self.xy[:, 1],
            s=self.sizes,
            c=self._colors,
            edgecolors="none",
            alpha=None,
            zorder=z,
            animated=True,
        )
        return self._coll

    def step(self) -> None:
        if self.xy.size == 0:
            return
        disp = self.rng.normal(0.0, self.sigma * math.sqrt(self.dt), size=self.xy.shape)
        self.xy += disp
        for d, lo, hi in [(0, 0.0, self.L), (1, 0.0, self.H)]:
            low = self.xy[:, d] < lo
            high = self.xy[:, d] > hi
            self.xy[low, d]  = 2 * lo - self.xy[low, d]
            self.xy[high, d] = 2 * hi - self.xy[high, d]
            self.xy[:, d] = np.clip(self.xy[:, d], lo, hi)

    def update_artist(self, t: float) -> None:
        if self._coll is None:
            return
        self._coll.set_offsets(self.xy)
        if self.flash and self._colors is not None:
            B = np.clip(self.bias + self.depth * np.sin(2 * np.pi * self.freq * t + self.phase), 0.0, 1.0)
            mixed = (1 - (1 - B)[:, None]) * self.base_rgb + (1 - B)[:, None] * np.array((1, 1, 1))
            self._colors[:, :3] = mixed
            self._coll.set_facecolors(self._colors)


class MyoglobinSites:
    def __init__(
        self,
        n: int,
        L: float,
        H: float,
        cap_h: float,
        seed: int = 0,
        draw_halos: bool = True,
        flash: bool = False,
        freq: float = 1.2,
        depth: float = 0.35,
        bias: float = 0.75,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.L, self.H, self.cap_h = L, H, cap_h
        self.y_top_tis = H - cap_h
        self.xy = self.rng.uniform([0.06 * L, 0.10 * self.y_top_tis],
                                   [0.94 * L, 0.95 * self.y_top_tis], size=(n, 2))
        self.r = 0.04 * self.y_top_tis  # modest radius; adjust if your SciAni14 uses a different factor
        self.patches: List[Circle] = []
        self.halos: List[Circle] = []
        self.draw_halos = draw_halos

        self.flash = flash
        self.freq = float(freq)
        self.depth = float(depth)
        self.bias = float(bias)
        self.phase = self.rng.uniform(0, 2 * np.pi, n)

    @staticmethod
    def _color_from_Ot(s: float) -> Tuple[float, float, float]:
        return _lerp3(_RED_LO, _RED_HI, s)

    def add_to_axes(self, ax, z: float = 4, alpha: float = 0.9) -> List[Circle]:
        for i in range(self.xy.shape[0]):
            if self.draw_halos:
                halo = Circle(tuple(self.xy[i]), radius=1.6 * self.r,
                              facecolor=_lerp3((1, 1, 1), _RED_LO, 0.55),
                              edgecolor="none", alpha=0.18, zorder=z - 0.2, animated=True)
                ax.add_patch(halo)
                self.halos.append(halo)
            c = Circle(tuple(self.xy[i]), radius=self.r,
                       facecolor=_RED_LO, edgecolor="none",
                       alpha=alpha, zorder=z, animated=True)
            ax.add_patch(c)
            self.patches.append(c)
        return self.patches

    def update(self, Ot_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, t: float) -> None:
        nx, ny = x_grid.size, y_grid.size
        Ot_n = normalize01(Ot_grid)
        for i, c in enumerate(self.patches):
            cx, cy = self.xy[i]
            ix = int(np.clip(np.searchsorted(x_grid, cx) - 1, 0, nx - 1))
            iy = int(np.clip(np.searchsorted(y_grid, cy) - 1, 0, ny - 1))
            s = float(Ot_n[iy, ix])
            base = np.array(self._color_from_Ot(s))
            if self.flash:
                B = float(np.clip(self.bias + self.depth * np.sin(2 * np.pi * self.freq * t + self.phase[i]), 0.0, 1.0))
                col = (1 - (1 - B)) * base + (1 - B) * np.array((1, 1, 1))
            else:
                col = base
            c.set_facecolor(tuple(col))
            if self.draw_halos:
                halo_col = _lerp3((1, 1, 1), tuple(col), 0.55)
                self.halos[i].set_facecolor(halo_col)
                self.halos[i].set_alpha(0.12 + (0.10 if self.flash else 0.0) * (B if self.flash else 1.0))


# ============================================================================
# Samplers
# ============================================================================

def make_acid_fun_from_Ot(Ot_grid: np.ndarray, y: np.ndarray, H: float, cap_h: float, L: float) -> Callable[[np.ndarray], np.ndarray]:
    y_int = H - cap_h
    row = np.clip(np.searchsorted(y, y_int) - 1, 1, len(y) - 2)
    line = Ot_grid[row, :]
    acid = 1.0 - normalize01(line)
    x_axis = np.linspace(0, L, line.size)
    def acid_fun(xs: np.ndarray) -> np.ndarray:
        return np.interp(xs, x_axis, acid)
    return acid_fun


def make_oc_sampler(Oc_line: np.ndarray, L: float) -> Callable[[np.ndarray], np.ndarray]:
    oc_norm = normalize01(Oc_line)
    x_axis = np.linspace(0, L, oc_norm.size)
    def sampler(xs: np.ndarray) -> np.ndarray:
        return np.interp(xs, x_axis, oc_norm)
    return sampler


# ============================================================================
# Main (precompute + playback)
# ============================================================================

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rbc_count = int(args.rbc * 10)
    myo_count = int(args.myoglobins * 10)

    # Load model
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

    cap_h = max(1e-3, float(args.cap_frac) * H)
    y_interface = H - cap_h

    # Figure & axes
    plt.rcParams["figure.figsize"] = (11.0, 7.0)
    fig, ax = plt.subplots()
    ax.set_xlim(0, L); ax.set_ylim(0, H)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

    tex = make_textures(args.nx, args.ny, L, H, cap_h, rng)
    ax.imshow(tex, extent=[0, L, 0, H], origin="lower", interpolation="bilinear", zorder=0)
    ax.axhline(y_interface, color=(0.15, 0.05, 0.18), linewidth=1.0, zorder=2)

    # Flow speeds
    u_meta = 10.0 * float(tp.get("u", 1.0))
    u_visual = max(0.1, u_meta * 0.1)

    # Entities
    rbc = RBCStream(n=rbc_count, L=L, H=H, cap_h=cap_h, u=u_meta, sim_dt=args.sim_dt,
                    seed=args.seed, vy_factor=args.vy_factor, vy_tau=args.vy_tau)
    rbc_art = RBCRenderer(ax, rbc, draw_halos=not args.no_halos, z=4, alpha=0.96,
                          flash=args.flash, freq=args.flash_freq, depth=args.flash_depth,
                          bias=args.flash_bias, seed=args.seed + 11)

    def u_fun(xs: np.ndarray) -> np.ndarray:
        return np.full_like(xs, u_visual, dtype=float)

    o2 = GasSpots(args.o2_spots, L, H, args.fps, color="lime",  size=7, seed=args.seed + 1,
                  flash=args.flash, freq=args.flash_freq, depth=args.flash_depth, bias=args.flash_bias)
    co = GasSpots(args.co_spots, L, H, args.fps, color="black", size=6, seed=args.seed + 2,
                  flash=args.flash, freq=args.flash_freq, depth=args.flash_depth, bias=args.flash_bias)
    o2_coll = o2.add_to_axes(ax, z=3, alpha=0.75)
    co_coll = co.add_to_axes(ax, z=3, alpha=0.70)

    myo = MyoglobinSites(myo_count, L, H, cap_h, seed=args.seed + 3, draw_halos=not args.no_halos,
                         flash=args.flash, freq=args.flash_freq, depth=args.flash_depth, bias=args.flash_bias)
    myo_patches = myo.add_to_axes(ax, z=4, alpha=0.92)

    fe = FieldsEvaluator(model, L, H, args.nx, args.ny)
    x_grid = fe.x; y_grid = fe.y

    # Timing
    draw_dt = args.sim_rate / args.fps

    # --- Axes-level text (blit-safe) ---
    time_text = ax.text(0.02, 0.08, "", transform=ax.transAxes,
                        va="top", ha="left", fontsize=10, color="k", animated=True)

    # -------------------- PRECOMPUTE --------------------
    times = [ (k+1)*draw_dt for k in range(args.frames) ]
    pre_Oc: List[np.ndarray] = []
    pre_Ot: List[np.ndarray] = []
    pre_RBC: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pre_O2:  List[np.ndarray] = []
    pre_CO:  List[np.ndarray] = []

    # Warm up model once
    _ = fe.predict(times[0])

    for k, t in enumerate(times, start=1):
        # Fields
        x, y, Oc_line, Ot_grid = fe.predict(t)
        pre_Oc.append(Oc_line.copy())
        pre_Ot.append(Ot_grid.copy())

        # Advance RBC to this frame
        acid_fun = make_acid_fun_from_Ot(Ot_grid, y, H, cap_h, L)
        oc_smpl  = make_oc_sampler(Oc_line, L)
        steps = max(1, int(math.ceil(draw_dt / rbc.dt)))
        for _ in range(steps):
            rbc.step(acid_fun, oc_smpl, u_fun=u_fun)
        pre_RBC.append((rbc.x.copy(), rbc.y.copy(), rbc.oxy.copy()))

        # Gas steps and snapshots
        o2.step(); co.step()
        pre_O2.append(o2.xy.copy())
        pre_CO.append(co.xy.copy())

        # Progress print every 10 steps
        if (k % 10) == 0:
            print(k, flush=True)

    # -------------------- INIT & PLAYBACK --------------------
    def init():
        # return all animated artists for blitting
        time_text.set_text("")
        artists: List = []
        artists.extend(rbc_art.outer); artists.extend(rbc_art.inner)
        if not args.no_halos:
            artists.extend(rbc_art.halos); artists.extend(myo.halos)
        artists.append(o2_coll); artists.append(co_coll)
        artists.extend(myo_patches); artists.append(time_text)
        return artists

    def update(i: int):
        # Time for this frame
        t = times[i]

        # Use cached fields
        Oc_line = pre_Oc[i]
        Ot_grid = pre_Ot[i]

        # Update myoglobin colors from cached Ot
        myo.update(Ot_grid, x_grid, y_grid, t=t)

        # Set precomputed RBC state and render
        x_r, y_r, oxy_r = pre_RBC[i]
        rbc.x[:] = x_r; rbc.y[:] = y_r; rbc.oxy[:] = oxy_r
        rbc_art.update(t)

        # Set cached gas positions and refresh collections
        o2.xy[:] = pre_O2[i]
        co.xy[:] = pre_CO[i]
        o2.update_artist(t); co.update_artist(t)

        # Title text: update EVERY frame (fixed width to reduce bbox churn)
        time_text.set_text(f"t={t:6.2f} s")

        artists: List = []
        artists.extend(rbc_art.outer)
        artists.extend(rbc_art.inner)
        if not args.no_halos:
            artists.extend(rbc_art.halos)
            artists.extend(myo.halos)
        artists.append(o2_coll); artists.append(co_coll)
        artists.extend(myo_patches)
        artists.append(time_text)
        return artists

    ani = FuncAnimation(fig, update, init_func=init,
                        frames=args.frames, interval=1000/args.fps,
                        blit=args.blit, repeat=False)

    # Re-cache blit background when the figure is redrawn or resized
    def _refresh_bg(_evt=None):
        if not args.blit:
            return
        try:
            ani._init_draw()
        except Exception:
            pass

    fig.canvas.mpl_connect('resize_event', _refresh_bg)
    fig.canvas.mpl_connect('draw_event',   _refresh_bg)

    if args.save:
        if not writers.is_available("ffmpeg"):
            print("[WARN] ffmpeg not found on PATH. Install ffmpeg to enable MP4 saving.")
            print("       Showing animation interactively instead."); plt.show(); return
        print(f"[INFO] Saving MP4 to: {args.save}")
        writer = FFMpegWriter(fps=args.fps, codec=args.codec, bitrate=args.bitrate, 
                              extra_args=["-crf", "18","-preset", "slow","-pix_fmt","yuv420p"])
        ani.save(args.saveFile, writer=writer, dpi=args.dpi); print("[INFO] Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
