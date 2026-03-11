#!/usr/bin/env python3
"""
extract_pressure.py
===================
Extract per-facet contact pressure from a FEBio .xplt file.

Outputs
-------
  <output_dir>/<stem>_<surface>_pressure.csv   — full time-series table
  <output_dir>/<stem>_<surface>_contour.png    — grid of contour snapshots
  <output_dir>/<stem>_<surface>_animation.gif  — animated contour (--animate)

CSV columns
-----------
  facet_id, surface_name, surface_id, speed_mm_s,
  facet_area_mm2, time_step, time_s, contact_pressure

Usage
-----
  # List surfaces / variables in the file first:
  .venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt --list

  # Full extraction (defaults: speed=5 mm/s, first surface found):
  .venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt

  # Specify surface, speed, output dir, and also produce an animation:
  .venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \\
      --surface "SlidingElastic1Primary" \\
      --speed 5.0 \\
      --output-dir results/ \\
      --animate
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive — safe for headless / SSH environments
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# ── make sure the project root is importable regardless of cwd ───────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from digital_twin_ui.extraction.xplt_parser import XpltParser  # noqa: E402
from digital_twin_ui.extraction.facet_tracker import FacetTracker  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroids(face_connectivity: np.ndarray, node_coords: np.ndarray) -> np.ndarray:
    """Return centroid of every face.  Shape: (n_faces, 3)."""
    # face_connectivity: (n_faces, nodes_per_face)  — 0-based node indices
    return node_coords[face_connectivity].mean(axis=1)


def project_cylindrical(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Project 3D face centroids onto a 2D (axial, angle) grid using PCA.

    Works for any roughly cylindrical/tubular surface regardless of orientation.

    Returns
    -------
    axial   : float array (n_faces,) — position along the main axis (mm)
    angle   : float array (n_faces,) — circumferential angle (degrees, -180..180)
    xlabel  : axis label string
    ylabel  : axis label string
    """
    center = centroids.mean(axis=0)
    centered = centroids - center

    # PCA: eigenvector with the largest eigenvalue = main (axial) direction
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]          # descending
    main_axis = eigenvectors[:, order[0]]
    perp1     = eigenvectors[:, order[1]]
    perp2     = eigenvectors[:, order[2]]

    axial = centered @ main_axis                   # mm along tube axis
    r1    = centered @ perp1
    r2    = centered @ perp2
    angle = np.degrees(np.arctan2(r2, r1))         # -180 … +180 °

    return axial, angle, "Axial position (mm)", "Circumferential angle (°)"


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(series_list, output_path: Path) -> None:
    """Flatten all FacetTimeSeries objects to rows and write CSV."""
    rows = []
    for ts in series_list:
        rows.extend(ts.as_rows())

    if not rows:
        print("  Warning: no data rows to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV saved  →  {output_path}  ({len(rows):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_triangulation(u: np.ndarray, v: np.ndarray) -> mtri.Triangulation:
    """Build a Delaunay triangulation from projected centroid coordinates."""
    return mtri.Triangulation(u, v)


def plot_static(
    u: np.ndarray,
    v: np.ndarray,
    times: np.ndarray,
    pressures_matrix: np.ndarray,   # shape (n_states, n_facets)
    xlabel: str,
    ylabel: str,
    surface_name: str,
    output_path: Path,
    n_panels: int = 12,
) -> None:
    """
    Save a grid of contact-pressure contour snapshots at evenly-spaced timesteps.

    Colours are normalised to the global min/max so panels are comparable.
    """
    n_states = len(times)
    n_panels = min(n_panels, n_states)
    state_indices = np.linspace(0, n_states - 1, n_panels, dtype=int)

    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    vmin = float(pressures_matrix.min())
    vmax = float(pressures_matrix.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9

    triang = _make_triangulation(u, v)
    last_tcf = None

    for panel_idx, state_idx in enumerate(state_indices):
        ax = axes_flat[panel_idx]
        p = pressures_matrix[state_idx]
        last_tcf = ax.tricontourf(triang, p, levels=15, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {times[state_idx]:.2f} s", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide any unused subplot slots
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    if last_tcf is not None:
        fig.colorbar(
            last_tcf,
            ax=axes_flat[:n_panels].tolist(),
            label="Contact pressure (MPa)",
            shrink=0.6,
        )

    fig.suptitle(f"Contact pressure — {surface_name}", fontsize=13, y=1.01)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Static plot saved  →  {output_path}")


def plot_animation(
    u: np.ndarray,
    v: np.ndarray,
    times: np.ndarray,
    pressures_matrix: np.ndarray,
    xlabel: str,
    ylabel: str,
    surface_name: str,
    output_path: Path,
    fps: int = 5,
) -> None:
    """Save an animated GIF cycling through every timestep (requires Pillow)."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("  Warning: animation skipped — PillowWriter not available. Install Pillow.")
        return

    vmin = float(pressures_matrix.min())
    vmax = float(pressures_matrix.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9

    triang = _make_triangulation(u, v)

    fig, ax = plt.subplots(figsize=(8, 5))
    tcf = ax.tricontourf(triang, pressures_matrix[0], levels=15, cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, label="Contact pressure (MPa)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_obj = ax.set_title(f"{surface_name}  —  t = {times[0]:.2f} s")

    def _update(frame: int):
        ax.cla()
        ax.tricontourf(triang, pressures_matrix[frame], levels=15, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"{surface_name}  —  t = {times[frame]:.2f} s")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    anim = FuncAnimation(fig, _update, frames=len(times), interval=1000 // fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Animation saved  →  {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inspection helper
# ─────────────────────────────────────────────────────────────────────────────

def list_surfaces(xplt_path: Path) -> None:
    """Print a summary of surfaces and variables found in the file, then exit."""
    data = XpltParser().parse(xplt_path)
    t = data.times
    print(f"\nFile     : {xplt_path}")
    print(f"Version  : {data.version}  |  Software : {data.software}")
    print(f"Nodes    : {data.n_nodes:,}")
    print(f"States   : {len(data.states)}  (t = {t[0]:.3f} → {t[-1]:.3f} s)")
    print("\nSurfaces:")
    for s in data.surfaces:
        print(f"  id={s.surface_id}  faces={s.n_faces:6,}  nodes/face={s.nodes_per_face}  name='{s.name}'")
    print(f"\nSurface variables : {[v.name for v in data.surface_vars]}")
    print(f"Nodal   variables : {[v.name for v in data.nodal_vars]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-facet contact pressure from a FEBio .xplt file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("xplt", type=Path, help="Path to the .xplt result file")
    ap.add_argument(
        "--speed", type=float, default=5.0,
        help="Insertion speed in mm/s (stored as metadata in CSV, does not affect extraction)",
    )
    ap.add_argument(
        "--surface", type=str, default=None,
        help="Surface name to extract (use --list to see available names). "
             "Defaults to the first surface in the file.",
    )
    ap.add_argument(
        "--variable", type=str, default="contact pressure",
        help="Surface variable name to extract",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for output files",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List surfaces and variables, then exit (no extraction)",
    )
    ap.add_argument(
        "--animate", action="store_true",
        help="Also save an animated GIF (requires Pillow)",
    )
    ap.add_argument(
        "--no-plot", action="store_true",
        help="Skip all plotting — CSV only",
    )
    args = ap.parse_args()

    if not args.xplt.exists():
        print(f"Error: file not found: {args.xplt}")
        sys.exit(1)

    # ── inspection mode ───────────────────────────────────────────────────────
    if args.list:
        list_surfaces(args.xplt)
        return

    # ── parse once to discover surfaces / validate surface name ───────────────
    print(f"\nParsing  {args.xplt} ...")
    xplt_data = XpltParser().parse(args.xplt)

    if not xplt_data.surfaces:
        print("Error: no surfaces found in the xplt file.")
        sys.exit(1)

    surface_name = args.surface
    if surface_name is None:
        surface_name = xplt_data.surfaces[0].name
        print(f"  No --surface given; defaulting to '{surface_name}'")

    if xplt_data.surface_by_name(surface_name) is None:
        available = [s.name for s in xplt_data.surfaces]
        print(f"Error: surface '{surface_name}' not found.\nAvailable: {available}")
        sys.exit(1)

    # ── extract per-facet time series ─────────────────────────────────────────
    print(f"  Extracting '{args.variable}' from '{surface_name}' ...")
    tracker = FacetTracker(variable_name=args.variable)
    series = tracker.extract(
        xplt_path=args.xplt,
        speed_mm_s=args.speed,
        surface_name=surface_name,
        facet_ids=None,          # all facets
    )

    if not series:
        print("Error: extraction returned no data.")
        sys.exit(1)

    n_facets = len(series)
    n_states = len(series[0].times)
    print(f"  Extracted: {n_facets:,} facets × {n_states} timesteps")

    # ── save CSV ──────────────────────────────────────────────────────────────
    stem = args.xplt.stem
    csv_path = args.output_dir / f"{stem}_{surface_name}_pressure.csv"
    save_csv(series, csv_path)

    if args.no_plot:
        print("\nDone (plots skipped).")
        return

    # ── build geometry for plots ──────────────────────────────────────────────
    surf = xplt_data.surface_by_name(surface_name)
    if surf is None or surf.face_connectivity is None or xplt_data.node_coords is None:
        print("  Warning: mesh geometry unavailable — skipping plots.")
        return

    centroids = compute_centroids(surf.face_connectivity, xplt_data.node_coords)
    u, v, xlabel, ylabel = project_cylindrical(centroids)

    # pressures_matrix[state_idx, facet_idx]
    times = series[0].times
    pressures_matrix = np.stack([ts.pressures for ts in series], axis=1)

    # ── static contour grid ───────────────────────────────────────────────────
    png_path = args.output_dir / f"{stem}_{surface_name}_contour.png"
    plot_static(u, v, times, pressures_matrix, xlabel, ylabel, surface_name, png_path)

    # ── animation (optional) ──────────────────────────────────────────────────
    if args.animate:
        gif_path = args.output_dir / f"{stem}_{surface_name}_animation.gif"
        plot_animation(u, v, times, pressures_matrix, xlabel, ylabel, surface_name, gif_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
