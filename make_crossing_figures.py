"""
Diagnostic figure: raw crossing counts per camera-day across all years.

Shows the actual bat detections before any geometric extrapolation, so you
can judge whether population estimate variation between years is plausible.

Three panels:
  1. Outward crossings per camera per date  (main diagnostic)
  2. Directional balance (out fraction)     (flags reversed cameras)
  3. Total outward crossings per date       (seasonal pattern)

Usage:
    python make_crossing_figures.py
    python make_crossing_figures.py --save ./figures
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, date

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OBS_ROOT = (
    r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme"
    r"\Eidolon_helvum\kasanka-bats\observations"
)

# Dates with negligible data
PRE_MIGRATION = {"30-Oct-2020"}
# 2022 dates with confirmed field issues — still shown, but with hollow markers
# Validated rounds: 20221101, 20221116, 20221213
UNCERTAIN_DATES = {"20221107", "20221124", "20221201", "20221219"}

# Duplicate date-sets: prefer the long-format name for uniqueness
# (16Nov == 16-Nov-2020, but they were compiled separately; we keep both
#  but flag short-format as a second run)
DATAVERSE_SUBDIR = "dataverse_files"

YEAR_COLOURS = {
    2019: "#0077BB",
    2020: "#EE7733",
    2021: "#009988",
    2022: "#CC3311",
}
YEAR_MARKERS = {2019: "o", 2020: "s", 2021: "^", 2022: "D"}

MONTH_ABBR = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
              "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(s):
    for fmt in ("%Y%m%d", "%d-%b-%Y"):
        try: return datetime.strptime(s.strip(), fmt).date()
        except ValueError: pass
    m = re.match(r"(\d+)([A-Za-z]+)", s.strip())
    if m:
        mon = MONTH_ABBR.get(m.group(2).lower())
        if mon: return date(2020, mon, int(m.group(1)))
    return None

def day_of_season(d):
    return (d - date(d.year, 10, 1)).days

# ---------------------------------------------------------------------------
# Load observations — deduplicated by (date_str, camera)
# ---------------------------------------------------------------------------

def load_observations(obs_root):
    """
    Return list of dicts, one per unique (date_str, camera) pair.
    Prefers long-format date names over short-format duplicates.
    Skips files found inside the dataverse_files subfolder.
    """
    seen = {}   # key: (date_str, cam_norm) -> dict

    all_files = sorted(
        glob.glob(os.path.join(obs_root, "*-observation-*.npy")) +
        glob.glob(os.path.join(obs_root, "*", "*-observation-*.npy"))
    )

    for fpath in all_files:
        # Skip dataverse_files — these duplicate the short-format dates
        parts = fpath.replace("\\", "/").split("/")
        if DATAVERSE_SUBDIR in parts:
            continue

        fname = os.path.basename(fpath)
        m = re.match(r"^(.+?)-observation-(.+)\.npy$", fname)
        if not m:
            continue
        date_str, camera = m.group(1), m.group(2)
        d = parse_date(date_str)
        if d is None:
            continue

        cam_norm = camera.lower().replace(" ", "")
        key = (date_str, cam_norm)

        # Prefer long-format date names (16-Nov-2020 over 16Nov)
        if key in seen:
            existing = seen[key]
            if len(date_str) > len(existing["date_str"]):
                pass   # new one is longer — overwrite below
            else:
                continue   # keep existing

        try:
            obs = np.load(fpath, allow_pickle=True).item()
        except Exception:
            continue

        directions = obs.get("direction", np.array([]))
        n_out  = int(np.sum(directions ==  1))
        n_in   = int(np.sum(directions == -1))
        n_total = len(directions)

        seen[key] = {
            "date_str":  date_str,
            "date":      d,
            "year":      d.year,
            "day":       day_of_season(d),
            "camera":    camera,
            "cam_norm":  cam_norm,
            "n_total":   n_total,
            "n_out":     n_out,
            "n_in":      n_in,
            "out_frac":  n_out / n_total if n_total > 0 else np.nan,
            "pre_migration": date_str in PRE_MIGRATION,
            "failed_deployment": date_str in UNCERTAIN_DATES,
        }

    return list(seen.values())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def jitter(x, scale=0.4):
    return x + np.random.uniform(-scale, scale, size=len(x))


def make_figure(rows, save_dir=None):
    years = sorted(set(r["year"] for r in rows))

    # Sort rows by date within year
    rows = sorted(rows, key=lambda r: (r["year"], r["day"]))

    # Unique dates per year → x positions
    date_keys = sorted(set((r["year"], r["date_str"], r["day"]) for r in rows),
                       key=lambda t: (t[0], t[2]))

    # Assign a global x position per (year, date)
    gap = 3   # gap between years in x-units
    x_map = {}     # (year, date_str) -> x position
    date_labels = {}  # x -> label
    x = 0
    prev_yr = None
    for yr, ds, day in date_keys:
        if prev_yr is not None and yr != prev_yr:
            x += gap
        if ds in PRE_MIGRATION:
            x += 0
        x_map[(yr, ds)] = x
        date_labels[x] = (ds, yr, day)
        x += 1
        prev_yr = yr

    # Year mid-points for labels
    year_xs = {}
    for (yr, ds), xpos in x_map.items():
        year_xs.setdefault(yr, []).append(xpos)
    year_mid = {yr: np.mean(xs) for yr, xs in year_xs.items()}
    year_span = {yr: (min(xs), max(xs)) for yr, xs in year_xs.items()}

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1.5]})
    fig.patch.set_facecolor("white")
    np.random.seed(42)

    # --- Panel 1: Outward crossings per camera (log scale) -----------------
    ax1 = axes[0]
    ax1.set_facecolor("#F8F8F8")
    ax1.set_yscale("log")

    for yr in years:
        yr_rows = [r for r in rows if r["year"] == yr and not r["pre_migration"]]
        xs_raw    = np.array([x_map[(r["year"], r["date_str"])] for r in yr_rows])
        ys        = np.array([max(r["n_out"], 1) for r in yr_rows])
        out_fracs = np.array([r["out_frac"] for r in yr_rows])
        uncertain = np.array([r["failed_deployment"] for r in yr_rows])

        # Three groups: certain/normal, certain/reversed, uncertain
        normal    = (out_fracs >= 0.30) & ~uncertain
        reversed_ = (out_fracs <  0.30) & ~np.isnan(out_fracs) & ~uncertain
        unc_mask  = uncertain

        # Certain validated points — solid filled
        if normal.any():
            ax1.scatter(jitter(xs_raw[normal]), ys[normal],
                        color=YEAR_COLOURS[yr], marker=YEAR_MARKERS[yr],
                        s=40, alpha=0.75, linewidths=0)
        # Reversed cameras — red X
        if reversed_.any():
            ax1.scatter(jitter(xs_raw[reversed_]), ys[reversed_],
                        color="red", marker="x", s=60, linewidths=1.5, zorder=5)
        # Uncertain deployment — hollow, faded
        if unc_mask.any():
            ax1.scatter(jitter(xs_raw[unc_mask]), ys[unc_mask],
                        color=YEAR_COLOURS[yr], marker=YEAR_MARKERS[yr],
                        s=35, alpha=0.35, linewidths=1,
                        facecolors="none", edgecolors=YEAR_COLOURS[yr])

    # Median per date
    for (yr, ds), xpos in x_map.items():
        vals = [r["n_out"] for r in rows
                if r["year"] == yr and r["date_str"] == ds and r["n_out"] > 0]
        if vals:
            ax1.hlines(np.median(vals), xpos - 0.4, xpos + 0.4,
                       color=YEAR_COLOURS.get(yr, "gray"), linewidth=2, zorder=4)

    ax1.set_ylabel("Outward crossings per camera", fontsize=11)
    ax1.set_title("Raw crossing counts per camera-day — all years",
                  fontsize=13, fontweight="bold", pad=8)
    ax1.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    # Legend: years
    legend_elements = [
        mlines.Line2D([0], [0], marker=YEAR_MARKERS[yr], color="w",
                      markerfacecolor=YEAR_COLOURS[yr], markersize=8, label=str(yr))
        for yr in years
    ]
    legend_elements.append(
        mlines.Line2D([0], [0], marker="x", color="red", markersize=8,
                      linewidth=1.5, label="Reversed camera (<30% outward)"))
    legend_elements.append(
        mlines.Line2D([0], [0], marker="o", color="gray", markersize=8,
                      markerfacecolor="none", markeredgewidth=1,
                      linewidth=0, alpha=0.5, label="Uncertain deployment"))
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9,
               frameon=True, framealpha=0.9)

    # Year background shading
    for yr, (x_lo, x_hi) in year_span.items():
        ax1.axvspan(x_lo - 0.5, x_hi + 0.5,
                    color=YEAR_COLOURS[yr], alpha=0.06, zorder=0)

    # --- Panel 2: Directional balance (out fraction) -----------------------
    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F8")

    for yr in years:
        yr_rows = [r for r in rows if r["year"] == yr and not r["pre_migration"]]
        xs_raw    = np.array([x_map[(r["year"], r["date_str"])] for r in yr_rows])
        fracs     = np.array([r["out_frac"] if not np.isnan(r["out_frac"]) else 0.5
                              for r in yr_rows])
        uncertain = np.array([r["failed_deployment"] for r in yr_rows])
        # Certain points solid, uncertain hollow/faded
        ax2.scatter(jitter(xs_raw[~uncertain], 0.3), fracs[~uncertain],
                    color=YEAR_COLOURS[yr], marker=YEAR_MARKERS[yr],
                    s=25, alpha=0.7, linewidths=0)
        if uncertain.any():
            ax2.scatter(jitter(xs_raw[uncertain], 0.3), fracs[uncertain],
                        color=YEAR_COLOURS[yr], marker=YEAR_MARKERS[yr],
                        s=22, alpha=0.3, linewidths=0.8,
                        facecolors="none", edgecolors=YEAR_COLOURS[yr])

    ax2.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax2.axhline(0.3, color="red",     linestyle=":",  linewidth=1, alpha=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction outward", fontsize=10)
    ax2.set_yticks([0, 0.3, 0.5, 1.0])
    ax2.set_yticklabels(["0", "0.3\n(flag)", "0.5", "1.0"], fontsize=8)
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    for yr, (x_lo, x_hi) in year_span.items():
        ax2.axvspan(x_lo - 0.5, x_hi + 0.5,
                    color=YEAR_COLOURS[yr], alpha=0.06, zorder=0)

    # --- Panel 3: Total outward crossings per date -------------------------
    ax3 = axes[2]
    ax3.set_facecolor("#F8F8F8")

    for (yr, ds), xpos in x_map.items():
        total_out = sum(r["n_out"] for r in rows
                        if r["year"] == yr and r["date_str"] == ds)
        col = YEAR_COLOURS.get(yr, "gray")
        ax3.bar(xpos, total_out / 1e3, color=col, alpha=0.8, width=0.7)

    ax3.set_ylabel("Total outward\ncrossings (thousands)", fontsize=10)
    ax3.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda y, _: f"{y:.0f}k"))
    ax3.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax3.spines[["top", "right"]].set_visible(False)

    for yr, (x_lo, x_hi) in year_span.items():
        ax3.axvspan(x_lo - 0.5, x_hi + 0.5,
                    color=YEAR_COLOURS[yr], alpha=0.06, zorder=0)

    # --- X-axis labels (date + year banners) --------------------------------
    all_xpos   = sorted(date_labels.keys())
    tick_labels = []
    for xpos in all_xpos:
        ds, yr, day = date_labels[xpos]
        if ds in PRE_MIGRATION:
            tick_labels.append(f"{ds}\n(pre)")
        elif re.fullmatch(r"\d{8}", ds):
            d = datetime.strptime(ds, "%Y%m%d").date()
            tick_labels.append(d.strftime("%-d %b") if sys.platform != "win32"
                               else d.strftime("%d %b").lstrip("0"))
        elif re.match(r"\d+-[A-Za-z]+-\d{4}", ds):
            d = datetime.strptime(ds, "%d-%b-%Y").date()
            tick_labels.append(d.strftime("%-d %b") if sys.platform != "win32"
                               else d.strftime("%d %b").lstrip("0"))
        else:
            tick_labels.append(ds)

    ax3.set_xticks(all_xpos)
    ax3.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax3.set_xlim(min(all_xpos) - 0.7, max(all_xpos) + 0.7)

    # Year banners above panel 1
    for yr, mid in year_mid.items():
        x_lo, x_hi = year_span[yr]
        x_frac = (mid - (min(all_xpos) - 0.7)) / \
                 ((max(all_xpos) + 0.7) - (min(all_xpos) - 0.7))
        ax1.text(x_frac, 1.02, str(yr),
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=YEAR_COLOURS[yr],
                 transform=ax1.transAxes)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.93, hspace=0.08)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            path = os.path.join(save_dir, f"kasanka_crossings_diagnostic.{ext}")
            fig.savefig(path, dpi=180)
            print(f"  Saved: {path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic plot of raw crossing counts across all years"
    )
    parser.add_argument("--obs", default=OBS_ROOT,
                        help="Observations root folder")
    parser.add_argument("--save", default=None, metavar="DIR")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "sans-serif"})

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try: sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError: pass

    print("Loading observations ...")
    rows = load_observations(args.obs)
    print(f"  {len(rows)} unique camera-days loaded")

    # Print per-year summary
    for yr in sorted(set(r["year"] for r in rows)):
        yr_rows = [r for r in rows if r["year"] == yr]
        outs = [r["n_out"] for r in yr_rows]
        n_reversed = sum(1 for r in yr_rows
                         if r["out_frac"] < 0.3 and r["n_total"] > 100)
        print(f"  {yr}: {len(yr_rows)} camera-days, "
              f"median out={int(np.median(outs)):,}, "
              f"max={max(outs):,}, "
              f"{n_reversed} possibly reversed cameras")

    print("\nGenerating figure ...")
    fig = make_figure(rows, save_dir=args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
