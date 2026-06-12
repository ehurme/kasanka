"""
Summary figures for Kasanka bat population estimates across all years.

Creates two figures:
  1. Time-series — median + 90% CI per date, each year a different colour,
     x-axis aligned by day-of-season (days since Oct 1)
  2. Year comparison — violin plots of the full replicate distribution,
     grouped by year, with seasonal-peak annotation

Usage:
    python make_summary_figures.py                    # show interactively
    python make_summary_figures.py --save ./figures   # save PNG + PDF
    python make_summary_figures.py --year 2020 2021   # subset of years
"""

import argparse
import os
import glob
import sys
from datetime import datetime, date

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COUNTS_ROOT = (
    r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme"
    r"\Eidolon_helvum\kasanka-bats\counts"
)

# Which subfolder to use per year (most complete, clean set)
YEAR_FOLDER = {
    "2019": "2019",      # 6 dates: late-Nov through mid-Dec
    "2020": "2020",      # 8 dates: Oct through Dec
    "2021": "2021",      # 9 dates: Oct through Dec (Dec incomplete)
    "2022": "2022",      # 7 dates: Nov through Dec (add once estimates run)
}

# Dates with no/negligible data — skipped entirely
EXCLUDE_DATES = {"30-Oct-2020"}

# 2022 dates with confirmed field issues (source: 22 Deployment.xlsx).
# Still included in figures but drawn with reduced emphasis (open markers, dashed CI).
# Validated rounds (Successful Full round? = Y): 20221101, 20221116, 20221213
UNCERTAIN_DATES = {
    "20221107",   # heavy rain, tripods blew over
    "20221124",   # multiple failures / battery dead
    "20221201",   # incomplete recordings
    "20221219",   # recording failures
}

# ---------------------------------------------------------------------------
# Colours per year  (colourblind-safe)
# ---------------------------------------------------------------------------

YEAR_COLOURS = {
    "2019": "#0077BB",   # blue
    "2020": "#EE7733",   # orange
    "2021": "#009988",   # teal
    "2022": "#CC3311",   # red
}

YEAR_LIGHT = {
    "2019": "#99CCEE",
    "2020": "#FFCC99",
    "2021": "#99DDD5",
    "2022": "#FFBBAA",
}

# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

MONTH_ABBR = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}

def parse_date(date_str):
    """Parse any of the three date formats used in the count files."""
    s = date_str.strip()
    # YYYYMMDD
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        pass
    # DD-Mon-YYYY
    try:
        return datetime.strptime(s, "%d-%b-%Y").date()
    except ValueError:
        pass
    # DDMon  (e.g. 16Nov) — assume 2020
    import re
    m = re.match(r"(\d+)([A-Za-z]+)", s)
    if m:
        mon = MONTH_ABBR.get(m.group(2).lower())
        if mon:
            return date(2020, mon, int(m.group(1)))
    raise ValueError(f"Cannot parse date: {date_str!r}")


def day_of_season(d):
    """Days since Oct 1 of the same year (so Oct 1 = 0, Nov 1 = 31 ...)."""
    season_start = date(d.year, 10, 1)
    delta = (d - season_start).days
    # If the date is before Oct 1 (shouldn't happen but just in case)
    return delta


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_year(year_str, counts_root, min_nonzero=10):
    """
    Load the all-camera count file for a year.
    Returns {date_obj: np.array_of_estimates} with empty/tiny dates removed.
    """
    folder = os.path.join(counts_root, YEAR_FOLDER[year_str])
    pattern = os.path.join(folder, "num_cameras-all-*-replicates-1000.npy")
    files = glob.glob(pattern)
    if not files:
        # Fall back to any replicates count
        pattern = os.path.join(folder, "num_cameras-all-*.npy")
        files = glob.glob(pattern)
    if not files:
        print(f"  Warning: no all-camera count file found for {year_str}")
        return {}

    # Pick the highest-replicate file
    files.sort(key=lambda f: int(
        f.split("replicates-")[-1].replace(".npy", "")) if "replicates-" in f else 0,
        reverse=True)
    data_raw = np.load(files[0], allow_pickle=True).item()

    result = {}
    for date_str, vals in data_raw.items():
        if date_str in EXCLUDE_DATES:
            continue
        nonzero = [v for v in vals if v > 0]
        if len(nonzero) < min_nonzero:
            continue
        try:
            d = parse_date(date_str)
        except ValueError:
            continue
        result[d] = np.array(nonzero)

    return result


def load_all_years(counts_root, year_filter=None):
    """Return {year_str: {date_obj: estimates_array}}."""
    years = year_filter or list(YEAR_FOLDER.keys())
    all_data = {}
    for yr in years:
        if yr not in YEAR_FOLDER:
            print(f"  Warning: no folder configured for year {yr}")
            continue
        d = load_year(yr, counts_root)
        if d:
            all_data[yr] = d
            print(f"  {yr}: {len(d)} dates loaded")
    return all_data


# ---------------------------------------------------------------------------
# Figure 1 — Time series aligned by day-of-season
# ---------------------------------------------------------------------------

def fig_time_series(all_data, save_dir=None):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    legend_handles = []

    for yr in sorted(all_data.keys()):
        dates_dict = all_data[yr]
        colour     = YEAR_COLOURS[yr]
        light      = YEAR_LIGHT[yr]

        sorted_dates = sorted(dates_dict.keys())

        # Split into validated vs uncertain (uncertain = flagged field issues)
        certain  = [d for d in sorted_dates
                    if d.strftime("%Y%m%d") not in UNCERTAIN_DATES]
        uncertain = [d for d in sorted_dates
                     if d.strftime("%Y%m%d") in UNCERTAIN_DATES]

        # --- Validated dates: solid fill + solid line + filled markers ---
        if certain:
            c_days = [day_of_season(d) for d in certain]
            c_meds = [np.median(dates_dict[d]) / 1e6 for d in certain]
            c_lo   = [np.percentile(dates_dict[d],  5) / 1e6 for d in certain]
            c_hi   = [np.percentile(dates_dict[d], 95) / 1e6 for d in certain]
            ax.fill_between(c_days, c_lo, c_hi, color=light, alpha=0.7, linewidth=0)
            ax.plot(c_days, c_meds, "o-", color=colour, linewidth=2,
                    markersize=6, markerfacecolor=colour,
                    markeredgewidth=1.5, zorder=3)

        # --- Uncertain dates: faint fill + dashed line + open markers ---
        if uncertain:
            u_days = [day_of_season(d) for d in uncertain]
            u_meds = [np.median(dates_dict[d]) / 1e6 for d in uncertain]
            u_lo   = [np.percentile(dates_dict[d],  5) / 1e6 for d in uncertain]
            u_hi   = [np.percentile(dates_dict[d], 95) / 1e6 for d in uncertain]
            ax.fill_between(u_days, u_lo, u_hi, color=light, alpha=0.25, linewidth=0)
            ax.plot(u_days, u_meds, "o--", color=colour, linewidth=1.2,
                    markersize=6, markerfacecolor="white",
                    markeredgewidth=1.5, alpha=0.6, zorder=3)

        legend_handles.append(mpatches.Patch(color=colour, label=yr))

    # X-axis ticks: month labels
    month_ticks  = [0, 31, 61, 92]          # Oct 1, Nov 1, Dec 1, Jan 1
    month_labels = ["1 Oct", "1 Nov", "1 Dec", "1 Jan"]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_xlim(-5, 100)

    ax.set_xlabel("Date (day of season)", fontsize=12)
    ax.set_ylabel("Population estimate (millions)", fontsize=12)
    ax.set_title("Kasanka bat colony — population estimates across all years",
                 fontsize=13, fontweight="bold", pad=10)

    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    import matplotlib.lines as mlines
    legend_handles.append(
        mlines.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
                      marker="o", markerfacecolor="white", markersize=6,
                      label="Uncertain deployment"))
    ax.legend(handles=legend_handles, title="Year",
              frameon=True, fontsize=10, title_fontsize=10)

    # Shade the canonical study window (Nov 16–20)
    ax.axvspan(46, 50, color="#DDDDDD", alpha=0.5, zorder=0)
    ax.text(48, ax.get_ylim()[1] * 0.97, "Published\nstudy",
            ha="center", va="top", fontsize=8, color="#888888")

    plt.tight_layout()
    _save(fig, "kasanka_population_time_series", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Violin / distribution comparison by year
# ---------------------------------------------------------------------------

def fig_year_comparison(all_data, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # --- Left panel: violin per date grouped by year ---
    ax = axes[0]
    ax.set_facecolor("#F8F8F8")

    x_pos   = []
    all_vals = []
    colours  = []
    labels   = []
    x_cursor = 0
    year_mid = {}      # for year-label placement
    year_ranges = {}   # for bracket drawing

    gap_between_years = 1.5

    for yr in sorted(all_data.keys()):
        dates_dict = all_data[yr]
        sorted_dates = sorted(dates_dict.keys())
        year_start = x_cursor

        for d in sorted_dates:
            is_uncertain = d.strftime("%Y%m%d") in UNCERTAIN_DATES
            vals = dates_dict[d] / 1e6
            # Clip to 1–99th percentile so extreme outliers don't blow
            # up the bounding box when bbox_inches is computed
            p01, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
            vals_plot = vals[(vals >= p01) & (vals <= p99)]

            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals_plot, bw_method=0.3)
            v_range = np.linspace(vals_plot.min(), vals_plot.max(), 200)
            density = kde(v_range)
            half_w  = density / density.max() * 0.4

            alpha = 0.25 if is_uncertain else 0.55
            ax.fill_betweenx(v_range,
                             x_cursor - half_w,
                             x_cursor + half_w,
                             color=YEAR_COLOURS[yr], alpha=alpha,
                             hatch="///" if is_uncertain else None,
                             edgecolor=YEAR_COLOURS[yr] if is_uncertain else None)
            med = np.median(vals)   # median from full distribution
            ax.hlines(med, x_cursor - 0.35, x_cursor + 0.35,
                      color=YEAR_COLOURS[yr],
                      linewidth=2 if not is_uncertain else 1,
                      linestyle="-" if not is_uncertain else "--",
                      zorder=4, alpha=1.0 if not is_uncertain else 0.5)

            # Date label: day + abbreviated month
            label = d.strftime("%-d %b") if sys.platform != "win32" \
                    else d.strftime("%d %b").lstrip("0")
            x_pos.append(x_cursor)
            labels.append(label)
            x_cursor += 1

        year_mid[yr]    = (year_start + x_cursor - 1) / 2
        year_ranges[yr] = (year_start, x_cursor - 1)
        x_cursor += gap_between_years

    # Fix xlim from known positions BEFORE any limit-dependent calculations
    x_lo = min(x_pos) - 0.7
    x_hi = max(x_pos) + 0.7
    ax.set_xlim(x_lo, x_hi)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Population estimate (millions)", fontsize=11)
    ax.set_title("Per-date distributions", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Colour x-tick labels by year
    tick_colour_map = {}
    for yr, (x_start, x_end) in year_ranges.items():
        for xp in x_pos:
            if x_start <= xp <= x_end:
                tick_colour_map[xp] = YEAR_COLOURS[yr]
    for tick, xp in zip(ax.get_xticklabels(), x_pos):
        tick.set_color(tick_colour_map.get(xp, "black"))

    # Shaded background stripes + year label at top of each group
    x_span = x_hi - x_lo
    for yr, (x_start, x_end) in year_ranges.items():
        ax.axvspan(x_start - 0.5, x_end + 0.5,
                   color=YEAR_COLOURS[yr], alpha=0.07, zorder=0)
        x_frac = ((x_start + x_end) / 2 - x_lo) / x_span
        ax.text(x_frac, 1.01, yr,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=YEAR_COLOURS[yr],
                transform=ax.transAxes)

    # --- Right panel: peak estimate per year ---
    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F8")

    year_list = sorted(all_data.keys())
    peaks     = []
    peak_lo   = []
    peak_hi   = []
    for yr in year_list:
        # Peak = highest median across dates in that year
        best_date = max(all_data[yr], key=lambda d: np.median(all_data[yr][d]))
        vals = all_data[yr][best_date] / 1e6
        peaks.append(np.median(vals))
        peak_lo.append(np.percentile(vals, 5))
        peak_hi.append(np.percentile(vals, 95))

    bar_cols = [YEAR_COLOURS[yr] for yr in year_list]
    bars = ax2.bar(year_list, peaks, color=bar_cols, width=0.5,
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax2.errorbar(year_list, peaks,
                 yerr=[np.array(peaks) - np.array(peak_lo),
                       np.array(peak_hi) - np.array(peaks)],
                 fmt="none", color="#333333", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, peaks):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15,
                 f"{val:.1f}M", ha="center", va="bottom", fontsize=10)

    ax2.set_ylabel("Peak season estimate (millions)", fontsize=11)
    ax2.set_title("Peak per year", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, max(peak_hi) * 1.2)

    fig.suptitle("Kasanka bat colony — population estimates 2019–2021",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.88, wspace=0.3)
    _save(fig, "kasanka_population_year_comparison", save_dir, tight=False)
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, name, save_dir, tight=True):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        kwargs = dict(dpi=200, bbox_inches="tight") if tight else dict(dpi=200)
        for ext in ("png", "pdf"):
            path = os.path.join(save_dir, f"{name}.{ext}")
            fig.savefig(path, **kwargs)
            print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Make Kasanka bat population summary figures"
    )
    parser.add_argument("--counts", default=COUNTS_ROOT,
                        help="Root folder containing year subfolders of count files")
    parser.add_argument("--year", nargs="+", default=None,
                        help="Limit to specific year(s), e.g. --year 2020 2021")
    parser.add_argument("--save", default=None, metavar="DIR",
                        help="Save figures as PNG + PDF to this folder")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display figures interactively")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    print("Loading count data ...")
    all_data = load_all_years(args.counts, args.year)

    if not all_data:
        print("No data found. Check --counts path.")
        return

    print("\nGenerating figures ...")
    fig1 = fig_time_series(all_data, save_dir=args.save)
    fig2 = fig_year_comparison(all_data, save_dir=args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
