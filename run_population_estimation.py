"""
Run Monte Carlo population estimates for all available observation dates.

Discovers all observation .npy files on the server, groups them by year,
and runs the same Monte Carlo procedure as population-estimation.ipynb.
Saves count files to counts/<year>/ alongside the existing ones.

Usage:
    python run_population_estimation.py --dry-run        # preview dates + cameras
    python run_population_estimation.py                  # run all years
    python run_population_estimation.py --year 2021      # single year
    python run_population_estimation.py --replicates 100 # quick test run
"""

import argparse
import glob
import os
import random
import re
import sys
import traceback

import cv2
import numpy as np
import rasterio
import utm

sys.path.insert(0, os.path.dirname(__file__))
import bat_functions as bf

# ---------------------------------------------------------------------------
# Paths — edit if your layout differs
# ---------------------------------------------------------------------------

BASE_ROOT        = r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme\Eidolon_helvum\kasanka-bats"
OBSERVATIONS_ROOT = os.path.join(BASE_ROOT, "observations")
COUNTS_ROOT      = os.path.join(BASE_ROOT, "counts")

# Local repo files (already in the kasanka/ folder)
_here = os.path.dirname(os.path.abspath(__file__))
MAP_FILE             = os.path.join(_here, "kasanka-utm.tiff")
WING_VALIDATION_FILE = os.path.join(_here, "combined_wing_validation_info.csv")

# ---------------------------------------------------------------------------
# Camera locations — consolidated from all EH population-estimation notebooks
# Keys must match exactly what is stored in obs['camera'] (= folder name)
# ---------------------------------------------------------------------------

CAMERA_LOCATIONS = {
    # Fibwe Parking variants
    "FibweParking":      [-12.5903393,  30.2525047],
    "FibweParking2":     [-12.5903393,  30.2525047],
    "Fibwe Parking":     [-12.5903393,  30.2525047],

    # Chinyangali variants
    "Chinyangali":       [-12.5851284,  30.245529],
    "Chinangali":        [-12.5851284,  30.245529],
    "Chyniangale":       [-12.5851284,  30.245529],
    "Chinangale":        [-12.5851284,  30.245529],
    "Chiniangale":       [-12.5851284,  30.245529],
    "ChinyangaliB":      [-12.5851284,  30.245529],   # assumed same position

    # BBC variants
    "BBC":               [-12.5863538,  30.2484985],
    "BBC2":              [-12.5863538,  30.2484985],  # assumed same position

    # Sunset
    "Sunset":            [-12.585784,   30.240003],

    # NotChinyangali variants
    "NotChinyangali":    [-12.5849206,  30.2436135],
    "Not Chinyangali":   [-12.5849206,  30.2436135],
    "NotChyniangale":    [-12.5849206,  30.2436135],
    "NotChiniangale":    [-12.5849206,  30.2436135],
    "NotChinangale":     [-12.5849206,  30.2436135],
    "Not Chinangali":    [-12.5849206,  30.2436135],
    "Not Chingyangali":  [-12.5849206,  30.2436135],
    "NotChinyangaliLocA":[-12.584804,   30.242267],
    "NotChinyangaliLocB":[-12.584967,   30.243829],
    "NotChinyangaliLocC":[-12.582486,   30.250048],

    # Musola Parking variants
    "MusolaParking":     [-12.58787,    30.2401],
    "MusoleParking":     [-12.58787,    30.2401],
    "Musola Parking":    [-12.58787,    30.2401],
    "Musole Parking":    [-12.58787,    30.2401],
    "NotMusolaParking":  [-12.590832,   30.238941],
    "Not Musola Parking":[-12.590832,   30.238941],

    # Musola Path variants
    "MusolaPath":        [-12.589544,   30.242488],
    "MusolePath":        [-12.589544,   30.242488],
    "MusolePath2":       [-12.589544,   30.242488],
    "MusolaPath2":       [-12.589544,   30.242488],
    "Musola Path":       [-12.589544,   30.242488],
    "Musole Path":       [-12.589544,   30.242488],
    "NotMusolaPath":     [-12.592674,   30.244078],
    "Not Musola Path":   [-12.592674,   30.244078],

    # Puku variants
    "Puku":              [-12.584838,   30.24137],
    "PukuLocB":          [-12.584838,   30.24137],    # assumed same area
    "PukuLocC":          [-12.582959,   30.241016],
    "Puku Loc C":        [-12.582959,   30.241016],

    # Fibwe Public variants
    "FibwePublic":       [-12.592537,   30.2515924],
    "Fibwe Public":      [-12.592537,   30.2515924],
    "Fibwe public":      [-12.592537,   30.2515924],
    "FibweManagement":   [-12.592537,   30.2515924],  # assumed Fibwe area

    # Musola Tower variants
    "MusolaTower":       [-12.589434,   30.244736],
    "MusoleTower":       [-12.589434,   30.244736],
    "Musola Tower":      [-12.589434,   30.244736],
    "Musole Tower":      [-12.589434,   30.244736],
    "KKCamera":          [-12.589434,   30.244736],   # KK's Camera = MusolaTower area
    "KK's Camera":       [-12.589434,   30.244736],

    # 2022 numbered camera names — GPS from 22 Deployment.xlsx (mean across valid rounds)
    # These positions differ from earlier years; cameras were repositioned each deployment.
    "1 Fibwe Parking":     [-12.591127,  30.252812],
    "2 BBC":               [-12.585629,  30.251253],
    "3 Chinyangali":       [-12.582324,  30.250244],
    "4 Not Chinyangali":   [-12.583024,  30.245623],
    "5 Puku":              [-12.583147,  30.240842],
    "6 Sunset":            [-12.585768,  30.239975],
    "7 Bupata":            [-12.590427,  30.238381],
    "8 Musola Path":       [-12.592753,  30.242034],
    "9 KK":                [-12.594063,  30.246678],
    "10 Fibwe Management": [-12.593827,  30.250030],
}

# ---------------------------------------------------------------------------
# Fixed model parameters (from EH notebooks)
# ---------------------------------------------------------------------------

FRAME_WIDTH  = 2704 - (2 * 48)   # 2608 px — sensor width minus padding
WINGSPAN     = 0.8                # metres
DARK_PARAMS  = [1.57454778e+01, 9.37398964e-01, 7.18914388e-02, -1.27575036e-04]

# 2022 dates with confirmed issues (source: 22 Deployment.xlsx).
# All 7 dates are still processed; these are flagged for context only.
# Validated (Successful Full round? = Y): 20221101, 20221116, 20221213
UNCERTAIN_DATES_2022 = {
    "20221107",   # Round 2: heavy rain, tripods blew over
    "20221124",   # Round 4: multiple failures, battery dead
    "20221201",   # Round 5: camera failures, incomplete recordings
    "20221219",   # Round 7: failed recordings
}

FOREST_BORDER = [
    [-12.585957, 30.242762],
    [-12.586763, 30.246229],
    [-12.589182, 30.245566],
    [-12.587557, 30.241598],
]

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(t):    return f"{GREEN}{t}{RESET}"
def warn(t):  return f"{YELLOW}{t}{RESET}"
def err(t):   return f"{RED}{t}{RESET}"
def bold(t):  return f"{BOLD}{t}{RESET}"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Setup — build forest mask and load wing correction once
# ---------------------------------------------------------------------------

def build_forest_mask(map_file):
    dataset = rasterio.open(map_file)
    width   = int(abs(dataset.bounds.left  - dataset.bounds.right))
    height  = int(abs(dataset.bounds.top   - dataset.bounds.bottom))
    area    = np.zeros((height, width), dtype=np.uint8)

    forest_utms = np.array([
        list(utm.from_latlon(*ll)[:2]) for ll in FOREST_BORDER
    ])
    norm = forest_utms.copy()
    norm[:, 0] -= dataset.bounds.left
    norm[:, 1] -= dataset.bounds.bottom
    mask = cv2.drawContours(area, [norm.astype(np.int32)], -1, 255, -1)
    return mask, dataset


def load_wing_correction(validation_file):
    return bf.get_wing_correction_distributions(
        validation_file, num_darkness_bins=4,
        kde_bw_scale=0.25, should_plot=False
    )


# ---------------------------------------------------------------------------
# Observation discovery
# ---------------------------------------------------------------------------

def discover_observations(obs_root):
    """
    Return {date: {camera: obs_dict}} for all observations found.
    Searches both obs_root/*.npy and obs_root/*/*.npy.
    """
    all_obs = {}
    patterns = [
        os.path.join(obs_root, "*-observation-*.npy"),
        os.path.join(obs_root, "*", "*-observation-*.npy"),
    ]
    for pat in patterns:
        for fpath in glob.glob(pat):
            fname = os.path.basename(fpath)
            m = re.match(r"^(.+?)-observation-(.+)\.npy$", fname)
            if not m:
                continue
            date, camera = m.group(1), m.group(2)
            try:
                obs = np.load(fpath, allow_pickle=True).item()
            except Exception:
                continue
            all_obs.setdefault(date, {})[camera] = obs
    return all_obs


def year_of(date_str):
    if re.fullmatch(r"\d{8}", date_str):
        return date_str[:4]
    if re.search(r"2022", date_str): return "2022"
    if re.search(r"2019", date_str): return "2019"
    if re.search(r"2020", date_str): return "2020"
    if re.search(r"2021", date_str): return "2021"
    if re.search(r"2022", date_str): return "2022"
    # Short names like 16Nov → assume 2020 (the published study)
    return "2020"


def pick_observations(camera_dict, num_cameras):
    if num_cameras == -1:
        return camera_dict
    available = list(camera_dict.keys())
    n = min(num_cameras, len(available))
    keys = random.sample(available, n)
    return {k: camera_dict[k] for k in keys}


# ---------------------------------------------------------------------------
# Single date estimation
# ---------------------------------------------------------------------------

def estimate_date(date, obs_by_camera, all_camera_utms,
                  forest_mask, forest_dataset,
                  wing_kdes, darkness_bins,
                  num_cameras, replicates,
                  vary_center, jitter, correct_wing):
    """Return list of `replicates` total-bat estimates for one date."""

    # Warn about cameras with unknown locations (skip them)
    known, unknown = {}, []
    for cam, obs in obs_by_camera.items():
        cam_key = obs.get("camera", cam)
        if cam_key in all_camera_utms:
            known[cam] = obs
        else:
            unknown.append(cam_key)
    if unknown:
        print(warn(f"    Skipping {len(unknown)} camera(s) with no location: "
                   f"{', '.join(unknown)}"))
    if not known:
        print(warn(f"    No cameras with known locations for {date}"))
        return [0.0] * replicates

    totals = []
    for _ in range(replicates):
        obs_sample = pick_observations(known, num_cameras)

        center = (bf.get_random_utm_in_mask(forest_mask, forest_dataset)
                  if vary_center else np.array([200450, 8606950]))

        cam_utms = bf.get_camera_locations(obs_sample, all_camera_utms, exclude=True)
        if not cam_utms:
            totals.append(0.0)
            continue

        cam_distances = bf.get_camera_distances(cam_utms, center)
        cam_fractions = bf.get_camera_fractions(cam_utms, center, jitter)

        day_total = 0.0
        for cam_name, obs in obs_sample.items():
            if len(obs["darkness"]) != len(obs["mean_wing"]):
                continue
            correction_scale = (
                bf.get_kde_samples(obs, wing_kdes, darkness_bins)[0]
                if correct_wing else 0
            )
            biased_wing = bf.correct_wingspan(obs["mean_wing"], correction_scale)
            biased_wing = np.maximum(biased_wing, 2)

            obs["multiplier"] = bf.combined_bat_multiplier(
                FRAME_WIDTH, WINGSPAN, biased_wing,
                cam_distances[obs["camera"]]
            )
            obs["fraction_total"] = cam_fractions[obs["camera"]]

            accum = bf.get_bat_accumulation(obs["frames"], obs, DARK_PARAMS)
            day_total += accum[-1]

        totals.append(day_total)

    return totals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(obs_root, counts_root, year_filter, replicates,
        vary_center, jitter, correct_wing, dry_run, verbose, force=False):

    print()
    print("=" * 64)
    print("  Kasanka — Population Estimation")
    print("=" * 64)
    print(f"  Observations : {obs_root}")
    print(f"  Counts out   : {counts_root}")
    print(f"  Replicates   : {replicates}")
    print(f"  vary_center={vary_center}  jitter={jitter}  "
          f"correct_wing={correct_wing}")
    if year_filter:
        print(f"  Year filter  : {year_filter}")
    if dry_run:
        print(warn("  DRY RUN — no files will be written"))
    print()

    # Load setup
    print("  Loading map and wing-correction data ...", end="", flush=True)
    forest_mask, forest_dataset = build_forest_mask(MAP_FILE)
    wing_kdes, darkness_bins    = load_wing_correction(WING_VALIDATION_FILE)
    all_camera_utms             = bf.latlong_dict_to_utm(CAMERA_LOCATIONS)
    print(" done.")

    # Discover observations
    all_obs = discover_observations(obs_root)
    print(f"  Found {len(all_obs)} dates across all years.\n")

    # Group by year
    by_year = {}
    for date, cams in all_obs.items():
        yr = year_of(date)
        by_year.setdefault(yr, {})[date] = cams

    if year_filter:
        by_year = {yr: dates for yr, dates in by_year.items()
                   if yr in year_filter}

    # Camera-count loop — same as the notebooks
    num_camera_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, -1]

    for yr in sorted(by_year.keys()):
        dates_in_year = by_year[yr]
        print(bold(f"Year {yr}  —  {len(dates_in_year)} dates"))

        out_dir = os.path.join(counts_root, yr)
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)

        # Show what cameras are available per date
        for date in sorted(dates_in_year):
            n_cams = len(dates_in_year[date])
            unknown = [c for c in dates_in_year[date]
                       if dates_in_year[date][c].get("camera", c)
                       not in all_camera_utms]
            tag = (warn(f"  {n_cams} cams, {len(unknown)} unknown")
                   if unknown else ok(f"  {n_cams} cams"))
            print(f"    {date}{tag}")
        print()

        if dry_run:
            continue

        for num_cameras in num_camera_options:
            nc_label = "all" if num_cameras == -1 else str(num_cameras)
            count_file = (
                f"num_cameras-{nc_label}"
                f"-jitter-{jitter}"
                f"-vary_center-{vary_center}"
                f"-correct_wing-{correct_wing}"
                f"-replicates-{replicates}.npy"
            )
            out_path = os.path.join(out_dir, count_file)

            if os.path.exists(out_path) and not force:
                print(f"  {nc_label} cameras: {ok('already exists')} — skipping "
                      f"(use --force to overwrite)")
                continue

            print(f"  {nc_label} cameras: running {replicates} replicates "
                  f"× {len(dates_in_year)} dates ...", flush=True)

            day_totals = {d: [] for d in dates_in_year}

            for date in sorted(dates_in_year):
                obs_by_cam = dates_in_year[date]
                print(f"    {date} ({len(obs_by_cam)} cameras) ...",
                      end="", flush=True)
                try:
                    totals = estimate_date(
                        date, obs_by_cam, all_camera_utms,
                        forest_mask, forest_dataset,
                        wing_kdes, darkness_bins,
                        num_cameras, replicates,
                        vary_center, jitter, correct_wing,
                    )
                    day_totals[date] = totals
                    med = int(np.median(totals))
                    print(ok(f" median {med:,}"))
                except Exception as e:
                    print(err(f" FAILED: {e}"))
                    if verbose:
                        traceback.print_exc()
                    day_totals[date] = [0.0] * replicates

            np.save(out_path, day_totals)
            print(f"  Saved → {out_path}\n")

    print(ok("  Done."))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run population estimates for all Kasanka observation dates"
    )
    parser.add_argument("--obs",  default=OBSERVATIONS_ROOT,
                        help="Observations root folder")
    parser.add_argument("--counts", default=COUNTS_ROOT,
                        help="Counts output root (year subfolders created automatically)")
    parser.add_argument("--year", nargs="+", default=None,
                        help="Restrict to specific year(s), e.g. --year 2021 2020")
    parser.add_argument("--replicates", type=int, default=1000,
                        help="Monte Carlo replicates per date (default 1000)")
    parser.add_argument("--no-vary-center", action="store_true",
                        help="Fix forest center instead of sampling randomly")
    parser.add_argument("--no-jitter",   action="store_true")
    parser.add_argument("--no-correct-wing", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing count files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run(
        obs_root    = args.obs,
        counts_root = args.counts,
        year_filter = args.year,
        replicates  = args.replicates,
        vary_center = not args.no_vary_center,
        jitter      = not args.no_jitter,
        correct_wing= not args.no_correct_wing,
        dry_run     = args.dry_run,
        verbose     = args.verbose,
        force       = args.force,
    )


if __name__ == "__main__":
    main()
