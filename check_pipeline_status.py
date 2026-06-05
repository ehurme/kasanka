"""
Pipeline status checker for the Kasanka bat counting workflow.

Auto-discovers the actual folder structure on the server, maps every
date/camera combination through all pipeline stages, and reports what
is done and what still needs to be run to reach final population counts.

Usage:
    python check_pipeline_status.py
    python check_pipeline_status.py --root "\\\\10.0.16.7\\share\\kasanka-bats"
    python check_pipeline_status.py --verbose
"""

import argparse
import os
import re
import glob
import sys

import numpy as np

# Force UTF-8 output on Windows so block/box characters render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(t):    return f"{GREEN}{t}{RESET}"
def warn(t):  return f"{YELLOW}{t}{RESET}"
def err(t):   return f"{RED}{t}{RESET}"
def hdr(t):   return f"{BOLD}{CYAN}{t}{RESET}"
def bold(t):  return f"{BOLD}{t}{RESET}"

# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------
STAGE_REQUIRED = {
    "Inference": ["centers.npy", "size.npy", "rects.npy"],
    "Tracking":  ["raw_tracks.npy"],
    "Crossing":  ["crossing_tracks.npy", "blue-means.npy"],
}

COUNT_FILE_RE = re.compile(
    r"num_cameras-(.+?)-jitter-(.+?)-vary_center-(.+?)-correct_wing-(.+?)"
    r"-replicates-(\d+)\.npy$"
)

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def looks_like_camera_folder(path):
    markers = ["centers.npy", "raw_tracks.npy", "crossing_tracks.npy"]
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def looks_like_date_name(name):
    months = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    return (bool(re.search(months, name.lower()))
            or bool(re.fullmatch(r"\d{6,8}", name)))


def discover_camera_days(root):
    """Walk root looking for (date, camera, path) triples."""
    entries = []
    skip_dirs = {"plots", "example-frames", "observations", "counts",
                 "__pycache__", ".git"}
    try:
        top_items = sorted(os.listdir(root))
    except (PermissionError, OSError):
        return entries

    for item in top_items:
        if item.lower() in skip_dirs:
            continue
        item_path = os.path.join(root, item)
        if not os.path.isdir(item_path):
            continue

        # Date folder → camera subfolders
        if looks_like_date_name(item):
            try:
                subs = sorted(os.listdir(item_path))
            except (PermissionError, OSError):
                continue
            for sub in subs:
                sub_path = os.path.join(item_path, sub)
                if os.path.isdir(sub_path) and looks_like_camera_folder(sub_path):
                    entries.append({
                        "date": item,
                        "camera": sub,
                        "path": sub_path
                    })
        # Direct camera folder under root
        elif looks_like_camera_folder(item_path):
            entries.append({
                "date": os.path.basename(root),
                "camera": item,
                "path": item_path
            })
    return entries


def discover_observation_files(obs_root):
    """
    Return {(date, camera_normalised): path}.
    Searches obs_root/ and obs_root/*/ (one level of date subfolders).
    """
    result = {}
    if not os.path.isdir(obs_root):
        return result
    patterns = [
        os.path.join(obs_root, "*-observation-*.npy"),
        os.path.join(obs_root, "*", "*-observation-*.npy"),
    ]
    for pat in patterns:
        for fpath in glob.glob(pat):
            fname = os.path.basename(fpath)
            m = re.match(r"^(.+?)-observation-(.+)\.npy$", fname)
            if m:
                date = m.group(1)
                cam  = m.group(2).replace(" ", "").lower()
                result[(date, cam)] = fpath
    return result


def discover_count_files(counts_root):
    """
    Return list of dicts for population-estimate .npy files.
    Searches both counts_root/ and counts_root/*/ (year subfolders).
    """
    found = []
    if not os.path.isdir(counts_root):
        return found
    search_paths = [counts_root]
    try:
        for sub in os.listdir(counts_root):
            sub_path = os.path.join(counts_root, sub)
            if os.path.isdir(sub_path):
                search_paths.append(sub_path)
    except (PermissionError, OSError):
        pass

    for folder in search_paths:
        try:
            fnames = os.listdir(folder)
        except (PermissionError, OSError):
            continue
        for fname in sorted(fnames):
            m = COUNT_FILE_RE.match(fname)
            if m:
                found.append({
                    "path": os.path.join(folder, fname),
                    "folder": os.path.basename(folder),
                    "num_cameras": m.group(1),
                    "replicates": int(m.group(5)),
                })
    return found


def normalise_camera(name):
    return name.replace(" ", "").lower()


def stage_status(camera_path):
    status = {}
    for stage, files in STAGE_REQUIRED.items():
        missing = [f for f in files
                   if not os.path.exists(os.path.join(camera_path, f))]
        status[stage] = (len(missing) == 0, missing)
    return status


def crossing_count(obs_path):
    try:
        obs = np.load(obs_path, allow_pickle=True).item()
        return len(obs.get("frames", []))
    except Exception:
        return None


def load_count_summary(count_path):
    try:
        data = np.load(count_path, allow_pickle=True).item()
        return {d: int(np.median(v)) for d, v in data.items() if v}
    except Exception:
        return {}


def find_obs_root(base):
    return os.path.join(base, "observations")


def find_counts_root(base):
    return os.path.join(base, "counts")


# ---------------------------------------------------------------------------
# Date normalisation for grouping
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def date_sort_key(date_str):
    """Return a sortable key so dates order chronologically."""
    s = date_str.lower()
    # YYYYMMDD
    if re.fullmatch(r"\d{8}", date_str):
        return date_str
    # YYYYMM
    if re.fullmatch(r"\d{6}", date_str):
        return date_str + "00"
    # DD-Mon-YYYY  e.g. 16-Nov-2020
    m = re.match(r"(\d+)-([a-z]+)-(\d{4})", s)
    if m:
        return f"{m.group(3)}{MONTH_MAP.get(m.group(2),'00')}{int(m.group(1)):02d}"
    # DDMon  e.g. 16Nov
    m = re.match(r"(\d+)([a-z]+)", s)
    if m:
        mon = MONTH_MAP.get(m.group(2), "00")
        dd  = int(m.group(1))
        # heuristic year (Nov/Dec = 2020 study; others vary)
        yr  = "2020"
        return f"{yr}{mon}{dd:02d}"
    return "99999999"


def infer_year_group(date_str):
    k = date_sort_key(date_str)
    if k.startswith("2019"):
        return "2019"
    if k.startswith("2020"):
        return "2020"
    if k.startswith("2021"):
        return "2021"
    if k.startswith("2022"):
        return "2022"
    return "other"


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run(base, verbose):
    print()
    print(hdr("=" * 72))
    print(hdr("  Kasanka Bat Pipeline — Status Report"))
    print(hdr("=" * 72))

    if not os.path.isdir(base):
        print(err(f"\n  Cannot access base folder: {base}"))
        return

    print(f"  Base: {base}")

    print("\n  Scanning...", end="", flush=True)

    obs_root    = find_obs_root(base)
    counts_root = find_counts_root(base)

    camera_days  = discover_camera_days(base)
    obs_map      = discover_observation_files(obs_root)
    count_files  = discover_count_files(counts_root)

    # De-duplicate
    seen, unique = set(), []
    for cd in camera_days:
        key = (cd["date"], cd["camera"], cd["path"])
        if key not in seen:
            seen.add(key)
            unique.append(cd)
    camera_days = unique

    print(f" {len(camera_days)} camera-days, "
          f"{len(obs_map)} observations, "
          f"{len(count_files)} count files.")

    # Group camera_days by date
    by_date = {}
    for cd in camera_days:
        by_date.setdefault(cd["date"], []).append(cd)

    stages = list(STAGE_REQUIRED.keys())
    all_stages = stages + ["Observation"]

    # Running totals
    totals = {s: [0, 0] for s in all_stages}
    blocking = {s: [] for s in all_stages}

    col_cam = 24
    col_stg = 12
    col_obs = 14

    header = (f"{'Camera':<{col_cam}}"
              + "".join(f"{s:^{col_stg}}" for s in stages)
              + f"{'Obs':^{col_obs}}"
              + f"{'Crossings':>10}")
    sep = "-" * len(header)

    # Group dates by year for section headers
    year_groups = {}
    for date in sorted(by_date.keys(), key=date_sort_key):
        yr = infer_year_group(date)
        year_groups.setdefault(yr, []).append(date)

    for yr in sorted(year_groups.keys()):
        print(f"\n{'='*72}")
        print(bold(f"  Year group: {yr}"))
        print(f"{'='*72}")

        for date in year_groups[yr]:
            print(f"\n  {bold(date)}")
            print(f"  {hdr(header)}")
            print(f"  {hdr(sep)}")

            for cd in sorted(by_date[date], key=lambda x: x["camera"]):
                cam  = cd["camera"]
                path = cd["path"]
                ss   = stage_status(path)

                # Find observation file (try various camera normalisations)
                cam_key = normalise_camera(cam)
                obs_path = obs_map.get((date, cam_key))
                if obs_path is None:
                    # Also try with spaces preserved
                    obs_path = obs_map.get((date, cam.lower()))
                if obs_path is None:
                    # Try alternate date formats
                    for (d, c), p in obs_map.items():
                        if c == cam_key and (
                            d.replace("-", "").lower().replace(" ", "") ==
                            date.replace("-", "").lower().replace(" ", "")
                        ):
                            obs_path = p
                            break

                obs_done = obs_path is not None

                row = [f"  {cam:<{col_cam - 2}}"]
                for s in stages:
                    done, missing = ss[s]
                    totals[s][1] += 1
                    if done:
                        totals[s][0] += 1
                        row.append(ok(f"{'OK':^{col_stg}}"))
                    else:
                        blocking[s].append((date, cam))
                        row.append(err(f"{'--':^{col_stg}}"))

                totals["Observation"][1] += 1
                if obs_done:
                    totals["Observation"][0] += 1
                    n = crossing_count(obs_path)
                    row.append(ok(f"{'OK':^{col_obs}}"))
                    row.append(f"{n:>10,}" if n is not None else f"{'?':>10}")
                else:
                    blocking["Observation"].append((date, cam))
                    row.append(err(f"{'--':^{col_obs}}"))
                    row.append(f"{'':>10}")

                print("".join(row))

    # -----------------------------------------------------------------------
    # Stage summary
    # -----------------------------------------------------------------------
    print(f"\n{hdr('=' * 72)}")
    print(hdr("  Stage Summary  (all dates combined)"))
    print(hdr("=" * 72))
    for s in all_stages:
        done, total = totals[s]
        pct  = 100 * done / total if total else 0
        bar  = "█" * done + "░" * (total - done)
        cfn  = ok if done == total else (warn if pct >= 60 else err)
        print(f"  {s:<14}  {cfn(bar)}  {done}/{total} ({pct:.0f}%)")

    # -----------------------------------------------------------------------
    # What's blocking
    # -----------------------------------------------------------------------
    total_blocking = sum(len(v) for v in blocking.values())
    if total_blocking:
        print(f"\n{hdr('=' * 72)}")
        print(hdr("  What Still Needs To Be Done"))
        print(hdr("=" * 72))
        for s in all_stages:
            items = blocking[s]
            if not items:
                continue
            print(f"\n  {warn(s)}  —  {len(items)} camera-day{'s' if len(items)>1 else ''}")
            by_d = {}
            for d, c in items:
                by_d.setdefault(d, []).append(c)
            for d in sorted(by_d.keys(), key=date_sort_key):
                cams = ", ".join(sorted(by_d[d]))
                print(f"    {d}: {cams}")
    else:
        print(f"\n{ok('  All camera-days are complete through observations!')}")

    # -----------------------------------------------------------------------
    # Count files
    # -----------------------------------------------------------------------
    print(f"\n{hdr('=' * 72)}")
    print(hdr("  Population Estimate Files"))
    print(hdr("=" * 72))

    if count_files:
        by_folder = {}
        for cf in count_files:
            by_folder.setdefault(cf["folder"], []).append(cf)
        for folder in sorted(by_folder.keys()):
            cfs = by_folder[folder]
            all_cam_file = next(
                (c for c in cfs if c["num_cameras"] == "all"), None)
            n_cams_max = max(
                (int(c["num_cameras"]) for c in cfs
                 if c["num_cameras"].isdigit()), default=0)
            reps = sorted(set(c["replicates"] for c in cfs))
            tag = (ok(f"all-cam + {n_cams_max} subsets")
                   if all_cam_file else warn(f"{n_cams_max} camera subsets only"))
            print(f"\n  counts/{folder}/   {len(cfs)} files  |  {tag}"
                  f"  |  replicates: {reps}")
            if verbose and all_cam_file:
                summary = load_count_summary(all_cam_file["path"])
                if summary:
                    for d, med in sorted(summary.items(), key=lambda x: date_sort_key(x[0])):
                        print(f"    {d:20s}  median {med:>10,}")
    else:
        print(f"\n  {err('No count files found.')}")

    # -----------------------------------------------------------------------
    # Readiness per year
    # -----------------------------------------------------------------------
    print(f"\n{hdr('=' * 72)}")
    print(hdr("  Readiness for Population Estimation"))
    print(hdr("=" * 72))

    for yr in sorted(year_groups.keys()):
        dates_in_yr = year_groups[yr]
        print(f"\n  {bold(yr)}")
        for date in dates_in_yr:
            cds = by_date[date]
            n_total = len(cds)
            ready, not_ready = [], []
            for cd in cds:
                cam = cd["camera"]
                ss  = stage_status(cd["path"])
                cam_key  = normalise_camera(cam)
                obs_found = obs_map.get((date, cam_key))
                if obs_found is None:
                    for (d, c), p in obs_map.items():
                        if c == cam_key and (
                            d.replace("-","").lower() ==
                            date.replace("-","").lower()
                        ):
                            obs_found = p
                            break
                if all(done for done, _ in ss.values()) and obs_found:
                    ready.append(cam)
                else:
                    not_ready.append(cam)

            if not not_ready:
                status_str = ok(f"READY  ({n_total}/{n_total} cameras)")
            elif ready:
                pct = 100 * len(ready) // n_total
                status_str = warn(f"{len(ready)}/{n_total} cameras ready  "
                                  f"| still needed: {', '.join(sorted(not_ready))}")
            else:
                status_str = err(f"0/{n_total} cameras ready")
            print(f"    {date:<22}  {status_str}")

        # Check if counts exist for this year
        yr_counts = [c for c in count_files
                     if c["folder"] in (yr, f"{yr}_new", f"{yr}_all")]
        has_all = any(c["num_cameras"] == "all" for c in yr_counts)
        if has_all:
            print(f"    {ok(f'Population estimates exist for {yr}.')}")
        else:
            n_ready = sum(1 for d in dates_in_yr
                          if not any(c for cd in by_date[d]
                                     for c in [blocking["Observation"]]
                                     if (d, cd["camera"]) in c))
            print(f"    {warn(f'No all-camera estimate for {yr} yet.')}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check Kasanka bat pipeline status"
    )
    parser.add_argument(
        "--root",
        default=r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme\Eidolon_helvum\kasanka-bats",
        help="Base kasanka-bats folder"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show median population estimates per date"
    )
    args = parser.parse_args()
    run(args.root, args.verbose)


if __name__ == "__main__":
    main()
