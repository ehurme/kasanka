"""
Batch processor for incomplete Kasanka bat pipeline stages.

Finds every camera-day that has raw_tracks.npy but is missing downstream
outputs, and runs the following stages automatically:

  Stage A — Blue-means assembly
    Combines any per-clip mean-blue-*.npy files (produced by
    get-observation-frame-darkness.ipynb) into a single blue-means.npy.
    NOTE: blue-means.npy itself still requires the original videos to be
    processed first. Run get-observation-frame-darkness.ipynb if you have
    the video files and blue-means.npy is missing entirely.

  Stage B — Crossing tracks
    Loads raw_tracks.npy → filters short tracks → finds midline crossings
    → saves crossing_tracks.npy.  Does NOT require the original videos.

  Stage C — Observation compilation
    Loads crossing_tracks.npy + blue-means.npy → extracts per-bat
    crossing metadata → saves <date>-observation-<camera>.npy to the
    shared observations folder.

Usage:
    python run_missing_analyses.py --dry-run        # show what would run
    python run_missing_analyses.py                  # run everything
    python run_missing_analyses.py --date 20211201  # single date
    python run_missing_analyses.py --stages B C     # skip blue-means assembly
"""

import argparse
import glob
import os
import sys
import traceback

import numpy as np

# bat_functions is imported lazily inside the functions that need it so that
# discovery and --dry-run work even when heavy deps (cv2, rasterio) are absent.
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Configuration — edit these paths to match your setup
# ---------------------------------------------------------------------------

BASE_ROOT = (
    r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme"
    r"\Eidolon_helvum\kasanka-bats"
)
OBSERVATIONS_ROOT = os.path.join(BASE_ROOT, "observations")

# Frame height in pixels. Must match the actual video resolution.
# 1520 is correct for the GoPro footage used in this project.
FRAME_HEIGHT = 1520

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(t):    return f"{GREEN}{t}{RESET}"
def warn(t):  return f"{YELLOW}{t}{RESET}"
def err(t):   return f"{RED}{t}{RESET}"

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Discovery helpers (same logic as check_pipeline_status.py)
# ---------------------------------------------------------------------------

def looks_like_date_name(name):
    import re
    months = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    return (bool(re.search(months, name.lower()))
            or bool(re.fullmatch(r"\d{6,8}", name)))


def looks_like_camera_folder(path):
    return any(
        os.path.exists(os.path.join(path, m))
        for m in ["centers.npy", "raw_tracks.npy", "crossing_tracks.npy"]
    )


def discover_camera_days(root):
    """Return list of {date, camera, path} dicts."""
    entries = []
    skip = {"plots", "example-frames", "observations", "counts", "__pycache__"}
    for item in sorted(os.listdir(root)):
        if item.lower() in skip:
            continue
        item_path = os.path.join(root, item)
        if not os.path.isdir(item_path):
            continue
        if looks_like_date_name(item):
            for sub in sorted(os.listdir(item_path)):
                sub_path = os.path.join(item_path, sub)
                if os.path.isdir(sub_path) and looks_like_camera_folder(sub_path):
                    entries.append({"date": item, "camera": sub, "path": sub_path})
    return entries


# ---------------------------------------------------------------------------
# Stage A — compute or assemble blue-means.npy
#
# Three sub-cases handled in priority order:
#   A1. mean-blue-<clip>.npy files already exist → combine them
#   A2. raw video files exist in/near the camera folder → read and compute
#   A3. nothing found → return False
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = ("*.MP4", "*.mp4", "*.MOV", "*.mov", "*.AVI", "*.avi")


def _find_video_files(camera_path, video_root=None, date=None, camera=None):
    """Return sorted list of video file paths for this camera-day."""
    found = []
    # Check directly inside the camera folder
    for ext in VIDEO_EXTENSIONS:
        found.extend(glob.glob(os.path.join(camera_path, ext)))
    # Check in a parallel video root tree: video_root/<date>/<camera>/
    if not found and video_root and date and camera:
        search = os.path.join(video_root, date, camera)
        for ext in VIDEO_EXTENSIONS:
            found.extend(glob.glob(os.path.join(search, ext)))
    return sorted(found)


def _compute_blue_means_from_video(video_path):
    """Read every frame of a video and return array of per-frame mean blue values."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 not available — install opencv-python-headless")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    means = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads as BGR; channel 0 is blue
        means.append(float(np.mean(frame[..., 0])))
        frame_count += 1
        if frame_count % 10_000 == 0:
            print(f"         {frame_count:,} frames processed ...", flush=True)

    cap.release()
    return np.array(means)


def compute_blue_means(camera_path, video_root=None, date=None,
                       camera=None, dry_run=False):
    """
    Ensure blue-means.npy exists for this camera folder.

    Priority:
      1. Already done → return True immediately
      2. Pre-computed per-clip mean-blue-*.npy files → assemble them
      3. Raw video files found → compute from scratch
      4. Nothing found → return False (needs manual attention)
    """
    out_file   = os.path.join(camera_path, "blue-means.npy")
    clip_files = sorted(glob.glob(os.path.join(camera_path, "mean-blue-*.npy")))
    video_files = _find_video_files(camera_path, video_root, date, camera)

    if os.path.exists(out_file):
        return True

    # --- A1: assemble pre-computed clips ---
    if clip_files:
        print(f"    [A] Assembling {len(clip_files)} clip file(s) → blue-means.npy")
        if dry_run:
            return True
        try:
            combined = np.hstack([np.load(f) for f in clip_files])
            np.save(out_file, combined)
            for f in clip_files:
                os.remove(f)
            print(ok(f"         {len(combined):,} frames assembled"))
            return True
        except Exception as e:
            print(err(f"    [A1] FAILED: {e}"))
            return False

    # --- A2: compute from raw video ---
    if video_files:
        print(f"    [A] Computing blue-means from {len(video_files)} video file(s) ...")
        if dry_run:
            return True
        all_means = []
        for vf in video_files:
            vname = os.path.splitext(os.path.basename(vf))[0]
            print(f"         {vname} ...", flush=True)
            try:
                means = _compute_blue_means_from_video(vf)
                all_means.append(means)
                print(ok(f"         {len(means):,} frames done"))
            except Exception as e:
                print(err(f"         FAILED: {e}"))
                if args_verbose:
                    traceback.print_exc()
        if not all_means:
            return False
        combined = np.hstack(all_means)
        np.save(out_file, combined)
        print(ok(f"         blue-means.npy saved ({len(combined):,} total frames)"))
        return True

    # --- A3: nothing available ---
    return False


# ---------------------------------------------------------------------------
# Stage B — compute crossing tracks
# ---------------------------------------------------------------------------

def _fix_track_contours(tracks):
    """Cast object-dtype contour arrays to float32 so cv2.minAreaRect accepts them.

    Contours saved from different pipeline runs sometimes end up as numpy
    object arrays rather than typed float/int arrays, which causes OpenCV to
    raise 'points data type = object is not supported'.
    """
    for track in tracks:
        if "contour" not in track:
            continue
        fixed = []
        for c in track["contour"]:
            if c is None or not hasattr(c, "dtype"):
                fixed.append(c)
            elif c.dtype == object:
                try:
                    fixed.append(np.array(c.tolist(), dtype=np.float32))
                except Exception:
                    fixed.append(None)  # drop unrecoverable contour
            else:
                fixed.append(c)
        track["contour"] = fixed
    return tracks

def compute_crossing_tracks(camera_path, frame_height, dry_run=False):
    """
    raw_tracks.npy → crossing_tracks.npy.
    Returns True if crossing_tracks.npy now exists (or already did).
    """
    raw_file      = os.path.join(camera_path, "raw_tracks.npy")
    crossing_file = os.path.join(camera_path, "crossing_tracks.npy")

    if os.path.exists(crossing_file):
        return True

    if not os.path.exists(raw_file):
        print(warn(f"    [B] raw_tracks.npy missing — cannot compute crossings"))
        return False

    print(f"    [B] Computing crossing tracks from raw_tracks.npy ...")
    if dry_run:
        return True

    try:
        from bat_functions import threshold_short_tracks, measure_crossing_bats
    except ImportError as e:
        print(err(f"    [B] Cannot import bat_functions ({e})"))
        print(err(f"         Make sure cv2, scipy, etc. are installed in this environment."))
        return False

    try:
        raw_tracks = np.load(raw_file, allow_pickle=True)
        print(f"         {len(raw_tracks):,} raw tracks loaded")

        tracks = threshold_short_tracks(raw_tracks, min_length_threshold=2)
        print(f"         {len(tracks):,} tracks after length filter")

        tracks = _fix_track_contours(tracks)
        crossing_tracks = measure_crossing_bats(tracks, frame_height=frame_height)
        print(ok(f"         {len(crossing_tracks):,} crossing tracks found"))

        np.save(crossing_file, np.array(crossing_tracks, dtype=object))
        return True

    except Exception as e:
        print(err(f"    [B] FAILED: {e}"))
        if args_verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Stage C — compile observation dict
# ---------------------------------------------------------------------------

def compile_observation(date, camera, camera_path,
                        observations_root, dry_run=False):
    """
    crossing_tracks.npy + blue-means.npy → <date>-observation-<camera>.npy.
    Returns True on success.
    """
    out_dir  = os.path.join(observations_root, date)
    out_file = os.path.join(out_dir, f"{date}-observation-{camera}.npy")

    if os.path.exists(out_file):
        return True

    crossing_file = os.path.join(camera_path, "crossing_tracks.npy")
    blue_file     = os.path.join(camera_path, "blue-means.npy")

    if not dry_run and not os.path.exists(crossing_file):
        print(warn(f"    [C] crossing_tracks.npy missing"))
        return False
    if not dry_run and not os.path.exists(blue_file):
        print(warn(f"    [C] blue-means.npy missing — "
                   f"run get-observation-frame-darkness.ipynb first"))
        return False

    print(f"    [C] Compiling observation ...")
    if dry_run:
        return True

    try:
        crossing_tracks = np.load(crossing_file, allow_pickle=True)
        darkness_means  = np.load(blue_file)

        frames, sizes, ids, directions, darkness, lengths = [], [], [], [], [], []

        for track_ind, track in enumerate(crossing_tracks):
            frame = track.get("crossed", 0)
            if frame == 0:
                continue
            frames.append(frame)
            sizes.append(track.get("mean_wing", np.nan))
            ids.append(track_ind)
            directions.append(1 if frame > 0 else -1)
            dark_frame = abs(frame)
            if dark_frame < len(darkness_means):
                darkness.append(darkness_means[dark_frame])
            else:
                darkness.append(np.nan)
            lengths.append(len(track["track"]) if "track" in track else 0)

        obs = {
            "date":         date,
            "camera":       camera,
            "frames":       np.array(frames),
            "mean_wing":    np.array(sizes),
            "ids":          np.array(ids),
            "direction":    np.array(directions),
            "darkness":     np.array(darkness),
            "track_length": np.array(lengths),
        }

        os.makedirs(out_dir, exist_ok=True)
        np.save(out_file, obs)
        print(ok(f"         Saved {len(frames):,} crossings → {out_file}"))
        return True

    except Exception as e:
        print(err(f"    [C] FAILED: {e}"))
        if args_verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_verbose = False   # set by argparse below

def run(base, obs_root, frame_height, stages, date_filter, dry_run, verbose,
        video_root=None):
    global args_verbose
    args_verbose = verbose

    print()
    print("=" * 64)
    print("  Kasanka — Run Missing Analyses")
    print("=" * 64)
    print(f"  Base root       : {base}")
    print(f"  Observations out: {obs_root}")
    print(f"  Frame height    : {frame_height} px")
    print(f"  Stages          : {', '.join(stages)}")
    if video_root:
        print(f"  Video root      : {video_root}")
    if date_filter:
        print(f"  Date filter     : {date_filter}")
    if dry_run:
        print(warn("  DRY RUN — no files will be written"))
    print()

    camera_days = discover_camera_days(base)
    if date_filter:
        camera_days = [cd for cd in camera_days if cd["date"] in date_filter]

    # Only work on camera-days that are incomplete
    todo = []
    for cd in camera_days:
        p      = cd["path"]
        date   = cd["date"]
        camera = cd["camera"]
        blue_done = os.path.exists(os.path.join(p, "blue-means.npy"))
        has_clips = bool(glob.glob(os.path.join(p, "mean-blue-*.npy")))
        has_video = bool(_find_video_files(p, video_root, date, camera))
        needs_A = not blue_done and (has_clips or has_video)
        needs_B = (os.path.exists(os.path.join(p, "raw_tracks.npy")) and
                   not os.path.exists(os.path.join(p, "crossing_tracks.npy")))
        out_file = os.path.join(obs_root, date, f"{date}-observation-{camera}.npy")
        needs_C = (os.path.exists(os.path.join(p, "crossing_tracks.npy")) or needs_B) and \
                  (blue_done or needs_A) and \
                  not os.path.exists(out_file)
        if (needs_A and "A" in stages) or \
           (needs_B and "B" in stages) or \
           (needs_C and "C" in stages):
            cd["needs_A"] = needs_A
            cd["needs_B"] = needs_B
            cd["needs_C"] = needs_C
            todo.append(cd)

    if not todo:
        print(ok("  Nothing to do — all stages are already complete!"))
        return

    print(f"  {len(todo)} camera-days need processing.\n")

    # Count outcomes
    counts = {"A_done": 0, "A_skip": 0, "B_done": 0, "B_skip": 0,
              "C_done": 0, "C_skip": 0, "errors": 0}

    current_date = None
    for cd in todo:
        date   = cd["date"]
        camera = cd["camera"]
        path   = cd["path"]

        if date != current_date:
            current_date = date
            print(f"\n  {date}")

        print(f"  {camera}")

        # Stage A
        if "A" in stages and cd["needs_A"]:
            ok_a = compute_blue_means(path, video_root=video_root,
                                      date=date, camera=camera, dry_run=dry_run)
            if ok_a:
                counts["A_done"] += 1
            else:
                counts["A_skip"] += 1

        # Stage B
        if "B" in stages and cd["needs_B"]:
            ok_b = compute_crossing_tracks(path, frame_height, dry_run)
            if ok_b:
                counts["B_done"] += 1
            else:
                counts["B_skip"] += 1
                # If crossing failed, C will also fail — skip it
                print(warn(f"    [C] Skipping (crossing not available)"))
                counts["C_skip"] += 1
                continue

        # Stage C — re-check availability after A and B may have run
        out_file = os.path.join(obs_root, date, f"{date}-observation-{camera}.npy")
        if "C" in stages and not os.path.exists(out_file):
            ok_c = compile_observation(date, camera, path,
                                       obs_root, dry_run)
            if ok_c:
                counts["C_done"] += 1
            else:
                counts["C_skip"] += 1

    # Summary
    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    if "A" in stages:
        print(f"  Blue-means computed  : {counts['A_done']} done, "
              f"{counts['A_skip']} skipped/failed")
    if "B" in stages:
        print(f"  Crossing tracks      : {counts['B_done']} done, "
              f"{counts['B_skip']} skipped/failed")
    if "C" in stages:
        print(f"  Observations saved   : {counts['C_done']} done, "
              f"{counts['C_skip']} skipped/failed")

    if counts["C_done"] > 0 and not dry_run:
        print()
        print(ok("  New observations are ready."))
        print("  Next step: run population-estimation.ipynb")
        print(f"  pointing at observations_root = {obs_root!r}")

    remaining_blue = sum(
        1 for cd in todo
        if not os.path.exists(os.path.join(cd["path"], "blue-means.npy"))
        and not glob.glob(os.path.join(cd["path"], "mean-blue-*.npy"))
        and not _find_video_files(cd["path"], video_root, cd["date"], cd["camera"])
    )
    if remaining_blue > 0:
        print()
        print(warn(f"  {remaining_blue} camera-days still need blue-means.npy."))
        print("  Options:")
        print("    1. Provide --video-root pointing to the raw video files")
        print("    2. Run get-observation-frame-darkness.ipynb on those cameras")
        print("       then re-run this script to compile observations.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run missing Kasanka bat pipeline stages (crossing + observations)"
    )
    parser.add_argument("--root", default=BASE_ROOT,
                        help="Base kasanka-bats folder")
    parser.add_argument("--obs", default=OBSERVATIONS_ROOT,
                        help="Observations output folder")
    parser.add_argument("--frame-height", type=int, default=FRAME_HEIGHT,
                        help="Frame height in pixels (default: 1520)")
    parser.add_argument("--stages", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"],
                        help="Which stages to run (A=blue-means assembly, "
                             "B=crossing tracks, C=observations)")
    parser.add_argument("--date", nargs="+", default=None,
                        help="Restrict to specific date(s), e.g. --date 20211201 20211207")
    parser.add_argument("--video-root", default=None,
                        help="Root folder containing raw videos in "
                             "<video-root>/<date>/<camera>/ structure. "
                             "Only needed if videos are not inside the camera data folders.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing any files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full tracebacks on errors")
    args = parser.parse_args()

    run(
        base=args.root,
        obs_root=args.obs,
        frame_height=args.frame_height,
        stages=args.stages,
        date_filter=args.date,
        dry_run=args.dry_run,
        verbose=args.verbose,
        video_root=args.video_root,
    )


if __name__ == "__main__":
    main()
