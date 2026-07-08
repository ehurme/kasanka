"""
Automated bat tracking for all camera-days that have detections but no raw_tracks.npy.

Loads centers.npy + contours per camera, splits the observation into 10 overlapping
temporal chunks, runs kbf.find_tracks() on each chunk (optionally in parallel),
then combines the chunks into a single raw_tracks.npy.

Usage:
    python run_tracking.py --dry-run
    python run_tracking.py
    python run_tracking.py --root "\\\\server\\kasanka-bats" --year 2022
    python run_tracking.py --workers 4   # parallel tracking jobs
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OUTPUT_ROOT = (r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme"
               r"\Eidolon_helvum\kasanka-bats")

OVERLAP_FRAMES   = 450   # 15 seconds at 30 fps — overlap between temporal chunks
N_CHUNKS         = 10    # number of temporal chunks per observation
DEFAULT_WORKERS  = 20    # AMD Ryzen Threadripper 3960X has 24 cores / 48 threads;
                         # 20 workers leaves headroom for OS and file I/O

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def looks_like_date(name):
    months = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    return (bool(re.search(months, name.lower()))
            or bool(re.fullmatch(r"\d{6,8}", name))
            or bool(re.match(r"^\d{6}\s+Bat\s+Count", name, re.IGNORECASE)))


def canonical_date(folder_name):
    m = re.match(r"^(\d{2})(\d{2})(\d{2})\s+Bat\s+Count", folder_name, re.IGNORECASE)
    if m:
        return f"20{m.group(1)}{m.group(2)}{m.group(3)}"
    return folder_name


def load_centers(camera_folder):
    """Load centers array — handles both pickle (new) and np.load (old) format."""
    for fname in ("centers.npy", "p_centers.npy"):
        path = os.path.join(camera_folder, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                return np.load(path, allow_pickle=True)
            except Exception:
                pass
    return None


def load_sizes(camera_folder):
    for fname in ("size.npy", "p_size.npy"):
        path = os.path.join(camera_folder, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                return np.load(path, allow_pickle=True)
            except Exception:
                pass
    return None


def has_detections(camera_folder):
    return (load_centers(camera_folder) is not None
            and bool(glob.glob(os.path.join(camera_folder, "contours-compressed-*.npy"))))


def needs_tracking(camera_folder):
    return (has_detections(camera_folder)
            and not os.path.exists(os.path.join(camera_folder, "raw_tracks.npy")))


def discover_todo(root, year_filter=None):
    todo = []
    skip = {"plots", "example-frames", "observations", "counts", "__pycache__"}
    try:
        top_items = sorted(os.listdir(root))
    except (PermissionError, OSError) as e:
        print(f"ERROR: cannot list {root}: {e}")
        return todo

    for item in top_items:
        if item.lower() in skip:
            continue
        item_path = os.path.join(root, item)
        if not os.path.isdir(item_path):
            continue
        if not looks_like_date(item):
            continue
        date_key = canonical_date(item)
        if year_filter and date_key[:4] not in year_filter:
            continue
        for cam in sorted(os.listdir(item_path)):
            cam_path = os.path.join(item_path, cam)
            if os.path.isdir(cam_path) and needs_tracking(cam_path):
                todo.append({"date": date_key, "camera": cam, "path": cam_path})
    return todo


# ---------------------------------------------------------------------------
# Core tracking logic (extracted from detections_to_tracks-2022.ipynb)
# ---------------------------------------------------------------------------

def build_chunk_dicts(camera_folder, n_chunks=N_CHUNKS, overlap=OVERLAP_FRAMES):
    """
    Return list of {camera_folder, first_frame, max_frame, tracks_file} dicts,
    one per temporal chunk, skipping chunks whose output file already exists.
    """
    centers = load_centers(camera_folder)
    if centers is None:
        return []

    n_frames = len(centers)
    breakpoints = np.linspace(0, n_frames, n_chunks + 1, dtype=int)
    first_frames = breakpoints[:-1].tolist()
    max_frames   = breakpoints[1:].tolist()
    max_frames[-1] = None   # last chunk runs to end

    # Apply overlap: shift first_frame back by overlap frames
    for i in range(1, len(first_frames)):
        first_frames[i] = max(first_frames[i] - overlap, 0)

    chunks = []
    for first_f, max_f in zip(first_frames, max_frames):
        max_str = f"{max_f:06d}" if max_f is not None else "None"
        basename = f"first_frame_{first_f:06d}_max_val_{max_str}_raw_tracks.npy"
        tracks_file = os.path.join(camera_folder, basename)
        if not os.path.exists(tracks_file):
            chunks.append({
                "camera_folder": camera_folder,
                "first_frame":   int(first_f),
                "max_frame":     max_f,
                "tracks_file":   tracks_file,
            })
    return chunks


def track_chunk(chunk_dict):
    """Run find_tracks on one temporal chunk. Safe to call from a Pool worker."""
    import bat_functions as kbf

    camera_folder = chunk_dict["camera_folder"]
    first_frame   = chunk_dict["first_frame"]
    max_frame     = chunk_dict["max_frame"]
    tracks_file   = chunk_dict["tracks_file"]

    if os.path.exists(tracks_file):
        return tracks_file

    centers = load_centers(camera_folder)
    sizes   = load_sizes(camera_folder)

    contour_files = sorted(
        glob.glob(os.path.join(camera_folder, "contours-compressed-*.npy"))
    )

    cam_name = os.path.basename(camera_folder)
    print(f"    [{cam_name}] chunk {first_frame}–{max_frame} ...", flush=True)

    try:
        kbf.find_tracks(
            first_frame,
            centers,
            contours_files=contour_files,
            sizes_list=sizes,
            tracks_file=tracks_file,
            max_frame=max_frame,
        )
    except Exception as e:
        print(f"    ERROR in chunk {first_frame}: {e}")
        return None

    return tracks_file


def combine_chunks(camera_folder):
    """
    Merge partial first_frame_*_raw_tracks.npy files into raw_tracks.npy,
    deduplicating by the overlap regions (keeps tracks whose first_frame is
    before the next group's start).
    """
    chunk_files = sorted(
        glob.glob(os.path.join(camera_folder, "first_frame*_raw_tracks.npy")),
        key=lambda f: int(os.path.basename(f).split("_")[2])
    )

    if not chunk_files:
        return False

    # Extract the start frame of each chunk from the filename
    start_frames = [int(os.path.basename(f).split("_")[2]) for f in chunk_files]
    # The overlap cutoff for each group is the start of the NEXT group
    cutoffs = start_frames[1:] + [None]

    all_tracks = []
    for chunk_file, cutoff in zip(chunk_files, cutoffs):
        track_group = np.load(chunk_file, allow_pickle=True)
        for track in track_group:
            if isinstance(track.get("track"), list):
                track["track"]     = np.stack(track["track"])
                track["pos_index"] = np.stack(track["pos_index"])
                if "size" in track:
                    track["size"] = np.stack(track["size"])
            # Keep only tracks that started before this chunk's cutoff
            if cutoff is None or track["first_frame"] < cutoff:
                all_tracks.append(track)

    out_file = os.path.join(camera_folder, "raw_tracks.npy")
    np.save(out_file, all_tracks)
    print(f"    Saved {len(all_tracks):,} tracks → {out_file}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated bat tracking")
    parser.add_argument("--root", default=OUTPUT_ROOT,
                        help="Root kasanka-bats folder")
    parser.add_argument("--year", nargs="+", default=None,
                        help="Limit to specific year(s), e.g. --year 2022")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel tracking workers (default: {DEFAULT_WORKERS} "
                             f"— tuned for 24-core Threadripper)")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try: sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError: pass

    print("\n" + "="*60)
    print("  Kasanka — Bat Tracking")
    print("="*60)
    print(f"  Root    : {args.root}")
    print(f"  Workers : {args.workers}")
    if args.dry_run:
        print("  DRY RUN — no files will be written")
    print()

    todo = discover_todo(args.root, args.year)

    if not todo:
        print("  Nothing to do — all cameras already have raw_tracks.npy.")
        print("  Next step: run_missing_analyses.py")
        return

    print(f"  {len(todo)} camera-days need tracking:\n")
    for e in todo:
        print(f"    {e['date']}/{e['camera']}")

    if args.dry_run:
        return

    # Build all chunk dicts across all cameras
    all_chunks = []
    for entry in todo:
        chunks = build_chunk_dicts(entry["path"])
        all_chunks.extend(chunks)

    pending = [c for c in all_chunks if not os.path.exists(c["tracks_file"])]
    print(f"\n  {len(pending)} chunk jobs to run "
          f"({len(all_chunks) - len(pending)} already done).")

    if pending:
        if args.workers > 1:
            from multiprocessing import Pool
            print(f"  Using {args.workers} parallel workers ...")
            with Pool(processes=args.workers) as pool:
                pool.map(track_chunk, pending)
        else:
            for chunk in pending:
                track_chunk(chunk)

    # Combine chunks for each camera
    print("\n  Combining chunks into raw_tracks.npy ...")
    done_combine, failed_combine = 0, 0
    for entry in todo:
        # Check all chunks now exist
        cam_chunks = build_chunk_dicts(entry["path"])
        missing = [c for c in cam_chunks
                   if not os.path.exists(c["tracks_file"])]
        if missing:
            print(f"    {entry['date']}/{entry['camera']}: "
                  f"{len(missing)} chunk(s) still missing — skipping combine")
            failed_combine += 1
            continue
        print(f"  {entry['date']}/{entry['camera']}")
        try:
            combine_chunks(entry["path"])
            done_combine += 1
        except Exception as e:
            print(f"    ERROR combining: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            failed_combine += 1

    print(f"\n  Tracking complete: {done_combine} cameras done, "
          f"{failed_combine} failed/incomplete.")
    print("\n  Next step: run_missing_analyses.py "
          "(crossing tracks + observations)")


if __name__ == "__main__":
    main()
