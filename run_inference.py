"""
Automated bat detection inference for all camera-days missing centers.npy.

Finds every camera folder that has video files but no centers.npy, then runs
the UNETTraditional deep-learning model to produce per-frame detections.

Outputs per camera folder:
  centers.npy, size.npy, rects.npy    — per-frame detection arrays (pickle)
  contours-compressed-00.npy …        — compressed contour files
  example-frames/<date>_<cam>_obs-ind_<N>.jpg

Requires: PyTorch + CUDA, cv2, the trained model .tar, and mean/std .npy files.

Usage:
    python run_inference.py --dry-run
    python run_inference.py
    python run_inference.py --cameras-root "E:\\KasankaCameras" --year 2022
"""

import argparse
import glob
import os
import pickle
import re
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults — edit these or override with CLI args
# ---------------------------------------------------------------------------

CAMERAS_ROOT = r"E:\KasankaCameras"
OUTPUT_ROOT  = (r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme"
                r"\Eidolon_helvum\kasanka-bats")

MODEL_FILE   = (r"C:\Users\Edward\Dropbox\bats-code\models"
                r"\model_UNETTraditional_epochs_100_batcheff_16_lr_0.01"
                r"_momentum_0.9_aug_better-norm-aug-2d-20Nov-big-dataset.tar")

DATA_DIR     = r"C:\Users\Edward\Dropbox\bats-code\data"

BAT_PROB_THRESH   = 0.6
SAVE_EVERY_N      = 1350   # save an example frame every N frames
CHANNEL           = 2      # blue channel (index 2 in RGB)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONTH_ABBR_NUM = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05",
                  "jun":"06","jul":"07","aug":"08","sep":"09","oct":"10",
                  "nov":"11","dec":"12"}

def canonical_date(folder_name):
    """'221101 Bat Count' or '221101BatCount' -> '20221101'. Others unchanged."""
    m = re.match(r"^(\d{2})(\d{2})(\d{2})\s*(?:Bat\s*Count)?$",
                 folder_name.strip(), re.IGNORECASE)
    if m and re.search(r"Bat", folder_name, re.IGNORECASE):
        return f"20{m.group(1)}{m.group(2)}{m.group(3)}"
    # Pure 6-digit: 221101 -> 20221101
    if re.fullmatch(r"\d{6}", folder_name.strip()):
        return f"20{folder_name.strip()}"
    return folder_name


def find_videos(camera_path):
    vids = []
    for ext in ("*.MP4", "*.mp4", "*.MOV", "*.mov", "*.AVI", "*.avi"):
        vids.extend(glob.glob(os.path.join(camera_path, ext)))
    return sorted(vids)


def needs_inference(output_cam_path):
    return not (os.path.exists(os.path.join(output_cam_path, "centers.npy"))
                or os.path.exists(os.path.join(output_cam_path, "p_centers.npy")))


def discover_todo(cameras_root, output_root, year_filter=None):
    """
    Return list of {date, camera, video_path, output_path} dicts for
    all camera-days that have videos but no centers.npy yet.
    """
    todo = []
    if not os.path.isdir(cameras_root):
        print(f"ERROR: cameras root not found: {cameras_root}")
        return todo

    for date_dir in sorted(os.listdir(cameras_root)):
        date_path = os.path.join(cameras_root, date_dir)
        if not os.path.isdir(date_path):
            continue
        date_key = canonical_date(date_dir)

        if year_filter and date_key[:4] not in year_filter:
            continue

        for cam_dir in sorted(os.listdir(date_path)):
            cam_path = os.path.join(date_path, cam_dir)
            if not os.path.isdir(cam_path):
                continue
            videos = find_videos(cam_path)
            if not videos:
                continue
            out_path = os.path.join(output_root, date_key, cam_dir)
            if needs_inference(out_path):
                todo.append({
                    "date": date_key,
                    "camera": cam_dir,
                    "video_path": cam_path,
                    "videos": videos,
                    "output_path": out_path,
                })
    return todo


# ---------------------------------------------------------------------------
# Model and dataset (lazy-loaded at runtime to avoid importing torch globally)
# ---------------------------------------------------------------------------

def load_model(model_file):
    import torch
    from bat_seg_models import UNETTraditional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETTraditional(1, 2, should_pad=False)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.train(False)
    return model, device


def make_augmentor(mean, std, channel):
    from frame_augmentors import (MaskCompose, Mask3dto2d, MaskToTensor,
                                  MaskNormalize)
    return MaskCompose([
        Mask3dto2d(channel_to_use=channel),
        MaskToTensor(),
        MaskNormalize(mean[channel] / 255, std[channel] / 255),
    ])


class BatVideoReader:
    """Iterate frames from a list of video files, skipping too-dark frames."""
    def __init__(self, video_files, augmentor, max_bad_reads=300):
        self.video_files  = video_files
        self.augmentor    = augmentor
        self.max_bad_reads = max_bad_reads

    def __iter__(self):
        for vf in self.video_files:
            cap = cv2.VideoCapture(vf)
            bad = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    bad += 1
                    if bad > self.max_bad_reads:
                        break
                    continue
                bad = 0
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame[2:-2, 2:-2]       # trim 2px each edge
                if np.mean(frame[::50, ::50, 2]) < 5:
                    break                       # too dark → end of bat flight
                sample = self.augmentor({"image": frame})
                yield sample, frame
            cap.release()


def run_one_camera(entry, model, device, mean, std, channel,
                   bat_prob_thresh, save_every_n, dry_run):
    import torch
    import torch.utils.data as data
    import bat_functions

    date    = entry["date"]
    camera  = entry["camera"]
    out_dir = entry["output_path"]

    print(f"\n  {date}/{camera}")
    print(f"    {len(entry['videos'])} video file(s)  →  {out_dir}")

    if dry_run:
        return True

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "example-frames"), exist_ok=True)

    augmentor = make_augmentor(mean, std, channel)
    reader    = BatVideoReader(entry["videos"], augmentor)

    centers_list, sizes_list, rects_list, contours_list = [], [], [], []
    num_frames = 0
    t0 = time.time()

    for sample, raw_frame in reader:
        im_tensor = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(im_tensor)
            mask   = (output[0, 1].cpu().numpy() > np.log(bat_prob_thresh)).astype(np.uint8)

        centers, areas, contours, _, _, rects = bat_functions.get_blob_info(mask)
        centers_list.append(centers)
        sizes_list.append(areas)
        rects_list.append(rects)
        contours_list.append(contours)

        if save_every_n and num_frames % save_every_n == 0:
            im_name = f"{date}_{camera}_obs-ind_{num_frames}.jpg"
            im_save = raw_frame[:, :, channel]   # single-channel for display
            cv2.imwrite(os.path.join(out_dir, "example-frames", im_name), im_save)

        num_frames += 1
        if num_frames % 10_000 == 0:
            elapsed = time.time() - t0
            fps = num_frames / elapsed
            print(f"    {num_frames:,} frames  ({fps:.0f} fps)", flush=True)

    print(f"    Done: {num_frames:,} frames in {time.time()-t0:.0f}s")

    # Save contours split across ~15 files
    num_files = 15
    chunk = max(1, len(contours_list) // num_files)
    for i in range(0, len(contours_list), chunk):
        file_num = i // chunk
        chunk_contours = []
        for cs in contours_list[i: i + chunk]:
            compressed = [np.squeeze(cv2.approxPolyDP(c, 0.1, closed=True))
                          for c in cs]
            chunk_contours.append(compressed)
        fname = os.path.join(out_dir, f"contours-compressed-{file_num:02d}.npy")
        np.save(fname, np.array(chunk_contours, dtype=object))

    for name, data_list in [("centers", centers_list),
                             ("size",    sizes_list),
                             ("rects",   rects_list)]:
        with open(os.path.join(out_dir, f"{name}.npy"), "wb") as f:
            pickle.dump(data_list, f)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated bat inference")
    parser.add_argument("--cameras-root", default=CAMERAS_ROOT,
                        help="Root folder containing date/camera/videos structure")
    parser.add_argument("--output-root",  default=OUTPUT_ROOT,
                        help="Root folder for processed output")
    parser.add_argument("--model-file",   default=MODEL_FILE)
    parser.add_argument("--data-dir",     default=DATA_DIR,
                        help="Folder containing mean.npy and std.npy")
    parser.add_argument("--year", nargs="+", default=None,
                        help="Limit to specific year(s), e.g. --year 2022")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try: sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError: pass

    print("\n" + "="*60)
    print("  Kasanka — Bat Inference")
    print("="*60)
    print(f"  Cameras root : {args.cameras_root}")
    print(f"  Output root  : {args.output_root}")
    print(f"  Model        : {os.path.basename(args.model_file)}")
    if args.dry_run:
        print("  DRY RUN — no files will be written")
    print()

    todo = discover_todo(args.cameras_root, args.output_root, args.year)

    if not todo:
        print("  Nothing to do — all cameras already have detections.")
        return

    print(f"  {len(todo)} camera-days need inference:\n")
    for e in todo:
        n = len(e["videos"])
        print(f"    {e['date']}/{e['camera']}  ({n} video{'s' if n>1 else ''})")

    if args.dry_run:
        return

    # Load model and normalization stats
    print("\n  Loading model ...", end="", flush=True)
    model, device = load_model(args.model_file)
    mean = np.load(os.path.join(args.data_dir, "mean.npy"))
    std  = np.load(os.path.join(args.data_dir, "std.npy"))
    print(f" done  (device: {device})")

    done, failed = 0, 0
    for entry in todo:
        try:
            run_one_camera(entry, model, device, mean, std, CHANNEL,
                           BAT_PROB_THRESH, SAVE_EVERY_N, args.dry_run)
            done += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()
            failed += 1

    print(f"\n  Inference complete: {done} done, {failed} failed.")
    print("  Next step: run_tracking.py")


if __name__ == "__main__":
    main()
