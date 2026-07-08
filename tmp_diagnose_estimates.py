"""
Check how vary_center affects the multiplier and total estimate.
Compares fixed center vs sampled center for a representative 2019 and 2020 date.
"""
import sys, os, glob, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import cv2, rasterio, utm as utm_lib
sys.path.insert(0, os.path.dirname(__file__))
import bat_functions as bf

OBS_ROOT  = r"\\10.0.16.7\grpdechmann\Postdoc-EdwardHurme\Eidolon_helvum\kasanka-bats\observations"
WING_FILE = "combined_wing_validation_info.csv"

FRAME_WIDTH = 2704 - (2 * 48)
WINGSPAN    = 0.8
DARK_PARAMS = [1.57454778e+01, 9.37398964e-01, 7.18914388e-02, -1.27575036e-04]

FOREST_BORDER = [[-12.585957,30.242762],[-12.586763,30.246229],
                 [-12.589182,30.245566],[-12.587557,30.241598]]
MAP_FILE = "kasanka-utm.tiff"

CAMERA_LOCS = {
    "FibweParking": [-12.5903393,30.2525047], "FibweParking2": [-12.5903393,30.2525047],
    "Fibwe Parking": [-12.5903393,30.2525047],
    "Chinyangali": [-12.5851284,30.245529], "Chinangali": [-12.5851284,30.245529],
    "Chyniangale": [-12.5851284,30.245529], "Chiniangale": [-12.5851284,30.245529],
    "BBC": [-12.5863538,30.2484985], "Sunset": [-12.585784,30.240003],
    "NotChinyangali": [-12.5849206,30.2436135], "Not Chinyangali": [-12.5849206,30.2436135],
    "NotChyniangale": [-12.5849206,30.2436135], "Not Chinangali": [-12.5849206,30.2436135],
    "Not Chingyangali": [-12.5849206,30.2436135],
    "MusolaParking": [-12.58787,30.2401], "MusoleParking": [-12.58787,30.2401],
    "Musola Parking": [-12.58787,30.2401], "Musole Parking": [-12.58787,30.2401],
    "MusolaPath": [-12.589544,30.242488], "MusolePath": [-12.589544,30.242488],
    "Musola Path": [-12.589544,30.242488], "MusolePath2": [-12.589544,30.242488],
    "Puku": [-12.584838,30.24137],
    "FibwePublic": [-12.592537,30.2515924], "Fibwe Public": [-12.592537,30.2515924],
    "Fibwe public": [-12.592537,30.2515924],
    "MusolaTower": [-12.589434,30.244736], "MusoleTower": [-12.589434,30.244736],
    "Musola Tower": [-12.589434,30.244736],
}
ALL_CAM_UTMS = bf.latlong_dict_to_utm(CAMERA_LOCS)
wing_kdes, darkness_bins = bf.get_wing_correction_distributions(
    WING_FILE, num_darkness_bins=4, kde_bw_scale=0.25, should_plot=False)

# Build forest mask
forest_utms = np.array([list(utm_lib.from_latlon(*ll)[:2]) for ll in FOREST_BORDER])
forest_dataset = rasterio.open(MAP_FILE)
w = int(abs(forest_dataset.bounds.left - forest_dataset.bounds.right))
h = int(abs(forest_dataset.bounds.top  - forest_dataset.bounds.bottom))
area = np.zeros((h, w), dtype=np.uint8)
norm = forest_utms.copy()
norm[:,0] -= forest_dataset.bounds.left
norm[:,1] -= forest_dataset.bounds.bottom
forest_mask = cv2.drawContours(area, [norm.astype(np.int32)], -1, 255, -1)

def load_date(date_str):
    obs = {}
    for f in glob.glob(os.path.join(OBS_ROOT, date_str, f"{date_str}-observation-*.npy")):
        m = re.match(r"^.+-observation-(.+)\.npy$", os.path.basename(f))
        if m:
            try: obs[m.group(1)] = np.load(f, allow_pickle=True).item()
            except: pass
    return obs

def estimate_with_center(day_obs, center):
    cam_utms = bf.get_camera_locations(day_obs, ALL_CAM_UTMS, exclude=True)
    if not cam_utms: return 0, {}
    dists     = bf.get_camera_distances(cam_utms, center)
    fractions = bf.get_camera_fractions(cam_utms, center, jitter=False)
    total = 0
    details = {}
    for cam, obs in day_obs.items():
        if cam not in cam_utms or len(obs['mean_wing']) == 0: continue
        corr, _ = bf.get_kde_samples(obs, wing_kdes, darkness_bins)
        biased = np.maximum(bf.correct_wingspan(obs['mean_wing'], corr), 2)
        obs['multiplier']     = bf.combined_bat_multiplier(FRAME_WIDTH, WINGSPAN, biased, dists[cam])
        obs['fraction_total'] = fractions[cam]
        accum = bf.get_bat_accumulation(obs['frames'], obs, DARK_PARAMS)
        details[cam] = (dists[cam], float(np.mean(obs['multiplier'])), accum[-1])
        total += accum[-1]
    return total, details

FIXED_CENTER = np.array([200450, 8606950])

print("=" * 70)
print("EFFECT OF vary_center ON ESTIMATES")
print("Fixed center vs 20 random forest center samples")
print("=" * 70)

for date in ["27-Nov-2019", "3-Nov-2020", "17-Nov-2020"]:
    day_obs = load_date(date)
    if not day_obs:
        print(f"\n{date}: no obs"); continue

    fixed_total, fixed_det = estimate_with_center(day_obs, FIXED_CENTER)
    print(f"\n{date}  fixed center → {int(fixed_total):>10,}")
    print(f"  {'Camera':<28} {'Dist':>7} {'Mult':>7} {'Contrib':>10}")
    for cam, (d, m, c) in sorted(fixed_det.items(), key=lambda x: -abs(x[1][2])):
        print(f"  {cam:<28} {d:>7.0f} {m:>7.1f} {c:>10,.0f}")

    # Sample 20 random centers
    random_totals = []
    for _ in range(500):
        center = bf.get_random_utm_in_mask(forest_mask, forest_dataset)
        t, _ = estimate_with_center(day_obs, center)
        random_totals.append(t)
    random_totals = [t for t in random_totals if t > 0]
    print(f"  vary_center (500 samples): "
          f"median={int(np.median(random_totals)):>10,}  "
          f"p5={int(np.percentile(random_totals,5)):>10,}  "
          f"p95={int(np.percentile(random_totals,95)):>10,}  "
          f"max={int(np.max(random_totals)):>10,}")
