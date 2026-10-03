import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_SESSIONS = list(range(1, 10))
DEFAULT_R2_THRESHOLD = 50.0
DEFAULT_OUTPUT = "roi_voxel_counts_r2_50.csv"

# Uses the ROI IDs provided in the request.
ROI_MAP = {
    "V1": [1],
    "V2": [2],
    "V3": [3],
    "IT": [132, 133, 134, 136, 138, 177],
}

ATLAS_PATH = "rois/sub_01/HCP_MMP1.nii.gz"
R2_PATH_TEMPLATE = "betas/sub_01/R2_session{session:02d}.nii.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report voxel counts for selected ROIs after average R2 thresholding."
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=DEFAULT_SESSIONS,
        help="Session numbers to include when averaging R2 maps.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=DEFAULT_R2_THRESHOLD,
        help="Average R2 threshold used to retain voxels.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV file path for the voxel count report.",
    )
    return parser.parse_args()


def load_atlas_data(atlas_path):
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    return nib.load(atlas_path).get_fdata().astype(int)


def load_average_r2(sessions):
    r2_maps = []
    loaded_sessions = []

    for session in sessions:
        r2_path = R2_PATH_TEMPLATE.format(session=session)
        if not os.path.exists(r2_path):
            print(f"  -> Skipping missing R2 map for session {session}: {r2_path}")
            continue

        r2_maps.append(nib.load(r2_path).get_fdata())
        loaded_sessions.append(session)

    if not r2_maps:
        raise FileNotFoundError("No R2 maps were found for the requested sessions.")

    avg_r2 = np.mean(np.stack(r2_maps, axis=-1), axis=-1)
    return avg_r2, loaded_sessions


def summarize_roi(atlas_data, avg_r2, roi_name, roi_ids, r2_threshold):
    anatomical_mask = np.isin(atlas_data, roi_ids)
    thresholded_mask = anatomical_mask & (avg_r2 >= r2_threshold)

    anatomical_count = int(np.sum(anatomical_mask))
    thresholded_count = int(np.sum(thresholded_mask))
    retained_fraction = thresholded_count / anatomical_count if anatomical_count else 0.0

    return {
        "roi_name": roi_name,
        "roi_ids": ",".join(str(roi_id) for roi_id in roi_ids),
        "anatomical_voxels": anatomical_count,
        "r2_thresholded_voxels": thresholded_count,
        "retained_fraction": retained_fraction,
        "r2_threshold": r2_threshold,
    }


def main():
    args = parse_args()

    print("=== ROI Voxel Count Report ===")
    print(f"Requested sessions: {args.sessions}")
    print(f"R2 threshold: {args.r2_threshold}")

    atlas_data = load_atlas_data(ATLAS_PATH)
    avg_r2, loaded_sessions = load_average_r2(args.sessions)

    print(f"Loaded R2 sessions: {loaded_sessions}")
    print("")

    rows = []
    for roi_name, roi_ids in ROI_MAP.items():
        row = summarize_roi(atlas_data, avg_r2, roi_name, roi_ids, args.r2_threshold)
        rows.append(row)
        print(
            f"{roi_name:<3} | IDs: {row['roi_ids']:<23} | "
            f"Anatomical: {row['anatomical_voxels']:<6} | "
            f"After R2 threshold: {row['r2_thresholded_voxels']:<6} | "
            f"Retained: {row['retained_fraction']:.3f}"
        )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(args.output, index=False)

    print("")
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
