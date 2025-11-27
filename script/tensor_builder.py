import os
import json
import yaml
import re
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Constants
# STATIC_CLASSES = ["bench", "chair", "swing", "slide"]
# DYNAMIC_CLASSES = ["dog", "sports ball", "tennis racket", "frisbee"]
STATIC_CLASSES = []
DYNAMIC_CLASSES = []
M = 6  # Max people per clip
C = 3  # x, y, conf
frame_w = 2560
frame_h = 1440

DEFAULT_MERGED_DIR = "../data/temp/merged/"
DEFAULT_OUTPUT_YAML = "../config/objects.yaml"
DEFAULT_OUTPUT_TENSORS = "../data/npy"

CAM_PAT = re.compile(r'(hundidocam\d+|columpios[_]?cam\d+|columpios_tierra)')


def extract_camera_name(filename: str) -> str:
    base = os.path.basename(filename)
    match = CAM_PAT.search(base)
    if match:
        return match.group(1)
    else:
        return "unknown_camera"


def create_objects_yaml(merged_dir=DEFAULT_MERGED_DIR, output_yaml=DEFAULT_OUTPUT_YAML):
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    camera_objects = {}

    for file in os.listdir(merged_dir):
        if not file.endswith("_merged.json"):
            continue

        cam_name = extract_camera_name(file)

        with open(os.path.join(merged_dir, file)) as f:
            data = json.load(f)

        centroids = {}
        for frame in data:
            for obj in frame.get("objects", []):
                cls = obj["class"].lower()
                if cls not in STATIC_CLASSES:
                    continue
                cx, cy = obj["centroid"]
                cx /= frame_w
                cy /= frame_h
                centroids.setdefault(cls, []).append((cx, cy))

        if centroids:
            camera_objects[cam_name] = {
                "static_objects": {
                    cls: [
                        float(np.mean([c[0] for c in coords])),
                        float(np.mean([c[1] for c in coords]))
                    ]
                    for cls, coords in centroids.items()
                }
            }

    with open(output_yaml, "w") as f:
        yaml.dump(camera_objects, f, sort_keys=False)

    print(f"Static object centroids saved to {output_yaml}")
    print(f"Cameras processed: {list(camera_objects.keys())}")

    return camera_objects


def ensure_keypoints(kpts):
    arr = np.zeros((17, 3), np.float32)
    for i, kp in enumerate(kpts[:17]):
        arr[i] = [kp.get("x", 0), kp.get("y", 0), kp.get("conf", 0)]
    return arr


def build_clip(json_path, static_cfg):
    cam = extract_camera_name(json_path)
    static_objs = static_cfg.get(cam, {}).get("static_objects", {})
    S = len(static_objs)
    D = len(DYNAMIC_CLASSES)

    with open(json_path) as f:
        frames = json.load(f)
    T = len(frames)
    skel = np.zeros((T, M, 17, C), np.float32)
    objs = np.zeros((T, S + D, C), np.float32)

    track_map = {}
    next_slot = 0

    for t, fr in enumerate(frames):
        for p in fr.get("people", []):
            tid = p.get("track_id")
            if tid not in track_map and next_slot < M:
                track_map[tid] = next_slot
                next_slot += 1
            if tid not in track_map:
                continue
            s = track_map[tid]
            k = ensure_keypoints(p["keypoints"])
            k[:, 0] /= frame_w
            k[:, 1] /= frame_h
            skel[t, s] = k

        for j, cls in enumerate(DYNAMIC_CLASSES):
            dets = [o for o in fr.get("objects", []) if o["class"] == cls]
            if not dets:
                continue
            cx = np.mean([o["centroid"][0] for o in dets]) / frame_w
            cy = np.mean([o["centroid"][1] for o in dets]) / frame_h
            conf = np.mean([o["conf"] for o in dets])
            objs[t, j] = [cx, cy, conf]

        # static objects (same every frame)
        for k, (cls, (sx, sy)) in enumerate(static_objs.items()):
            objs[t, D + k] = [sx, sy, 1.0]

    return skel, objs


def build_tensors(static_cfg, merged_dir=DEFAULT_MERGED_DIR, output_tensors=DEFAULT_OUTPUT_TENSORS):
    os.makedirs(output_tensors, exist_ok=True)

    files_processed = 0
    for f in tqdm(os.listdir(merged_dir)):
        if not f.endswith("_merged.json"):
            continue
        path = os.path.join(merged_dir, f)
        skel, objs = build_clip(path, static_cfg)

        skel_path = os.path.join(output_tensors, f.replace("_merged.json", "_data.npy"))
        obj_path = os.path.join(output_tensors, f.replace("_merged.json", "_object_data.npy"))

        np.save(skel_path, skel)
        np.save(obj_path, objs)
        files_processed += 1

    return files_processed


def validate_npy_files(output_tensors=DEFAULT_OUTPUT_TENSORS):
    npy_files = [f for f in os.listdir(output_tensors) if f.endswith(".npy")]

    if not npy_files:
        print("Error: No .npy files were created!")
        return False

    # Check pairs of files
    data_files = [f for f in npy_files if f.endswith("_data.npy")]
    object_files = [f for f in npy_files if f.endswith("_object_data.npy")]

    print(f"Created {len(data_files)} skeleton data files")
    print(f"Created {len(object_files)} object data files")

    if len(data_files) != len(object_files):
        print("Warning: Mismatch between number of skeleton and object files")

    data_bases = [f.replace("_data.npy", "") for f in data_files]
    obj_bases = [f.replace("_object_data.npy", "") for f in object_files]

    missing_obj_files = [b for b in data_bases if b not in obj_bases]
    missing_data_files = [b for b in obj_bases if b not in data_bases]

    if missing_obj_files:
        print(f"Warning: {len(missing_obj_files)} skeleton files don't have matching object files")

    if missing_data_files:
        print(f"Warning: {len(missing_data_files)} object files don't have matching skeleton files")

    # Validate sample files
    validation_results = []

    # Try to validate up to 3 sample files
    sample_count = min(3, len(data_files))
    for i in range(sample_count):
        if i >= len(data_files):
            break

        sample_path = os.path.join(output_tensors, data_files[i])
        base_name = data_files[i].replace("_data.npy", "")
        obj_file = f"{base_name}_object_data.npy"

        if obj_file not in object_files:
            print(f"Warning: No matching object file for {data_files[i]}")
            continue

        obj_sample_path = os.path.join(output_tensors, obj_file)

        try:
            pose = np.load(sample_path)
            objs = np.load(obj_sample_path)

            print(f"\nValidation sample {i+1}:")
            print(f"File: {data_files[i]}")
            print(f"Pose data shape: {pose.shape}")
            print(f"Expected shape: (frames, {M}, 17, {C})")

            # Verify dimensions
            if pose.shape[1] != M:
                print(f"Warning: Expected {M} people, but found {pose.shape[1]}")

            if pose.shape[2] != 17 or pose.shape[3] != C:
                print(f"Warning: Unexpected keypoint dimensions: {pose.shape[2:]} (expected 17, {C})")

            print(f"\nFile: {obj_file}")
            print(f"Object data shape: {objs.shape}")

            # Check if frames match
            if pose.shape[0] != objs.shape[0]:
                print(f"Warning: Frame count mismatch: {pose.shape[0]} vs {objs.shape[0]}")

            print(f"\nPose min/max: {pose.min():.4f}, {pose.max():.4f}")
            print(f"Object min/max: {objs.min():.4f}, {objs.max():.4f}")

            # Check if values are normalized (should be between 0 and 1)
            if pose.max() > 1.0 or objs.max() > 1.0:
                print(f"Warning: Some values are not normalized (> 1.0)")

            # Check for empty tensors
            if np.all(pose == 0):
                print("Warning: Pose tensor is all zeros")

            if np.all(objs == 0):
                print("Warning: Object tensor is all zeros")

            validation_results.append(True)
        except Exception as e:
            print(f"Error validating .npy files: {e}")
            validation_results.append(False)

    return any(validation_results) if validation_results else False


def parse_args():
    parser = argparse.ArgumentParser(description="Build tensors from merged files")

    parser.add_argument("--merged-dir", type=str, default=DEFAULT_MERGED_DIR,
                        help=f"Path to directory containing merged JSON files [default: {DEFAULT_MERGED_DIR}]")
    parser.add_argument("--output-yaml", type=str, default=DEFAULT_OUTPUT_YAML,
                        help=f"Path to output YAML file for static objects [default: {DEFAULT_OUTPUT_YAML}]")
    parser.add_argument("--output-tensors", type=str, default=DEFAULT_OUTPUT_TENSORS,
                        help=f"Path to output directory for .npy files [default: {DEFAULT_OUTPUT_TENSORS}]")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation of .npy files")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting tensor builder...")
    print(f"Merged directory: {args.merged_dir}")
    print(f"Output YAML: {args.output_yaml}")
    print(f"Output tensors directory: {args.output_tensors}")

    print("\n1. Creating objects.yaml...")
    static_cfg = create_objects_yaml(merged_dir=args.merged_dir, output_yaml=args.output_yaml)

    print("\n2. Building tensors...")
    files_processed = build_tensors(
        static_cfg, 
        merged_dir=args.merged_dir, 
        output_tensors=args.output_tensors
    )
    print(f"Processed {files_processed} files")

    if not args.skip_validation:
        print("\n3. Validating .npy files...")
        validation_success = validate_npy_files(output_tensors=args.output_tensors)

        if validation_success:
            print("\nTensor building completed successfully!")
        else:
            print("\nTensor building completed with warnings or errors.")
    else:
        print("\nSkipping validation as requested.")
        print("\nTensor building completed!")


if __name__ == "__main__":
    main()
