import os
import pickle
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold

from .select_top_m_people import select_top_m_people

class Playground_Reader:
    def __init__(self,
                 dataset_root_folder,
                 out_folder,
                 label_csv_path,
                 num_frame=48,
                 max_person=4,
                 n_obj_max=None,
                 max_joint=17,
                 max_channel=3,
                 split_strategy="auto",
                 **kwargs):

        self.max_channel = max_channel
        self.num_frame = num_frame
        self.max_joint = max_joint
        self.max_person = max_person
        self.n_obj_max = n_obj_max

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder

        df = pd.read_csv(label_csv_path)
        df["clip_name"] = df["trimmed_name"].str.replace(".mp4", "", regex=False)
        self.df = df

        # Drop sparse/unused categories before building label maps
        excluded_labels = {"Negative_Contact", "Play_Object_Risk", "No_Activity", "Adult_Assisting"}
        df = df[~df["activity_label"].isin(excluded_labels)].copy()
        self.excluded_labels = excluded_labels - {"Negative_Contact"}
        self.label_map = dict(zip(df["clip_name"], df["activity_label"]))

        self.class2idx = {
            'Transit': 0,
            'Social_People': 1,
            'Play_Object_Normal': 2,
        }

        if split_strategy == "auto":
            self.folds = self._stratified_split(df)
        else:
            raise ValueError("Only stratified split is supported.")

    def _stratified_split(self, df):
        """Create repeated stratified K-fold splits and persist clip lists for reproducibility."""

        clip_names = df["clip_name"].tolist()
        labels = df["activity_label"].map(self.class2idx).tolist()

        # Basic sanity check to ensure stratification can work
        class_counts = pd.Series(labels).value_counts()
        min_count = class_counts.min()
        n_splits = 5
        if min_count < n_splits:
            raise ValueError(
                f"Not enough samples per class for {n_splits} folds. "
                f"Lowest class count: {min_count}"
            )

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
        folds = []
        summary = {}

        os.makedirs(self.out_folder, exist_ok=True)

        for fold_id, (train_idx, eval_idx) in enumerate(rskf.split(clip_names, labels)):
            train_clips = [clip_names[i] for i in train_idx]
            eval_clips = [clip_names[i] for i in eval_idx]

            # Save fold membership for downstream reproducibility
            with open(os.path.join(self.out_folder, f"fold_{fold_id:02d}_train.txt"), "w") as f:
                f.write("\n".join(train_clips))
            with open(os.path.join(self.out_folder, f"fold_{fold_id:02d}_eval.txt"), "w") as f:
                f.write("\n".join(eval_clips))

            fold_map = {name: "train" for name in train_clips}
            fold_map.update({name: "eval" for name in eval_clips})
            folds.append(fold_map)

            # Track counts per class for both splits
            train_labels = [labels[i] for i in train_idx]
            eval_labels = [labels[i] for i in eval_idx]
            counts = {
                "train": self._count_labels(train_labels),
                "eval": self._count_labels(eval_labels),
            }

            missing_classes = [
                lbl for lbl, idx in self.class2idx.items()
                if counts["train"].get(lbl, 0) == 0 or counts["eval"].get(lbl, 0) == 0
            ]
            if missing_classes:
                raise ValueError(
                    f"Fold {fold_id} is missing classes in train/eval: {missing_classes}"
                )

            summary[f"fold_{fold_id:02d}"] = counts

        with open(os.path.join(self.out_folder, "fold_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return folds

    def _count_labels(self, label_ids):
        inv_map = {v: k for k, v in self.class2idx.items()}
        counts = {name: 0 for name in self.class2idx.keys()}
        for lid in label_ids:
            counts[inv_map[lid]] = counts.get(inv_map[lid], 0) + 1
        return counts

    def _get_max_object_count(self, phase_dir, obj_files):
        n_obj_max = 0
        for f in tqdm(obj_files, desc=f"Scanning object files in {phase_dir}"):
            obj_path = os.path.join(phase_dir, f)
            try:
                obj = np.load(obj_path)
                n_obj_max = max(n_obj_max, obj.shape[1])
            except Exception as e:
                print(f"Error reading {f}: {e}")
        print(f"Max number of object nodes in {phase_dir}: {n_obj_max}")
        return n_obj_max

    def _select_object_files(self, base_name, object_map):
        entry = object_map.get(base_name)
        if not entry:
            return []
        if entry['new']:
            return entry['new']
        return entry['legacy']

    def _normalize_object_array(self, obj):
        obj = np.asarray(obj)
        if obj.ndim == 1:
            obj = obj[:, np.newaxis, np.newaxis]
        elif obj.ndim == 2:
            if obj.shape[1] == self.max_channel:
                obj = obj[:, np.newaxis, :]
            else:
                obj = obj[:, :, np.newaxis]
        elif obj.ndim > 3:
            obj = obj.reshape(obj.shape[0], -1, obj.shape[-1])

        if obj.shape[-1] < self.max_channel:
            pad = np.zeros((*obj.shape[:-1], self.max_channel - obj.shape[-1]), dtype=obj.dtype)
            obj = np.concatenate([obj, pad], axis=-1)
        elif obj.shape[-1] > self.max_channel:
            obj = obj[..., :self.max_channel]
        return obj.astype(np.float32)

    def gendata(self, fold_id, split_map):
        res = {
            "train": {"pose": [], "obj": [], "labels": []},
            "eval": {"pose": [], "obj": [], "labels": []},
        }

        all_files = os.listdir(self.dataset_root_folder)
        pose_files = []
        object_map = defaultdict(lambda: {'new': [], 'legacy': []})

        for fname in all_files:
            if not fname.endswith(".npy"):
                continue
            if "_object_data" in fname:
                base = fname.split("_object_data")[0]
                object_map[base]['new'].append(fname)
            elif "_object" in fname:
                base = fname.split("_object")[0]
                object_map[base]['legacy'].append(fname)
            elif fname.endswith("_data.npy"):
                pose_files.append(fname)

        print(f"\n=== FILE DISCOVERY ===")
        print(f"Found {len(pose_files)} pose files in {self.dataset_root_folder}")

        matched_obj_files = []
        unmatched = []
        for pose_file in pose_files:
            base = pose_file.replace("_data.npy", "")
            selected = self._select_object_files(base, object_map)
            if selected:
                matched_obj_files.extend(selected)
            else:
                unmatched.append(pose_file)

        if unmatched:
            print(f"⚠️ {len(unmatched)} pose files have no matching object files (legacy or new).")
        else:
            print("✅ All pose files have at least one matching object file.")

        matched_obj_files = sorted(set(matched_obj_files))

        print(f"\nGenerating fold {fold_id:02d} splits...")

        n_obj_max = self._get_max_object_count(self.dataset_root_folder, matched_obj_files) \
            if self.n_obj_max is None else self.n_obj_max
        print(f"Using n_obj_max = {n_obj_max}")

        for pose_file in tqdm(pose_files):
            base_name = pose_file.replace("_data.npy", "")
            phase = split_map.get(base_name)
            if phase not in {"train", "eval"}:
                continue

            pose_path = os.path.join(self.dataset_root_folder, pose_file)
            obj_candidates = self._select_object_files(base_name, object_map)
            if not obj_candidates:
                print(f"Missing object file for {base_name}, skipping.")
                continue

            pose_data = np.load(pose_path)
            obj_arrays = []
            for obj_file in obj_candidates:
                obj_path = os.path.join(self.dataset_root_folder, obj_file)
                obj_arr = np.load(obj_path)
                obj_arr = self._normalize_object_array(obj_arr)
                obj_arrays.append(obj_arr)

            if not obj_arrays:
                print(f"Unable to load object data for {base_name}, skipping.")
                continue
            min_frames = min(arr.shape[0] for arr in obj_arrays)
            obj_arrays = [arr[:min_frames] for arr in obj_arrays]
            obj_data = np.concatenate(obj_arrays, axis=1) if len(obj_arrays) > 1 else obj_arrays[0]

            label_name = self.label_map.get(base_name)
            if label_name is None:
                print(f"Skipping {base_name}: missing label or excluded class.")
                continue

            label_id = self.class2idx.get(label_name)
            if label_id is None:
                print(f"Skipping {base_name}: label '{label_name}' not in target classes.")
                continue

            # Trim or pad to num_frame
            T = min(self.num_frame, pose_data.shape[0], obj_data.shape[0])
            pose_data = pose_data[:T]
            obj_data = obj_data[:T]

            pose_data = select_top_m_people(pose_data, M_target=self.max_person)

            if pose_data.shape[0] < self.num_frame:
                pad_T = self.num_frame - pose_data.shape[0]
                pad_pose = np.zeros((pad_T, self.max_person, self.max_joint, self.max_channel))
                pad_obj = np.zeros((pad_T, obj_data.shape[1], self.max_channel))
                pose_data = np.concatenate([pose_data, pad_pose], axis=0)
                obj_data = np.concatenate([obj_data, pad_obj], axis=0)

            n_obj = obj_data.shape[1]
            if n_obj < n_obj_max:
                pad_obj = np.zeros((self.num_frame, n_obj_max - n_obj, self.max_channel))
                obj_data = np.concatenate([obj_data, pad_obj], axis=1)
            elif n_obj > n_obj_max:
                obj_data = obj_data[:, :n_obj_max, :]

            if obj_data.shape != (self.num_frame, n_obj_max, self.max_channel):
                fixed_obj_data = np.zeros((self.num_frame, n_obj_max, self.max_channel))
                t_range = min(obj_data.shape[0], self.num_frame)
                n_range = min(obj_data.shape[1], n_obj_max)
                c_range = min(obj_data.shape[2], self.max_channel)
                fixed_obj_data[:t_range, :n_range, :c_range] = obj_data[:t_range, :n_range, :c_range]
                obj_data = fixed_obj_data

            res[phase]["pose"].append(pose_data)
            res[phase]["obj"].append(obj_data)
            res[phase]["labels"].append([label_id, base_name])

        fold_dir = os.path.join(self.out_folder, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_counts = {}

        for phase in ["train", "eval"]:
            res_pose = res[phase]["pose"]
            res_obj = res[phase]["obj"]
            labels = res[phase]["labels"]

            if not res_pose:
                print(f"No samples found for {phase} in fold {fold_id:02d}, skipping save.")
                continue

            shapes = [r.shape for r in res_obj]
            print(f"Unique object shapes found for fold {fold_id:02d} {phase}: {set(shapes)}")

            try:
                res_pose = np.array(res_pose, dtype=np.float32)
                res_obj = np.array(res_obj, dtype=np.float32)
            except ValueError as e:
                print(f"⚠️ Error converting to numpy array: {e}")
                print("Attempting to fix inconsistent shapes...")

                expected_pose_shape = (self.num_frame, self.max_person, self.max_joint, self.max_channel)
                expected_obj_shape = (self.num_frame, n_obj_max, self.max_channel)

                for i, obj in enumerate(res_obj):
                    if obj.shape != expected_obj_shape:
                        print(f"Fixing inconsistent shape at index {i}: {obj.shape} → {expected_obj_shape}")
                        fixed_obj = np.zeros(expected_obj_shape)
                        t_range = min(obj.shape[0], expected_obj_shape[0])
                        n_range = min(obj.shape[1], expected_obj_shape[1])
                        c_range = min(obj.shape[2], expected_obj_shape[2])
                        fixed_obj[:t_range, :n_range, :c_range] = obj[:t_range, :n_range, :c_range]
                        res_obj[i] = fixed_obj

                res_pose = np.array(res_pose, dtype=np.float32)
                res_obj = np.array(res_obj, dtype=np.float32)

            np.save(os.path.join(fold_dir, f"{phase}_data.npy"), res_pose)
            np.save(os.path.join(fold_dir, f"{phase}_object_data.npy"), res_obj)
            with open(os.path.join(fold_dir, f"{phase}_label.pkl"), "wb") as f:
                pickle.dump(labels, f)

            label_ids = [lbl for lbl, _ in labels]
            fold_counts[phase] = self._count_labels(label_ids)

            print(f"{phase.capitalize()} set saved → {fold_dir}")

        return fold_counts

    def start(self):
        print("Starting dataset build from CSV with repeated stratified K-folds...")
        fold_summaries = {}
        for fold_id, split_map in enumerate(self.folds):
            counts = self.gendata(fold_id, split_map)
            fold_summaries[f"fold_{fold_id:02d}"] = counts

        summary_path = os.path.join(self.out_folder, "fold_summary_actual.json")
        with open(summary_path, "w") as f:
            json.dump(fold_summaries, f, indent=2)
        print(f"Saved fold class distribution summary → {summary_path}")
