import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

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

        self.label_map = dict(zip(df["clip_name"], df["activity_label"]))

        self.class2idx = {
            'Transit': 0,
            'Social_People': 1,
            'Play_Object_Normal': 2,
            'Play_Object_Risk': 3,
            'Adult_Assisting': 4,
            'Negative_Contact': 5,
            'No_Activity': 6
        }

        if split_strategy == "auto":
            df["split"] = df["camera"].apply(lambda c: "eval" if "cam4" in str(c) else "train")
        self.split_map = dict(zip(df["clip_name"], df["split"]))

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

    def gendata(self, phase):
        res_pose, res_obj, labels = [], [], []

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

        print(f"\nGenerating {phase} split...")

        n_obj_max = self._get_max_object_count(self.dataset_root_folder, matched_obj_files) \
            if self.n_obj_max is None else self.n_obj_max
        print(f"Using n_obj_max = {n_obj_max}")

        for pose_file in tqdm(pose_files):
            base_name = pose_file.replace("_data.npy", "")
            if self.split_map.get(base_name, "train") != phase:
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

            label_name = self.label_map.get(base_name, "No_Activity")
            label_id = self.class2idx.get(label_name, self.class2idx["No_Activity"])

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

            res_pose.append(pose_data)
            res_obj.append(obj_data)
            labels.append([label_id, base_name])

        shapes = [r.shape for r in res_obj]
        print(f"Unique object shapes found for {phase}: {set(shapes)}")

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

        os.makedirs(self.out_folder, exist_ok=True)
        np.save(os.path.join(self.out_folder, f"{phase}_data.npy"), res_pose)
        np.save(os.path.join(self.out_folder, f"{phase}_object_data.npy"), res_obj)
        with open(os.path.join(self.out_folder, f"{phase}_label.pkl"), "wb") as f:
            pickle.dump(labels, f)

        print(f"{phase.capitalize()} set saved → {self.out_folder}")

    def start(self):
        print("Starting dataset build from CSV...")
        for phase in ["train", "eval"]:
            self.gendata(phase)
