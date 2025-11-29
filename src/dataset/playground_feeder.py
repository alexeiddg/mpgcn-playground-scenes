import logging
import os
import pickle
import re
import numpy as np
from torch.utils.data import Dataset

from .utils import graph_processing, multi_input
from .augment import SkeletonAugmenter


AREA_TO_ID = {
    "columpios": 0,
    "unknown": 1,
}

STREAM_ALIAS_MAP = {
    "JOINT": "J",
    "JOINT-MOTION": "JM",
    "BONE": "B",
    "BONE-MOTION": "BM",
}


def extract_area_name(clip_name):
    m = re.search(r"(hundidocam\d+|columpios[_]?cam\d+|columpios_tierra)", clip_name)
    if not m:
        return "unknown"
    token = m.group(1)
    if token.startswith("columpios"):
        return "columpios"
    # hundido and anything else collapse to unknown
    return "unknown"


class Playground_Feeder(Dataset):
    def __init__(
            self,
            phase,
            graph,
            root_folder,
            inputs,
            debug,
            augment=True,
            object_folder=None,
            window=None,
            processing='default',
            person_id=None,
            input_dims=3,
            num_frame=48,
            num_object=0,
            stream_type=None,
            **kwargs,
    ):
        self.phase = phase
        self.inputs = inputs
        self.processing = processing
        self.debug = debug
        self.augment = augment
        self.input_dims = input_dims
        self.window = window or [0, num_frame]
        self.num_object = num_object
        self.augmenter = SkeletonAugmenter(enable=(phase == "train") and self.augment)

        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person

        self.root_folder = root_folder
        self.object_folder = object_folder or root_folder
        self.stream_type = stream_type.upper() if stream_type else None
        self.single_stream = self.stream_type is not None
        self.stream_order = self._determine_stream_order()
        self.num_input = 1 if self.single_stream else len(self.stream_order)

        data_path = os.path.join(root_folder, f'{phase}_data.npy')
        label_path = os.path.join(root_folder, f'{phase}_label.pkl')
        object_path = os.path.join(self.object_folder, f'{phase}_object_data.npy')

        try:
            logging.info('Loading {} pose data from {}'.format(phase, data_path))
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)
            if self.num_object:
                logging.info('Loading {} object data from {}'.format(phase, object_path))
                self.object_data = np.load(object_path, mmap_mode='r')
            else:
                self.object_data = None
            area_path = os.path.join(self.object_folder, f'{phase}_area.npy')
            if os.path.exists(area_path):
                self.area_ids = np.load(area_path, mmap_mode='r')
                if len(self.area_ids) != len(self.label):
                    raise ValueError('Area id count does not match labels for phase {}'.format(phase))
            else:
                self.area_ids = None
        except Exception as exc:
            logging.error('Error loading dataset splits: {}'.format(exc))
            raise

        labels_array = np.array([lbl for lbl, _ in self.label], dtype=int)
        if labels_array.min() < 0 or labels_array.max() >= 3:
            raise ValueError('Unexpected label ids found; expected only 0..2 after filtering.')

        if person_id is None:
            default_person = min(self.data.shape[2], graph.num_person)
            self.person_indices = list(range(default_person))
        else:
            self.person_indices = person_id
        self.M = len(self.person_indices)
        self.person_indices = np.array(self.person_indices)

        if self.debug:
            self.data = self.data[:64]
            self.label = self.label[:64]
            if self.object_data is not None:
                self.object_data = self.object_data[:64]

        self.datashape = self.get_datashape()

    def __len__(self):
        return len(self.label)

    def _ensure_temporal_consistency(self, pose_data, object_clip=None):
        """Ensure data matches the expected temporal window after augmentation."""
        start, end = self.window
        expected_frames = end - start

        current_frames = pose_data.shape[0]

        if current_frames > expected_frames:
            # Trim excess frames
            pose_data = pose_data[:expected_frames]
            if object_clip is not None:
                object_clip = object_clip[:expected_frames]
        elif current_frames < expected_frames:
            # Pad with repetition of last frame
            pad_frames = expected_frames - current_frames
            last_pose = np.repeat(pose_data[-1:], pad_frames, axis=0)
            pose_data = np.concatenate([pose_data, last_pose], axis=0)

            if object_clip is not None:
                last_obj = np.repeat(object_clip[-1:], pad_frames, axis=0)
                object_clip = np.concatenate([object_clip, last_obj], axis=0)

        return pose_data, object_clip

    def __getitem__(self, idx):
        sample_idx = idx
        pose_data = self.data[sample_idx].copy()  # (T, M, V, C)

        if self.object_data is not None:
            object_clip = self.object_data[sample_idx].copy()  # (T, O, C)

            # Apply augmentation only during training
            if self.augmenter.enable:
                pose_data, object_clip = self.augmenter(pose_data, object_clip,
                                                        T_target=self.window[1] - self.window[0])
                # Ensure temporal consistency after augmentation
                pose_data, object_clip = self._ensure_temporal_consistency(pose_data, object_clip)

            # Convert to model input format
            pose_data = pose_data.transpose(3, 0, 2, 1)  # (C, T, V, M)
            pose_data = pose_data[:, :, :, self.person_indices]
            start, end = self.window
            pose_data = pose_data[:, start:end, :, :]

            object_clip = object_clip.transpose(2, 0, 1)  # (C, T, O)
            max_obj = min(self.num_object, object_clip.shape[2])
            object_clip = object_clip[:, start:end, :max_obj]
            object_clip = np.expand_dims(object_clip, axis=-1)  # (C, T, O, 1)
            object_clip = np.tile(object_clip, (1, 1, 1, self.M))
            pose_data = np.concatenate((pose_data, object_clip), axis=2)
        else:
            # No object data, just process pose data
            if self.augmenter.enable:
                # For pose-only data, we need to handle augmentation differently
                # Since augmenter expects both pose and object data, we create dummy object data
                dummy_object = np.zeros((pose_data.shape[0], 1, pose_data.shape[-1]))
                pose_data, _ = self.augmenter(pose_data, dummy_object, T_target=self.window[1] - self.window[0])
                pose_data, _ = self._ensure_temporal_consistency(pose_data, None)

            pose_data = pose_data.transpose(3, 0, 2, 1)  # (C, T, V, M)
            pose_data = pose_data[:, :, :, self.person_indices]
            start, end = self.window
            pose_data = pose_data[:, start:end, :, :]

        data = graph_processing(pose_data, self.graph, self.processing)
        data_new = multi_input(data, self.conn, self.inputs, self.center)

        if self.single_stream:
            stream_map = {"J": 0, "JM": 1, "B": 2, "BM": 3}
            stream_idx = stream_map.get(self.stream_type)
            if stream_idx is None:
                raise ValueError(f"stream_type '{self.stream_type}' not in '{self.stream_order}'")
            data_new = data_new[stream_idx]
            if data_new.ndim == len(self.datashape) and data_new.shape[-1] == 1:
                data_new = data_new.squeeze(-1)

        # More robust shape checking with detailed error information
        expected_shape = self.datashape[1:] if self.single_stream else self.datashape
        actual_shape = list(data_new.shape)

        if actual_shape != expected_shape:
            logging.error('Shape mismatch - Expected: {}, Got: {}'.format(expected_shape, actual_shape))
            logging.error('Debug info - Window: {}, M: {}, inputs: {}, phase: {}'.format(
                self.window, self.M, self.inputs, self.phase))

            # Attempt to fix minor shape mismatches
            if len(actual_shape) == len(expected_shape):
                # Try to pad/trim each dimension to match expected shape
                fixed_data = data_new
                for dim_idx, (actual_size, expected_size) in enumerate(zip(actual_shape, expected_shape)):
                    if actual_size != expected_size:
                        if actual_size < expected_size:
                            # Pad dimension
                            pad_width = [(0, 0)] * len(actual_shape)
                            pad_width[dim_idx] = (0, expected_size - actual_size)
                            fixed_data = np.pad(fixed_data, pad_width, mode='constant', constant_values=0)
                        elif actual_size > expected_size:
                            # Trim dimension
                            slices = [slice(None)] * len(actual_shape)
                            slices[dim_idx] = slice(0, expected_size)
                            fixed_data = fixed_data[tuple(slices)]

                data_new = fixed_data
                logging.warning('Shape fixed from {} to {}'.format(actual_shape, list(data_new.shape)))
            else:
                raise ValueError('Critical shape mismatch that cannot be automatically fixed.')

        label, name = self.label[sample_idx]
        if self.area_ids is not None:
            area_id = int(self.area_ids[sample_idx])
        else:
            area_name = extract_area_name(name)
            area_id = AREA_TO_ID.get(area_name, AREA_TO_ID["unknown"])
        return data_new, label, name, area_id

    def _determine_stream_order(self):
        if isinstance(self.inputs, str) and self.inputs.upper() == "JVBM":
            return ["J", "JM", "B", "BM"]
        if isinstance(self.inputs, (list, tuple)):
            order = []
            for item in self.inputs:
                key = str(item).upper()
                order.append(STREAM_ALIAS_MAP.get(key, key))
            return order
        if isinstance(self.inputs, str):
            key = self.inputs.upper()
            return [STREAM_ALIAS_MAP.get(key, key)]
        return ["J"]

    def get_datashape(self):
        I = self.num_input
        C = self.input_dims if self.inputs in [
            'joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims * 2
        T = len(range(*self.window))
        V = self.num_node
        M = self.M // self.num_person
        return [I, C, T, V, M]
