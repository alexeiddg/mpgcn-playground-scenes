import logging
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

from .utils import graph_processing, multi_input


class Playground_Feeder(Dataset):
    def __init__(
        self,
        phase,
        graph,
        root_folder,
        inputs,
        debug,
        object_folder=None,
        window=None,
        processing='default',
        person_id=None,
        input_dims=3,
        num_frame=48,
        num_object=0,
        **kwargs,
    ):
        self.phase = phase
        self.inputs = inputs
        self.processing = processing
        self.debug = debug
        self.input_dims = input_dims
        self.window = window or [0, num_frame]
        self.num_object = num_object

        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person

        self.root_folder = root_folder
        self.object_folder = object_folder or root_folder

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
        except Exception as exc:
            logging.error('Error loading dataset splits: {}'.format(exc))
            raise

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

    def __getitem__(self, idx):
        pose_data = self.data[idx]  # (T, M, V, C)
        pose_data = pose_data.transpose(3, 0, 2, 1)  # (C, T, V, M)
        pose_data = pose_data[:, :, :, self.person_indices]
        start, end = self.window
        pose_data = pose_data[:, start:end, :, :]

        if self.object_data is not None:
            object_clip = self.object_data[idx]  # (T, O, C)
            object_clip = object_clip.transpose(2, 0, 1)  # (C, T, O)
            max_obj = min(self.num_object, object_clip.shape[2])
            object_clip = object_clip[:, start:end, :max_obj]
            object_clip = np.expand_dims(object_clip, axis=-1)  # (C, T, O, 1)
            object_clip = np.tile(object_clip, (1, 1, 1, self.M))
            pose_data = np.concatenate((pose_data, object_clip), axis=2)

        data = graph_processing(pose_data, self.graph, self.processing)
        data_new = multi_input(data, self.conn, self.inputs, self.center)

        try:
            assert list(data_new.shape) == self.datashape
        except AssertionError:
            logging.error('data_new.shape: {} vs expected {}'.format(data_new.shape, self.datashape))
            raise ValueError('Mismatch between computed tensor shape and graph definition.')

        label, name = self.label[idx]
        return data_new, label, name

    def get_datashape(self):
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims if self.inputs in [
            'joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims * 2
        T = len(range(*self.window))
        V = self.num_node
        M = self.M // self.num_person
        return [I, C, T, V, M]
