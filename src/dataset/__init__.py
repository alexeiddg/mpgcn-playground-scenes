import logging
import os

import numpy as np

from .graphs import Graph
from .playground_feeder import Playground_Feeder

__data_args = {
    'playground': {'class': 3, 'feeder': Playground_Feeder},
}


def _maybe_infer_num_object(kwargs):
    if kwargs.get('num_object', 0) > 0:
        return
    root = kwargs.get('object_folder') or kwargs.get('root_folder')
    if not root:
        return
    candidate_path = os.path.join(root, 'train_object_data.npy')
    if not os.path.exists(candidate_path):
        return
    try:
        arr = np.load(candidate_path, mmap_mode='r')
        kwargs['num_object'] = arr.shape[2]
    except Exception:
        logging.warning('Warning: Unable to infer num_object from {}'.format(candidate_path))


def create(dataset, **kwargs):
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except KeyError as exc:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError() from exc

    _maybe_infer_num_object(kwargs)

    graph = Graph(dataset, **kwargs)
    kwargs.pop('graph', None)

    feeders = {
        'train': data_args['feeder'](phase='train', graph=graph, **kwargs),
        'eval': data_args['feeder'](phase='eval', graph=graph, **kwargs),
    }
    data_shape = feeders['train'].datashape

    return feeders, data_shape, num_class, graph.A, graph.parts
