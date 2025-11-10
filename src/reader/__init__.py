import logging
from .playground_reader import Playground_Reader

__generator = {
    'playground': Playground_Reader
}

def create(args):
    dataset = args.dataset
    dataset_args = args.dataset_args
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](**dataset_args)