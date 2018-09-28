from collections import OrderedDict
from functools import partial
from itertools import count
from pprint import pformat
import datajoint as dj

from attorch.dataset import H5SequenceSet

from neuro_data.movies.transforms import Subsequence
from neuro_data.utils.measures import corr
from .mixins import TraceMixin
from .. import logger as log
import imageio
import io
import numpy as np
import cv2
import json
from .schema_bridge import *
from tqdm import tqdm
from scipy.signal import convolve2d
from ..utils.data import SplineMovie, FilterMixin, SplineCurve, h5cached, NaNSpline, fill_nans
from .data_schemas import MovieMultiDataset, MovieScan
from .configs import DataConfig

schema = dj.schema('neurodata_moviestats', locals())
data_schemas = dj.create_virtual_module('data_schemas', 'neurodata_movies')


@schema
class Activity(dj.Computed):
    definition = """
    # oracle computation for hollymonet data

    -> MovieMultiDataset
    -> DataConfig
    ---
    """

    class FractionActive(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        -> MovieScan.Unit
        ---
        activity           : float    # fraction of time points, the neuron had non-zero activity in the entire set
        """

    class Average(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        -> MovieScan.Unit
        ---
        avg               : float    # fraction of time points, the neuron had non-zero activity in the entire set
        """


    def make(self, key):
        log.info('Populating' + repr(key))
        # --- load data
        trainsets, trainloaders = DataConfig().load_data(key, tier=None)

        self.insert1(dict(key))
        for readout_key, loader in trainloaders.items():
            log.info('Computing activity for ' + readout_key)
            out = None
            for *_, outputs in tqdm(loader):
                outputs = outputs.numpy().squeeze()
                if out is None:
                    out = outputs
                else:
                    out = np.vstack((out, outputs))

            activity = (out > 1e-3).mean(axis=0)
            avg = out.mean(axis=0)

            unit_ids = trainsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(activity) == outputs.shape[-1], 'Neuron numbers do not add up'
            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)

            self.FractionActive().insert(
                [dict(member_key, activity=a, unit_id=u) \
                 for u, a in tqdm(zip(unit_ids, activity), total=len(unit_ids))],
                ignore_extra_fields=True)

            self.Average().insert(
                [dict(member_key, avg=a, unit_id=u) \
                 for u, a in tqdm(zip(unit_ids, avg), total=len(unit_ids))],
                ignore_extra_fields=True)


@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for hollymonet data

    -> MovieMultiDataset
    -> DataConfig
    ---
    """

    class Pearson(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        ---
        n_neurons         : int       # number of neurons averaged over
        pearson           : float     # mean test correlation
        """

    class UnitPearson(dj.Part):
        definition = """
        -> master.Pearson
        -> MovieScan.Unit
        ---
        pearson           : float     # mean test correlation
        """

    def _make_tuples(self, key):
        log.info('Populating ' + repr(key))
        # --- load data
        testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)

        self.insert1(dict(key))
        for readout_key, loader in testloaders.items():
            log.info('Computing oracle for ' + readout_key)
            oracles, data = [], []
            for inputs, *_, outputs in loader:
                inputs = inputs.numpy()
                assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
                    'Video inputs of oracle trials does not match'
                outputs = outputs.numpy()
                new_shape = (-1, outputs.shape[-1])
                r, _, n = outputs.shape  # responses X neurons
                mu = outputs.mean(axis=0, keepdims=True)
                oracle = (mu - outputs / r) * r / (r - 1)
                oracles.append(oracle.reshape(new_shape))
                data.append(outputs.reshape(new_shape))

            pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)
            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Pearson().insert1(dict(member_key, pearson=np.mean(pearson), n_neurons=len(pearson)),
                                   ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(pearson) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitPearson().insert(
                [dict(member_key, pearson=c, unit_id=u) for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
                ignore_extra_fields=True)


def load_dataset(key):
    from neuro_data.movies.data_schemas import InputResponse, Eye, Treadmill, MovieSet
    for mkey in (InputResponse & key).fetch(dj.key, order_by='animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC'):
        include_behavior = bool(Eye() * Treadmill() & mkey)
        data_names = ['inputs', 'responses'] if not include_behavior \
            else ['inputs',
                  'behavior',
                  'eye_position',
                  'responses']
        
        filename = InputResponse().get_filename(mkey)

        return MovieSet(filename, *data_names)


@schema
class OracleStims(dj.Computed):
    definition = """
    -> data_schemas.InputResponse
    ---
    stimulus_type           : varchar(64)   # {stimulus.Frame, ~stimulus.Frame, stimulus.Frame|~stimulus.Frame}
    condition_hashes_json   : varchar(8000) # Json (list) of condition_hashes that has at least 4 (Arbitary) repeats
    num_oracle_stims        : int           # num of unique stimuli that have >= 4 repeat presentations
    min_trial_repeats       : int           # The min_num_of_occurances in the condition_hashes array
    min_frames              : int           # min_num of frames in the condtion_hash set
    """

    @property
    def key_source(self):
        from .data_schemas import MovieMultiDataset, InputResponse
        return InputResponse & (MovieMultiDataset.Member & 'group_id!=11' & 'group_id!=12' & 'group_id!=13')

    def make(self, key):
        min_num_of_repeats = 4 # Arbitary requirment

        dataset = load_dataset(key)
        dataset_condition_hashes = dataset.condition_hashes
        dataset_stimulus_type = dataset.types

        # Find conditions_hashes that repeats more than min_num_of_repeats
        unique_condition_hashes, counts = np.unique(dataset_condition_hashes, return_counts=True)
        mask = counts > min_num_of_repeats

        condition_hashes = unique_condition_hashes[mask]

        # Determine stimulus type
        unique_stimulus_types = np.unique(dataset_stimulus_type[np.isin(dataset_condition_hashes, condition_hashes)])

        if 'stimulus.Clip' in unique_stimulus_types:
            stimulus_type = 'stimulus.Clip'
            if unique_stimulus_types.size > 1:
                stimulus_type += '|~stimulus.Clip'
        elif unique_stimulus_types.size >= 1:
            stimulus_type = '~stimulus.Clip'
        else:
            raise Exception('Dataset does not contain trial repeats')

        # compute min_frames
        target_indices = np.where(np.isin(dataset_condition_hashes, condition_hashes))[0]
        frames_count = np.empty(shape=[target_indices.size])

        for i, index in enumerate(target_indices):
            frames_count[i] = dataset[index].responses.shape[0]
            
        # Convert conditon_hashes into json object
        condition_hashes_json = json.dumps(condition_hashes.tolist())
        assert len(condition_hashes_json) < 8000, 'condition hashes exceeds 8000 characters'
        
        key['stimulus_type'] = stimulus_type
        key['condition_hashes_json'] = condition_hashes_json
        key['num_oracle_stims'] = condition_hashes.size
        key['min_trial_repeats'] = counts[mask].min()
        key['min_frames'] = frames_count.min()

        self.insert1(key)