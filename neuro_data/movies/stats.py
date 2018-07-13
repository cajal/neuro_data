from collections import OrderedDict
from functools import partial
from itertools import count
from pprint import pformat

from attorch.dataset import H5SequenceSet

from neuro_data.movies.transforms import Subsequence
from neuro_data.utils.measures import corr
from .mixins import TraceMixin
from .. import logger as log
import imageio
import io
import numpy as np
import cv2
from .schema_bridge import *
from tqdm import tqdm
from scipy.signal import convolve2d
from ..utils.data import SplineMovie, FilterMixin, SplineCurve, h5cached, NaNSpline, fill_nans
from .data_schemas import MovieMultiDataset, MovieScan
from .configs import DataConfig

schema = dj.schema('neurodata_moviestats', locals())


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
