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
import pandas as pd
import json
from .schema_bridge import *
from tqdm import tqdm
from scipy import stats
from scipy.signal import convolve2d
from ..utils.data import SplineMovie, FilterMixin, SplineCurve, h5cached, NaNSpline, fill_nans
from .data_schemas import InputResponse, MovieMultiDataset, MovieScan, MovieSet
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
            assert len(unit_ids) == len(
                activity) == outputs.shape[-1], 'Neuron numbers do not add up'
            member_key = (MovieMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)

            self.FractionActive().insert(
                [dict(member_key, activity=a, unit_id=u)
                 for u, a in tqdm(zip(unit_ids, activity), total=len(unit_ids))],
                ignore_extra_fields=True)

            self.Average().insert(
                [dict(member_key, avg=a, unit_id=u)
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
            member_key = (MovieMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Pearson().insert1(dict(member_key, pearson=np.mean(pearson), n_neurons=len(pearson)),
                                   ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(
                pearson) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitPearson().insert(
                [dict(member_key, pearson=c, unit_id=u)
                 for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
                ignore_extra_fields=True)


@schema
class ScanOracle(dj.Computed):
    definition = """
    # oracle computation for each scan
    -> InputResponse
    ---
    n_neurons           : int       # number of neurons in scan
    pearson             : float     # mean test correlation
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> MovieScan.Unit
        ---
        pearson             : float     # mean test correlation
        """

    def make(self, key):
        fname = InputResponse().get_filename(key)
        dset = MovieSet(fname, 'inputs', 'responses')
        test_index = np.where(dset.tiers == 'test')[0]
        condition_hashes = dset.condition_hashes
        hashes, counts = np.unique(condition_hashes, return_counts=True)
        repeat_hashes = hashes[counts > 2]

        oracles, data = [], []
        for cond_hash in repeat_hashes:
            repeat_index = np.where(condition_hashes == cond_hash)[0]
            index = np.intersect1d(repeat_index, test_index).tolist()
            if len(index) < 3:
                continue
            inputs = np.stack([dset.inputs[str(i)][()] for i in index], axis=0)
            outputs = np.stack([dset.responses[str(i)][()] for i in index], axis=0)
            assert (np.diff(inputs, axis=0) == 0).all(), 'Video inputs of oracle trials do not match'
            new_shape = (-1, outputs.shape[-1])
            r = outputs.shape[0]
            mu = outputs.mean(axis=0, keepdims=True)
            oracle = (mu * r - outputs) / (r - 1)
            oracles.append(oracle.reshape(new_shape))
            data.append(outputs.reshape(new_shape))
        pearsons = corr(np.vstack(data), np.vstack(oracles), axis=0)
        unit_ids = dset._fid['neurons']['unit_ids'][()]

        self.insert1(dict(key, n_neurons=len(pearsons), pearson=np.mean(pearsons)))
        self.Unit.insert([dict(key, unit_id=u, pearson=p) for u, p in zip(unit_ids, pearsons)])


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
        min_num_of_repeats = 4  # Arbitary requirment

        dataset = load_dataset(key)
        dataset_condition_hashes = dataset.condition_hashes
        dataset_stimulus_type = dataset.types

        # Find conditions_hashes that repeats more than min_num_of_repeats
        unique_condition_hashes, counts = np.unique(
            dataset_condition_hashes, return_counts=True)
        mask = counts > min_num_of_repeats

        condition_hashes = unique_condition_hashes[mask]

        # Determine stimulus type
        unique_stimulus_types = np.unique(
            dataset_stimulus_type[np.isin(dataset_condition_hashes, condition_hashes)])

        if 'stimulus.Clip' in unique_stimulus_types:
            stimulus_type = 'stimulus.Clip'
            if unique_stimulus_types.size > 1:
                stimulus_type += '|~stimulus.Clip'
        elif unique_stimulus_types.size >= 1:
            stimulus_type = '~stimulus.Clip'
        else:
            raise Exception('Dataset does not contain trial repeats')

        # Convert conditon_hashes into json object
        condition_hashes_json = json.dumps(condition_hashes.tolist())
        assert len(
            condition_hashes_json) < 8000, 'condition hashes exceeds 8000 characters'

        key['stimulus_type'] = stimulus_type
        key['condition_hashes_json'] = condition_hashes_json
        key['num_oracle_stims'] = condition_hashes.size
        key['min_trial_repeats'] = counts[mask].min()
        key['min_frames'] = np.min([dataset[index].responses.shape[0] for index in np.where(
            np.isin(dataset_condition_hashes, condition_hashes))[0]])

        self.insert1(key)


@schema
class BootstrapOracleSeed(dj.Lookup):
    definition = """
    oracle_bootstrap_seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        for seed in list(range(100)):
            yield (seed,)


@schema
class BootstrapOracle(dj.Computed):
    definition = """
    -> OracleStims
    -> BootstrapOracleSeed
    ---
    """

    class Score(dj.Part):
        definition = """
        -> master
        ---
        boostrap_score_true			    : float
        boostrap_score_null			    : float
        """

    class UnitScore(dj.Part):
        definition = """
        -> master
        -> MovieScan.Unit
        ---
        boostrap_unit_score_true		: float
        boostrap_unit_score_null		: float
        """

    @property
    def key_source(self):
        from .data_schemas import MovieMultiDataset
        return super().key_source & (
            MovieMultiDataset.Member & 'group_id in (0, 1, 2, 9, 15, 16, 17)')

    def sample_from_condition_hash(self, target_hash, dataset, sample_size):
        return np.random.choice(np.where(dataset == target_hash)[0], sample_size, replace=False)

    def check_input(self, target_indices, dataset, min_frames):
        dataset_images_shape = dataset[target_indices[0]].inputs.shape
        inputs = np.empty(shape=[len(target_indices), dataset_images_shape[0],
                                 min_frames, dataset_images_shape[2], dataset_images_shape[3]])
        for i, index in enumerate(target_indices):
            inputs[i] = dataset[index].inputs[0][0:min_frames]
        assert np.all(np.abs(np.diff(inputs, axis=0)) ==
                      0), 'Images of oracle trials do not match'

    def sample_frames_from_dataset(self, target_indices, dataset, min_frames, num_of_neurons):
        # Compute start_index
        starting_index = 0

        response_matrix = np.empty(
            shape=[target_indices.size, min_frames, num_of_neurons])
        for i, index in enumerate(target_indices):
            response_matrix[i] = dataset[index].responses[starting_index:starting_index+min_frames]

        return response_matrix

    def compute_oracle(self, outputs):
        r = outputs.shape[0]
        mu = outputs.mean(axis=0, keepdims=True)
        oracles = ((mu - outputs / r) * r / (r - 1)
                   ).reshape(-1, outputs.shape[-1])
        return oracles

    def sample_and_compute_oracle(self, dataset, condition_hashes, sample_size, min_frames):
        num_of_neurons = dataset[0].responses.shape[1]
        dataset_condition_hashes = dataset.condition_hashes

        # Oracle compuatation
        true_responses = np.empty(
            shape=[len(condition_hashes), sample_size * min_frames, num_of_neurons])
        true_oracles = np.empty(
            shape=[len(condition_hashes), sample_size * min_frames, num_of_neurons])

        null_responses = np.empty(
            shape=[len(condition_hashes), sample_size * min_frames, num_of_neurons])
        null_oracles = np.empty(
            shape=[len(condition_hashes), sample_size * min_frames, num_of_neurons])

        for i in range(0, len(condition_hashes)):
            # True Oracle Computation
            # For each condition_hashes, sample (sample_size) trials to construct the true_response_matrix
            # Select (sample_size) trials

            true_target_indices = self.sample_from_condition_hash(
                condition_hashes[i], dataset_condition_hashes, sample_size)

            # Check inputs for true_oracles
            self.check_input(true_target_indices, dataset, min_frames)

            response_matrix = self.sample_frames_from_dataset(
                true_target_indices, dataset, min_frames, num_of_neurons)
            true_responses[i] = response_matrix.reshape(
                -1, response_matrix.shape[-1])
            true_oracles[i] = self.compute_oracle(response_matrix)

            # Null Oracle Computation
            # Select (samples_size) hashes and sample from them
            target_hashes = np.random.choice(
                dataset_condition_hashes, sample_size, replace=False)

            # Get null_target_indices
            null_target_indices = np.array([self.sample_from_condition_hash(
                h, dataset_condition_hashes, 1)[0] for h in target_hashes])

            # Sample for each target index
            response_matrix = self.sample_frames_from_dataset(
                null_target_indices, dataset, min_frames, num_of_neurons)
            null_responses[i] = response_matrix.reshape(
                -1, response_matrix.shape[-1])
            null_oracles[i] = self.compute_oracle(response_matrix)

        true_responses = true_responses.reshape(-1, num_of_neurons)
        true_oracles = true_oracles.reshape(-1, num_of_neurons)
        null_responses = null_responses.reshape(-1, num_of_neurons)
        null_oracles = null_oracles.reshape(-1, num_of_neurons)

        return corr(true_responses, true_oracles, axis=0), corr(null_responses, null_oracles, axis=0)

    def make(self, key):
        log.info('Populating {}'.format(key))

        dataset = load_dataset(key)

        stim_tup = OracleStims & key
        condition_hashes = json.loads(stim_tup.fetch1('condition_hashes_json'))
        sample_size = min(stim_tup.fetch1(
            'num_oracle_stims', 'min_trial_repeats'))
        min_frames = stim_tup.fetch1('min_frames')
        # Add this later once you get the table
        np.random.seed(key['oracle_bootstrap_seed'])

        true_pearson, null_pearson = self.sample_and_compute_oracle(
            dataset, condition_hashes, sample_size, min_frames)

        self.insert1(key)
        # Inserting pearson mean scores to Score table
        self.Score().insert1(dict(key, boostrap_score_true=true_pearson.mean(),
                                  boostrap_score_null=null_pearson.mean()))
        # Inserting unit pearson scores
        self.UnitScore().insert([dict(key, unit_id=u, boostrap_unit_score_true=t, boostrap_unit_score_null=n)
                                 for u, t, n in zip(dataset.neurons.unit_ids, true_pearson, null_pearson)])


@schema
class BootstrapOracleTTest(dj.Computed):
    definition = """
    -> OracleStims
    ---
    """

    class PValue(dj.Part):
        definition = """
        -> master
        ---
        p_value             : float     # bootstrap oracle t-test p_value for dataset
        """

    class UnitPValue(dj.Part):
        definition = """
        -> master
        -> MovieScan.Unit
        ---
        unit_p_value        : float     # bootstrap oracle t-test p_value for single neuron
        """

    @property
    def key_source(self):
        return super().key_source & BootstrapOracle

    def make(self, key):
        num_seeds = len(BootstrapOracleSeed())

        dset = load_dataset(key)
        unit_ids = dset.neurons.unit_ids
        unit_scores = pd.DataFrame((BootstrapOracle.UnitScore & key).fetch())

        assert len(unit_scores) == num_seeds * len(unit_ids)

        # Computeing p_values for means
        mean_scores = unit_scores.groupby('unit_id').mean()
        _, p_value = stats.ttest_rel(
            mean_scores.boostrap_unit_score_true.values, mean_scores.boostrap_unit_score_null.values)

        # Computing unit scores
        scores_true = unit_scores.pivot(
            index='unit_id', columns='oracle_bootstrap_seed',
            values='boostrap_unit_score_true')
        scores_null = unit_scores.pivot(
            index='unit_id', columns='oracle_bootstrap_seed',
            values='boostrap_unit_score_null')
        _, unit_p_values = stats.ttest_ind(
            scores_true.values, scores_null.values,
            axis=1, equal_var=False)
        assert np.array_equal(scores_true.index.values, scores_null.index.values)
        assert np.array_equal(scores_true.index.values, unit_ids)

        self.insert1(key)
        self.PValue().insert1(dict(key, p_value=p_value))
        self.UnitPValue().insert([dict(key, unit_id=u, unit_p_value=upv)
                                  for u, upv in zip(unit_ids, unit_p_values)])
