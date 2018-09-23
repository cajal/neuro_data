import datajoint as dj
from tqdm import tqdm

import numpy as np
from scipy import stats

from neuro_data.utils.measures import corr
from .configs import DataConfig

from .data_schemas import StaticMultiDataset, StaticScan

from .. import logger as log

schema = dj.schema('neurodata_static_stats', locals())


@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for static images

    -> StaticMultiDataset
    -> DataConfig
    ---
    """

    @property
    def key_source(self):
        return StaticMultiDataset() * DataConfig()

    class Scores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member
        ---
        pearson           : float     # mean test correlation
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> StaticScan.Unit
        ---
        pearson           : float     # mean test correlation
        """

    def make(self, key):
        # --- load data
        testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)

        self.insert1(dict(key))
        for readout_key, loader in testloaders.items():
            log.info('Computing oracle for ' + readout_key)
            oracles, data = [], []
            for inputs, *_, outputs in loader:
                inputs = inputs.numpy()
                outputs = outputs.numpy()
                assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
                    'Images of oracle trials does not match'
                r, n = outputs.shape  # responses X neurons
                log.info('\t    {} responses for {} neurons'.format(r, n))
                assert r > 4, 'need more than 4 trials for oracle computation'
                mu = outputs.mean(axis=0, keepdims=True)
                oracle = (mu - outputs / r) * r / (r - 1)
                oracles.append(oracle)
                data.append(outputs)
            if len(data) == 0:
                log.error('Found no oracle trials! Skipping ...')
                return
            pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

            member_key = (StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Scores().insert1(dict(member_key, pearson=np.mean(pearson)), ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(pearson) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitScores().insert(
            [dict(member_key, pearson=c, unit_id=u) for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
            ignore_extra_fields=True)

@schema 
class OracleStims(dj.Computed):
    definition = """
    ->InputResponse
    ---
    condition_hashes    : longblob      # Array of condition_hashes that has at least 4 (Arbitary) repeats
    stimulus_type       : varchar(64)   # {stimulus.Frame, ~stimulus.Frame, stimulus.Frame|~stimulus.Frame} corresponding to
    num_cond_hashes     : int           # num of condition_hashes that meet the Arbitary requirement
    min_num_of_occurances     : int     # The min_num_of_occurances in the condition_hashes array
    """

    @property
    def key_source(self):
       return InputResponse()

    def make(self, key):
        min_num_of_repeats = 4 # Arbitary requirment

        # Extract data from database with respect to the given key
        include_behavior = bool(Eye.proj() * Treadmill().proj() & key)
        data_names = ['images', 'responses'] if not include_behavior \
            else ['images',
                  'behavior',
                  'pupil_center',
                  'responses']
        h5filename = InputResponse().get_filename(key)
        dataset = StaticImageSet(h5filename, *data_names)

        # Find unique hashes and its number of occurances
        hash_count = Counter(dataset.condition_hashes)

        condtion_hashes = []
        min_num_of_occurances = len(hash_count)

        # Find smallest_num_of_occurances among hashes that has >= than min_num_of_repeats
        for hash in hash_count:
             if (hash_count[hash] >= min_num_of_repeats):
                if (hash_count[hash] < min_num_of_occurances):
                    smallest_num_of_occurances = hash_count[hash]
                condtion_hashes.append(hash)
        
        # Determine stimulus_type
        if ('stimulus.Frame' in dataset.types):
            if('stimulus.TrippyFrame' in dataset.types or 'stimulus.MonetFrame' in dataset.types):
                stimulus_type = 'stimulus.Frame|~stimulus.Frame'
            else:
                stimulus_type = 'stimulus.Frame'
        else:
            stimulus_type = '~stimulus.Frame'

        key['condition_hashes'] = condtion_hashes
        key['stimulus_type'] = stimulus_type
        key['num_cond_hashes'] = len(condtion_hashes)
        key['min_num_of_occurances'] = min_num_of_occurances
