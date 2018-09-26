import datajoint as dj
from tqdm import tqdm
import json
import numpy as np
from scipy import stats

from neuro_data.utils.measures import corr
from .configs import DataConfig

from .data_schemas import StaticMultiDataset, StaticScan

from .. import logger as log

schema = dj.schema('neurodata_static_stats', locals())
data_schemas = dj.create_virtual_module('data_schemas', 'neurodata_static')


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

            member_key = (StaticMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Scores().insert1(dict(member_key, pearson=np.mean(pearson)), ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(
                pearson) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitScores().insert(
                [dict(member_key, pearson=c, unit_id=u)
                 for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
                ignore_extra_fields=True)


@schema
class OracleStims(dj.Computed):
    definition = """
    -> data_schemas.InputResponse
    ---
    stimulus_type           : varchar(64)   # {stimulus.Frame, ~stimulus.Frame, stimulus.Frame|~stimulus.Frame}
    frame_image_ids         : longblob      # Array of frame_iamge_ids that has at least 4 (Arbitary) repeats
    condition_hashes_json   : varchar(8000) # Json (list) of condition_hashes that has at least 4 (Arbitary) repeats
    num_oracle_stims        : int           # num of unique stimuli that have >= 4 repeat presentations
    min_trial_repeats       : int           # The min_num_of_occurances in the condition_hashes array
    """

    @property
    def key_source(self):
        from .data_schemas import StaticMultiDataset, InputResponse
        return InputResponse & StaticMultiDataset.Member

    def make(self, key):
        from .data_schemas import InputResponse, Eye, Treadmill
        from .datasets import StaticImageSet

        min_num_of_repeats = 4  # Arbitary requirment

        # Extract data from database with respect to the given key
        include_behavior = bool(Eye.proj() * Treadmill().proj() & key)
        data_names = ['images', 'responses'] if not include_behavior \
            else ['images',
                'behavior',
                'pupil_center',
                'responses']
        h5filename = InputResponse().get_filename(key)
        dataset = StaticImageSet(h5filename, *data_names)

        # Get all frame_image_ids for repeated stimulus.image that repeates more than 4
        all_stimulus_unique_frame_id, all_stimulus_unique_frame_id_count = np.unique(
            dataset.info.frame_image_id[dataset.types == 'stimulus.Frame'], return_counts=True)
        frame_image_ids = all_stimulus_unique_frame_id[
            all_stimulus_unique_frame_id_count >= min_num_of_repeats]

        # Get all condition_hash for repeated ~stimulus.image that repeates more than 4
        all_not_stimulus_unique_frames, all_not_stimulus_unique_frames_count = np.unique(
            dataset.condition_hashes[dataset.types != 'stimulus.Frame'], return_counts=True)
        condition_hashes = all_not_stimulus_unique_frames[
            all_not_stimulus_unique_frames_count >= min_num_of_repeats]

        # Compute min_trial_repeats for both natural images and noise, also determine stimulus.type
        stimulus_type = ''

        temp = all_stimulus_unique_frame_id_count[all_stimulus_unique_frame_id_count >=
                                                min_num_of_repeats]
        if temp.size > 0:
            minumum_natural_image_trials = temp.min()
            stimulus_type = 'stimulus.Frame'
        else:
            minumum_natural_image_trials = 0

        temp = all_not_stimulus_unique_frames_count[
            all_not_stimulus_unique_frames_count >= min_num_of_repeats]
        if temp.size > 0:
            minumum_noise_image_trials = temp.min()
            if stimulus_type == 'stimulus.Frame':
                stimulus_type += '|~stimulus.Frame'
            else:
                stimulus_type = '~stimulus.Frame'
        else:
            minumum_noise_image_trials = 0

        # Deteremine min_trial_repeats based on the values above
        min_trial_repeats = np.array([minumum_natural_image_trials, minumum_noise_image_trials])
        min_trial_repeats = min_trial_repeats[min_trial_repeats > 0]
        if min_trial_repeats.size == 0:
            min_trial_repeats = 0
        elif min_trial_repeats.size == 2:
            min_trial_repeats = min(min_trial_repeats)
        else:
            min_trial_repeats = min_trial_repeats[0]

        chashes_json = json.dumps(condition_hashes.tolist())
        assert len(chashes_json) < 8000, 'condition hashes exceeds 8000 characters'

        # Fill in table
        key['stimulus_type'] = stimulus_type
        key['frame_image_ids'] = frame_image_ids
        key['condition_hashes_json'] = chashes_json
        key['num_oracle_stims'] = frame_image_ids.size + condition_hashes.size
        key['min_trial_repeats'] = min_trial_repeats
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
        -> StaticScan.Unit
        ---
        boostrap_unit_score_true		: float
        boostrap_unit_score_null		: float
        """

    def sample_from_x(self, x, dataset, min_trial_repeats):
        return np.random.choice(np.where(np.isin(dataset, x[np.random.randint(0, len(x) - 1)]))[0], min_trial_repeats, replace=False)

    def compute_oracle(self, response_matrix, min_trial_repeats):
        num_of_trials, num_of_responses = response_matrix.shape
        means = response_matrix.mean(axis=0, keepdims=True)
        oracle_matrix = (means - response_matrix / min_trial_repeats) * min_trial_repeats / (min_trial_repeats - 1)
    
        return corr(response_matrix, oracle_matrix, axis=0)

    def sample_and_compute_oracle(self, frame_image_ids, condition_hashes, dataset, stimulus_type, min_trial_repeats):
        # Sample both true and null oracles
    
        if stimulus_type == 'stimulus.Frame|~stimulus.Frame':
            # Select a unique condition_hashes or frame_image_ids
            if np.random.randint(0, 2) == 0:
                # Select from frame_image_ids
                true_target_indices = self.sample_from_x(frame_image_ids, dataset.info.frame_image_id, min_trial_repeats)
            else:
                # Select from condition_hashes
                true_target_indices = self.sample_from_x(condition_hashes, dataset.condition_hashes, min_trial_repeats)
            
            # Select min_trial_repeats from either condition_hashes or frame_image_ids
            null_target_indices = np.empty(shape=[min_trial_repeats], dtype=int)
            for i in range(0, min_trial_repeats):
                if np.random.randint(0, 2) == 0:
                    # Select from frame_image_ids
                    null_target_indices[i] = self.sample_from_x(frame_image_ids, dataset.info.frame_image_id, 1)
                else:
                    # Select from condition_hashes
                    null_target_indices[i] = self.sample_from_x(condition_hashes, dataset.condition_hashes, 1)

        elif stimulus_type == 'stimulus.Frame':
            # True
            true_target_indices = self.sample_from_x(frame_image_ids, dataset.info.frame_image_id, min_trial_repeats)
        
            # Null
            null_target_indices = np.empty(shape=[min_trial_repeats], dtype=int)
            for i in range(0, min_trial_repeats):
                null_target_indices[i] = self.sample_from_x(frame_image_ids, dataset.info.frame_image_id, 1)
            
        elif stimulus_type == '~stimulus.Frame':
            # True
            true_target_indices = self.sample_from_x(condition_hashes, dataset.condition_hashes, min_trial_repeats)
        
            # Null
            null_target_indices = np.empty(shape=[min_trial_repeats], dtype=int)
            for i in range(0, min_trial_repeats):
                # Select from condition_hashes
                null_target_indices[i] = self.sample_from_x(condition_hashes, dataset.condition_hashes, 1)
        
        # Get Responses    
        true_response_matrix = np.take(dataset.responses, true_target_indices, axis=0)
        null_response_matrix = np.take(dataset.responses, null_target_indices, axis=0)
    
        # Compute Oracle Score
        return self.compute_oracle(true_response_matrix, min_trial_repeats), self.compute_oracle(null_response_matrix, min_trial_repeats)

    def make(self, key):
        from .data_schemas import InputResponse, Eye, Treadmill
        from .datasets import StaticImageSet

        self.insert1(key)

        attrs = OracleStims.heading.dependent_attributes
        attr_dict = dict(zip(attrs, (OracleStims & key).fetch1(*attrs)))

        frame_image_ids = attr_dict['frame_image_ids']
        condition_hashes = json.loads(attr_dict['condition_hashes_json'])
        stimulus_type = attr_dict['stimulus_type']
        min_trial_repeats = attr_dict['min_trial_repeats']

        # Extract data from database with respect to the given key
        include_behavior = bool(Eye.proj() * Treadmill().proj() & key)
        data_names = ['images', 'responses'] if not include_behavior \
            else ['images',
                'behavior',
                'pupil_center',
                'responses']
        h5filename = InputResponse().get_filename(key)
        dataset = StaticImageSet(h5filename, *data_names)

        true_pearson, null_pearson = self.sample_and_compute_oracle(frame_image_ids, condition_hashes, dataset, stimulus_type, min_trial_repeats)

        # Inserting pearson mean scores to Score table
        self.Score().insert1(dict(key, boostrap_score_true=true_pearson.mean(), boostrap_score_null=null_pearson.mean()))

        # Inserting unit pearson scores
        self.UnitScore().insert([dict(key, unit_id=u, boostrap_unit_score_true=t, boostrap_unit_score_null=n) for u, t, n in zip(dataset.neurons.unit_ids, true_pearson, null_pearson)])