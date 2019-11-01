from collections import OrderedDict
from itertools import product, count
from attorch.dataloaders import RepeatsBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from .data_schemas import StaticMultiDataset
from .transforms import Subsample, Normalizer, ToTensor
from ..utils.sampler import SubsetSequentialSampler, BalancedSubsetSampler
from ..utils.config import ConfigBase
from ..common import configs as common_configs
import datajoint as dj
from .. import logger as log
import warnings
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os import path






experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

schema = dj.schema('neurodata_static_configs')


try:
    models = dj.create_virtual_module('models', 'neurostatic_models')


    @schema
    class ModelCollection(dj.Lookup):
        definition = """
        model_collection_id: smallint   # collection id
        ---
        collection_description: varchar(255)  # description of the collection
        """
        contents = [(0, 'Best CNN model')]

        class Entry(dj.Part):
            definition = """
            -> master
            -> StaticMultiDataset
            ---
            -> models.Model
            """
except:
    pass



class BackwardCompatibilityMixin:
    """
    Backward compatibility layer: namely use of the buggy Normalizer
    """

    @staticmethod
    def add_transforms(key, datasets, exclude=None):
        warnings.warn('You are using an outdated `add_transform` kept for backward compatibility. Do not use this in new networks.')
        if exclude is not None:
            log.info('Excluding "{}" from normalization'.format(
                '", "'.join(exclude)))
        for k, dataset in datasets.items():
            transforms = []
            if key['normalize']:
                transforms.append(Normalizer(
                    dataset, stats_source=key['stats_source'],
                    buggy=True, normalize_per_image=True, exclude=exclude))
            transforms.append(ToTensor())
            dataset.transforms = transforms

        return datasets

class StimulusTypeMixin:
    _stimulus_type = None

    @staticmethod
    def add_transforms(key, datasets, exclude=None):
        if exclude is not None:
            log.info('Excluding "{}" from normalization'.format(
                '", "'.join(exclude)))
        for k, dataset in datasets.items():
            transforms = []

            if key.get('normalize', True):
                transforms.append(Normalizer(
                    dataset,
                    stats_source=key.get('stats_source', 'all'),
                    normalize_per_image=key.get('normalize_per_image', False),
                    exclude=exclude))
            transforms.append(ToTensor())
            dataset.transforms = transforms

        return datasets

    @staticmethod
    def get_constraint(dataset, stimulus_type, tier=None):
        """
        Find subentries of dataset that matches the given `stimulus_type` specification and `tier` specification.
        `stimulus_type` is of the format `stimulus.Frame|~stimulus.Monet|...`. This function returns a boolean array
        suitable to be used for boolean indexing to obtain only entries with data types and tiers matching the
        specified condition.
        """
        constraint = np.zeros(len(dataset.types), dtype=bool)
        for const in map(lambda s: s.strip(), stimulus_type.split('|')):
            if const.startswith('~'):
                log.info('Using all trial but from {}'.format(const[1:]))
                tmp = (dataset.types != const[1:])
            else:
                log.info('Using all trial from {}'.format(const))
                tmp = (dataset.types == const)
            constraint = constraint | tmp
        if tier is not None:
            constraint = constraint & (dataset.tiers == tier)
        return constraint

    @staticmethod
    def get_sampler_class(tier, balanced=False):
        """
        Given data `tier` return a default Sampler class suitable for that tier. If `tier="train"` and `balanced=True`,
        returns BalancedSubsetSampler
        Args:
            tier: dataset tier - 'train', 'validation', 'test', or None
            balanced: if True and tier='train', returns balanced version

        Returns:
            A subclass of Sampler

        """
        assert tier in ['train', 'validation', 'test', None]
        if tier == 'train':
            if not balanced:
                Sampler = SubsetRandomSampler
            else:
                Sampler = BalancedSubsetSampler
        else:
            Sampler = SubsetSequentialSampler
        return Sampler

    @staticmethod
    def log_loader(loader):
        """
        A helper function that when given an instance of DataLoader, print out a log detailing its configuration
        """
        log.info('Loader sampler is {}'.format(
            loader.sampler.__class__.__name__))
        log.info('Number of samples in the loader will be {}'.format(
            len(loader.sampler)))
        log.info(
            'Number of batches in the loader will be {}'.format(int(np.ceil(len(loader.sampler) / loader.batch_size))))

    def get_loaders(self, datasets, tier, batch_size, stimulus_types, Sampler):
        """

        Args:
            datasets: a dictionary of H5ArrayDataSets
            tier: tier of data to be loaded. Can be 'train', 'validation', 'test', or None
            batch_size: size of a batch to be returned by the data loader
            stimulus_types: stimulus type specification like 'stimulus.Frame|~stimulus.Monet'
            Sampler: sampler to be placed on the data loader. If None, defaults to a sampler chosen based on the tier

        Returns:
            A dictionary of DataLoader's, key paired to each dataset
        """

        # if Sampler not given, use a default one specified for each tier
        if Sampler is None:
            Sampler = self.get_sampler_class(tier)

        # if only a single stimulus_types string was given, apply to all datasets
        if not isinstance(stimulus_types, list):
            log.info('Using {} as stimulus type for all datasets'.format(
                stimulus_types))
            stimulus_types = len(datasets) * [stimulus_types]

        log.info('Stimulus sources: "{}"'.format('","'.join(stimulus_types)))

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            log.info('Selecting trials from {} and tier={} for dataset {}'.format(
                stimulus_type, tier, k))
            ix = np.where(constraint)[0]
            log.info('Found {} active trials'.format(constraint.sum()))
            if Sampler is BalancedSubsetSampler:
                sampler = Sampler(ix, dataset.types, mode='longest')
            else:
                sampler = Sampler(ix)
            loaders[k] = DataLoader(
                dataset, sampler=sampler, batch_size=batch_size)
            self.log_loader(loaders[k])
        return loaders

    def load_data(self, key, tier=None, batch_size=1, key_order=None,
                  exclude_from_normalization=None, stimulus_types=None, Sampler=None):
        log.info('Loading {} dataset with tier={}'.format(
            self._stimulus_type, tier))
        datasets = StaticMultiDataset().fetch_data(key, key_order=key_order)
        for k, dat in datasets.items():
            if 'stats_source' in key:
                log.info(
                    'Adding stats_source "{stats_source}" to dataset'.format(**key))
                dat.stats_source = key['stats_source']

        log.info('Using statistics source ' + key['stats_source'])

        datasets = self.add_transforms(
            key, datasets, exclude=exclude_from_normalization)

        loaders = self.get_loaders(
            datasets, tier, batch_size, stimulus_types, Sampler)
        return datasets, loaders


class AreaLayerRawMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None, Sampler=None, **kwargs):
        log.info('Ignoring input arguments: "' +
                 '", "'.join(kwargs.keys()) + '"' + 'when creating datasets')
        exclude = key.pop('exclude').split(',')
        stimulus_types = key.pop('stimulus_type')
        datasets, loaders = super().load_data(key, tier, batch_size, key_order,
                                              exclude_from_normalization=exclude,
                                              stimulus_types=stimulus_types,
                                              Sampler=Sampler)

        log.info('Subsampling to layer {} and area(s) "{}"'.format(key['layer'],
                                                                   key.get('brain_area') or key['brain_areas']))
        for readout_key, dataset in datasets.items():
            layers = dataset.neurons.layer
            areas = dataset.neurons.area

            layer_idx = (layers == key['layer'])
            desired_areas = ([key['brain_area'], ] if 'brain_area' in key else
                             (common_configs.BrainAreas.BrainArea & key).fetch('brain_area'))
            area_idx = np.stack([areas == da for da in desired_areas]).any(axis=0)
            idx = np.where(layer_idx & area_idx)[0]
            if len(idx) == 0:
                log.warning('Empty set of neurons. Deleting this key')
                del datasets[readout_key]
                del loaders[readout_key]
            else:
                dataset.transforms.insert(-1, Subsample(idx))
        return datasets, loaders


class AreaLayerModelMixin:
    def load_data(self, key, prep_cuda=True, prep_batch_size=1, **kwargs):

        from staticnet_experiments.models import Model
        from staticnet_experiments import configs

        entry_key = (ModelCollection.Entry.proj() & key).fetch1('KEY')
        net_key = (Model & (ModelCollection.Entry & entry_key)).fetch1('KEY')


        data_key = (DataConfig & (configs.NetworkConfig.CorePlusReadout & (Model & net_key))).fetch1('KEY')
        data_key['group_id'] = key['group_id']
        datasets, loaders = DataConfig().load_data(data_key, **kwargs)

        cache_path = '/external/model_resp_cache/{}-{}.pt'.format(key['group_id'], key['data_hash'])

        if path.exists(cache_path):
            # if cache exists, get saved responses loaded
            total_response_dict = torch.load(cache_path)
            print('Loaded data for {} {} from cache!'.format(key['group_id'], key['data_hash']))

        else:
            net = Model().load_network(net_key)
            net.eval()
            if prep_cuda:
                net.cuda()

            total_response_dict = {}
            for k, ds in datasets.items():
                dl = DataLoader(ds, batch_size=prep_batch_size)
                resp = []
                for input, beh, eye, _ in tqdm(dl):
                    with torch.no_grad():
                        if prep_cuda:
                            input, beh, eye = input.cuda(), beh.cuda(), eye.cuda()
                        resp.append(net(input, readout_key=k, behavior=beh, eye_pos=eye).data.cpu().numpy())

                total_response_dict[k] = np.concatenate(resp, axis=0)

            torch.save(total_response_dict, cache_path)

        for k, ds in datasets.items():

            ds.responses_override = total_response_dict[k]

            # also exclude responses from normalization because the model the model response already accounts for this
            for t in ds.transforms:
                if isinstance(t, Normalizer):
                    if 'responses' not in t.exclude:
                        t.exclude.append('responses')

        return datasets, loaders







class AreaLayerNoiseMixin(AreaLayerRawMixin):
    def load_data(self, key, balanced=False, **kwargs):
        tier = kwargs.pop('tier', None)
        Sampler = self.get_sampler_class(tier, balanced)
        datasets, loaders = super().load_data(key, tier=None, Sampler=Sampler, **kwargs)
        if tier is not None:
            for k, dataset in datasets.items():
                np.random.seed(key['split_seed'])
                train_hashes, val_hashes, test_hashes = [], [], []
                for noise_type in ['stimulus.MonetFrame', 'stimulus.TrippyFrame']:
                    unique_condition_hashes = np.unique(
                        dataset.info.condition_hash[dataset.types == noise_type])
                    assert unique_condition_hashes.size > 1, 'Dataset does not contain sufficient {}'.format(
                        noise_type)
                    num_hashes = unique_condition_hashes.size
                    num_train_hashes = np.round(
                        num_hashes * key['train_fraction']).astype(np.int)
                    num_val_hashes = np.round(
                        num_hashes * key['val_fraction']).astype(np.int)
                    train_hashes.append(np.random.choice(
                        unique_condition_hashes, num_train_hashes, replace=False))
                    val_hashes.append(np.random.choice(
                        unique_condition_hashes[~np.isin(unique_condition_hashes, train_hashes)], num_val_hashes, replace=False))
                    test_hashes.append(unique_condition_hashes[
                        (~np.isin(unique_condition_hashes, train_hashes)) & (~np.isin(unique_condition_hashes, val_hashes))])
                cond_hashes = dict(
                    train=np.concatenate(train_hashes),
                    validation=np.concatenate(val_hashes),
                    test=np.concatenate(test_hashes))
                if tier == 'test':
                    tier_bool = np.isin(
                        dataset.info.condition_hash, cond_hashes[tier])
                else:
                    tier_bool = np.logical_or(
                        np.isin(dataset.info.condition_hash,
                                cond_hashes[tier]),
                        dataset.tiers == tier)
                if not balanced or tier != 'train':
                    loaders[k].sampler.indices = np.flatnonzero(tier_bool)
                else:
                    loaders[k].sampler.configure_sampler(
                        np.flatnonzero(tier_bool), dataset.types, mode='longest')
                self.log_loader(loaders[k])
        return datasets, loaders


@schema
class DataConfig(ConfigBase, dj.Lookup):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, cuda=False, oracle=False, **kwargs):
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            log.info('Placing oracle data samplers')
            for readout_key, loader in loaders.items():
                ix = loader.sampler.indices
                types = np.unique(datasets[readout_key].types[ix])
                if len(types) == 1 and types[0] == 'stimulus.Frame':
                    condition_hashes = datasets[readout_key].info.frame_image_id
                elif len(types) == 2 and types[0] in ('stimulus.MonetFrame',  'stimulus.TrippyFrame'):
                    condition_hashes = datasets[readout_key].condition_hashes
                else:
                    raise ValueError(
                        'Do not recognize types={}'.format(*types))
                log.info('Replacing ' + loader.sampler.__class__.__name__ +
                         ' with RepeatsBatchSampler')
                Loader = loader.__class__
                loaders[readout_key] = Loader(loader.dataset,
                                              batch_sampler=RepeatsBatchSampler(condition_hashes, subset_index=ix))

                removed = []
                keep = []
                for tr in datasets[readout_key].transforms:
                    if isinstance(tr, (Subsample, ToTensor)):
                        keep.append(tr)
                    else:
                        removed.append(tr.__class__.__name__)
                datasets[readout_key].transforms = keep
                if len(removed) > 0:
                    log.warning('Removed the following transforms: "{}"'.format(
                        '", "'.join(removed)))

        log.info('Setting cuda={}'.format(cuda))
        for dat in datasets.values():
            for tr in dat.transforms:
                if isinstance(tr, ToTensor):
                    tr.cuda = cuda

        return datasets, loaders

    class CorrectedAreaLayer(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(50)   # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalizer or not
        normalize_per_image     : bool          # whether to normalize each input separately
        -> experiment.Layer
        -> anatomy.Area
        """

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses', ''],
                             [True],
                             [True, False],
                             ['L4', 'L2/3'],
                             ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))


    class ModeledAreaLayer(dj.Part, AreaLayerModelMixin):
        definition = """
        -> master
        ---
        -> ModelCollection
        """

        @property
        def content(self):
            for p in [
                (0,)
            ]:
                yield dict(zip(self.heading.dependent_attributes, p))


    class MultipleAreasOneLayer(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source                : varchar(50)   # normalization source
        stimulus_type               : varchar(50)   # type of stimulus
        exclude                     : varchar(512)  # what inputs to exclude from normalization
        normalize                   : bool          # whether to use a normalizer or not
        normalize_per_image         : bool          # whether to normalize each input separately
        -> experiment.Layer
        -> common_configs.BrainAreas
        """
        def describe(self, key):
            return ('{brain_areas} {layer} on {stimulus_type}. normalize={normalize} on '
                    '{stats_source} (except "{exclude}")').format(**key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses', ''],
                             [True],
                             [True, False],
                             ['L4', 'L2/3'],
                             ['all-unknown']):
                yield dict(zip(self.heading.dependent_attributes, p))

    ############ Below are data configs that were using the buggy normalizer #################
    class AreaLayer(dj.Part, BackwardCompatibilityMixin, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(50)   # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        -> experiment.Layer
        -> anatomy.Area
        """

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses', ''],
                             [True],
                             ['L4', 'L2/3'],
                             ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerPercentOracle(dj.Part, BackwardCompatibilityMixin, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(50)   # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        (oracle_source) -> master
        -> experiment.Layer
        -> anatomy.Area
        percent_low                 : tinyint       # percent oracle lower cutoff
        percent_high                 : tinyint      # percent oracle upper cutoff
        """

        def describe(self, key):
            return "Like AreaLayer but only {percent_low}-{percent_high} percent best oracle neurons computed on {oracle_source}".format(
                **key)

        @property
        def content(self):
            oracle_source = list((DataConfig.AreaLayer() &
                                  [dict(brain_area='V1', layer='L2/3',
                                        normalize=True, stats_source='all',
                                        stimulus_type='~stimulus.Frame',
                                        exclude='images,responses'),
                                   dict(brain_area='V1', layer='L2/3',
                                        normalize=True, stats_source='all',
                                        stimulus_type='stimulus.Frame',
                                        exclude='images,responses')]).fetch('data_hash'))
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [25],
                             [75]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [75],
                             [100]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [0],
                             [100]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None, Sampler=None):
            from .stats import Oracle
            datasets, loaders = super().load_data(
                key, tier=tier, batch_size=batch_size, key_order=key_order,
                stimulus_types=stimulus_types, Sampler=Sampler)
            for rok, dataset in datasets.items():
                member_key = (StaticMultiDataset.Member() & key &
                              dict(name=rok)).fetch1(dj.key)

                okey = dict(key, **member_key)
                okey['data_hash'] = okey.pop('oracle_source')
                units, pearson = (Oracle.UnitScores() & okey).fetch(
                    'unit_id', 'pearson')
                assert len(pearson) > 0, 'You forgot to populate oracle for data_hash="{}"'.format(
                    key['oracle_source'])
                assert len(units) == len(
                    dataset.neurons.unit_ids), 'Number of neurons has changed'
                assert np.all(
                    units == dataset.neurons.unit_ids), 'order of neurons has changed'

                low, high = np.percentile(
                    pearson, [key['percent_low'], key['percent_high']])
                selection = (pearson >= low) & (pearson <= high)
                log.info(
                    'Subsampling to {} neurons above {:.2f} and below {} oracle'.format(selection.sum(), low, high))
                dataset.transforms.insert(-1,
                                          Subsample(np.where(selection)[0]))

                assert np.all(dataset.neurons.unit_ids ==
                              units[selection]), 'Units are inconsistent'
            return datasets, loaders

    class AreaLayerNoise(dj.Part, BackwardCompatibilityMixin, AreaLayerNoiseMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(128)  # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        split_seed              : tinyint       # train/validation random split seed
        train_fraction          : float         # fraction of noise dataset for training
        val_fraction            : float         # fraction of noise dataset for validation
        -> experiment.Layer
        -> anatomy.Area
        """

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame | stimulus.MonetFrame | stimulus.TrippyFrame'],
                             ['images,responses'],
                             [True],
                             [0, 1, 2],
                             [0.4],
                             [0.2],
                             ['L2/3'],
                             ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, **kwargs):
            return super().load_data(key, balanced=False, **kwargs)

    class AreaLayerNoiseBalanced(dj.Part, BackwardCompatibilityMixin, AreaLayerNoiseMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(128)  # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        split_seed              : tinyint       # train/validation random split seed
        train_fraction          : float         # fraction of noise dataset for training
        val_fraction            : float         # fraction of noise dataset for validation
        -> experiment.Layer
        -> anatomy.Area
        """

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame | stimulus.MonetFrame | stimulus.TrippyFrame'],
                             ['images,responses'],
                             [True],
                             [0, 1, 2],
                             [0.4],
                             [0.2],
                             ['L2/3'],
                             ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, **kwargs):
            return super().load_data(key, balanced=True, **kwargs)

    class AreaLayerReliable(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512) # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        -> experiment.Layer
        -> anatomy.Area
        p_val_power             : tinyint
        """

        def describe(self, key):
            return "Like AreaLayer but only neurons that have significantly different (p-val < {:.0E}) response-oracle correlations to the same stimuli vs different stimuli".format(
                np.power(10, float(key['p_val_power'])))

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             ['L2/3'],
                             ['V1'],
                             [-3]):
                yield dict(zip(self.heading.dependent_attributes, p))
        
        def load_data(self, key, tier=None, batch_size=1,
                      Sampler=None, t_first=False, cuda=False):
            from .stats import BootstrapOracleTTest
            assert tier in [None, 'train', 'validation', 'test']
            datasets, loaders = super().load_data(
                key, tier=tier, batch_size=batch_size, Sampler=Sampler,
                cuda=cuda)
            for rok, dataset in datasets.items():
                member_key = (StaticMultiDataset.Member() & key &
                            dict(name=rok)).fetch1(dj.key)
                all_units, all_pvals = (
                    BootstrapOracleTTest.UnitPValue & member_key).fetch(
                        'unit_id', 'unit_p_value')
                assert len(all_pvals) > 0, \
                    'You forgot to populate BootstrapOracleTTest for group_id={}'.format(
                    member_key['group_id'])
                units_mask = np.isin(all_units, dataset.neurons.unit_ids)
                units, pvals = all_units[units_mask], all_pvals[units_mask]
                assert np.all(
                    units == dataset.neurons.unit_ids), 'order of neurons has changed'
                pval_thresh = np.power(10, float(key['p_val_power']))
                selection = pvals < pval_thresh
                log.info('Subsampling to {} neurons with BootstrapOracleTTest p-val < {:.0E}'.format(
                    selection.sum(), pval_thresh))
                dataset.transforms.insert(
                    -1, Subsample(np.where(selection)[0]))

                assert np.all(dataset.neurons.unit_ids ==
                            units[selection]), 'Units are inconsistent'
            return datasets, loaders
