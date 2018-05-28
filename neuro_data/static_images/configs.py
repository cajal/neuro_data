from collections import OrderedDict
from itertools import product, count
from attorch.dataloaders import RepeatsBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from .data_schemas import StaticMultiDataset
from .transforms import Subsample, Normalizer, ToTensor
from ..utils.sampler import SubsetSequentialSampler
from ..utils.config import ConfigBase
import datajoint as dj
from .. import logger as log

import numpy as np
from torch.utils.data import DataLoader

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

schema = dj.schema('neurodata_static_configs', locals())


class StimulusTypeMixin:
    _stimulus_type = None

    def add_transforms(self, key, datasets, tier, exclude=None):
        if exclude is not None:
            log.info('Excluding "{}" from normalization'.format('", "'.join(exclude)))
        for k, dataset in datasets.items():
            transforms = []
            transforms.extend([Normalizer(dataset, stats_source=key['stats_source'], exclude=exclude), ToTensor()])
            dataset.transforms = transforms

        return datasets

    def get_constraint(self, dataset, stimulus_type, tier=None):
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

    def get_loaders(self, datasets, tier, batch_size, stimulus_types=None):
        if stimulus_types is None:
            log.info('Using {} as stimulus type for all datasets'.format(self._stimulus_type))
            stimulus_types = len(datasets) * [self._stimulus_type]
        if not isinstance(stimulus_types, list):
            log.info('Using {} as stimulus type for all datasets'.format(self._stimulus_type))
            stimulus_types = len(datasets) * [stimulus_types]

        log.info('Stimulus sources: "{}"'.format('","'.join(stimulus_types)))

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            log.info('Selecting trials from {} and tier={}'.format(stimulus_type, tier))
            ix = np.where(constraint)[0]
            log.info('Found {} active trials'.format(constraint.sum()))
            if tier == 'train':
                log.info("Configuring random subset sampler for " + k)
                loaders[k] = DataLoader(dataset, sampler=SubsetRandomSampler(ix), batch_size=batch_size)
            else:
                log.info("Configuring sequential subset sampler for " + k)
                loaders[k] = DataLoader(dataset, sampler=SubsetSequentialSampler(ix), batch_size=batch_size)
                log.info('Number of samples in the loader will be {}'.format(len(loaders[k].sampler)))
                log.info('Number of batches in the loader will be {}'.format(
                    int(np.ceil(len(loaders[k].sampler) / loaders[k].batch_size))))
        return loaders

    def load_data(self, key, tier=None, batch_size=1, key_order=None,
                  exclude_from_normalization=None, stimulus_types=None):
        log.info('Loading {} dataset with tier={}'.format(self._stimulus_type, tier))
        datasets = StaticMultiDataset().fetch_data(key, key_order=key_order)
        for k, dat in datasets.items():
            if 'stats_source' in key:
                log.info('Adding stats_source "{stats_source}" to dataset'.format(**key))
                dat.stats_source = key['stats_source']

        log.info('Using statistics source ' +  key['stats_source'])
        datasets = self.add_transforms(key, datasets, tier, exclude=exclude_from_normalization)
        loaders = self.get_loaders(datasets, tier, batch_size, stimulus_types=stimulus_types)
        return datasets, loaders


class AreaLayerRawMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None):
        datasets, loaders = super().load_data(key, tier, batch_size, key_order,
                                              exclude_from_normalization=self._exclude_from_normalization,
                                              stimulus_types=stimulus_types)

        log.info('Subsampling to layer "{layer}" and area "{brain_area}"'.format(**key))
        for readout_key, dataset in datasets.items():
            layers = dataset.neurons.layer
            areas = dataset.neurons.area
            idx = np.where((layers == key['layer']) & (areas == key['brain_area']))[0]
            dataset.transforms.insert(-1, Subsample(idx))
        return datasets, loaders


@schema
class DataConfig(ConfigBase, dj.Lookup):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, oracle=False, **kwargs):
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            log.info('Placing oracle data samplers')
            for readout_key, loader in loaders.items():
                ix = loader.sampler.indices
                condition_hashes = datasets[readout_key].condition_hashes
                log.info('Replacing', loader.sampler.__class__.__name__, 'with RepeatsBatchSampler', depth=1)
                loader.sampler = None

                datasets[readout_key].transforms = \
                    [tr for tr in datasets[readout_key].transforms if isinstance(tr, (Subsample, ToTensor))]
                loader.batch_sampler = RepeatsBatchSampler(condition_hashes, subset_index=ix)
        return datasets, loaders

    class AreaLayerRawNatural(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Frame'
        _exclude_from_normalization = ['images', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} and {} only. Unnormalized images and responses.".format(self._stimulus_type,
                                                                                                **key)

        @property
        def content(self):
            for p in product(['all'], ['L4', 'L2/3'], ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerRawNoise(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = '~stimulus.Frame'
        _exclude_from_normalization = ['images', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} and {} only. Unnormalized images and responses.".format(self._stimulus_type,
                                                                                                **key)

        @property
        def content(self):
            for p in product(['all'], ['L4', 'L2/3'], ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))

