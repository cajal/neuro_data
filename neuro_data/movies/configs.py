from collections import OrderedDict
from itertools import product, count
from pprint import pformat

import datajoint as dj
import numpy as np
from attorch.dataloaders import RepeatsBatchSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .data_schemas import MovieMultiDataset
from .schema_bridge import stimulus, experiment
from .transforms import Subsample, Normalizer, ToTensor, Subsequence, ScaleInput, Resize
from .. import logger as log
from ..utils.config import ConfigBase, fixed_seed
from ..utils.sampler import BalancedSubsetSampler, SampledSubsetRandomSampler, SampledSubsetSequentialSampler
from ..common import configs as common_configs

schema = dj.schema('neurodata_movie_configs', locals())


class DataLoaderTFirst(DataLoader):
    def __iter__(self):
        for x in super().__iter__():
            yield list(
                map(lambda xf: xf[0].permute(2, 0, 1, 3, 4)
                    if xf[1] == 'inputs' else xf[0].permute(1, 0, 2),
                    zip(x, self.dataset.data_point._fields)))


class StimulusTypeMixin:

    def add_transforms(self, key, datasets, tier, exclude=None, normalize=True, cuda=False):
        log.info('Adding transforms'.ljust(80, '-'))
        if exclude is not None:
            log.info('Excluding "' + '", "'.join(exclude) +
                     '" from normalization')
        for k, dataset in datasets.items():
            for ex in exclude:
                assert ex in dataset.data_groups, '{} not in data_groups'.format(
                    ex)
            transforms = []
            if 'seq_len' in key and key['seq_len'] is not None:
                transforms.append(Subsequence(key['seq_len']))
            elif tier == 'train' and 'train_seq_len' in key:
                transforms.append(Subsequence(key['train_seq_len']))
            else:
                log.warning('No subsquence transform will be added to the dataset!')

            if normalize:
                log.info('Using normalization={}'.format(normalize))
                transforms.append(Normalizer(
                    dataset, stats_source=key['stats_source'], exclude=exclude))
            transforms.append(ToTensor(cuda=cuda))
            dataset.transforms = transforms

        return datasets

    def get_constraint(self, dataset, stimulus_type, tier=None):
        constraint = np.zeros(len(dataset.types), dtype=bool)
        for const in map(lambda s: s.strip(), stimulus_type.split('|')):
            if '(' in const:
                start, stop = const.index('('), const.index(')')
                const, modifier = const[:start], const[start + 1:stop]
                assert const == 'stimulus.Clip', 'Do not support modifiers for movies other than stimulus.Clip'
            else:
                modifier = None

            if const.startswith('~'):
                tmp = (dataset.types != const[1:])
            else:
                tmp = (dataset.types == const)

            if modifier is not None:
                if modifier.startswith('~'):
                    comp = '<>'
                    modifier = modifier[1:]
                else:
                    comp = '='

                rhashes = (stimulus.Clip() * stimulus.Movie()
                           & 'movie_class {} "{}"'.format(comp, modifier)).fetch('condition_hash')
                tmp &= np.isin(dataset.condition_hashes, rhashes)
                ch = dataset.condition_hashes[tmp]
                rel = stimulus.Clip * \
                    stimulus.Movie & 'condition_hash in ("{}")'.format(
                        '","'.join(ch))
                log.info('\tRestricted movie classes are: {}'.format(
                    ', '.join(np.unique(rel.fetch('movie_class')))))

            constraint = constraint | tmp
        if tier is not None:
            constraint = constraint & (dataset.tiers == tier)
        return constraint

    def get_sampler_class(self, tier, balanced=False):
        assert tier in ['train', 'validation', 'test', None]
        if tier == 'train':
            if not balanced:
                Sampler = SampledSubsetRandomSampler
            else:
                Sampler = BalancedSubsetSampler
        else:
            Sampler = SampledSubsetSequentialSampler
        return Sampler

    def log_loader(self, loader):
        log.info('Loader sampler is {}'.format(
            loader.sampler.__class__.__name__))
        log.info('Number of samples in the loader will be {}'.format(
            len(loader.sampler)))
        log.info(
            'Number of batches in the loader will be {}'.format(int(np.ceil(len(loader.sampler) / loader.batch_size))))

    def get_loaders(self, datasets, tier, batch_size, stimulus_types, balanced=False,
                    merge_noise_types=True, shrink_to_same_size=False, Sampler=None,
                    t_first=False, train_iterations=None):
        if Sampler is None:
            Sampler = self.get_sampler_class(tier, balanced)

        log.info('Assembling loaders'.ljust(80, '-'))
        if not isinstance(stimulus_types, list):
            log.info('Using ' + stimulus_types +
                     ' as stimulus type for all datasets')
            stimulus_types = len(datasets) * [stimulus_types]
        elif len(stimulus_types) == 1 and len(datasets) > 1:
            log.info('Using ' + stimulus_types[0] +
                     ' as stimulus type for all datasets')
            stimulus_types = len(datasets) * stimulus_types
        elif len(datasets) % len(stimulus_types) == 0:
            log.info('Using [' + ",".join(stimulus_types) +
                     '] as stimulus type for all datasets')
            stimulus_types = (
                len(datasets) // len(stimulus_types)) * stimulus_types
        else:
            assert len(stimulus_types) == len(datasets), \
                'Number of requested types does not match number of datasets. You need to choose a different group'

        log.info('Stimulus sources: ' + ', '.join(stimulus_types))

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]
        if shrink_to_same_size:
            if tier is None or tier in ('train', 'validation'):
                min_n = np.min([e.sum() for e in constraints])
                new_con = []
                for i, c, st in zip(count(), constraints, stimulus_types):
                    if not '|' in st:
                        c2 = c & (c.cumsum() <= min_n)
                    else:
                        c2 = c
                    untouched = c == c2
                    for c3 in constraints[i + 1:]:
                        c3 &= untouched
                    new_con.append(c2)
                constraints = new_con
                log.info(
                    'Shrinking each type in {} sets to same size of {}'.format(tier, min_n))

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            log.info('Selecting trials from ' +
                     stimulus_type + ' and tier=' + repr(tier))
            ix = np.where(constraint)[0]
            log.info('Found {} active trials'.format(constraint.sum()))
            if Sampler is BalancedSubsetSampler:
                log.info("Balancing Clip vs. Rest")
                types = np.array(
                    [('Clip' if t == 'stimulus.Clip' else 'Noise') for t in dataset.types])
                sampler = Sampler(ix, types)
            elif Sampler in (SampledSubsetRandomSampler, SampledSubsetSequentialSampler):
                if (train_iterations is None) or (tier != 'train'):
                    num_samples = len(ix)
                else:
                    num_samples = batch_size * train_iterations
                log.info('Number samples per epoch = {}'.format(num_samples))
                sampler = Sampler(ix, num_samples=num_samples)
            else:
                sampler = Sampler(ix)
            if t_first:
                log.info('Time in first dimension')
                loaders[k] = DataLoaderTFirst(
                    dataset, sampler=sampler, batch_size=batch_size)
            else:
                log.info('Batch in first dimension')
                loaders[k] = DataLoader(
                    dataset, sampler=sampler, batch_size=batch_size)
            self.log_loader(loaders[k])
        return loaders

    def load_data(self, key, stimulus_types, tier=None, batch_size=1, key_order=None,
                  normalize=True, exclude_from_normalization=None,
                  balanced=False, shrink_to_same_size=False, cuda=False,
                  Sampler=None, t_first=False, train_iterations=None):
        log.info('Loading {} datasets with tier {}'.format(
            pformat(stimulus_types, indent=20), tier))
        datasets = MovieMultiDataset().fetch_data(key, key_order=key_order)
        log.info('Adding stats source'.ljust(80, '-'))
        for k, dat in datasets.items():
            if 'stats_source' in key:
                log.info(
                    'Adding stats_source "{stats_source}" to dataset '.format(**key))
                dat.stats_source = key['stats_source']

        log.info('Using statistics source ' + key['stats_source'])
        datasets = self.add_transforms(key, datasets, tier, exclude=exclude_from_normalization,
                                       normalize=normalize, cuda=cuda)
        loaders = self.get_loaders(
            datasets, tier, batch_size, stimulus_types=stimulus_types,
            balanced=balanced, shrink_to_same_size=shrink_to_same_size,
            Sampler=Sampler, t_first=t_first, train_iterations=train_iterations)
        return datasets, loaders


class AreaLayerMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, cuda=False,
                  Sampler=None, t_first=False, train_iterations=None, **kwargs):
        log.info('Ignoring {} when loading {}'.format(
            pformat(kwargs, indent=20), self.__class__.__name__))
        shrink = key.pop('shrink', False)
        balanced = key.pop('balanced', False)

        datasets, loaders = super().load_data(key, key.pop('stimulus_type').split(','),
                                              tier, batch_size, key_order,
                                              exclude_from_normalization=key.pop(
                                                  'exclude').split(','),
                                              normalize=key.pop('normalize'),
                                              balanced=balanced,
                                              shrink_to_same_size=shrink,
                                              cuda=cuda, Sampler=Sampler, t_first=t_first,
                                              train_iterations=train_iterations)

        def area_layer_idx(areas, layers):
            if 'brain_area' in key:
                log.info('Subsampling to layer "{layer}" and area "{brain_area}"'.format(**key))
                idx = np.where((layers == key['layer']) & (areas == key['brain_area']))[0]
            elif 'brain_areas' in key:
                log.info('Subsampling to layer "{layer}" and areas "{brain_areas}"'.format(**key))
                brain_areas_mask = False
                for brain_area in (common_configs.BrainAreas.BrainArea & key).fetch('brain_area'):
                    brain_areas_mask = brain_areas_mask | (areas == brain_area)
                idx = np.where((layers == key['layer']) & brain_areas_mask)[0]
            else:
                raise Exception('brain area key not recognized')
            return idx

        for readout_key, dataset in datasets.items():
            areas = dataset.neurons.area
            layers = dataset.neurons.layer
            idx = area_layer_idx(areas, layers)
            if len(idx) == 0:
                log.warning('Empty set of neurons. Deleting this key')
                del datasets[readout_key]
                del loaders[readout_key]
            else:
                dataset.transforms.insert(-1, Subsample(idx))

        return datasets, loaders


class AreaLayerReliableMixin(AreaLayerMixin):
    def load_data(self, key, tier=None, batch_size=1, seq_len=None, Sampler=None, t_first=False,
                  cuda=False, scale_input=False, train_iterations=None, **kwargs):
        log.info('Ignoring {} when loading {}'.format(
            pformat(kwargs, indent=20), self.__class__.__name__))

        from .stats import BootstrapOracleTTest
        key['seq_len'] = seq_len
        assert tier in [None, 'train', 'validation', 'test']
        datasets, loaders = super().load_data(
            key, tier=tier, batch_size=batch_size, Sampler=Sampler,
            t_first=t_first, cuda=cuda, train_iterations=train_iterations)
        for rok, dataset in datasets.items():
            member_key = (MovieMultiDataset.Member() & key &
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
            if scale_input:
                log.info('Scaling Input to [0, 1]')
                dataset.transforms.insert(-1, ScaleInput())
            assert np.all(dataset.neurons.unit_ids ==
                          units[selection]), 'Units are inconsistent'
        return datasets, loaders


@schema
class DataConfig(ConfigBase, dj.Lookup):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, oracle=False, keep_transforms=False, **kwargs):
        assert not keep_transforms or oracle, 'keep_transforms should only be true when oracle is true'
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            log.info('Placing oracle data samplers')
            for readout_key, loader in loaders.items():
                ix = loader.sampler.indices
                condition_hashes = datasets[readout_key].condition_hashes
                log.info('Replacing ' + loader.sampler.__class__.__name__ +
                         ' with RepeatsBatchSampler')
                Loader = loader.__class__
                loaders[readout_key] = Loader(loader.dataset,
                                              batch_sampler=RepeatsBatchSampler(condition_hashes, subset_index=ix))

                if not keep_transforms:
                    datasets[readout_key].transforms = \
                        [tr for tr in datasets[readout_key].transforms if isinstance(
                            tr, (Subsample, ToTensor))]
        return datasets, loaders

    class AreaLayer(dj.Part, AreaLayerMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512) # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip',
                              'stimulus.Clip(unreal)', 'stimulus.Clip(~unreal)',
                              'stimulus.Clip|~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             [30 * 5],
                             ['L2/3'],
                             ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerPercentOracle(dj.Part, AreaLayerMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512)  # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        (oracle_source) -> master
        -> experiment.Layer
        -> anatomy.Area
        percent_low             : tinyint       # percent oracle lower cutoff
        percent_high            : tinyint      # percent oracle upper cutoff
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "Like AreaLayer but only {percent_low}-{percent_high} percent best oracle neurons computed on {oracle_source}".format(
                **key)

        @property
        def content(self):
            oracle_source = list((DataConfig.AreaLayer() &
                                  [dict(brain_area='V1', layer='L2/3',
                                        train_seq_len=150, normalize=True,
                                        stats_source='all', stimulus_type='stimulus.Clip',
                                        exclude='inputs,responses'),
                                   dict(brain_area='V1', layer='L2/3',
                                        train_seq_len=150, normalize=True,
                                        stats_source='all', stimulus_type='~stimulus.Clip',
                                        exclude='inputs,responses')]).fetch('data_hash'))
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [25],
                             [75]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [75],
                             [100]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             oracle_source,
                             ['L2/3'],
                             ['V1'],
                             [0],
                             [100]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, seq_len=None, Sampler=None, t_first=False, cuda=False,
                      **kwargs):
            log.info('Ignoring {} when loading {}'.format(
                pformat(kwargs, indent=20), self.__class__.__name__))

            from .stats import Oracle
            key['seq_len'] = seq_len
            datasets, loaders = super().load_data(
                key, tier=tier, batch_size=batch_size, Sampler=Sampler,
                t_first=t_first, cuda=cuda)
            for rok, dataset in datasets.items():
                member_key = (MovieMultiDataset.Member() & key &
                              dict(name=rok)).fetch1(dj.key)

                okey = dict(key, **member_key)
                okey['data_hash'] = okey.pop('oracle_source')
                units, pearson = (Oracle.UnitPearson() & okey).fetch(
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
                dataset.transforms.insert(
                    -1, Subsample(np.where(selection)[0]))

                assert np.all(dataset.neurons.unit_ids ==
                              units[selection]), 'Units are inconsistent'
            return datasets, loaders

    class AreaLayerReliable(dj.Part, AreaLayerReliableMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512) # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        -> experiment.Layer
        -> anatomy.Area
        p_val_power             : tinyint      # 10^(p_val_power) is p-val threshold
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "Like AreaLayer but only neurons that have significantly different (p-val < {:.0E}) response-oracle correlations to the same stimuli vs different stimuli".format(
                np.power(10, float(key['p_val_power'])))

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             ['L2/3'],
                             ['V1'],
                             [-3]):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreasLayerReliable(dj.Part, AreaLayerReliableMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512) # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        -> experiment.Layer
        -> common_configs.BrainAreas
        p_val_power             : tinyint       # 10^(p_val_power) is p-val threshold
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "Like AreaLayer but only neurons that have significantly different (p-val < {:.0E}) response-oracle correlations to the same stimuli vs different stimuli".format(
                np.power(10, float(key['p_val_power'])))

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             ['L2/3'],
                             ['V1+LM+LI+AL+RL'],
                             [-3]):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerSubset(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512)  # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        data_seed               : int          # seed for subset selection
        seconds                 : int          # maximal length in seconds
        neurons                 : int          # maximal length in seconds
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return """subset from {brain_area} {layer} on {stimulus_type}. normalize={normalize}
            on {stats_source} (except '{exclude}') with {seconds}s duration and {neurons} neurons
             and seed {data_seed}""".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip|~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             [30 * 5],
                             ['L2/3'],
                             ['V1'],
                             [42],
                             [s * 60 for s in [20, 40, 60]],
                             [100, 250, 500, 1000, 2000]
                             ):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None, cuda=False, **kwargs):
            log.info('Ignoring {} when loading {}'.format(
                pformat(kwargs, indent=20), self.__class__.__name__))

            datasets, loaders = super().load_data(key, key.pop('stimulus_type').split(','),
                                                  tier, batch_size, key_order,
                                                  exclude_from_normalization=key.pop(
                                                      'exclude').split(','),
                                                  normalize=key.pop(
                                                      'normalize'),
                                                  balanced=False,
                                                  shrink_to_same_size=False,
                                                  cuda=cuda)
            log.info(
                'Subsampling to layer "{layer}", area "{brain_area}", {neurons} neurons'.format(**key))

            for rok, loader, dataset in zip(loaders.keys(), loaders.values(), datasets.values()):

                with fixed_seed(key['data_seed']):
                    # subsample neurons
                    layers = dataset.neurons.layer
                    areas = dataset.neurons.area
                    idx = np.where((layers == key['layer']) & (
                        areas == key['brain_area']))[0]
                    assert len(
                        idx) >= key['neurons'], 'number of requested neurons exceeds available neurons'
                    selection = np.random.permutation(
                        len(idx))[:key['neurons']]
                    idx = idx[selection]
                    dataset.transforms.insert(-1, Subsample(idx))

                if tier is None or tier == 'train':
                    log.info(
                        'Subsampling to {seconds}s of trial time'.format(**key))
                    with fixed_seed(key['data_seed']):
                        idx = loader.sampler.indices
                        durations = dataset.durations[idx]

                        assert durations.sum(
                        ) > key['seconds'], 'trial durations are not enough to cover {seconds}s'.format(**key)

                        selection = np.random.permutation(len(durations))
                        total_duration = np.cumsum(durations[selection])
                        selection = selection[total_duration <= key['seconds']]

                        gap = durations[selection].sum() - key['seconds']
                        if gap != 0:
                            log.warning(
                                '{gap}s gap between requested stimulus length and actual stimulus length.'.format(
                                    gap=gap))
                        log.info('Using {} trials'.format(
                            (total_duration <= key['seconds'])).sum())
                        log.info(
                            'Replacing ' + loader.sampler.__class__.__name__ + ' with RandomSubsetSampler')
                        Loader = loader.__class__
                        loaders[rok] = Loader(
                            loader.dataset, sampler=SubsetRandomSampler(idx[selection]))

            return datasets, loaders

    class AreaLayerMultiSource(dj.Part, AreaLayerMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512)  # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        train_seq_len           : smallint     # training sequence length in frames
        balanced                : bool         # whether to use balanced samplers
        shrink                  : bool         # whether to shrink all datasets to the same number of trial
        -> experiment.Layer
        -> anatomy.Area
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}'). balanced={balanced} and shrink={shrink}".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip',
                              '~stimulus.Clip,stimulus.Clip,stimulus.Clip|~stimulus.Clip',
                              'stimulus.Clip(unreal),stimulus.Clip(~unreal),stimulus.Clip',
                              'stimulus.Clip(~unreal),stimulus.Clip(unreal),stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             [30 * 5],
                             [True],
                             [False],
                             ['L2/3'],
                             ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))
