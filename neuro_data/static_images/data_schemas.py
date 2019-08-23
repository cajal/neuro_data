from collections import OrderedDict
from functools import partial
from itertools import count, compress
from pprint import pformat

import datajoint as dj
import numpy as np
import pandas as pd

from .datasets import StaticImageSet
from .. import logger as log
from ..utils.data import h5cached, SplineCurve, FilterMixin, fill_nans, NaNSpline

dj.config['external-data'] = dict(
    protocol='file',
    location='/external/')

# Day 1: 2-24 ImageNet - used to generate MEIs, 3-7 Repeat ImageNet
# Day 2: 4-19 MEIs - incorrect depths, 4-29 Repeat ImageNet - incorrect depths
# Day 3: 5-26 MEIs,  6-1 Repeat ImageNet
# Day 4: 7-23 MEIs, 7-29 Repeat ImageNet
STATIC = [
    '(animal_id=11521 AND session=7 AND scan_idx=1)',
    '(animal_id=11521 AND session=7 AND scan_idx=2)',
    '(animal_id=16157 AND session=5 AND scan_idx=5)',
    '(animal_id=16157 AND session=5 AND scan_idx=6)',
    '(animal_id=16312 AND session=3 AND scan_idx=20)',
    '(animal_id=18765 AND session=4 AND scan_idx=6)',
    '(animal_id=18765 AND session=7 AND scan_idx=17)',
    '(animal_id=21067 AND session=15 AND scan_idx=9)',  # 360 images x 20 repeatitions (for Zhe)
]

MEI_STATIC = [
    '(animal_id=20505 AND session=2 AND scan_idx=24)', # loop 0 day 1 (Tue) source ImageNet
    '(animal_id=20505 AND session=3 AND scan_idx=7)',  # loop 0 day 1 (Tue) repeat ImageNet repeat
    '(animal_id=20505 AND session=5 AND scan_idx=26)', # loop 0 day 3 (Thu) MEI
    '(animal_id=20505 AND session=6 AND scan_idx=1)',  # loop 0 day 3 (Thu) repeat ImageNet
    '(animal_id=20505 AND session=7 AND scan_idx=23)', # loop 0 day 4 (Fri) MEI
    '(animal_id=20505 AND session=7 AND scan_idx=29)', # loop 0 day 4 (Fri) repeat ImageNet

    '(animal_id=20457 AND session=5 AND scan_idx=9)',  # loop 1 day 1 (Thu) source ImageNet
    '(animal_id=20457 AND session=5 AND scan_idx=17)', # loop 1 day 1 (Thu) repeat ImageNet
    '(animal_id=20457 AND session=5 AND scan_idx=27)', # loop 1 day 1 (Thu) Monet
    #'(animal_id=20457 AND session=7 AND scan_idx=4)',  # loop 1 day 2 (Fri) MEI, sync failed
    '(animal_id=20457 AND session=7 AND scan_idx=10)', # loop 1 day 2 (Fri) repeat ImageNet,
    '(animal_id=20457 AND session=7 AND scan_idx=16)', # loop 1 day 2 (Fri) Monet,
    '(animal_id=20457 AND session=8 AND scan_idx=9)',  # loop 1 day 3 (Mon) MEI,
    '(animal_id=20457 AND session=8 AND scan_idx=12)', # loop 1 day 3 (Mon) repeat ImageNet
    '(animal_id=20457 AND session=8 AND scan_idx=22)', # loop 1 day 3 (Mon) Monet

    '(animal_id=20505 AND session=10 AND scan_idx=14)',  # loop 2 day 1 (Tue) source ImageNet
    '(animal_id=20505 AND session=10 AND scan_idx=19)',  # loop 2 day 1 (Tue) repeat ImageNet
    #'(animal_id=20505 AND session=11 AND scan_idx=7)',   # loop 2 day 2 (Wed) MEI - BAD: mouse not awake
    '(animal_id=20505 AND session=11 AND scan_idx=16)',  # loop 2 day 2 (Wed) repeat ImageNet
    '(animal_id=20505 AND session=12 AND scan_idx=16)',  # loop 2 day 3 (Thu) MEI
    '(animal_id=20505 AND session=12 AND scan_idx=29)',  # loop 2 day 3 (Thu) repeat ImageNet
    '(animal_id=20505 AND session=14 AND scan_idx=4)',   # loop 2 day 4 (Thu) MEI
    '(animal_id=20505 AND session=14 AND scan_idx=33)',  # loop 2 day 4 (Thu) repeat ImageNet

    '(animal_id=20210 AND session=4 AND scan_idx=11)',  # loop 3 day 1 (Tue) source ImageNet
    #'(animal_id=20210 AND session=4 AND scan_idx=20)',  # loop 3 day 1 (Tue) ImageNet (alternative set of images)
    #'(animal_id=20210 AND session=5 AND scan_idx=26)',  # loop 3 day 2 (Wed) MEI, eye secretion for half the scan
    '(animal_id=20210 AND session=5 AND scan_idx=16)',  # loop 3 day 2 (Wed) repeat ImageNet
    '(animal_id=20210 AND session=7 AND scan_idx=10)',  # loop 3 day 3 (Thu) MEI
    '(animal_id=20210 AND session=7 AND scan_idx=14)',  # loop 3 day 3 (Thu) repeat ImageNet
    #'(animal_id=20210 AND session=8 AND scan_idx=11)',  # loop 3 day 4 (Fri) Masked MEI vs Masked ImageNet, masking was wrong
    '(animal_id=20210 AND session=8 AND scan_idx=17)',  # loop 3 day 4 (Fri) repeat ImageNet

    '(animal_id=20892 AND session=3 AND scan_idx=14)',  # loop 4 day 1 (Tue, Jan 29) source ImageNet
    #'(animal_id=20892 AND session=4 AND scan_idx=11)',  # loop 4 day 2 (Wed) MEI, kind of big bubble
    '(animal_id=20892 AND session=4 AND scan_idx=16)',  # loop 4 day 2 (Wed) repeat ImageNet, small bubble
    '(animal_id=20892 AND session=5 AND scan_idx=18)',  # loop 4 day 3 (Thu) MEI
    #'(animal_id=20892 AND session=5 AND scan_idx=29)',  # loop 4 day 3 (Thu) repeat ImageNet, mouse was sleep half of the time
    '(animal_id=20892 AND session=6 AND scan_idx=17)',  # loop 4 day 4 (Fri) MEI, small bubble
    '(animal_id=20892 AND session=6 AND scan_idx=24)',  # loop 4 day 4 (Fri) repeat ImageNet

    '(animal_id=21067 AND session=9 AND scan_idx=17)',  # loop 5 day 1 (Tue) source ImageNet
    # '(animal_id=21067 AND session=9 AND scan_idx=23)',  # loop 5 day 1 (Tue) ImageNet (alternative set of images), Sync failed, do not use
    '(animal_id=21067 AND session=10 AND scan_idx=14)', # loop 5 day 2 (Wed) MEI
    '(animal_id=21067 AND session=10 AND scan_idx=18)', # loop 5 day 2 (Wed) repeat ImageNet
    # '(animal_id=21067 AND session=11 AND scan_idx=12)', # loop 5 day 3 (Thu) MEI vs Gabor
    '(animal_id=21067 AND session=11 AND scan_idx=21)', # loop 5 day 3 (Thu) repeat ImageNet
    #'(animal_id=21067 AND session=12 AND scan_idx=11)', # loop 5 day 4 (Fri) Masked MEI vs Masked ImageNet
    '(animal_id=21067 AND session=12 AND scan_idx=15)', # loop 5 day 4 (Fri) repeat ImageNet
    # '(animal_id=21067 AND session=13 AND scan_idx=10)', # loop 5 day 5 (Mon) Masked MEI vs Unmasked Imagenet
    '(animal_id=21067 AND session=13 AND scan_idx=14)', # loop 5 day 5 (Mon) repeat ImageNet
]

HIGHER_AREAS = [
    '(animal_id=20892 AND session=9 AND scan_idx=10)', # ImageNet, single depth, big FOV, mostly V1
    '(animal_id=20892 AND session=9 AND scan_idx=11)', # ImageNet, single depth, big FOV, mostly V1
    '(animal_id=20892 AND session=10 AND scan_idx=10)', # ImageNet, V1+LM+AL+RL in a single rectangular FOV
    '(animal_id=21553 AND session=11 AND scan_idx=10)', # ImageNet, V1+LM+AL+RL in a single rectangular FOV
    '(animal_id=21844 AND session=2 AND scan_idx=12)', # ImageNet, V1+LM+AL+RL in four distinct rois
    '(animal_id=22085 AND session=2 AND scan_idx=20)', # ImageNet, V1+LM+AL+RL in four distinct rois
    '(animal_id=22083 AND session=7 AND scan_idx=21)', # ImageNet, V1+LM+AL+RL in four distinct rois
]

STATIC = STATIC + MEI_STATIC + HIGHER_AREAS

# set of attributes that uniquely identifies the frame content
UNIQUE_FRAME = {
    'stimulus.Frame': ('image_id', 'image_class'),
    'stimulus.MonetFrame': ('rng_seed', 'orientation'),
    'stimulus.TrippyFrame': ('rng_seed',),
}

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
beh = dj.create_virtual_module('behavior', 'pipeline_behavior')
pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
vis = dj.create_virtual_module('vis', 'pipeline_vis')
maps = dj.create_virtual_module('maps', 'pipeline_map')
shared = dj.create_virtual_module('shared', 'pipeline_shared')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
mesonet = dj.create_virtual_module('mesonet', 'cortex_ex_machina_mesonet_data')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')

schema = dj.schema('neurodata_static')

extra_info_types = {
    'condition_hash':'S',
    'trial_idx':int,
    'trial_idx':int,
    'animal_id':int,
    'session':int,
    'scan_idx':int,
    'image_class':'S',
    'image_id':int,
    'pre_blank_period':float,
    'presentation_time':float,
    'last_flip':int,
    'trial_ts':'S',
    'contrast_x':float,
    'rng_seed_x':float,
    'pattern_width':float,
    'pattern_aspect':float,
    'ori_coherence':float,
    'ori_mix':float,
    'orientation':float,
    'contrast_y':float,
    'rng_seed_y':float,
    'tex_ydim':float,
    'tex_xdim':float,
    'xnodes':float,
    'ynodes':float,
    'up_factor':float,
    'spatial_freq':float
}


@schema
class StaticScan(dj.Computed):
    definition = """
    # gatekeeper for scan and preprocessing settings
    
    -> fuse.ScanDone
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master        
        unit_id              : int                          # unique per scan & segmentation method
        ---
        -> fuse.ScanSet.Unit
        """

    key_source = fuse.ScanDone() & STATIC & 'spike_method=5 and segmentation_method=6'

    @staticmethod
    def complete_key(key):
        return dict((dj.U('segmentation_method', 'pipe_version') \
                     & (meso.ScanSet.Unit() & key)).fetch1(dj.key), **key)

    def make(self, key):
        self.insert(fuse.ScanDone() & key, ignore_extra_fields=True)
        pipe = (fuse.ScanDone() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)
        self.Unit().insert(fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type & key
                           & dict(pipe_version=1, segmentation_method=6, spike_method=5, type='soma'),
                           ignore_extra_fields=True)


@schema
class Tier(dj.Lookup):
    definition = """
    tier        : varchar(20)
    ---
    """

    @property
    def contents(self):
        yield from zip(["train", "test", "validation"])


@schema
class ExcludedTrial(dj.Manual):
    definition = """
    # trials to be excluded from analysis
    -> stimulus.Trial
    ---
    exclusion_comment='': varchar(64)   # reasons for exclusion
    """

@schema
class ConditionTier(dj.Computed):
    definition = """
    # split into train, test, validation

    -> stimulus.Condition
    -> StaticScan
    ---
    -> Tier
    """

    @property
    def dataset_compositions(self):
        return dj.U('animal_id', 'session', 'scan_idx', 'stimulus_type', 'tier').aggr(
            self * stimulus.Condition(), n='count(*)')

    @property
    def key_source(self):
        # all static scan with at least on recorded trial
        return StaticScan() & stimulus.Trial()

    def check_train_test_split(self, frames, cond):
        stim = getattr(stimulus, cond['stimulus_type'].split('.')[-1])
        train_test = dj.U(*UNIQUE_FRAME[cond['stimulus_type']]).aggr(frames * stim, train='sum(1-test)',
                                                                     test='sum(test)') \
                     & 'train>0 and test>0'
        assert len(train_test) == 0, 'Train and test clips do overlap'

    def fill_up(self, tier, frames, cond, key, m):
        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            # all hashes that are in clips but not registered for that animal and have the right tier
            candidates = dj.U('condition_hash') & \
                         (self & (dj.U('condition_hash') & (frames - self)) & dict(tier=tier))
            keys = candidates.fetch(dj.key)
            d = m - n
            update = min(len(keys), d)

            log.info('Inserting {} more existing {} trials'.format(update, tier))
            for k in keys[:update]:
                k = (frames & k).fetch1(dj.key)
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            keys = (frames - self).fetch(dj.key)
            update = m - n
            log.info('Inserting {} more new {} trials'.format(update, tier))

            for k in keys[:update]:
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

    def make(self, key):
        log.info(80 * '-')
        log.info('Processing ' + pformat(key))
        # count the number of distinct conditions presented for each one of three stimulus types:
        # "stimulus.Frame","stimulus.MonetFrame", "stimulus.TrippyFrame"
        conditions = dj.U('stimulus_type').aggr(stimulus.Condition() & (stimulus.Trial() & key),
                                                count='count(*)') \
                     & 'stimulus_type in ("stimulus.Frame","stimulus.MonetFrame", "stimulus.TrippyFrame")'
        for cond in conditions.fetch(as_dict=True):
            # hack for compatibility with previous datasets
            if cond['stimulus_type'] == 'stimulus.Frame':

                # deal with ImageNet frames first
                log.info('Inserting assignment from Mesonet')
                assignment = dj.U('tier', 'image_id') & (stimulus.Frame * mesonet.MesonetSplit.proj(tier='type') & 'image_class = "imagenet"')

                targets = StaticScan * stimulus.Frame * assignment & (stimulus.Trial & key) & 'image_class = "imagenet"'
                print('Inserting {} imagenet conditions!'.format(len(targets)))
                self.insert(targets,
                            ignore_extra_fields=True)

                # deal with MEI images, assigning tier test for all images
                assignment = (stimulus.Frame() & 'image_class in ("cnn_mei", "lin_rf", "multi_cnn_mei", "multi_lin_rf")').proj(tier='"train"')
                self.insert(StaticScan * stimulus.Frame * assignment & (stimulus.Trial & key), ignore_extra_fields=True)

                # make sure that all frames were assigned
                remaining = (stimulus.Trial * stimulus.Frame & key) - self
                assert len(remaining) == 0, 'There are still unprocessed Frames'
                continue

            log.info('Checking condition {stimulus_type} (n={count})'.format(**cond))
            frames = (stimulus.Condition() * StaticScan() & key & cond).aggr(stimulus.Trial(), repeats="count(*)",
                                                                             test='count(*) > 4')
            self.check_train_test_split(frames, cond)

            m = len(frames)
            m_test = m_val = len(frames & 'test > 0') or max(m * 0.075, 1)
            log.info('Minimum test and validation set size will be {}'.format(m_test))
            log.info('Processing test conditions')

            # insert repeats as test trials
            self.insert((frames & dict(test=1)).proj(tier='"test"'), ignore_extra_fields=True)
            self.fill_up('test', frames, cond, key, m_test)

            log.info('Processing validation conditions')
            self.fill_up('validation', frames, cond, key, m_val)

            log.info('Processing training conditions')
            self.fill_up('train', frames, cond, key, m - m_test - m_val)


@schema
class Preprocessing(dj.Lookup):
    definition = """
    # settings for movie preprocessing

    preproc_id       : tinyint # preprocessing ID
    ---
    offset           : decimal(6,4) # offset to stimulus onset in s
    duration         : decimal(6,4) # window length in s
    row              : smallint     # row size of movies
    col              : smallint     # col size of movie
    filter           : varchar(24)  # filter type for window extraction
    gamma            : boolean      # whether to convert images to luminance values rather than pixel intensities
    """
    contents = [
        {'preproc_id': 0, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': False},  # this one was still processed with cropping
        {'preproc_id': 1, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': False},
        {'preproc_id': 2, 'offset': 0.05, 'duration': 0.5, 'row': 72, 'col': 128,
         'filter': 'hamming', 'gamma': False},
        {'preproc_id': 3, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': True},
    ]


def process_frame(preproc_key, frame):
    """
    Helper function that preprocesses a frame
    """
    import cv2
    imgsize = (Preprocessing() & preproc_key).fetch1('col', 'row')  # target size of movie frames
    log.info('Downsampling frame')
    if not frame.shape[0] / imgsize[1] == frame.shape[1] / imgsize[0]:
        log.warning('Image size would change aspect ratio.')
        # if frame.shape == (126, 216):
        #     log.warning('Using center crop')
        #     frame = frame[4:4 + 117, 4:4 + 208]
        # else:
        #     raise ValueError('Frame shape {} cannot be processed'.format(frame.shape))

    return cv2.resize(frame, imgsize, interpolation=cv2.INTER_AREA).astype(np.float32)



@schema
class Frame(dj.Computed):
    definition = """ # frames downsampled

    -> stimulus.Condition
    -> Preprocessing
    ---
    frame                : external-data   # frame processed
    """

    @property
    def key_source(self):
        return stimulus.Condition() * Preprocessing() & ConditionTier()

    def load_frame(self, key):

        if stimulus.Frame & key:
            assert (stimulus.Frame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.StaticImage.Image & (stimulus.Frame & key)).fetch1('image')
        elif stimulus.MonetFrame & key:
            assert (stimulus.MonetFrame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.MonetFrame & key).fetch1('img')
        elif stimulus.TrippyFrame & key:
            assert (stimulus.TrippyFrame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.TrippyFrame & key).fetch1('img')
        else:
            raise KeyError('Cannot find matching stimulus relation')

    def make(self, key):


        log.info(80 * '-')
        log.info('Processing key ' + pformat(dict(key)))

        # get original frame
        frame = self.load_frame(key)

        # preprocess the frame
        frame = process_frame(key, frame)

        # --- generate response sampling points and sample movie frames relative to it
        self.insert1(dict(key, frame=frame))


@h5cached('/external/cache/', mode='array', transfer_to_tmp=False,
          file_format='static{animal_id}-{session}-{scan_idx}-preproc{preproc_id}.h5')
@schema
class InputResponse(dj.Computed, FilterMixin):
    definition = """
    # responses of one neuron to the stimulus

    -> StaticScan
    -> Preprocessing
    ---
    """

    key_source = StaticScan() * Preprocessing() & Frame()

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> Frame
            ---
            row_id           : int             # row id in the response block
            """

    class ResponseBlock(dj.Part):
        definition = """
            -> master
            ---
            responses           : external-data   # response of one neurons for all bins
            """

    class ResponseKeys(dj.Part):
        definition = """
            -> master.ResponseBlock
            -> fuse.Activity.Trace
            ---
            col_id           : int             # col id in the response block
            """

    def load_traces_and_frametimes(self, key):
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        frame_times = (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

        soma = pipe.MaskClassification.Type() & dict(type='soma')

        spikes = (dj.U('field', 'channel') * pipe.Activity.Trace() * StaticScan.Unit() \
                  * pipe.ScanSet.UnitInfo() & soma & key)
        traces, ms_delay, trace_keys = spikes.fetch('trace', 'ms_delay', dj.key,
                                                    order_by='animal_id, session, scan_idx, unit_id')
        delay = np.fromiter(ms_delay / 1000, dtype=np.float)
        frame_times = (delay[:, None] + frame_times[None, :])
        traces = np.vstack([fill_nans(tr.astype(np.float32)).squeeze() for tr in traces])
        traces, frame_times = self.adjust_trace_len(traces, frame_times)
        return traces, frame_times, trace_keys

    def adjust_trace_len(self, traces, frame_times):
        trace_len, nframes = traces.shape[1], frame_times.shape[1]
        if trace_len < nframes:
            frame_times = frame_times[:, :trace_len]
        elif trace_len > nframes:
            traces = traces[:, :nframes]
        return traces, frame_times

    def get_trace_spline(self, key, sampling_period):
        traces, frame_times, trace_keys = self.load_traces_and_frametimes(key)
        log.info('Loaded {} traces'.format(len(traces)))

        log.info('Generating lowpass filters to {}Hz'.format(1 / sampling_period))
        h_trace = self.get_filter(sampling_period, np.median(np.diff(frame_times)), 'hamming',
                                  warning=False)
        # low pass filter
        trace_spline = SplineCurve(frame_times,
                                   [np.convolve(trace, h_trace, mode='same') for trace in traces], k=1, ext=1)
        return trace_spline, trace_keys, frame_times.min(), frame_times.max()

    @staticmethod
    def stimulus_onset(flip_times, duration):
        n_ft = np.unique([ft.size for ft in flip_times])
        assert len(n_ft) == 1, 'Found inconsistent number of fliptimes'
        n_ft = int(n_ft)
        log.info('Found {} flip times'.format(n_ft))

        assert n_ft in (2, 3), 'Cannot deal with {} flip times'.format(n_ft)

        stimulus_onset = np.vstack(flip_times)  # columns correspond to  clear flip, onset flip
        ft = stimulus_onset[np.argsort(stimulus_onset[:, 0])]
        if n_ft == 2:
            assert np.median(ft[1:, 0] - ft[:-1, 1]) < duration + 0.05, 'stimulus duration off by more than 50ms'
        else:
            assert np.median(ft[:, 2] - ft[:, 1]) < duration + 0.05, 'stimulus duration off by more than 50ms'
        stimulus_onset = stimulus_onset[:, 1]

        return stimulus_onset

    def make(self, scan_key):
        self.insert1(scan_key)
        # integration window size for responses
        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Sampling neural responses at {}s intervals'.format(duration))

        trace_spline, trace_keys, ftmin, ftmax = self.get_trace_spline(scan_key, duration)
        # exclude trials marked in ExcludedTrial
        log.info('Excluding {} trials based on ExcludedTrial'.format(len(ExcludedTrial() & scan_key)))
        flip_times, trial_keys = (Frame * (stimulus.Trial - ExcludedTrial) & scan_key).fetch('flip_times', dj.key,
                                                                           order_by='condition_hash')
        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(valid):
            log.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~valid).sum()))

        stimulus_onset = self.stimulus_onset(flip_times, duration)
        log.info('Sampling {} responses {}s after stimulus onset'.format(valid.sum(), sample_point))
        R = trace_spline(stimulus_onset[valid] + sample_point, log=True).T

        self.ResponseBlock.insert1(dict(scan_key, responses=R))
        self.ResponseKeys.insert([dict(scan_key, **trace_key, col_id=i) for i, trace_key in enumerate(trace_keys)])
        self.Input.insert([dict(scan_key, **trial_key, row_id=i)
                           for i, trial_key in enumerate(compress(trial_keys, valid))])

    def compute_data(self, key):
        key = dict((self & key).fetch1(dj.key), **key)
        log.info('Computing dataset for\n' + pformat(key, indent=20))

        # meso or reso?
        pipe = (fuse.ScanDone() * StaticScan() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)

        # get data relation
        include_behavior = bool(Eye.proj() * Treadmill.proj() & key)

        assert include_behavior, 'Behavior data is missing!'

        # make sure that including areas and layers does not decrease number of neurons
        assert len(pipe.ScanSet.UnitInfo() * experiment.Layer() * anatomy.AreaMembership() * anatomy.LayerMembership() & key) == \
               len(pipe.ScanSet.UnitInfo() & key), "AreaMembership decreases number of neurons"

        responses = (self.ResponseBlock & key).fetch1('responses')
        trials = Frame() * ConditionTier() * self.Input() * stimulus.Condition().proj('stimulus_type') & key
        hashes, trial_idxs, tiers, types, images = trials.fetch('condition_hash', 'trial_idx', 'tier',
                                                                'stimulus_type', 'frame', order_by='row_id')
        images = np.stack(images)
        if len(images.shape) == 3:
            log.info('Adding channel dimension')
            images = images[:, None, ...]
        hashes = hashes.astype(str)
        types = types.astype(str)

        # gamma correction
        if (Preprocessing & key).fetch1('gamma'):
            log.info('Gamma correcting images.')
            from staticnet_analyses import multi_mei

            if len(multi_mei.ClosestCalibration & key) == 0:
                raise ValueError('No ClosestMonitorCalibration for this scan.')
            f, f_inv = (multi_mei.ClosestCalibration & key).get_fs()
            images = f(images)


        # --- extract infomation for each trial
        extra_info = pd.DataFrame({'condition_hash':hashes, 'trial_idx':trial_idxs})
        dfs = OrderedDict()

        # add information about each stimulus
        for t in map(lambda x: x.split('.')[1], np.unique(types)):
            stim = getattr(stimulus, t)
            rel = stim() * stimulus.Trial() & key
            df = pd.DataFrame(rel.proj(*rel.heading.non_blobs).fetch())
            dfs[t] = df

        on = ['animal_id', 'condition_hash', 'scan_idx', 'session', 'trial_idx']
        for t, df in dfs.items():
            mapping = {c:(t.lower() + '_' + c) for c in set(df.columns) - set(on)}
            dfs[t] = df.rename(str, mapping)
        df = list(dfs.values())[0]
        for d in list(dfs.values())[1:]:
            df = df.merge(d, how='outer', on=on)
        extra_info = extra_info.merge(df, on=['condition_hash','trial_idx']) # align rows to existing data
        assert len(extra_info) == len(trial_idxs), 'Extra information changes in length'
        assert np.all(extra_info['condition_hash'] == hashes), 'Hash order changed'
        assert np.all(extra_info['trial_idx'] == trial_idxs), 'Trial idx order changed'
        row_info = {}

        for k in extra_info.columns:
            dt = extra_info[k].dtype
            if isinstance(extra_info[k][0], str):
                row_info[k] = np.array(extra_info[k], dtype='S')
            elif dt == np.dtype('O') or dt == np.dtype('<M8[ns]'):
                row_info[k] = np.array(list(map(repr, extra_info[k])), dtype='S')
            else:
                row_info[k] = np.array(extra_info[k])

        # extract behavior
        if include_behavior:
            pupil, dpupil, pupil_center, valid_eye = (Eye & key).fetch1('pupil', 'dpupil', 'center', 'valid')
            pupil_center = pupil_center.T
            treadmill, valid_treadmill = (Treadmill & key).fetch1('treadmill', 'valid')
            valid = valid_eye & valid_treadmill
            if np.any(~valid):
                log.warning('Found {} invalid trials. Reducing data.'.format((~valid).sum()))
                hashes = hashes[valid]
                images = images[valid]
                responses = responses[valid]
                trial_idxs = trial_idxs[valid]
                tiers = tiers[valid]
                types = types[valid]
                pupil = pupil[valid]
                dpupil = dpupil[valid]
                pupil_center = pupil_center[valid]
                treadmill = treadmill[valid]
                for k in row_info:
                    row_info[k] = row_info[k][valid]
            behavior = np.c_[pupil, dpupil, treadmill]

        areas, layers, animal_ids, sessions, scan_idxs, unit_ids = (self.ResponseKeys
                                                                    * anatomy.AreaMembership
                                                                    * anatomy.LayerMembership & key).fetch('brain_area',
                                                                                                           'layer',
                                                                                                           'animal_id',
                                                                                                           'session',
                                                                                                           'scan_idx',
                                                                                                           'unit_id',
                                                                                                           order_by='col_id ASC')

        assert len(np.unique(unit_ids)) == len(unit_ids), \
            'unit ids are not unique, do you have more than one preprocessing method?'

        neurons = dict(
            unit_ids=unit_ids.astype(np.uint16),
            animal_ids=animal_ids.astype(np.uint16),
            sessions=sessions.astype(np.uint8),
            scan_idx=scan_idxs.astype(np.uint8),
            layer=layers.astype('S'),
            area=areas.astype('S')
        )

        def run_stats(selector, types, ix, axis=None):
            ret = {}
            for t in np.unique(types):
                if not np.any(ix & (types == t)):
                    continue
                data = selector(ix & (types == t))

                ret[t] = dict(
                    mean=data.mean(axis=axis).astype(np.float32),
                    std=data.std(axis=axis, ddof=1).astype(np.float32),
                    min=data.min(axis=axis).astype(np.float32),
                    max=data.max(axis=axis).astype(np.float32),
                    median=np.median(data, axis=axis).astype(np.float32)
                )
            data = selector(ix)
            ret['all'] = dict(
                mean=data.mean(axis=axis).astype(np.float32),
                std=data.std(axis=axis, ddof=1).astype(np.float32),
                min=data.min(axis=axis).astype(np.float32),
                max=data.max(axis=axis).astype(np.float32),
                median=np.median(data, axis=axis).astype(np.float32)
            )
            return ret

        # --- compute statistics
        log.info('Computing statistics on training dataset')
        response_statistics = run_stats(lambda ix: responses[ix], types, tiers == 'train', axis=0)

        input_statistics = run_stats(lambda ix: images[ix], types, tiers == 'train')

        statistics = dict(
            images=input_statistics,
            responses=response_statistics
        )

        if include_behavior:
            # ---- include statistics
            behavior_statistics = run_stats(lambda ix: behavior[ix], types, tiers == 'train', axis=0)
            eye_statistics = run_stats(lambda ix: pupil_center[ix], types, tiers == 'train', axis=0)

            statistics['behavior'] = behavior_statistics
            statistics['pupil_center'] = eye_statistics

        retval = dict(images=images,
                      responses=responses,
                      types=types.astype('S'),
                      condition_hashes=hashes.astype('S'),
                      trial_idx=trial_idxs.astype(np.uint32),
                      neurons=neurons,
                      item_info=row_info,
                      tiers=tiers.astype('S'),
                      statistics=statistics
                      )
        if include_behavior:
            retval['behavior'] = behavior
            retval['pupil_center'] = pupil_center
        return retval


class BehaviorMixin:
    def load_frame_times(self, key):
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

    def load_eye_traces(self, key):
        #r, center = (pupil.FittedPupil.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
        r, center = (pupil.FittedPupil.Circle() & key).fetch('radius', 'center',
                                                             order_by='frame_id')
        detectedFrames = ~np.isnan(r)
        xy = np.full((len(r), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])
        xy = np.vstack(map(partial(fill_nans, preserve_gap=3), xy.T))
        if np.any(np.isnan(xy)):
            log.info('Keeping some nans in the pupil location trace')
        pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
        if np.any(np.isnan(pupil_radius)):
            log.info('Keeping some nans in the pupil radius trace')

        eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
        return pupil_radius, xy, eye_time

    def load_behavior_timing(self, key):
        log.info('Loading behavior frametimes')
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.BehaviorSync() & key).fetch1('frame_times').squeeze()[0::ndepth]

    def load_treadmill_velocity(self, key):
        t, v = (treadmill.Treadmill() & key).fetch1('treadmill_time', 'treadmill_vel')
        return v.squeeze(), t.squeeze()


@schema
class Eye(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse
    ---
    -> pupil.FittedPupil                 # tracking_method as a secondary attribute
    pupil              : external-data   # pupil dilation trace
    dpupil             : external-data   # derivative of pupil dilation trace
    center             : external-data   # center position of the eye
    valid              : external-data   # valid trials
    """

    @property
    def key_source(self):
        return InputResponse & pupil.FittedPupil & stimulus.BehaviorSync

    def make(self, scan_key):
        scan_key = {**scan_key, 'tracking_method': 2}
        log.info('Populating '+ pformat(scan_key))
        radius, xy, eye_time = self.load_eye_traces(scan_key)
        frame_times = self.load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)

        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Downsampling eye signal to {}Hz'.format(1 / duration))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(duration, deye, 'hamming', warning=True)
        h_deye = self.get_filter(duration, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(eye_time,
                                 np.convolve(radius, h_eye, mode='same'), k=1, ext=0)

        dpupil_spline = NaNSpline(eye_time,
                                  np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(eye_time,
                                    np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]),
                                    k=1, ext=0)

        flip_times = (InputResponse.Input * Frame * stimulus.Trial & scan_key).fetch('flip_times',
                                                                                     order_by='row_id ASC')

        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        t = fr2beh(stimulus_onset + sample_point)
        pupil = pupil_spline(t)
        dpupil = dpupil_spline(t)
        center = center_spline(t)
        valid = ~np.isnan(pupil + dpupil + center.sum(axis=0))
        if not np.all(valid):
            log.warning('Found {} NaN trials. Setting to -1'.format((~valid).sum()))
            pupil[~valid] = -1
            dpupil[~valid] = -1
            center[:, ~valid] = -1

        self.insert1(dict(scan_key, pupil=pupil, dpupil=dpupil, center=center, valid=valid))


@schema
class Treadmill(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse
    -> treadmill.Treadmill
    ---
    treadmill          : external-data   # treadmill speed (|velcolity|)
    valid              : external-data   # valid trials
    """

    @property
    def key_source(self):
        rel = InputResponse
        return rel & treadmill.Treadmill() & stimulus.BehaviorSync()

    def make(self, scan_key):
        log.info('Populating\n' + pformat(scan_key))
        v, treadmill_time = self.load_treadmill_velocity(scan_key)
        frame_times = self.load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.warning('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Downsampling treadmill signal to {}Hz'.format(1 / duration))

        h_tread = self.get_filter(duration, np.nanmedian(np.diff(treadmill_time)), 'hamming', warning=True)
        treadmill_spline = NaNSpline(treadmill_time, np.abs(np.convolve(v, h_tread, mode='same')), k=1, ext=0)

        flip_times = (InputResponse.Input * Frame * stimulus.Trial & scan_key).fetch('flip_times',
                                                                                     order_by='row_id ASC')

        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        tm = treadmill_spline(fr2beh(stimulus_onset + sample_point))
        valid = ~np.isnan(tm)
        if not np.all(valid):
            log.warning('Found {} NaN trials. Setting to -1'.format((~valid).sum()))
            tm[~valid] = -1

        self.insert1(dict(scan_key, treadmill=tm, valid=valid))


@schema
class StaticMultiDataset(dj.Manual):
    definition = """
    # defines a group of datasets

    group_id    : smallint  # index of group
    ---
    description : varchar(255) # short description of the data
    """

    class Member(dj.Part):
        definition = """
        -> master
        -> InputResponse
        ---
        name                    : varchar(50) unique # string description to be used for training
        """

    _template = 'group{group_id:03d}-{animal_id}-{session}-{scan_idx}-{preproc_id}'


    def fill(self):
        selection = [
            ('11521-7-1', dict(animal_id=11521, session=7, scan_idx=1, preproc_id=0)),
            ('11521-7-2', dict(animal_id=11521, session=7, scan_idx=2, preproc_id=0)),
            ('16157-5-5', dict(animal_id=16157, session=5, scan_idx=5, preproc_id=0)),
            ('16157-5-6', dict(animal_id=16157, session=5, scan_idx=6, preproc_id=0)),
            ('16157-5-5-scaled', dict(animal_id=16157, session=5, scan_idx=5, preproc_id=2)),
            ('16312-3-20', dict(animal_id=16312, session=3, scan_idx=20, preproc_id=0)),
            ('11521-7-1-scaled', dict(animal_id=11521, session=7, scan_idx=1, preproc_id=2)),
            ('11521-7-2-scaled', dict(animal_id=11521, session=7, scan_idx=2, preproc_id=2)),
            ('18765-4-6', dict(animal_id=18765, session=4, scan_idx=6, preproc_id=0)),
            ('16157-5', [dict(animal_id=16157, session=5, scan_idx=5, preproc_id=0),
                         dict(animal_id=16157, session=5, scan_idx=6, preproc_id=0)]),
            ('20505-2-24', dict(animal_id=20505, session=2, scan_idx=24, preproc_id=0)),
            ('20505-3-7', dict(animal_id=20505, session=3, scan_idx=7, preproc_id=0)),
            ('20505-6-1', dict(animal_id=20505, session=6, scan_idx=1, preproc_id=0)),
            ('20505-7-29', dict(animal_id=20505, session=7, scan_idx=29, preproc_id=0)),
            ('20457-5-9', dict(animal_id=20457, session=5, scan_idx=9, preproc_id=0)),
            ('20505-10-14', dict(animal_id=20505, session=10, scan_idx=14, preproc_id=0)),
            ('20457-7-10', dict(animal_id=20457, session=7, scan_idx=10, preproc_id=0)),
            ('20457-8-12', dict(animal_id=20457, session=8, scan_idx=12, preproc_id=0)),
            ('20505-12-29', dict(animal_id=20505, session=12, scan_idx=29, preproc_id=0)),
            ('20505-14-33', dict(animal_id=20505, session=14, scan_idx=33, preproc_id=0)),
            ('20505-11-16', dict(animal_id=20505, session=11, scan_idx=16, preproc_id=0)),
            ('20210-4-11', dict(animal_id=20210, session=4, scan_idx=11, preproc_id=0)),
            ('20892-3-14', dict(animal_id=20892, session=3, scan_idx=14, preproc_id=0)),
            ('20892-9-10', dict(animal_id=20892, session=9, scan_idx=10, preproc_id=0)),
            ('20210-5-16', dict(animal_id=20210, session=5, scan_idx=16, preproc_id=0)),
            ('20210-7-14', dict(animal_id=20210, session=7, scan_idx=14, preproc_id=0)),
            ('20210-8-17', dict(animal_id=20210, session=8, scan_idx=17, preproc_id=0)),
            ('20892-6-24', dict(animal_id=20892, session=6, scan_idx=24, preproc_id=0)),
            ('20505-10-14-gamma', dict(animal_id=20505, session=10, scan_idx=14, preproc_id=3)),
            ('21067-9-17', dict(animal_id=21067, session=9, scan_idx=17, preproc_id=0)),
            ('21067-15-9', dict(animal_id=21067, session=15, scan_idx=9, preproc_id=0)),
            ('20892-10-10', dict(animal_id=20892, session=10, scan_idx=10, preproc_id=0)),
            ('20457-5-17', dict(animal_id=20457, session=5, scan_idx=17, preproc_id=0)),
            ('20505-10-19', dict(animal_id=20505, session=10, scan_idx=19, preproc_id=0)),
            ('20892-4-16', dict(animal_id=20892, session=4, scan_idx=16, preproc_id=0)),
            ('21067-10-18', dict(animal_id=21067, session=10, scan_idx=18, preproc_id=0)),
            ('21067-11-21', dict(animal_id=21067, session=11, scan_idx=21, preproc_id=0)),
            ('21067-12-15', dict(animal_id=21067, session=12, scan_idx=15, preproc_id=0)),
            ('21067-13-14', dict(animal_id=21067, session=13, scan_idx=14, preproc_id=0)),
            ('21553-11-10', dict(animal_id=21553, session=11, scan_idx=10, preproc_id=0)),
            ('20892-9-11', dict(animal_id=20892, session=9, scan_idx=11, preproc_id=0)),
            ('21844-2-12', dict(animal_id=21844, session=2, scan_idx=12, preproc_id=0)),
            ('22085-2-20', dict(animal_id=22085, session=2, scan_idx=20, preproc_id=0)),
            ('22083-7-21', dict(animal_id=22083, session=7, scan_idx=21, preproc_id=0)),
        ]
        for group_id, (descr, key) in enumerate(selection):
            entry = dict(group_id=group_id, description=descr)
            if entry in self:
                print('Already found entry', entry)
            else:
                with self.connection.transaction:
                    if not (InputResponse() & key):
                        ValueError('Dataset not found')
                    self.insert1(entry)
                    for k in (InputResponse() & key).fetch(dj.key):
                        k = dict(entry, **k)
                        name = self._template.format(**k)
                        self.Member().insert1(dict(k, name=name), ignore_extra_fields=True)

    def fetch_data(self, key, key_order=None):
        assert len(self & key) == 1, 'Key must refer to exactly one multi dataset'
        ret = OrderedDict()
        log.info('Fetching data for ' +  repr(key))
        for mkey in (self.Member() & key).fetch(dj.key,
                                                order_by='animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC'):
            name = (self.Member() & mkey).fetch1('name')
            include_behavior = bool(Eye().proj() * Treadmill().proj() & mkey)
            data_names = ['images', 'responses'] if not include_behavior \
                else ['images',
                      'behavior',
                      'pupil_center',
                      'responses']
            log.info('Data will be ({})'.format(','.join(data_names)))

            h5filename = InputResponse().get_filename(mkey)
            log.info('Loading dataset {} --> {}'.format(name, h5filename))
            ret[name] = StaticImageSet(h5filename, *data_names)
        if key_order is not None:
            log.info('Reordering datasets according to given key order {}'.format(', '.join(key_order)))
            ret = OrderedDict([
                (k, ret[k]) for k in key_order
            ])
        return ret
