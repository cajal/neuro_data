from collections import OrderedDict
from functools import partial
from itertools import compress
from pprint import pformat

import datajoint as dj
import numpy as np
import pandas as pd

from neuro_data import logger as log
from neuro_data.utils.data import h5cached, SplineCurve, FilterMixin, fill_nans, NaNSpline
from neuro_data.static_images import datasets

dj.config['external-data'] = {'protocol': 'file', 'location': '/external/'}

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
reso = dj.create_virtual_module('reso', 'pipeline_reso')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
shared = dj.create_virtual_module('shared', 'pipeline_shared')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')

schema = dj.schema('neurodata_static')

# set of attributes that uniquely identifies the frame content
UNIQUE_FRAME = {
    'stimulus.Frame': ('image_id', 'image_class'),
    'stimulus.MonetFrame': ('rng_seed', 'orientation'),
    'stimulus.TrippyFrame': ('rng_seed',),
    'stimulus.ColorFrameProjector': ('image_id', 'image_class'),
}

IMAGE_CLASSES = 'image_class in ("imagenet", "imagenet_v2_gray")' # all valid natural image classes

@schema
class StaticScanCandidate(dj.Manual):
    definition = """ # list of scans to process
    
    -> fuse.ScanDone
    ---
    candidate_notes='' : varchar(1024)
    """
    @staticmethod
    def fill(key, candidate_notes='', segmentation_method=6, spike_method=5,
             pipe_version=1):
        """ Fill an entry with key"""
        StaticScanCandidate.insert1({'segmentation_method': segmentation_method,
                                     'spike_method': spike_method,
                                     'pipe_version': pipe_version, **key,
                                     'candidate_notes': candidate_notes},
                                    skip_duplicates=True)

@schema
class StaticScan(dj.Computed):
    definition = """ # gatekeeper for scan and preprocessing settings
    
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

    key_source = fuse.ScanDone() & StaticScanCandidate & 'spike_method=5 and segmentation_method=6'

    @staticmethod
    def complete_key(key):
        return dict((dj.U('segmentation_method', 'pipe_version') &
                     (meso.ScanSet.Unit() & key)).fetch1(dj.key), **key)

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

# based on mesonet.MesoNetSplit
@schema
class ImageNetSplit(dj.Lookup):
    definition = """ # split imagenet frames into train, test, validation

    -> stimulus.StaticImage.Image
    ---
    -> Tier
    """
    def fill(self, scan_key):
        """ Assign each imagenet frame in the current scan to train/test/validation set.

        Arguments:
            scan_key: An scan (animal_id, session, scan_idx) that has stimulus.Trials
                created. Usually one where the stimulus was presented.

        Note:
            Each image is assigned to one set and that holds true for all our scans and
            collections. Once an image has been assigned (and models have been trained
            with that split), it cannot be changed in the future (this is problematic if
            images are reused as those from collection 2 or collection 3 with a different
            purpose).

            The exact split assigned will depend on the scans used in fill and the order
            that this table was filled. Not ideal.
        """
        # Find out whether we are using the old pipeline (grayscale only) or the new version
        if stimulus.Frame & (stimulus.Trial & scan_key):
            frame_table = stimulus.Frame
        elif stimulus.ColorFrameProjector & (stimulus.Trial & scan_key):
            frame_table = stimulus.ColorFrameProjector
        else:
            print('Static images were not shown for this scan')

        # Get all image ids in this scan
        all_frames = frame_table * stimulus.Trial & scan_key & IMAGE_CLASSES
        unique_frames = dj.U('image_id', 'image_class').aggr(all_frames, repeats='COUNT(*)')
        image_ids, image_classes = unique_frames.fetch('image_id', 'image_class', order_by='repeats DESC')
        num_frames = len(image_ids)
        # * NOTE: this fetches all oracle images first and the rest in a "random" order;
        # we use that random order to make the validation/training division below.

        # Get number of repeated frames
        n = int(np.median(unique_frames.fetch('repeats')))  # HACK
        num_oracles = len(unique_frames & 'repeats > {}'.format(n))  # repeats
        if num_oracles == 0:
            raise ValueError('Could not find repeated frames to use for oracle.')

        # Compute number of validation examples
        num_validation = int(np.ceil((num_frames - num_oracles) * 0.1))  # 10% validation examples

        # Insert
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                     zip(image_ids[:num_oracles], image_classes[:num_oracles])],
                    skip_duplicates=True)
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'validation'} for
                     iid, ic in zip(image_ids[num_oracles: num_oracles + num_validation],
                                    image_classes[num_oracles: num_oracles + num_validation])])
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                     zip(image_ids[num_oracles + num_validation:],
                         image_classes[num_oracles + num_validation:])])


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
        train_test = (dj.U(*UNIQUE_FRAME[cond['stimulus_type']]).aggr(frames * stim,
                                                                      train='sum(1-test)',
                                                                      test='sum(test)') &
                      'train>0 and test>0')
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
                     & 'stimulus_type in ("stimulus.Frame", "stimulus.MonetFrame", "stimulus.TrippyFrame", "stimulus.ColorFrameProjector")'
        for cond in conditions.fetch(as_dict=True):
            # hack for compatibility with previous datasets
            if cond['stimulus_type'] in ['stimulus.Frame', 'stimulus.ColorFrameProjector']:
                frame_table = (stimulus.Frame if cond['stimulus_type'] == 'stimulus.Frame'
                               else stimulus.ColorFrameProjector)

                # deal with ImageNet frames first
                log.info('Inserting assignment from ImageNetSplit')
                targets = StaticScan * frame_table * ImageNetSplit & (stimulus.Trial & key) & IMAGE_CLASSES
                print('Inserting {} imagenet conditions!'.format(len(targets)))
                self.insert(targets, ignore_extra_fields=True)

                # deal with MEI images, assigning tier test for all images
                assignment = (frame_table & 'image_class in ("cnn_mei", "lin_rf", "multi_cnn_mei", "multi_lin_rf")').proj(tier='"train"')
                self.insert(StaticScan * frame_table * assignment & (stimulus.Trial & key), ignore_extra_fields=True)

                # make sure that all frames were assigned
                remaining = (stimulus.Trial * frame_table & key) - self
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
        {'preproc_id': 4, 'offset': 0.03, 'duration': 0.5, 'row': 36, 'col': 64,
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
        if stimulus.ColorFrameProjector & key:
            assert (stimulus.ColorFrameProjector & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.StaticImage.Image & (stimulus.ColorFrameProjector & key)).fetch1('image')
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


def get_traces(key):
    """ Get spike traces for all cells in these scan (along with their times in stimulus
    clock).

    Arguments:
        key (dict): Key for a scan (or field).

    Returns:
        traces (np.array): A (num_units x num_scan_frames) array with all spike traces.
            Traces are restricted to those classified as soma and ordered by unit_id.
        unit_ids (list): A (num_units) list of unit_ids in traces.
        trace_times (np.array): A (num_units x num_scan_frames) array with the time (in
            seconds) for each unit's trace in stimulus clock (same clock as times in
            stimulus.Trial).

    Note: On notation
        What is called a frametime in stimulus.Sync and stimulus.Trial is actually the
        time each depth of scanning started. So for a scan with 1000 frames and four
        depths per frame/volume, there will be 4000 "frametimes".

    Note 2:
        For a scan with 10 depths, a frame i is considered complete if all 10 depths were
        recorded and saved in the tiff file, frame_times however save the starting time of
        each depth independently (for instance if 15 depths were recorded there will be
        one scan frame but 15 frame times, the last 5 have to be ignored).
    """
    # Pick right pipeline for this scan (reso or meso)
    pipe_name = (fuse.ScanDone & key).fetch1('pipe')
    pipe = reso if pipe_name == 'reso' else meso

    # Get traces
    units = pipe.ScanSet.Unit() & key & (pipe.MaskClassification.Type & {'type': 'soma'})
    spikes = pipe.Activity.Trace() * pipe.ScanSet.UnitInfo() & units.proj()
    unit_ids, traces, ms_delays = spikes.fetch('unit_id', 'trace', 'ms_delay',
                                               order_by='unit_id')

    # Get time of each scan frame for this scan (in stimulus clock; same as in Trial)
    depth_times = (stimulus.Sync & key).fetch1('frame_times')
    num_frames = (pipe.ScanInfo & key).fetch1('nframes')
    num_depths = len(dj.U('z') & (pipe.ScanInfo.Field.proj('z', nomatch='field') & key))
    if len(depth_times) / num_depths < num_frames or (len(depth_times) / num_depths >
                                                      num_frames + 1):
        raise ValueError('Mismatch between frame times and tiff frames')
    frame_times = depth_times[:num_depths * num_frames:num_depths]  # one per frame

    # Add per-cell delay to each frame_time
    trace_times = np.add.outer(ms_delays / 1000, frame_times)  # num_traces x num_frames

    return np.stack(traces), np.stack(unit_ids), trace_times


def trapezoid_integration(x, y, x0, xf):
    """ Integrate y (recorded at points x) from x0 to xf.

    Arguments:
        x (np.array): Timepoints (num_timepoints) when y was recorded.
        y (np.array): Signal (num_timepoints).
        x0 (float or np.array): Starting point(s). Could be a 1-d array (num_samples).
        xf (float or np.array): Final point. Same shape as x0.

    Returns:
        Integrated signal from x0 to xf:
            a 0-d array (i.e., float) if x0 and xf are floats
            a 1-d array (num_samples) if x0 and xf are 1-d arrays
    """
    # Basic checks
    if np.any(xf <= x0):
        raise ValueError('xf has to be higher than x0')
    if np.any(x0 < x[0]) or np.any(xf > x[-1]):
        raise ValueError('Cannot integrate outside the original range x of the signal.')

    # Compute area under each trapezoid
    trapzs = np.diff(x) * (y[:-1] + y[1:]) / 2  # index i is trapezoid from point i to point i + 1

    # Find timepoints right before x0 and xf
    idx_before_x0 = np.searchsorted(x, x0) - 1
    idx_before_xf = np.searchsorted(x, xf) - 1

    # Compute y at the x0 and xf points
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # index i is slope from p_i to p_{i+1}
    y0 = y[idx_before_x0] + slopes[idx_before_x0] * (x0 - x[idx_before_x0])
    yf = y[idx_before_xf] + slopes[idx_before_xf] * (xf - x[idx_before_xf])

    # Sum area of all interior trapezoids
    indices = np.stack([idx_before_x0 + 1, idx_before_xf], axis=-1).ravel()  # interleaved x0 and xf for all samples
    integral = np.add.reduceat(trapzs, indices, axis=-1)[::2].squeeze()

    # Add area of edge trapezoids (ones that go from x0 to first_x_sample and from last_x_sample to xf)
    integral += (x[idx_before_x0 + 1] - x0) * (y0 + y[idx_before_x0 + 1]) / 2
    integral += (xf - x[idx_before_xf]) * (y[idx_before_xf] + yf) / 2

    # Deal with edge case where both x0 and xf are in the same trapezoid
    same_trapezoid = idx_before_x0 == idx_before_xf
    integral[same_trapezoid] = ((xf - x0) * (y0 + yf) / 2)[same_trapezoid]

    return integral


@h5cached('/external/cache/', mode='array', transfer_to_tmp=False,
          file_format='static{animal_id}-{session}-{scan_idx}-preproc{preproc_id}.h5')
@schema
class InputResponse(dj.Computed):
    definition = """ # responses of each neuron to images (stimulus.Frame) in stimuli
    -> StaticScan
    -> Preprocessing
    """
    @property
    def key_source(self):
        return StaticScan * Preprocessing & Frame

    class ResponseBlock(dj.Part):
        definition = """
        -> master
        ---
        responses       : external-data # response of one neurons for all bins
        """

    class Input(dj.Part):
        definition = """
        -> master
        -> stimulus.Trial
        -> Frame
        ---
        row_id          : int           # row id in the response block
        """

    class ResponseKeys(dj.Part):
        definition = """
        -> master.ResponseBlock
        -> fuse.Activity.Trace
        ---
        col_id          : int           # col id in the response block
        """

    def make(self, key):
        # Check new preprocessing
        if key['preproc_id'] < 4:
            raise ValueError('Deprecated preprocessing, use preproc_id > 4 or downgrade '
                             'code to access previous preprocessings.')

        # Get all traces for this scan
        log.info('Getting traces...')
        traces, unit_ids, trace_times = get_traces(key)

        # Get trial times for frames in Scan.Frame (excluding bad trials)
        log.info('Getting onset and offset times for each image...')
        trials_rel = stimulus.Trial * Frame - ExcludedTrial & key
        flip_times, trial_ids, cond_hashes = trials_rel.fetch('flip_times', 'trial_idx',
                                                              'condition_hash',
                                                              order_by='condition_hash',
                                                              squeeze=True)
        if any([len(ft) < 2 or len(ft) > 3 for ft in flip_times]):
            raise ValueError('Only works for stimulus.Frames with 2 or 3 flips')

        # Find start and duration of image frames
        monitor_fps = 60
        image_onset = np.stack([ft[1] for ft in flip_times]) + 1 / monitor_fps  # start of image
        image_duration = float((Preprocessing & key).fetch1('duration'))  # np.stack([ft[2] for ft in flip_times]) - image_onset
        """
        Each trial is a stimulus.Frame.
        A single stimulus.Frame is composed of a flip (1/60 secs), a blanking period (0.3 
        - 0.5 secs), another flip, the image (0.5 secs) and another flip. During flips 
        screen is gray (as during blanking).
        """

        # Add a shift to the onset times to account for the time it takes for the image to
        # travel from the retina to V1
        image_onset += float((Preprocessing & key).fetch1('offset'))
        # Wiskott, L. How does our visual system achieve shift and size invariance?. Problems in Systems Neuroscience, 2003.

        # Sample responses (trace by trace) with a rectangular window
        log.info('Sampling responses...')
        image_resps = np.stack([trapezoid_integration(tt, t, image_onset, image_onset +
                                                      image_duration) / image_duration for
                                tt, t in zip(trace_times, traces)], axis=-1)

        # Insert
        log.info('Inserting...')
        self.insert1(key)
        self.ResponseBlock.insert1({**key, 'responses': image_resps.astype(np.float32)})
        self.Input.insert([{**key, 'trial_idx': trial_idx, 'condition_hash': cond_hash,
                            'row_id': i} for i, (trial_idx, cond_hash) in enumerate(zip(
            trial_ids, cond_hashes))])
        self.ResponseKeys.insert([{**key, 'unit_id': unit_id, 'col_id': i,
                                   'field': (fuse.Activity.Trace & key &
                                             {'unit_id': unit_id}).fetch1('field'),
                                   'channel': (fuse.Activity.Trace & key &
                                               {'unit_id': unit_id}).fetch1('channel')}
                                  for i, unit_id in enumerate(unit_ids)])


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
    

# Patch job for the hardcoding mess that was StaticMultiDataset.fill()
# Instead of editing the code each time, the user will enter they scan with the desire group_id into here then call StaticMultiDataset.fill()
@schema
class StaticMultiDatasetGroupAssignment(dj.Manual):
    definition = """
    group_id : int unsigned
    -> InputResponse
    ---
    description = '' : varchar(1024)
    """

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

    @staticmethod
    def fill():
        _template = 'group{group_id:03d}-{animal_id}-{session}-{scan_idx}-{preproc_id}'
        for scan in StaticMultiDatasetGroupAssignment.fetch(as_dict=True):
            # Check if the scan has been added to StaticMultiDataset.Member, if not then do it
            if len(StaticMultiDataset & dict(group_id = scan['group_id'])) == 0:
                # Group id has not been added into StaticMultiDataset, thus add it
                StaticMultiDataset.insert1(dict(group_id = scan['group_id'], description = scan['description']))

            # Handle instertion into Member table
            if len(StaticMultiDataset.Member() & scan) == 0:
                StaticMultiDataset.Member().insert1(dict(scan, name = _template.format(**scan)), ignore_extra_fields=True)

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
            ret[name] = datasets.StaticImageSet(h5filename, *data_names)
        if key_order is not None:
            log.info('Reordering datasets according to given key order {}'.format(', '.join(key_order)))
            ret = OrderedDict([
                (k, ret[k]) for k in key_order
            ])
        return ret