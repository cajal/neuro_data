import io
from collections import OrderedDict
from functools import partial
from itertools import count
from pprint import pformat

import cv2
import imageio
import numpy as np
from attorch.dataset import H5SequenceSet
from scipy.signal import convolve2d
from tqdm import tqdm

from neuro_data.movies.transforms import Subsequence
from .mixins import TraceMixin
from .schema_bridge import *
from .. import logger as log
from ..utils.data import SplineMovie, FilterMixin, SplineCurve, NaNSpline, fill_nans, h5cached

dj.config['external-data'] = dict(
    protocol='file',
    location='/external/movie-data/')

try:
    # Virtual Modules
    synicix_latent_variables = dj.create_virtual_module('synicix_latent_variables', 'synicix_latent_variables')
except:
    synicix_latent_variables = None
    print('Failed to load synicix_latent_variables virutal module, please check your permissions')

STACKS = [
    dict(animal_id=17977, stack_session=2, stack_idx=8, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17886, stack_session=2, stack_idx=1, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17797, stack_session=6, stack_idx=9, pipe_version=1, volume_id=1, registration_method=2),
    dict(animal_id=17795, stack_session=3, stack_idx=1, pipe_version=1, volume_id=1, registration_method=2),
]

UNIQUE_CLIP = {
    'stimulus.Clip': ('movie_name', 'clip_number', 'cut_after', 'skip_time'),
    'stimulus.Monet': ('rng_seed',),
    'stimulus.Monet2': ('rng_seed',),
    'stimulus.Trippy': ('rng_seed',),
    'stimulus.Matisse2': ('condition_hash',)
}

schema = dj.schema('neurodata_movies', locals())

MOVIESCANS = [  # '(animal_id=16278 and session=11 and scan_idx between 5 and 9)',  # hollymonet
    # '(animal_id=15685 and session=2 and scan_idx between 11 and 15)',  # hollymonet
    'animal_id=17021 and session=18 and scan_idx=11',  # platinum (scan_idx in (10, 14))
    'animal_id=9771 and session=1 and scan_idx in (1,2)',  # madmonet
    'animal_id=17871 and session=4 and scan_idx=13',  # palindrome mouse
    'animal_id=17358 and session=5 and scan_idx=3',  # platinum
    'animal_id=17358 and session=9 and scan_idx=1',  # platinum
    platinum.CuratedScan() & dict(animal_id=18142, scan_purpose='trainable_platinum_classic', score=4),
    platinum.CuratedScan() & dict(animal_id=17797, scan_purpose='trainable_platinum_classic') & 'score > 2',
    'animal_id=16314 and session=3 and scan_idx=1',
    experiment.Scan() & (stimulus.Trial & stimulus.Condition() & stimulus.Monet()) & dict(animal_id=8973),
    'animal_id=18979 and session=2 and scan_idx=7',
    'animal_id=18799 and session=3 and scan_idx=14',
    'animal_id=18799 and session=4 and scan_idx=18',
    'animal_id=18979 and session=2 and scan_idx=5',
    # start with segmentation method 6
    'animal_id=20457 and session=1 and scan_idx=15',
    'animal_id=20457 and session=2 and scan_idx=20',
    'animal_id=20501 and session=1 and scan_idx=10',
    'animal_id=20458 and session=3 and scan_idx=5',
]


@schema
class MovieScan(dj.Computed):
    definition = """
    # smaller primary key table for data

    -> fuse.ScanDone
    ---
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        -> fuse.ScanSet.Unit
        """

    key_source = (fuse.ScanDone() & MOVIESCANS & dict(segmentation_method=3,
                                                      spike_method=5) & 'animal_id < 19000').proj() + \
                 (fuse.ScanDone() & MOVIESCANS & dict(segmentation_method=6,
                                                      spike_method=5) & 'animal_id > 19000').proj()

    def _make_tuples(self, key):
        self.insert(fuse.ScanDone() & key, ignore_extra_fields=True)
        pipe = (fuse.ScanDone() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)
        if key['animal_id'] < 19000:
            self.Unit().insert(fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type & key
                               & dict(pipe_version=1, segmentation_method=3, spike_method=5, type='soma'),
                               ignore_extra_fields=True)
        else:
            self.Unit().insert(fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type & key
                               & dict(pipe_version=1, segmentation_method=6, spike_method=5, type='soma'),
                               ignore_extra_fields=True)


@schema
class Preprocessing(dj.Lookup):
    definition = """
    # settings for movie preprocessing

    preproc_id       : tinyint # preprocessing ID
    ---
    resampl_freq     : decimal(3,1)  # resampling refrequency of stimuli and behavior
    behavior_lowpass : decimal(3,1)  # low pass cutoff of behavior signals Hz
    row              : tinyint # row size of movies
    col              : tinyint # col size of movie
    """

    @property
    def contents(self):
        yield from zip(count(), [30, 30, 30], [2.5, 2.5, 2.5], [36, 36, 36], [64, 64, 64])


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
class ConditionTier(dj.Computed):
    definition = """
    # split into train, test, validation

    -> stimulus.Condition
    -> MovieScan
    ---
    -> Tier
    """

    @property
    def dataset_compositions(self):
        return dj.U('animal_id', 'session', 'scan_idx', 'stimulus_type', 'tier').aggr(
            self * stimulus.Condition(), n='count(*)')

    @property
    def key_source(self):
        return MovieScan() & stimulus.Trial()

    def check_train_test_split(self, clips, cond):
        stim = getattr(stimulus, cond['stimulus_type'].split('.')[-1])
        train_test = dj.U(*UNIQUE_CLIP[cond['stimulus_type']]).aggr(clips * stim, train='sum(1-test)', test='sum(test)') \
                     & 'train>0 and test>0'
        assert len(train_test) == 0, 'Train and test clips do overlap'

    def fill_up(self, tier, clips, cond, key, m):
        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            # all hashes that are in clips but not registered for that animal and have the right tier
            candidates = dj.U('condition_hash') & \
                         (self & (dj.U('condition_hash') & (clips - self)) & dict(tier=tier))
            keys = candidates.fetch(dj.key)
            d = m - n
            update = min(len(keys), d)

            log.info('Inserting {} more existing {} trials'.format(update, tier))
            for k in keys[:update]:
                k = (clips & k).fetch1(dj.key)
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            keys = (clips - self).fetch(dj.key)
            update = m - n
            log.info('Inserting {} more new {} trials'.format(update, tier))
            for k in keys[:update]:
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

    def make(self, key):
        log.info('Processing ' + repr(key))
        conditions = dj.U('stimulus_type').aggr(stimulus.Condition() & (stimulus.Trial() & key),
                                                count='count(*)') \
                     & 'stimulus_type in ("stimulus.Clip","stimulus.Monet", "stimulus.Monet2", "stimulus.Trippy", "stimulus.Matisse2")'
        for cond in conditions.fetch(as_dict=True):
            log.info('Checking condition {stimulus_type} (n={count})'.format(**cond))
            clips = (stimulus.Condition() * MovieScan() & key & cond).aggr(stimulus.Trial(), repeats="count(*)",
                                                                           test='count(*) > 4')
            self.check_train_test_split(clips, cond)

            m = len(clips)
            m_test = m_val = len(clips & 'test > 0') or max(m // 10, 1)
            log.info('Minimum test and validation set size will be {}'.format(m_test))

            # insert repeats as test trials
            self.insert((clips & dict(test=1)).proj(tier='"test"'), ignore_extra_fields=True)
            self.fill_up('test', clips, cond, key, m_test)
            self.fill_up('validation', clips, cond, key, m_val)
            self.fill_up('train', clips, cond, key, m - m_test - m_val)


@schema
class MovieClips(dj.Computed, FilterMixin):
    definition = """
    # movies subsampled

    -> stimulus.Condition
    -> Preprocessing
    ---
    fps0                 : float           # original framerate
    frames               : external-data   # input movie downsampled
    sample_times         : external-data   # sample times for the new frames
    duration             : float           # duration in seconds
    """

    def get_frame_rate(self, key):
        stimulus_type = (stimulus.Condition() & key).fetch1('stimulus_type')
        if stimulus_type == 'stimulus.Clip':
            assert len(stimulus.Clip() & key) == 1, 'key must specify exactly one clip'
            frame_rate = (stimulus.Movie() * stimulus.Clip() & key).fetch1('frame_rate')
        else:
            movie_rel = getattr(stimulus, stimulus_type.split('.')[-1])
            frame_rate = (movie_rel() & key).fetch1('fps')
        return float(frame_rate)  # in case it was a decimal

    def load_movie(self, key):
        # --- get correct stimulus relation
        log.info('Loading movie {condition_hash}'.format(**key))
        stimulus_type = (stimulus.Condition() & key).fetch1('stimulus_type')

        if stimulus_type == 'stimulus.Clip':
            assert len(stimulus.Clip() & key) == 1, 'key must specify exactly one clip'
            movie, frame_rate = (stimulus.Movie() * stimulus.Movie.Clip()
                                 * stimulus.Clip() & key).fetch1('clip', 'frame_rate')
            vid = imageio.get_reader(io.BytesIO(movie.tobytes()), 'ffmpeg')
            # convert to grayscale and stack to movie in width x height x time
            m = vid.get_length()
            movie = np.stack([vid.get_data(i).mean(axis=-1) for i in range(m)], axis=2)
        else:
            movie_rel = getattr(stimulus, stimulus_type.split('.')[-1])
            assert len(movie_rel() & key) == 1, 'key must specify exactly one clip'
            movie, frame_rate = (movie_rel() & key).fetch1('movie', 'fps')

        frame_rate = float(frame_rate)  # in case it was a decimal

        return movie, frame_rate

    @property
    def key_source(self):
        return stimulus.Condition() * Preprocessing() & ConditionTier()

    def adjust_duration(self, key, base):
        if stimulus.Clip() & key:
            duration, skip_time = map(float, (stimulus.Clip() & key).fetch1('cut_after', 'skip_time'))
            duration = min(base.max(), duration)
            log.info('Stimulus duration is cut to {}s with {}s skiptime'.format(duration, skip_time))
        else:
            duration = base.max()
            skip_time = 0
            log.info('Stimulus duration is {}s (full length)'.format(duration))
        return duration, skip_time

    def _make_tuples(self, key):
        log.info(80 * '-')
        log.info('Processing key ' + repr(key))
        sampling_period = float((Preprocessing() & key).proj(period='1/resampl_freq').fetch1('period'))
        imgsize = (Preprocessing() & key).fetch1('col', 'row')  # target size of movie frames

        log.info('Downsampling movie to {}'.format(repr(imgsize)))
        movie, frame_rate = self.load_movie(key)

        # --- downsampling movie
        h_movie = self.get_filter(sampling_period, 1 / frame_rate, 'hamming', warning=False)

        if not movie.shape[0] / imgsize[1] == movie.shape[1] / imgsize[0]:
            log.warning('Image size changes aspect ratio.')

        movie2 = np.stack([cv2.resize(m, imgsize, interpolation=cv2.INTER_AREA) \
                           for m in movie.squeeze().transpose([2, 0, 1])],
                          axis=0)
        movie = movie2.astype(np.float32).transpose([1, 2, 0])
        # low pass filter movie
        movie = np.apply_along_axis(lambda m: np.convolve(m, h_movie, mode='same'), axis=-1, arr=movie)
        base = np.arange(movie.shape[-1]) / frame_rate  # np.vstack([ft - ft[0] for ft in flip_times]).mean(axis=0)

        duration, skip_time = self.adjust_duration(key, base)
        samps = np.arange(0, duration, sampling_period)  # samps is relative to fliptime 0

        movie_spline = SplineMovie(base, movie, k=1, ext=1)
        movie = movie_spline(samps + skip_time).astype(np.float32)

        # --- generate response sampling points and sample movie frames relative to it
        self.insert1(dict(key, frames=movie.transpose([2, 0, 1]), sample_times=samps, fps0=frame_rate))


@h5cached('/external/cache/', mode='groups', transfer_to_tmp=False,
          file_format='movies{animal_id}-{session}-{scan_idx}-pre{preproc_id}-pipe{pipe_version}-seg{segmentation_method}-spike{spike_method}.h5')
@schema
class InputResponse(dj.Computed, FilterMixin, TraceMixin):
    definition = """
    # responses of one neuron to the stimulus

    -> MovieScan
    -> Preprocessing
    ---
    """

    key_source = MovieScan() * Preprocessing() & MovieClips()

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> MovieClips
            ---
            """

    class ResponseBlock(dj.Part):
        definition = """
            -> master
            -> master.Input
            ---
            responses           : external-data   # reponse of one neurons for all bins
            """

    class ResponseKeys(dj.Part):
        definition = """
            -> master.ResponseBlock
            -> fuse.Activity.Trace
            row_id           : int             # row id in the response block
            ---
            """

    def get_trace_spline(self, key, sampling_period):
        traces, frame_times, trace_keys = self.load_traces_and_frametimes(key)
        log.info('Loaded {} traces'.format(len(traces)))

        log.info('Generating lowpass filters with cutoff {:.3f}Hz'.format(1 / sampling_period))
        h_trace = self.get_filter(sampling_period, np.median(np.diff(frame_times)), 'hamming',
                                  warning=False)
        # low pass filter
        trace_spline = SplineCurve(frame_times,
                                   [np.convolve(trace, h_trace, mode='same') for trace in traces], k=1, ext=1)
        return trace_spline, trace_keys, frame_times.min(), frame_times.max()

    def make(self, scan_key):
        log.info(80 * '-')
        log.info('Populating {}'.format(repr(scan_key)).ljust(80, '-'))
        self.insert1(scan_key)
        # integration window size for responses
        sampling_period = (Preprocessing() & scan_key).proj(period='1/resampl_freq').fetch1('period')

        log.info('Sampling neural responses at {}s intervals'.format(sampling_period))

        trace_spline, trace_keys, ftmin, ftmax = self.get_trace_spline(scan_key, sampling_period)

        flip_times, sample_times, fps0, trial_keys = \
            (MovieScan() * MovieClips() * stimulus.Trial() & scan_key).fetch('flip_times', 'sample_times', 'fps0',
                                                                             dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        nodrop = np.array([np.diff(ft).max() < 1.99 / frame_rate for ft, frame_rate in zip(flip_times, fps0)])
        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(nodrop & valid):
            log.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~(nodrop & valid)).sum()))
        for trial_key, flips, samps, take in tqdm(zip(trial_keys, flip_times, sample_times, nodrop & valid),
                                                  total=len(trial_keys), desc='Trial '):
            if take:
                self.Input().insert1(dict(scan_key, **trial_key),
                                     ignore_extra_fields=True, skip_duplicates=True)
                self.ResponseBlock().insert1(dict(trial_key, responses=trace_spline(flips[0] + samps)),
                                             ignore_extra_fields=True)
                self.ResponseKeys().insert(
                    [dict(trial_key, row_id=i, **k) for i, k in enumerate(trace_keys)], ignore_extra_fields=True
                )

    def compute_data(self, key):
        key = dict((self & key).fetch1(dj.key), **key)
        log.info('Computing dataset for {}'.format(repr(key)))

        # meso or reso?
        pipe = (fuse.ScanDone() * MovieScan() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)

        # get data relation
        include_behavior = bool(Eye() * Treadmill() & key)

        # make sure that including areas does not decreas number of neurons
        assert len(pipe.ScanSet.UnitInfo() * experiment.Layer() * anatomy.AreaMembership() & key) == \
               len(pipe.ScanSet.UnitInfo() * experiment.Layer() & key), "AreaMembership decreases number of neurons"

        data_rel = MovieClips() * ConditionTier() \
                   * self.Input() * self.ResponseBlock() * stimulus.Condition().proj('stimulus_type')

        if include_behavior:  # restrict trials to those that do not have NaNs in Treadmill or Eye
            data_rel = data_rel & Eye & Treadmill

        # Including lv from synicix_latent_variable schema
        include_lvs = synicix_latent_variables is not None and bool(synicix_latent_variables.LatentVariableVideoClip & (stimulus.Trial & key))
        # Include_lvs is valid, thus use it to restrict data_rel
        print(key)
        print(data_rel & key)
        if include_lvs:
            data_rel = data_rel & synicix_latent_variables.LatentVariableVideoClip
        print(data_rel & key)

        response = self.ResponseKeys() * (pipe.ScanSet.UnitInfo() * experiment.Layer() * anatomy.AreaMembership()
                                          & key & '(um_z >= z_start) and (um_z < z_end)')

        # --- fetch all stimuli and classify into train/test/val
        inputs, hashes, stim_keys, tiers, types, trial_idx, durations = \
            (data_rel & key).fetch('frames', 'condition_hash', dj.key,
                                   'tier', 'stimulus_type', 'trial_idx', 'duration',
                                   order_by='condition_hash ASC, trial_idx ASC')
        train_idx = np.array([t == 'train' for t in tiers], dtype=bool)
        test_idx = np.array([t == 'test' for t in tiers], dtype=bool)
        val_idx = np.array([t == 'validation' for t in tiers], dtype=bool)

        # ----- extract trials

        unit_ids_tmp = animal_ids_tmp = sessions_tmp = scan_idx_tmp = layer_tmp = area_tmp = None

        responses, behavior, eye_position = [], [], []
        latent_variable = {}
        print(len(stim_keys))
        for stim_key in tqdm(stim_keys):
            response_block = (self.ResponseBlock() & stim_key).fetch1('responses')
            sessions, animal_ids, unit_ids, scan_idx, layer, area = \
                (response & key & stim_key).fetch('session', 'animal_id',
                                                  'unit_id', 'scan_idx',
                                                  "layer", "brain_area",
                                                  order_by='row_id ASC')
            if include_behavior:
                pupil, dpupil, treadmill, center = (Eye() * Treadmill() & key
                                                    & stim_key).fetch1('pupil', 'dpupil', 'treadmill', 'center')

                behavior.append(np.vstack([pupil, dpupil, treadmill]).T)
                eye_position.append(center.T)

            if include_lvs:
                latent_variable_ids, processed_lv_frames = (synicix_latent_variables.LatentVariableVideoClip & key & stim_key).fetch('latent_variable_id', 'processed_lv_frames')
                for k, v in zip(latent_variable_ids, processed_lv_frames):
                    latent_variable.setdefault(str(k), []).append(v)

            assert area_tmp is None or np.all(area_tmp == area), 'areas do not match'
            assert layer_tmp is None or np.all(layer_tmp == layer), 'layers do not match'
            assert unit_ids_tmp is None or np.all(unit_ids_tmp == unit_ids), 'unit ids do not match'
            assert animal_ids_tmp is None or np.all(animal_ids_tmp == animal_ids), 'animal ids do not match'
            assert sessions_tmp is None or np.all(sessions_tmp == sessions), 'sessions do not match'
            assert scan_idx_tmp is None or np.all(scan_idx_tmp == scan_idx), 'scan_idx do not match'
            unit_ids_tmp, animal_ids_tmp, sessions_tmp, scan_idx_tmp, layer_tmp = \
                unit_ids, animal_ids, sessions, scan_idx, layer
            responses.append(response_block.T.astype(np.float32))
        assert len(np.unique(unit_ids)) == len(unit_ids), \
            'unit ids are not unique, do you have more than one preprocessing method?'

        neurons = dict(
            unit_ids=unit_ids.astype(np.uint16),
            animal_ids=animal_ids.astype(np.uint16),
            sessions=sessions.astype(np.uint8),
            scan_idx=scan_idx.astype(np.uint8),
            layer=layer.astype('S'),
            area=area.astype('S')
        )

        # insert channel dimension
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inputs[i] = inp[None, ...]

        hashes = hashes.astype(str)
        types = types.astype(str)

        def run_stats(selector, types, ix, axis=None):
            ret = {}
            for t in np.unique(types):
                train_responses = selector(ix & (types == t))
                ret[t] = dict(
                    mean=train_responses.mean(axis=axis).astype(np.float32),
                    std=train_responses.std(axis=axis, ddof=1).astype(np.float32),
                    min=train_responses.min(axis=axis).astype(np.float32),
                    max=train_responses.max(axis=axis).astype(np.float32),
                    median=np.median(train_responses, axis=axis).astype(np.float32)
                )
            train_responses = selector(ix)
            ret['all'] = dict(
                mean=train_responses.mean(axis=axis).astype(np.float32),
                std=train_responses.std(axis=axis, ddof=1).astype(np.float32),
                min=train_responses.min(axis=axis).astype(np.float32),
                max=train_responses.max(axis=axis).astype(np.float32),
                median=np.median(train_responses, axis=axis).astype(np.float32)
            )
            return ret

        # --- compute statistics
        log.info('Computing statistics on training dataset')
        response_selector = lambda ix: np.concatenate([r for take, r in zip(ix, responses) if take], axis=0)
        response_statistics = run_stats(response_selector, types, train_idx, axis=0)

        input_selector = lambda ix: np.hstack([r.ravel() for take, r in zip(ix, inputs) if take])
        input_statistics = run_stats(input_selector, types, train_idx)

        statistics = dict(
            inputs=input_statistics,
            responses=response_statistics
        )

        if include_behavior:
            # ---- include statistics
            behavior_selector = lambda ix: np.concatenate([r for take, r in zip(ix, behavior) if take], axis=0)
            behavior_statistics = run_stats(behavior_selector, types, train_idx, axis=0)

            eye_selector = lambda ix: np.concatenate([r for take, r in zip(ix, eye_position) if take], axis=0)
            eye_statistics = run_stats(eye_selector, types, train_idx, axis=0)

            statistics['behavior'] = behavior_statistics
            statistics['eye_position'] = eye_statistics

        retval = dict(inputs=inputs,
                      responses=responses,
                      types=types.astype('S'),
                      train_idx=train_idx,
                      val_idx=val_idx,
                      test_idx=test_idx,
                      condition_hashes=hashes.astype('S'),
                      durations=durations.astype(np.float32),
                      trial_idx=trial_idx.astype(np.uint32),
                      neurons=neurons,
                      tiers=tiers.astype('S'),
                      statistics=statistics
                      )
        if include_behavior:
            retval['behavior'] = behavior
            retval['eye_position'] = eye_position

        if include_lvs:
            retval['latent_variables'] = latent_variable
        return retval


class BehaviorMixin:
    def load_eye_traces(self, key):
        r, center = (pupil.FittedContour.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
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

    -> InputResponse.Input
    -> pupil.FittedContour
    ---
    pupil              : external-data   # pupil dilation trace
    dpupil             : external-data   # derivative of pupil dilation trace
    center             : external-data   # center position of the eye
    """

    @property
    def key_source(self):
        return InputResponse & pupil.FittedContour & stimulus.BehaviorSync

    def _make_tuples(self, scan_key):
        log.info('Populating\n' + pformat(scan_key, indent=10))
        radius, xy, eye_time = self.load_eye_traces(scan_key)
        frame_times = InputResponse().load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.', depth=1)
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        sampling_period = float((Preprocessing() & scan_key).proj(period='1/behavior_lowpass').fetch1('period'))
        log.info('Downsampling eye signal to {}Hz'.format(1 / sampling_period))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(sampling_period, deye, 'hamming', warning=True)
        h_deye = self.get_filter(sampling_period, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(eye_time,
                                 np.convolve(radius, h_eye, mode='same'), k=1, ext=0)

        dpupil_spline = NaNSpline(eye_time,
                                  np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(eye_time,
                                    np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]),
                                    k=1, ext=0)

        flip_times, sample_times, trial_keys = \
            (InputResponse.Input() * MovieClips() * stimulus.Trial() & scan_key).fetch('flip_times', 'sample_times',
                                                                                       dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        for trial_key, flips, samps in tqdm(zip(trial_keys, flip_times, sample_times),
                                            total=len(trial_keys), desc='Trial '):
            t = fr2beh(flips[0] + samps)
            pupil = pupil_spline(t)
            dpupil = dpupil_spline(t)
            center = center_spline(t)
            nans = np.array([np.isnan(e).sum() for e in [pupil, dpupil, center]])
            if np.any(nans > 0):
                log.info('Found {} NaNs in one of the traces. Skipping trial {}'.format(np.max(nans),
                                                                                        pformat(trial_key, indent=5),
                                                                                        ))
            else:
                self.insert1(dict(scan_key, **trial_key,
                                  pupil=pupil,
                                  dpupil=dpupil,
                                  center=center),
                             ignore_extra_fields=True)



@schema
class Treadmill(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse.Input
    -> treadmill.Treadmill
    ---
    treadmill          : external-data   # treadmill speed (|velcolity|)
    """

    @property
    def key_source(self):
        rel = InputResponse
        return rel & treadmill.Treadmill() & stimulus.BehaviorSync()

    def _make_tuples(self, scan_key):
        print('Populating', pformat(scan_key))
        v, treadmill_time = self.load_treadmill_velocity(scan_key)
        frame_times = InputResponse().load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        sampling_period = float((Preprocessing() & scan_key).proj(period='1/behavior_lowpass').fetch1('period'))
        log.info('Downsampling treadmill signal to {}Hz'.format(1 / sampling_period))

        h_tread = self.get_filter(sampling_period, np.nanmedian(np.diff(treadmill_time)), 'hamming', warning=True)
        treadmill_spline = NaNSpline(treadmill_time, np.abs(np.convolve(v, h_tread, mode='same')), k=1, ext=0)

        flip_times, sample_times, trial_keys = \
            (InputResponse.Input() * MovieClips() * stimulus.Trial() & scan_key).fetch('flip_times', 'sample_times',
                                                                                       dj.key)
        flip_times = [ft.squeeze() for ft in flip_times]
        for trial_key, flips, samps in tqdm(zip(trial_keys, flip_times, sample_times),
                                            total=len(trial_keys), desc='Trial '):
            tm = treadmill_spline(fr2beh(flips[0] + samps))
            nans = np.isnan(tm)
            if np.any(nans):
                log.info('Found {} NaNs in one of the traces. Skipping trial {}'.format(nans.sum(),
                                                                                        pformat(trial_key, indent=5),
                                                                                        ))

            else:
                self.insert1(dict(scan_key, **trial_key, treadmill=tm),
                             ignore_extra_fields=True)


@schema
class MovieMultiDataset(dj.Manual):
    definition = """
    # defines a group of movie datasets

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

    _template = 'group{group_id:03d}-{animal_id}-{session}-{scan_idx}-pre{preproc_id}-seg{segmentation_method}-spi{spike_method}-pip{pipe_version}'

    def fill(self):
        selection = [
            ('17358-5-3', [
                dict(animal_id=17358, session=5, scan_idx=3, preproc_id=0, pipe_version=1, segmentation_method=3,
                     spike_method=5)]),
            ('17797-8-5', [
                dict(animal_id=17797, session=8, scan_idx=5, preproc_id=0, pipe_version=1, segmentation_method=3,
                     spike_method=5)]),
            ('18142-6-3', [
                dict(animal_id=18142, session=6, scan_idx=3, preproc_id=0, pipe_version=1, segmentation_method=3,
                     spike_method=5)]),
            ('17358-5-3-triple', dj.AndList([
                dict(animal_id=17358, session=5, scan_idx=3, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('17797-8-5-triple', dj.AndList([
                dict(animal_id=17797, session=8, scan_idx=5, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('18142-6-3-triple', dj.AndList([
                dict(animal_id=17358, session=5, scan_idx=3, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('9771-1-1-triple', dj.AndList([
                dict(animal_id=9771, session=1, scan_idx=1, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('9771-1-2-triple', dj.AndList([
                dict(animal_id=9771, session=1, scan_idx=2, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('16314-3-1-triple', dj.AndList([
                dict(animal_id=16314, session=3, scan_idx=1, pipe_version=1, segmentation_method=3, spike_method=5),
                'preproc_id in (0,1,2)'])),
            ('16314-3-1', [
                dict(animal_id=16314, session=3, scan_idx=1, preproc_id=0, pipe_version=1, segmentation_method=3,
                     spike_method=5)]),
            ('18142-platinum', [
                dict(animal_id=18142, pipe_version=1, segmentation_method=3, spike_method=5)]),
            ('8973-golden', dj.AndList(['animal_id=8973 and session=1 and scan_idx in (2,3,4,5,6,9,11,12)',
                                        dict(pipe_version=1, segmentation_method=3, spike_method=5, preproc_id=0)])),
            ('18979-2-7-jiakun',
             dict(animal_id=18979, session=2, scan_idx=7, pipe_version=1, segmentation_method=3, spike_method=5)),
            ('18799-3-14-jiakun',
             dict(animal_id=18799, session=3, scan_idx=14, pipe_version=1, segmentation_method=3, spike_method=5)),
            ('18142-all', dict(animal_id=18142, pipe_version=1, segmentation_method=3, spike_method=5, preproc_id=0)),
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
        log.info('Fetching data for\n' + pformat(key, indent=10))
        for mkey in (self.Member() & key).fetch(dj.key,
                                                order_by='animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC'):
            name = (self.Member() & mkey).fetch1('name')
            include_behavior = bool(Eye() * Treadmill() & mkey)
            data_names = ['inputs', 'responses'] if not include_behavior \
                else ['inputs',
                      'behavior',
                      'eye_position',
                      'responses']
            log.info('Data will be ({})'.format(','.join(data_names)))

            filename = InputResponse().get_filename(mkey)
            log.info('Loading dataset ' + name + '-->' + filename)

            ret[name] = MovieSet(filename, *data_names)

        if key_order is not None:
            log.info('Reordering datasets according to given key order')
            ret = OrderedDict([
                (k, ret[k]) for k in key_order
            ])
        return ret


class AttributeTransformer:
    def __init__(self, name, h5_handle, transforms):
        assert name in h5_handle, '{} must be in {}'.format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle
        self.transforms = transforms

    def __getattr__(self, item):
        if not item in self.h5_handle[self.name]:
            raise AttributeError('{} is not among the attributes'.format(item))
        else:
            ret = self.h5_handle[self.name][item].value
            if ret.dtype.char == 'S':  # convert bytes to univcode
                ret = ret.astype(str)
            for tr in self.transforms:
                ret = tr.column_transform(ret)
            return ret


class MovieSet(H5SequenceSet):
    def __init__(self, filename, *data_keys, transforms=None, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}
        self.stats_source = stats_source if stats_source is not None else 'all'

    @property
    def n_neurons(self):
        return self[0].responses.shape[1]

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def img_shape(self):
        return (1,) + self[0].inputs.shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source

        tmp = [np.atleast_1d(self.statistics['{}/{}/mean'.format(dk, stats_source)].value)
               for dk in self.data_groups]
        return self.transform(self.data_point(*tmp), exclude=Subsequence)

    def rf_base(self, stats_source='all'):
        N, c, t, w, h = self.img_shape
        t = min(t, 150)
        mean = lambda dk: self.statistics['{}/{}/mean'.format(dk, stats_source)].value
        d = dict(
            inputs=np.ones((1, c, t, w, h)) * np.array(mean('inputs')),
            eye_position=np.ones((1, t, 1)) * mean('eye_position')[None, None, :],
            behavior=np.ones((1, t, 1)) * mean('behavior')[None, None, :],
            responses=np.ones((1, t, 1)) * mean('responses')[None, None, :]
        )
        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)

    def rf_noise_stim(self, m, t, stats_source='all'):
        """
        Generates a Gaussian white noise stimulus filtered with a 3x3 Gaussian filter
        for the computation of receptive fields. The mean and variance of the Gaussian
        noise are set to the mean and variance of the stimulus ensemble.

        The behvavior, eye movement statistics, and responses are set to their respective means.
        Args:
            m: number of noise samples
            t: length in time

        Returns: tuple of input, behavior, eye, and response

        """
        N, c, _, w, h = self.img_shape
        stat = lambda dk, what: self.statistics['{}/{}/{}'.format(dk, stats_source, what)].value
        mu, s = stat('inputs', 'mean'), stat('inputs', 'std')
        h_filt = np.float64([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]]
        )
        noise_input = np.stack([convolve2d(np.random.randn(w, h), h_filt, mode='same')
                                for _ in range(m * t * c)]).reshape((m, c, t, w, h)) * s + mu

        mean_beh = np.ones((m, t, 1)) * stat('behavior', 'mean')[None, None, :]
        mean_eye = np.ones((m, t, 1)) * stat('eye_position', 'mean')[None, None, :]
        mean_resp = np.ones((m, t, 1)) * stat('responses', 'mean')[None, None, :]

        d = dict(
            inputs=noise_input.astype(np.float32),
            eye_position=mean_eye.astype(np.float32),
            behavior=mean_beh.astype(np.float32),
            responses=mean_resp.astype(np.float32)
        )

        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)

    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][
                                           str(item if g not in self.shuffle_dims else self.shuffle_dims[g][item])])
                              for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __repr__(self):
        return 'MovieSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups)) \
               + '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']' \
               + (
                   ('\n\t[Shuffled Features: ' + ', '.join(self.shuffle_dims) + ']') if len(
                       self.shuffle_dims) > 0 else '') + \
               ('\n\t[Stats source: {}]'.format(self.stats_source) if self.stats_source is not None else '')


schema.spawn_missing_classes()
