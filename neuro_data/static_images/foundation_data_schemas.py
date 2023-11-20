import datajoint as dj
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from foundation.fnn.data import Data
from foundation.fnn.model import Model, Instance
from foundation.fnn.train import Objective, Train
from foundation.utility.resize import Resize
from foundation.stimulus.video import FrameList
from foundation.recording.scan import ScanUnitOrder
from foundation.recording.trace import Trace
from neuro_data import logger as log
from neuro_data.static_images.data_schemas import Preprocessing, SplineCurve, FilterMixin, stimulus, fuse

schema = dj.schema('neurodata_foundation_static')

@schema
class FoundationInputResponse(dj.Computed, FilterMixin):
    definition = """ # foundation model predicted responses of all neurons to all images
    -> Model
    -> FrameList
    -> Preprocessing
    ---
    """
    
    class Input(dj.Part):
        definition = """
        -> master
        -> stimulus.Frame
        ---
        row_id              : int         # row id in the response block
        -> stimulus.StaticImage.Image
        """

    class ResponseBlock(dj.Part):
        definition = """
        -> master
        ---
        responses           : blob@data   # response of all neurons to all trials in shape of (n_frames, n_neurons)
        """

    class ResponseKeys(dj.Part):
        definition = """
        -> master
        -> fuse.Activity.Trace
        ---
        col_id              : int         # col id in the response block
        """
    
    @property
    def key_source(self):
        return Model * FrameList * Preprocessing & 'preproc_id = 8'
    
    def make(self, key):
        # Get stimulus parameters used for foundation model training
        model = (Model & key).model(device="cuda")
        model_period = (Data & key).link.compute.sampling_period
        model_offset = (Data & key).link.compute.unit_offset
        height, width = (Data & key).link.compute.resolution
        resize_id = (Data & key).link.compute.resize_id
        objective = Objective & (Train & (Instance() & key).link).link
        assert ('burnin_frames' in (objective.link).fetch1(dj.key)), '"burnin_frames" is not a valid attribute in the {} objective!'.format(objective.fetch1('objective_type'))
        burnin_frames = ((Objective & (Train & (Instance() & key).link).link).link).fetch1('burnin_frames')
        
        # Create stimulus generator
        video = (FrameList() & key).compute.video
        rvideo = (Resize & {"resize_id": resize_id}).link.resize(video=video, height=height, width=width)
        stimuli = rvideo.generate(period=model_period)

        # And we'll feed that to our model, collecting responses in a list
        traces = []
        for r in model.generate_response(stimuli=stimuli):
            traces.append(r)
        traces = np.stack(traces, axis=0)

        # Since we have the number of frames, the sampling period and offset, we know the timing of the response
        frame_times = np.arange(traces.shape[0]) * model_period + model_offset
        
        # Integration window size for the predicted trace
        duration, offset = map(float, (Preprocessing & key).fetch1('duration', 'offset'))
        filter_type = (Preprocessing & key).fetch1('filter')
        sample_point = offset + duration / 2

        log.info('Generating lowpass filters to {}Hz'.format(1 / duration))
        downsample_filter = self.get_filter(duration, model_period, filter_type, warning=False)
        
        # Low pass filter the traces after removing the burnin frames
        R = [] 
        for trace in traces.T:
            trace_spline = SplineCurve(frame_times[burnin_frames:],
                                       [np.convolve(trace[burnin_frames:], downsample_filter, mode='same')], k=1, ext=1)
        
            # Compute onset time of each trial (i.e. when pre_blank ends and the image starts), see details of how video.times is computed  
            # at foundation.stimulus.video.FrameList.compute
            stimulus_onset = np.array(video.times[1:])[::2]
            if stimulus_onset[0] < frame_times[burnin_frames]:
                warnings.warn('First trial onset is within the burn-in period!')
            
            # Get interpolated trial responses
            _R = trace_spline(stimulus_onset + sample_point, log=False)
            R.append(_R.squeeze())
        R = np.stack(R).T
        
        # Get info of input images and neurons
        input_tups = (FrameList.Member * stimulus.Frame & key).fetch('condition_hash', 'image_class', 'image_id', order_by='framelist_index ASC', as_dict=True)
        assert ('trace_filterset_id' in (Data & key).link.fetch1(dj.key)), 'Trace restriction has not been implemented for the {} data type!'.format((Data & key).fetch1('data_type'))
        unit_tups = (fuse.Activity.Trace * Trace.ScanUnit * (ScanUnitOrder & (Data & key).link)).fetch(as_dict=True, order_by='unit_id ASC')

        # Re-order responses by ascending unit_id
        order = np.array([tup['trace_order'] for tup in unit_tups])
        R = R[:, order]
        
        self.insert1(key)
        self.ResponseBlock.insert1(dict(**key, responses=R))
        self.ResponseKeys.insert([dict(**key, **tup, col_id=cid) for cid, tup in enumerate(unit_tups)], ignore_extra_fields=True)
        self.Input.insert([dict(**key, **tup, row_id=rid) for rid, tup in enumerate(input_tups)])
