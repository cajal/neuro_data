import datajoint as dj
import numpy as np

from neuro_data.utils.data import fill_nans
from .schema_bridge import *

class TraceMixin:
    def load_frame_times(self, key):
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

    def load_traces_and_frametimes(self, key):
        from .data_schemas import MovieScan
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        frame_times = (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

        soma = pipe.MaskClassification.Type() & dict(type='soma')

        spikes = (dj.U('field', 'channel') * pipe.Activity.Trace() * MovieScan.Unit() \
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