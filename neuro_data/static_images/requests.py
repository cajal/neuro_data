from os import sched_getaffinity
from .dataset_config import InputConfig, ResponseConfig, TierConfig, LayerConfig, AreaConfig, StatsConfig
from .data_schemas import StaticScan
from .ds_pipe import DynamicScan
import datajoint as dj

schema = dj.schema("neurodata_static")

@schema
class DynamicStaticNoBehRequest(dj.Manual):
    definition = """
    -> DynamicScan.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> ResponseConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """