from os import sched_getaffinity
from .dataset_config import (
    InputConfig,
    TierConfig,
    LayerConfig,
    AreaConfig,
    StatsConfig,
)
from .data_schemas import StaticScan
from .ds_pipe import DvScanInfo
import datajoint as dj

schema = dj.schema("neurodata_static")


@schema
class DynamicStaticNoBehRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """
