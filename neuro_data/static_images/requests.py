from .dataset_config import (
    InputConfig,
    ResponseConfig,
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
<<<<<<< HEAD

@schema
class DynamicStaticNoBehAugRespRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> ResponseConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """
=======
>>>>>>> parent of 87497e9 (Merge branch 'master' of github.com:cajal/neuro_data)
