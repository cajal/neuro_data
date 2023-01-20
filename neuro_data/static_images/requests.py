from .dataset_config import (
    InputConfig,
    TierConfig,
    LayerConfig,
    AreaConfig,
    StatsConfig,
)
from .data_schemas import StaticScan
from .ds_pipe import DvScanInfo, DvModelConfig
import datajoint as dj

schema = dj.schema("neurodata_static")


@schema
class DvScanInfoRequest(dj.Manual):
    definition = """
    -> StaticScan
    -> DvModelConfig
    """


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

@schema
class DynamicStaticNoBehDiffAnimalRequest(dj.Manual):
    definition = """
    -> DvScanInfo.proj(dynamic_animal_id='animal_id', dynamic_session='session', dynamic_scan_idx='scan_idx')
    -> StaticScan.proj(static_animal_id='animal_id', static_session='session', static_scan_idx='scan_idx')
    -> InputConfig
    -> TierConfig
    -> LayerConfig
    -> AreaConfig
    -> StatsConfig
    """
