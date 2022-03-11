import datajoint as dj

class DynamicStaticRequest(dj.Manual):
    definition = """ # dynamic model responses to static images shown in a static scan, units in the dataset are from the dynamic scan
    -> DynamicScan
    -> StaticScan
    -> InputConfig
    -> ResponseConfig
    -> BehaviorConfig
    -> TierConfig
    -> UnitConfig
    """