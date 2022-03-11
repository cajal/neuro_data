import datajoint as dj
from neuro_data.utils.config import ConfigBase
from .data_schemas import Frame, StaticScan, Preprocessing, InputResponse, ConditionTier
from . import ds_pipe_requests
import pandas as pd

experiment = dj.create_virtual_module("experiment", "pipeline_experiment")
fuse = dj.create_virtual_module("fuse", "pipeline_fuse")
meso = dj.create_virtual_module("meso", "pipeline_meso")
shared = dj.create_virtual_module("shared", "pipeline_shared")
anatomy = dj.create_virtual_module("anatomy", "pipeline_anatomy")

dv_nn6_architecture = dj.create_virtual_module("dv_nn6_architecture", "dv_nns_v6_architecture")
dv_nn6_train = dj.create_virtual_module("dv_nn6_train", "dv_nns_v6_train")
dv_nn6_model = dj.create_virtual_module("dv_nn6_model", "dv_nns_v6_model")
dv_nn6_scan = dj.create_virtual_module("dv_nn6_scan", "dv_nns_v6_scan")
dv_nn6_pipe = dj.create_virtual_module("dv_nn6_pipe", "dv_nns_v6_ds_pipe")

schema = dj.schema("zhuokun_neurodata_static")

from utils.datajoint import config

def dv_nn6_models():
    keys = dv_nn6_model.ModelConfig.Scan1 * dv_nn6_scan.ScanModel * dv_nn6_scan.ModelScans & "n_scans=1"
    keys = dj.U("architecture_hash", "train_hash") * keys
    return keys.proj(dynamic_session="session", dynamic_scan_idx="scan_idx")


@config(schema)
class DvConfig(dj.Lookup):
    class Nn6(dj.Part):
        definition = """
        -> master
        ---
        -> dv_nn6_scan.ScanConfig
        -> dv_nn6_architecture.ArchitectureConfig
        -> dv_nn6_train.TrainConfig
        -> dv_nn6_model.Instance
        -> dv_nn6_model.OutputConfig
        -> dv_nn6_pipe.ResponseConfig
        """

        @property
        def content(self):
            return dj.U(*self.heading.secondary_attributes) & (dv_nn6_models() * dv_nn6_pipe.Response)

        def unit_keys(self, animal_id, dynamic_session, dynamic_scan_idx):
            scan_key = dict(
                animal_id=animal_id,
                session=dynamic_session,
                scan_idx=dynamic_scan_idx,
            )
            scan_key, n_units = (dv_nn6_scan.Scan & scan_key & self).fetch1(dj.key, "n_units")
            units = (self * dv_nn6_scan.Scan.Unit & scan_key).proj(
                ..., dynamic_session="session", dynamic_scan_idx="scan_idx", dynamic_unit_id="unit_id"
            )
            unit_keys = units.fetch(*DvInfo.Unit.primary_key, as_dict=True, order_by="nn_response_index")

            assert len(unit_keys) == n_units

            return unit_keys

        def response(self, animal_id, dynamic_session, dynamic_scan_idx, image_class, image_id):
            response_key = dict(
                animal_id=animal_id,
                session=dynamic_session,
                scan_idx=dynamic_scan_idx,
                image_class=image_class,
                image_id=image_id,
            )
            response = dv_nn6_models() * dv_nn6_pipe.Response & self & response_key
            return response.fetch1("response")


@schema
class DvInfo(dj.Computed):
    definition = """
    -> experiment.Scan.proj(dynamic_session="session", dynamic_scan_idx="scan_idx")
    -> DvConfig
    ---
    n_units         : int unsigned      # number of units
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> fuse.ScanSet.Unit.proj(dynamic_session="session", dynamic_scan_idx="scan_idx", dynamic_unit_id="unit_id")
        ---
        response_index      : int unsigned      # index of unit in response vector
        """

    @property
    def key_source(self):
        keys = (experiment.Scan * DvConfig).proj(dynamic_session="session", dynamic_scan_idx="scan_idx")
        key = [
            dv_nn6_models() * DvConfig.Nn6,
        ]
        return keys & key

    def make(self, key):
        unit_keys = (
            DvConfig().part_table(key).unit_keys(key["animal_id"], key["dynamic_session"], key["dynamic_scan_idx"])
        )

        self.insert1(dict(key, n_units=len(unit_keys)))

        self.Unit.insert([dict(unit_key, response_index=i) for i, unit_key in enumerate(unit_keys)])

    def response(self, image_class, image_id, key=None):
        if key is None:
            key, n_units = self.fetch1(dj.key, "n_units")
        else:
            key, n_units = (self & key).fetch1(dj.key, "n_units")

        dv_conf = DvConfig().part_table(key)
        response = dv_conf.response(
            key["animal_id"], key["dynamic_session"], key["dynamic_scan_idx"], image_class, image_id
        )
        assert len(response) == n_units
        return response


@schema
class DynamicScanCandidate(dj.Manual):
    definition = """ # list of scans to process

    -> fuse.ScanDone
    ---
    candidate_notes='' : varchar(1024)
    """

    @staticmethod
    def fill(
        key, candidate_notes="", segmentation_method=6, spike_method=6, pipe_version=1
    ):
        """Fill an entry with key"""
        DynamicScanCandidate().insert1(
            {
                "segmentation_method": segmentation_method,
                "spike_method": spike_method,
                "pipe_version": pipe_version,
                **key,
                "candidate_notes": candidate_notes,
            },
            skip_duplicates=True,
        )


@schema
class DynamicScan(dj.Computed):
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

    key_source = (
        fuse.ScanDone()
        & DynamicScanCandidate
        & "spike_method=6 and segmentation_method=6"
    )

    @staticmethod
    def complete_key(key):
        return dict(
            (
                dj.U("segmentation_method", "pipe_version")
                & (meso.ScanSet.Unit() & key)
            ).fetch1(dj.key),
            **key
        )

    def make(self, key):
        self.insert(fuse.ScanDone() & key, ignore_extra_fields=True)
        pipe = (fuse.ScanDone() & key).fetch1("pipe")
        pipe = dj.create_virtual_module(pipe, "pipeline_" + pipe)
        self.Unit().insert(
            fuse.ScanDone * pipe.ScanSet.Unit
            & key
            & dict(pipe_version=1, segmentation_method=6, spike_method=5),
            ignore_extra_fields=True,
        )


@schema
class UnitConfig(ConfigBase, dj.Lookup):
    _config_type = "unit"

    class MaskType(dj.Part):
        definition = """
        -> master
        ---
        -> shared.MaskType
        """
        contents = [
            ["soma"],
        ]

        def unit(self, scan_key):
            pipe = (fuse.ScanDone() & scan_key).fetch1("pipe")
            pipe = dj.create_virtual_module(pipe, "pipeline_" + pipe)
            target_type = self.fetch1("mask_type")
            units = (
                fuse.ScanSet.Unit().proj()
                * pipe.ScanSet.Unit
                * pipe.MaskClassification.Type
                & scan_key
                & {"type": target_type}
            ).fetch("KEY")
            return units


@schema
class ResponseConfig(dj.Lookup):
    _config_type = "response"

    def response(self, dynamic_scan, trial_idx, cond, dynamic_unit_ids):
        self.part_table().response()

    class DvNns6(dj.Part):
        ### each tuple here should represent a dynamic model configuration, models trained on different dynamic scans with same core/architecture should share the same tuple
        definition = """
        -> master
        ---
        ???
        """

        def response(self, dynamic_scan, trial_idx, cond, dynamic_unit_ids):
            """
            dynamic_scan: dict(animal_id, session, scan_idx)
            trial_idx: list of trial indices
            cond: list of condition_hash
            dynamic_unit_ids: list of unit_ids of interest
            """
            # ???
            return responses


@schema
class InputConfig(ConfigBase, dj.Lookup):
    _config_type = "input"

    def frame(self, scan_key):
        self.part_table().frame(scan_key)

    class NeuroStaticFrame(dj.Part):
        definition = """
        -> master
        ---
        -> Preprocessing
        """
        content = []

        def frame(self, scan_key):
            params = self.fetch1()
            trial_idx, cond, frame = (InputResponse.Input & scan_key & params).fetch(
                "trial_idx", "condition_hash", "frame"
            )
            return trial_idx, cond, frame


@schema
class BehaviorConfig(ConfigBase, dj.Lookup):
    _config_type = "behavior"

    def behavior(self, scan_key):
        self.part_table().behavior(scan_key)

    class No(dj.Part):
        definition = """
        -> master
        ---
        """
        content = []

        def behavior(self, **kwargs):
            return None


@schema
class TierConfig(ConfigBase, dj.Lookup):
    _config_type = "tier"

    def tier(self, condition_hashes):
        self.part_table().tier(condition_hashes)

    class NeuroStaticTier(dj.Part):
        definition = """
        -> master
        ---
        """
        content = []

        def tier(self, scan_key, condition_hashes):
            cond_tier_df = (
                (ConditionTier & scan_key)
                .fetch("condition_hash", "tier", format="frame")
                .reset_index()
            )
            cond_df = pd.DataFrame(dict(condition_hash=condition_hashes))
            cond_df = cond_df.merge(cond_tier_df, on="condition_hash", how="left")
            assert cond_df.tier.notnull().all(), "Missing tier for some conditions!"
            return cond_df.tier.values


@schema
class LayerConfig(ConfigBase, dj.Lookup):
    _config_type = "layer"

    def layer(self, scan_key, unit_ids):
        self.part_table().layer(scan_key, unit_ids)

    class LayerMembership(dj.Part):
        definition = """
        -> master
        ---
        """
        content = []

        def layer(self, scan_key, unit_ids):
            unit_layer_df = (
                (anatomy.LayerMembership & scan_key)
                .fetch("layer", "unit_id", format="frame")
                .reset_index()
            )
            unit_df = pd.DataFrame(dict(unit_id=unit_ids))
            unit_df = unit_df.merge(unit_layer_df, on="unit_id", how="left")
            assert unit_df.layer.notnull().all(), "Missing layer for some units!"
            return unit_df.layer.values


@schema
class AreaConfig(ConfigBase, dj.Lookup):
    _config_type = "area"

    def area(self, scan_key, unit_ids):
        self.part_table().area(scan_key, unit_ids)

    class AreaMembership(dj.Part):
        definition = """
        -> master
        ---
        """
        content = []

        def area(self, scan_key, unit_ids):
            unit_area_df = (
                (anatomy.AreaMembership & scan_key)
                .fetch("area", "unit_id", format="frame")
                .reset_index()
            )
            unit_df = pd.DataFrame(dict(unit_id=unit_ids))
            unit_df = unit_df.merge(unit_area_df, on="unit_id", how="left")
            assert unit_df.area.notnull().all(), "Missing area for some units!"
            return unit_df.area.values

@schema
class StatsConfig(ConfigBase, dj.Lookup):
    _config_type = "stats"

    def stats(self, scan_key, unit_ids):
        self.part_table().stats(scan_key, unit_ids)

    class (dj.Part):
        definition = """
        -> master
        ---
        """
        content = []

        def stats(self, scan_key, unit_ids):
            unit_stats_df = (
                (anatomy.Stats & scan_key)
                .fetch("unit_id", "mean", "std", "max", "min", "n", format="frame")
                .reset_index()
            )
            unit_df = pd.DataFrame(dict(unit_id=unit_ids))
            unit_df = unit_df.merge(unit_stats_df, on="unit_id", how="left")
            assert unit_df.n.notnull().all(), "Missing stats for some units!"
            return unit_df.n.values

@schema
class DatasetConfig(ConfigBase, dj.Lookup):
    _config_type = "dataset"

    def compute_data(self, key):
        self.part_table().compute_data(key)

    class DynamicStatic(dj.Part):
        definition = """ # dynamic model responses to static images shown in a static scan, units in the dataset are from the dynamic scan
        -> DynamicScan
        -> StaticScan
        -> InputConfig
        -> ResponseConfig
        -> BehaviorConfig
        -> TierConfig
        -> UnitConfig
        -> LayerConfig
        -> AreaConfig
        -> StatsConfig
        """
        content = ds_pipe_requests.DynamicStaticRequest().fetch(as_dict=True)

        def compute_data(self, key):
            dynamic_scan = DynamicScan() & key
            static_scan = StaticScan() & key
            trial_idx, cond, frame = (InputConfig() & key).input(static_scan)
            dynamic_unit_ids = (UnitConfig() & key).unit(dynamic_scan)
            response = (ResponseConfig() & key).response(
                dynamic_scan, trial_idx, cond, dynamic_unit_ids
            )
            behavior = (BehaviorConfig() & key).behavior(static_scan, trial_idx)
            tier = (TierConfig() & key).tier(static_scan, cond)
            layer = (LayerConfig() & key).layer(static_scan, dynamic_unit_ids)
            area = (AreaConfig() & key).area(static_scan, dynamic_unit_ids)
            return dynamic_scan, dynamic_unit_ids, response, behavior, tier, layer, area

    class Static(dj.Part):
        definition = """ # static datasets
        -> StaticScan
        -> InputConfig
        -> ResponseConfig
        -> BehaviorConfig
        -> TierConfig
        -> UnitConfig
        -> LayerConfig
        -> AreaConfig
        -> StatsConfig
        """


@schema
class ScanDataset(dj.Computed):
    definition = """
    -> DatasetConfig
    ---
    group_id
    """

    def make(self, key, insert=True):
        (
            scan_key,
            unit_ids,
            response,
            behavior,
            tier,
            layer,
            area,
        ) = DatasetConfig.compute_data(key)
        return super().make(key)

    def static_scan(self):
        return
