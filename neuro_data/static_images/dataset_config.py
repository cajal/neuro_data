from collections import OrderedDict

import datajoint as dj
import numpy as np
import pandas as pd
from neuro_data import logger as log
from neuro_data.static_images import datasets
from neuro_data.utils.config import ConfigBase
from neuro_data.utils.data import h5cached

from .data_schemas import (
    FF_CLASSES,
    ConditionTier,
    Frame,
    InputResponse,
    Preprocessing,
    StaticMultiDataset,
    StaticScan,
    schema,
)
from .ds_pipe import DvScanInfo

stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
base = dj.create_virtual_module("base", "neurostatic_base")



@schema
class InputConfig(ConfigBase, dj.Lookup):
    _config_type = "input"

    def frame(self, scan_key):
        self.part_table().frame(scan_key)

    class NeuroStaticFrame(dj.Part):
        # only load stimulus_type=='stimulus.Frame'
        definition = """
        -> master
        ---
        -> Preprocessing
        """
        content = [
            {"preproc_id": 9},
        ]

        def input(self, scan_key):
            params = (self * Preprocessing).fetch1()
            if not (params["gamma"] or params["linear_mon"]):
                trial_idx, cond, frame = (
                    InputResponse.Input * Frame & stimulus.Frame & scan_key & params
                ).fetch(
                    "trial_idx",
                    "condition_hash",
                    "frame",
                    order_by="trial_idx",
                )
                # reshape inputs
                frame = np.stack(frame)
            else:
                raise NotImplementedError(
                    f'InputConfig: gamma={params["gamma"]}, linear_mon={params["linear_mon"]} not implemented!'
                )
            assert (
                len(frame.shape) == 3
            ), f"Images has shape not supported: {frame.shape}!"
            frame = frame[:, None, ...]
            return trial_idx, cond, frame, np.full(len(trial_idx), "stimulus.Frame")


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
        content = [
            {},
        ]

        def behavior(self, **kwargs):
            return None


@schema
class TierConfig(ConfigBase, dj.Lookup):
    _config_type = "tier"

    def tier(self, condition_hashes):
        self.part_table().tier(condition_hashes)

    class ConditionTier(dj.Part):
        definition = """
        -> master
        ---
        """
        content = [
            {},
        ]

        def tier(self, scan_key, condition_hashes):
            cond_tier_df = pd.DataFrame(
                (
                    (ConditionTier & scan_key).fetch(
                        "condition_hash", "tier", as_dict=True
                    )
                )
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
        content = [
            {},
        ]

        def layer(self, unit_keys):
            anatomy = dj.create_virtual_module("anatomy", "pipeline_anatomy")
            unit_layer_df = pd.DataFrame(
                (
                    (anatomy.LayerMembership & unit_keys).fetch(
                        "animal_id",
                        "session",
                        "scan_idx",
                        "layer",
                        "unit_id",
                        as_dict=True,
                    )
                )
            )
            unit_df = pd.DataFrame(unit_keys)
            unit_df = unit_df.merge(unit_layer_df, how="left")
            assert unit_df.layer.notnull().all(), "Missing layer for some units!"
            return unit_df.layer.values

    class Constant(dj.Part):
        definition = """
        -> master
        ---
        layer         :varchar(256)        # layer name
        """
        content = [
            {"layer": "L2/3"},
        ]

        def layer(self, unit_keys):
            return np.full(len(unit_keys), self.fetch1("layer"))


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
        content = [
            {},
        ]

        def area(self, unit_keys):
            anatomy = dj.create_virtual_module("anatomy", "pipeline_anatomy")
            unit_area_df = pd.DataFrame(
                (
                    (anatomy.AreaMembership & unit_keys).fetch(
                        "animal_id",
                        "session",
                        "scan_idx",
                        "brain_area",
                        "unit_id",
                        as_dict=True,
                    )
                )
            )
            unit_df = pd.DataFrame(unit_keys)
            unit_df = unit_df.merge(unit_area_df, how="left")
            assert unit_df.area.notnull().all(), "Missing area for some units!"
            return unit_df.area.values

    class Constant(dj.Part):
        definition = """
        -> master
        ---
        brain_area         :varchar(256)        # brain area name
        """
        content = [
            {"brain_area": "V1"},
        ]

        def area(self, unit_keys):
            return np.full(len(unit_keys), self.fetch1("brain_area"))


@schema
class StatsConfig(ConfigBase, dj.Lookup):
    _config_type = "stats"

    class NoBehFullFieldFrame(dj.Part):
        # implemented only for full field stimulus with stimulus_type=='stimulus.Frame'
        definition = """
        # mimic the behavior of InputResponse with the corresponding preprocess_id to compute the statistics of input and responses, do not compute stats for behavior variables
        -> master
        ---
        stats_tier="train"                 : enum("train", "test", "validation", "all")               # tier used for computing stats
        stats_per_input                    : tinyint                                                  # whether to compute stats per input
        """
        content = [
            dict(stats_tier="train", stats_per_input=1),
        ]

        @staticmethod
        def run_stats_resp(data, ix, axis=0):
            ret = {}
            data = data[ix]
            ret["all"] = dict(
                mean=data.mean(axis=axis).astype(np.float32),
                std=data.std(axis=axis, ddof=1).astype(np.float32),
                min=data.min(axis=axis).astype(np.float32),
                max=data.max(axis=axis).astype(np.float32),
                median=np.median(data, axis=axis).astype(np.float32),
            )
            ret["stimulus.Frame"] = ret["all"]
            return ret

        @staticmethod
        def run_stats_input(data, ix, per_input=False):
            ret = {}
            data = data[ix]
            ret["all"] = dict(
                mean=data.mean(axis=(-1, -2)).mean().astype(np.float32)
                if per_input
                else data.mean().astype(np.float32),
                std=data.std(axis=(-1, -2))
                .mean()
                .astype(
                    np.float32
                )  # ddof is not set here to match the behavior of InputResponse
                if per_input
                else data.std(ddof=1).astype(np.float32),
                min=data.min().astype(np.float32),
                max=data.max().astype(np.float32),
                median=np.median(data).astype(np.float32),
            )
            ret["stimulus.Frame"] = ret["all"]
            return ret

        def stats(self, condition_hashes, images, responses, tiers):

            key = self.fetch1()
            # check if the method is eligible for condition_hashes requested
            assert (
                (
                    stimulus.Condition
                    & "condition_hash in {}".format(tuple(condition_hashes))
                ).fetch("stimulus_type")
                == "stimulus.Frame"
            ).all(), "StatsConfig.NeuroStaticNoBehFrame is only implemented for stimulus.Frame"
            image_classes = (
                stimulus.Condition * stimulus.Frame
                & "condition_hash in {}".format(tuple(condition_hashes))
            ).fetch("image_class")
            assert set(image_classes) <= set(
                FF_CLASSES
            ), "StatsConfig.NeuroStaticNoBehFrame is only implemented for full-field stimulus"

            # reshape inputs
            images = np.stack(images)
            if len(images.shape) == 3:
                log.info("Adding channel dimension")
                images = images[:, None, ...]
            elif len(images.shape) == 4:
                images = images.transpose(0, 3, 1, 2)

            # compute stats
            if key["stats_tier"] in ("train", "validation", "test"):
                ix = tiers == key["stats_tier"]
            elif key["stats_tier"] == "all":
                ix = np.ones_like(tiers, dtype=bool)
            else:
                raise NotImplementedError(
                    "stats_tier must be one of train, validation, test, all"
                )

            response_statistics = self.run_stats_resp(responses, ix, axis=0)
            input_statistics = self.run_stats_input(
                images, ix, per_input=key["stats_per_input"]
            )
            statistics = dict(images=input_statistics, responses=response_statistics)
            return statistics


@schema
class DatasetConfig(ConfigBase, dj.Lookup):
    _config_type = "dataset"

    def get_filename(self, key=None):
        key = self.fetch1() if key is None else key
        return self.part_table(key).get_filename()

    def compute_data(self, key=None):
        key = self.fetch1() if key is None else key
        self.part_table(key).compute_data()

    @h5cached(
        "/external/cache/dynamic-static",
        mode="array",
        transfer_to_tmp=False,
        file_format="dynamic-static-{animal_id}-{dynamic_session}-{dynamic_scan_idx}-{static_session}-{static_scan_idx}-{dataset_hash}.h5",
    )
    class DvStaticNoBeh(dj.Part):
        definition = """ # dynamic model responses to static images shown in a static scan, units in the dataset are from the dynamic scan
        -> master
        ---
        -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
        -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
        -> InputConfig
        -> TierConfig
        -> LayerConfig
        -> AreaConfig
        -> StatsConfig
        """

        data_names = ["images", "responses"]

        @property
        def content(self):
            from . import requests

            return requests.DynamicStaticNoBehRequest

        @property
        def static_scan(self):
            """returns the scan that would be injected into InputResponse, the scan key should match the scan key returned by compute_data"""
            return (
                StaticScan
                & self.proj(..., session="dynamic_session", scan_idx="dynamic_scan_idx")
            ).fetch1("KEY")

        @property
        def preprocessing(self):
            return (Preprocessing & "preproc_id=8").fetch1("KEY")

        def name(self, group_id, key=None, **kwargs):
            key = self.fetch1() if key is None else key
            return f'{group_id}-{key["animal_id"]}-{key["dynamic_session"]}-{key["dynamic_scan_idx"]}-{key["static_session"]}-{key["static_scan_idx"]}'

        def compute_data(self, key=None):
            key = self.fetch1() if key is None else key
            dynamic_scan = (
                DvScanInfo
                & {
                    "animal_id": key["animal_id"],
                    "session": key["dynamic_session"],
                    "scan_idx": key["dynamic_scan_idx"],
                }
            ).fetch1("KEY")
            static_scan = (
                StaticScan()
                & {
                    "animal_id": key["animal_id"],
                    "session": key["static_session"],
                    "scan_idx": key["static_scan_idx"],
                }
            ).fetch1("KEY")
            log.info("Fecthing images")
            trial_idx, condition_hashes, images, types = (
                InputConfig().part_table(key).input(static_scan)
            )
            log.info("Fetching responses")
            responses = (DvScanInfo & key).responses(
                trial_idx=trial_idx,
                condition_hashes=condition_hashes,
            )
            dynamic_unit_keys = (DvScanInfo & key).unit_keys()
            log.info("Fecthing tiers")
            tiers = TierConfig().part_table(key).tier(static_scan, condition_hashes)
            log.info("Fecthing layer information")
            layer = LayerConfig().part_table(key).layer(dynamic_unit_keys)
            log.info("Fecthing area information")
            area = AreaConfig().part_table(key).area(dynamic_unit_keys)
            log.info("Computing stats")
            statistics = (
                StatsConfig()
                .part_table(key)
                .stats(condition_hashes, images, responses, tiers)
            )
            neurons = dict(
                unit_ids=np.array([k["unit_id"] for k in dynamic_unit_keys]).astype(
                    np.uint16
                ),
                animal_ids=np.array([k["animal_id"] for k in dynamic_unit_keys]).astype(
                    np.uint16
                ),
                sessions=np.array([k["session"] for k in dynamic_unit_keys]).astype(
                    np.uint8
                ),
                scan_idx=np.array([k["scan_idx"] for k in dynamic_unit_keys]).astype(
                    np.uint8
                ),
                layer=layer.astype("S"),
                area=area.astype("S"),
            )
            return dict(
                images=images,
                responses=responses,
                types=types.astype("S"),
                condition_hashes=condition_hashes.astype("S"),
                trial_idx=trial_idx.astype(np.uint32),
                neurons=neurons,
                tiers=tiers.astype("S"),
                statistics=statistics,
            )


@schema
class DatasetInputResponse(dj.Computed):
    # Inject datasets back to InputResponse table. Mutltiple datasets with the same
    # static scan but different dataset_hash may share the same InputResponse entry.
    # The entry in InputResponse is only inserted to provide necessary dependencies for downstream tables, e.g. StaticMultiDataset.Member
    # Part tables of InputResponse is left empty if the master entry is inserted from this table.
    definition = """
    -> DatasetConfig
    ---
    -> InputResponse
    """

    def make(self, key):
        input_response_key = {
            **DatasetConfig().part_table(key).static_scan,
            **DatasetConfig().part_table(key).preprocessing,
        }
        key = {
            **key,
            **input_response_key,
        }
        InputResponse().insert1(
            key,
            allow_direct_insert=True,
            ignore_extra_fields=True,
            skip_duplicates=True,
        )
        self.insert1(key, ignore_extra_fields=True)


@schema
class MultiDataset(dj.Manual):
    definition = """
    -> StaticMultiDataset             
    ---
    description             : varchar(255)               # description of the dataset
    """

    class Member(dj.Part):
        definition = """
        -> master
        -> DatasetConfig
        ---
        name                    : varchar(50)            # string description to be used for training
        """

    @property
    def next_group_id(self):
        return StaticMultiDataset.fetch("group_id").max() + 1

    def fill(self, member_key, description):
        key = dict(
            group_id=self.next_group_id,
            description="Inserted with MultiDataset: " + description,
        )
        mkey = (DatasetConfig & member_key).fetch("KEY", as_dict=True)
        mkey = [{**k, **key, **(DatasetInputResponse & k).fetch1()} for k in mkey]
        mkey = [{**k, "name": DatasetConfig().part_table(k).name(**k)} for k in mkey]

        with self.connection.transaction:
            StaticMultiDataset.insert1(key, ignore_extra_fields=True)
            StaticMultiDataset.Member.insert(mkey, ignore_extra_fields=True)
            self.insert1(key, ignore_extra_fields=True)
            self.Member().insert(mkey, ignore_extra_fields=True)

    def fetch_data(self, key, key_order=None):
        ret = OrderedDict()
        log.info("Fetching data for " + repr(key))
        for mkey in (self.Member() * DatasetInputResponse & key).fetch(
            dj.key, order_by="animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC"
        ):
            name = (self.Member() & mkey).fetch1("name")
            data_names = DatasetConfig().part_table(mkey).data_names
            log.info("Data will be ({})".format(",".join(data_names)))

            h5filename = DatasetConfig().part_table(mkey).get_filename()
            log.info("Loading dataset {} --> {}".format(name, h5filename))
            ret[name] = datasets.StaticImageSet(h5filename, *data_names)
        if key_order is not None:
            log.info(
                "Reordering datasets according to given key order {}".format(
                    ", ".join(key_order)
                )
            )
            ret = OrderedDict([(k, ret[k]) for k in key_order])
        return ret
