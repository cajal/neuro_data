from collections import OrderedDict
from re import T

import datajoint as dj
import numpy as np
import pandas as pd
from tqdm import tqdm
from neuro_data import logger as log
from neuro_data.static_images import datasets
from neuro_data.utils.config import ConfigBase
from neuro_data.utils.data import h5cached
from neuro_data.utils.stimuli import frame2_make_mask

from .data_schemas import (
    FF_CLASSES,
    ConditionTier,
    Frame,
    InputResponse,
    Eye,
    Treadmill,
    Preprocessing,
    StaticMultiDataset,
    StaticScan,
    schema,
    stimulus,
)
from .ds_pipe import DvScanInfo


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
        warning = "NeuroStaticFrame: The output frame format was not compatible with StatsConfig and led to an underestimated std of the images. This part table is deprecated and only kept for record keeping purposes."

        def __new__(cls, *args, **kwargs):
            log.warning(cls.warning)
            return super().__new__(cls, *args, **kwargs)

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

    class NeuroStaticValidFrame(dj.Part):
        # only load stimulus_type=='stimulus.Frame'
        definition = """
        -> master
        ---
        -> Preprocessing
        """
        content = [
            {"preproc_id": 9},
            {"preproc_id": 0},
        ]

        def input(self, scan_key):
            """Load all frames included in InputResponse.Input and filtered by `valid` in Eye and Treadmill"""
            params = (self * Preprocessing).fetch1()
            if not (params["gamma"] or params["linear_mon"]):
                trial_idx, cond, frame, types = (
                    InputResponse.Input * Frame * stimulus.Condition & scan_key & params
                ).fetch(
                    "trial_idx",
                    "condition_hash",
                    "frame",
                    "stimulus_type",
                    order_by="row_id",  # order by row_id to ensure the order matches Eye and Treadmill
                )
                valid_eye = (Eye & scan_key & params).fetch1("valid")
                valid_treadmill = (Treadmill & scan_key & params).fetch1("valid")
                valid = valid_eye & valid_treadmill
                # keep only valid trials
                trial_idx = trial_idx[valid]
                cond = cond[valid]
                frame = np.stack(frame)[valid]
                types = types[valid]
                # check if all frames have the same shape
                assert (
                    len(frame.shape) == 3
                    and frame.shape[1] == params["row"]
                    and frame.shape[2] == params["col"]
                ), "dimension mismatch, only support 3-D frame (B,H,W)"
                # adjust frame shape to (B,1,W,H)
                frame = frame[:, None, ...]
            else:
                raise NotImplementedError(
                    f'InputConfig: gamma={params["gamma"]}, linear_mon={params["linear_mon"]} not implemented!'
                )
            return trial_idx, cond, frame, types

## Sample from the continuous analogue of a Poisson distribution with mean = Lambda
def poisson_pdf(x, Lambda):
    import math
    return (Lambda**x) * (np.e**(-Lambda)) /np.array([math.gamma(i+1) for i in x])
def poisson_cdf(x, Lambda):
    temp = poisson_pdf(x, Lambda)
    return np.cumsum(temp / temp.sum())
def poisson_sample(Lambda, size=1, a_min=0, a_max=150, n=1000):
    from scipy.interpolate import interp1d
    x = np.linspace(a_min, a_max-1, n)
    cdf = poisson_cdf(x, Lambda)
    f = interp1d(cdf, x, bounds_error=False, fill_value=(1e-6, max(x)))
    return f(np.random.random(size))
def normal_sample(mean, size=1):
    return np.random.normal(mean, np.sqrt(mean), size)
def sampling(resp):
    if resp <= 100:
        return poisson_sample(resp).item()
    else:
        return normal_sample(resp).item()

@schema
class ResponseConfig(ConfigBase, dj.Lookup):
    _config_type = "response"

    def response(self, scan_key):
        self.part_table().response(scan_key)

    class NoNoise(dj.Part):
        definition = """
        -> master
        ---
        """
        content = [
            {},
        ]

        def response(self, scan_key, trial_idx, condition_hashes):
            responses = (DvScanInfo & scan_key).responses(
                        trial_idx=trial_idx,
                        condition_hashes=condition_hashes,
                        )
            return responses

    class ContPoissonNoise(dj.Part):
        definition = """
        -> master
        ---
        """
        content = [
            {},
        ]

        def response(self, scan_key, trial_idx, condition_hashes):
            responses = (DvScanInfo & scan_key).responses(
                        trial_idx=trial_idx,
                        condition_hashes=condition_hashes,
                        )
            import os
            from multiprocessing import Pool
            n = os.cpu_count()
            r_shape = responses.shape
            responses = responses.ravel()
            with Pool(n) as p:
                responses_noisy = list(tqdm(p.imap(sampling, responses)))

            return np.array(responses_noisy).reshape(*r_shape)

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
            assert len(images.shape) == 4, "images dimension must be [B,C,H,W]"
            assert images.shape[1] == 1, "only support single channel images"
            assert (
                images.shape[0] == len(condition_hashes) == len(responses) == len(tiers)
            ), "number of trials mismatch"
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

    class NoBehFullFieldFrame2(dj.Part):
        # implemented only for full field stimulus with stimulus_type=='stimulus.Frame2'
        definition = """
        # mimic the behavior of InputResponse with the corresponding preprocess_id to compute the statistics of input and responses, do not compute stats for behavior variables
        -> master
        ---
        stats_tier="train"                 : enum("train", "test", "validation", "all")               # tier used for computing stats
        stats_per_input                    : tinyint                                                  # whether to compute stats per input
        """
        content = [
            dict(stats_tier="train", stats_per_input=0),
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
            ret["stimulus.Frame2"] = ret["all"]
            return ret

        def run_stats_input(self, data, info_df, ix, per_input):
            if per_input:
                raise NotImplementedError("per_input is not implemented for Frame2")
            ret = {}
            data = data[ix]
            info_df = info_df.loc[ix]
            from statsmodels.stats.weightstats import DescrStatsW as wstats

            sizes = [d.squeeze().shape for d in data]
            assert len(set(sizes)) == 1, "All images must have the same size"
            masks = [
                frame2_make_mask(
                    row["aperture_x"],
                    row["aperture_y"],
                    row["aperture_r"],
                    row["aperture_transition"],
                    sizes[0],
                )
                for _, row in info_df.iterrows()
            ]
            data_flat = np.hstack([d.flatten() for d in data])
            data_mask = np.hstack([m.flatten() for m in masks])

            data_mean = wstats(data_flat, data_mask).mean
            data_std = wstats(data_flat, data_mask).std

            ret["stimulus.Frame2"] = dict(
                mean=data_mean.astype(np.float32),
                std=data_std.astype(np.float32),
                min=data.min().astype(np.float32),
                max=data.max().astype(np.float32),
                median=np.median(data).astype(np.float32),  # median isn't computed with correct masking, downstream computation shouldn't use this value.
            )
            ret["all"] = ret["stimulus.Frame2"]
            return ret

        def stats(self, condition_hashes, images, responses, tiers):
            assert len(images.shape) == 4, "images dimension must be [B,C,H,W]"
            assert images.shape[1] == 1, "only support single channel images"
            key = self.fetch1()
            # check if the method is eligible for condition_hashes requested and collect stimulus info
            assert (
                (
                    stimulus.Condition
                    & "condition_hash in {}".format(tuple(condition_hashes))
                ).fetch("stimulus_type")
                == "stimulus.Frame2"
            ).all(), "StatsConfig.NeuroStaticNoBehFrame2 is only implemented for stimulus.Frame2"
            info_df = pd.DataFrame(dict(condition_hash=condition_hashes))
            info_df = info_df.merge(
                pd.DataFrame((stimulus.Frame2 & info_df).fetch(as_dict=True)),
                how="left",
            )
            assert not info_df.isna().any().any(), "Missing stimulus info"

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
                images, info_df, ix, per_input=key["stats_per_input"]
            )
            statistics = dict(images=input_statistics, responses=response_statistics)
            return statistics


@schema
class DatasetConfig(ConfigBase, dj.Lookup):
    _config_type = "dataset"

    def get_filename(self, key=None, **kwargs):
        key = self.fetch1() if key is None else key
        return self.part_table(key).get_filename(**kwargs)

    def compute_data(self, key=None):
        key = self.fetch1() if key is None else (self & key).fetch1()
        return self.part_table(key).compute_data(key)

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

        def describe(self, key):
            input_type = (InputConfig() & key).fetch1("input_type")
            tier_type = (TierConfig() & key).fetch1("tier_type")
            layer_type = (LayerConfig() & key).fetch1("layer_type")
            area_type = (AreaConfig() & key).fetch1("area_type")
            stats_type = (StatsConfig() & key).fetch1("stats_type")
            desc = f"DvScanInfo|StaticScan|InputConfig.{input_type}|TierConfig.{tier_type}|LayerConfig.{layer_type}|AreaConfig.{area_type}|StatsConfig.{stats_type}"
            return desc

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
            key = self.fetch1() if key is None else (self & key).fetch1()
            static_scan = (
                StaticScan()
                & {
                    **key,
                    "animal_id": key["animal_id"],
                    "session": key["static_session"],
                    "scan_idx": key["static_scan_idx"],
                }
            ).fetch1("KEY")
            dynamic_scan = (
                DvScanInfo()
                & {
                    **key,
                    "animal_id": key["animal_id"],
                    "session": key["dynamic_session"],
                    "scan_idx": key["dynamic_scan_idx"],
                }
            ).fetch1("KEY")
            log.info("Fecthing images")
            trial_idx, condition_hashes, images, types = (
                InputConfig().part_table(key).input(static_scan)
            )
            log.info("Fetching responses")
            responses = (DvScanInfo & dynamic_scan).responses(
                trial_idx=trial_idx,
                condition_hashes=condition_hashes,
            )
            dynamic_unit_keys = (DvScanInfo & dynamic_scan).unit_keys()
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

    @h5cached(
    "/dj-stor01/cache/dynamic-static",
    mode="array",
    transfer_to_tmp=False,
    file_format="dynamic-static-{animal_id}-{dynamic_session}-{dynamic_scan_idx}-{static_session}-{static_scan_idx}-{dataset_hash}.h5",
    )
    class DvStaticNoBehAugResp(dj.Part):
        definition = """ # DvStaticNoBeh with augmented responses (e.g. noisy responses)
        -> master
        ---
        -> DvScanInfo.proj(dynamic_session='session', dynamic_scan_idx='scan_idx')
        -> StaticScan.proj(static_session='session', static_scan_idx='scan_idx')
        -> InputConfig
        -> ResponseConfig
        -> TierConfig
        -> LayerConfig
        -> AreaConfig
        -> StatsConfig
        """

        data_names = ["images", "responses"]

        def describe(self, key):
            input_type = (InputConfig() & key).fetch1("input_type")
            response_type = (ResponseConfig() & key).fetch1("response_type")
            tier_type = (TierConfig() & key).fetch1("tier_type")
            layer_type = (LayerConfig() & key).fetch1("layer_type")
            area_type = (AreaConfig() & key).fetch1("area_type")
            stats_type = (StatsConfig() & key).fetch1("stats_type")
            desc = f"DvScanInfo|StaticScan|InputConfig.{input_type}|ResponseConfig.{response_type}|TierConfig.{tier_type}|LayerConfig.{layer_type}|AreaConfig.{area_type}|StatsConfig.{stats_type}"
            return desc

        @property
        def content(self):
            from . import requests

            return requests.DynamicStaticNoBehAugRespRequest

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
            key = self.fetch1() if key is None else (self & key).fetch1()
            static_scan = (
                StaticScan()
                & {
                    **key,
                    "animal_id": key["animal_id"],
                    "session": key["static_session"],
                    "scan_idx": key["static_scan_idx"],
                }
            ).fetch1("KEY")
            dynamic_scan = (
                DvScanInfo()
                & {
                    **key,
                    "animal_id": key["animal_id"],
                    "session": key["dynamic_session"],
                    "scan_idx": key["dynamic_scan_idx"],
                }
            ).fetch1("KEY")
            log.info("Fecthing images")
            trial_idx, condition_hashes, images, types = (
                InputConfig().part_table(key).input(static_scan)
            )
            log.info("Fetching responses")
            responses = ResponseConfig().part_table(key).response(dynamic_scan, trial_idx, condition_hashes)

            # (DvScanInfo & dynamic_scan).responses(
            #     trial_idx=trial_idx,
            #     condition_hashes=condition_hashes,
            # )
            dynamic_unit_keys = (DvScanInfo & dynamic_scan).unit_keys()
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
        print(f"Inserting {key} to InputResponse")
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
        # check if dataset is already in MultiDataset
        existing_dataset = MultiDataset().aggr(
            MultiDataset.Member & member_key,
            n_dataset="count(*)"
        ) & f'n_dataset={len(DatasetConfig & member_key)}'
        if existing_dataset:
            print(f'Dataset already exists in MultiDataset: {existing_dataset.fetch1("KEY")}')
            return
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
