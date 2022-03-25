import datajoint as dj
from neuro_data.utils.config import ConfigBase
import pandas as pd
import numpy as np
from neuro_data.static_images.data_schemas import StaticScan

experiment = dj.create_virtual_module("experiment", "pipeline_experiment")
fuse = dj.create_virtual_module("fuse", "pipeline_fuse")
meso = dj.create_virtual_module("meso", "pipeline_meso")
shared = dj.create_virtual_module("shared", "pipeline_shared")
anatomy = dj.create_virtual_module("anatomy", "pipeline_anatomy")
stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
dv_nn6_architecture = dj.create_virtual_module(
    "dv_nn6_architecture", "dv_nns_v6_architecture"
)
dv_nn6_train = dj.create_virtual_module("dv_nn6_train", "dv_nns_v6_train")
dv_nn6_model = dj.create_virtual_module("dv_nn6_model", "dv_nns_v6_model")
dv_nn6_scan = dj.create_virtual_module("dv_nn6_scan", "dv_nns_v6_scan")
dv_nn6_pipe = dj.create_virtual_module("dv_nn6_pipe", "dv_nns_v6_ds_pipe")
dv_scan1_scan_dataset = dj.create_virtual_module(
    "dv_scan1_scan_dataset", "dv_scans_v1_scan_dataset"
)
dv_scan1_scan = dj.create_virtual_module("dv_scan1_scan", "dv_scans_v1_scan")

schema = dj.schema("neurodata_static")


def dv_nn6_models():
    keys = (
        dv_nn6_model.ModelConfig.Scan1 * dv_nn6_scan.ScanModel * dv_nn6_scan.ModelScans
        & "n_scans=1"
    )
    keys = dj.U("architecture_hash", "train_hash") * keys
    return keys.proj(dynamic_session="session", dynamic_scan_idx="scan_idx")


@schema
class DvModelConfig(ConfigBase, dj.Lookup):

    _config_type = "dv_model"

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
            return dj.U(*self.heading.secondary_attributes) & (
                dv_nn6_models() * dv_nn6_pipe.Response
            )

        def unit_keys(self, dynamic_scan):
            dynamic_scan, n_units = (dv_nn6_scan.Scan & dynamic_scan & self).fetch1(
                dj.key, "n_units"
            )
            units = (
                self
                * dv_nn6_scan.Scan.Unit
                * dv_nn6_scan.ScanConfig().Scan1()
                * dv_scan1_scan_dataset.Preprocess
                * dv_scan1_scan.ResponseId
                & dynamic_scan
            )
            unit_keys = units.fetch(
                *(fuse.ScanDone * fuse.ScanSet.Unit).primary_key,
                as_dict=True,
                order_by="nn_response_index"
            )

            assert len(unit_keys) == n_units

            return unit_keys

        def responses(self, dynamic_scan, trial_idx, condition_hashes):
            assert len(trial_idx) == len(condition_hashes)
            cond_df = pd.DataFrame({"condition_hash": condition_hashes})
            resp_key_df = pd.DataFrame(
                (stimulus.Frame & cond_df).fetch(
                    "image_class", "image_id", "condition_hash", as_dict=True
                )
            )
            response = (
                dv_nn6_models() * dv_nn6_pipe.Response
                & dynamic_scan
                & self
                & resp_key_df
            )
            resp_df = pd.DataFrame(
                (response & resp_key_df).fetch(
                    "image_class", "image_id", "response", as_dict=True
                )
            )
            resp_df = cond_df.merge(
                resp_key_df.merge(resp_df, how="left"), how="left", validate="m:1"
            )
            assert len(cond_df) == len(resp_df)
            return np.stack(resp_df.response.values)  # (n_images, n_units)


@schema
class DvScanInfo(dj.Computed):
    definition = """
    -> StaticScan
    -> DvModelConfig
    ---
    n_units         : int unsigned      # number of units
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> StaticScan.Unit
        ---
        response_index      : int unsigned      # index of unit in response vector
        """

    @property
    def key_source(self):
        keys = fuse.ScanDone * DvModelConfig
        key = [
            dv_nn6_models() * DvModelConfig.Nn6 * dv_nn6_scan.Scan,
        ]
        return keys & key

    def make(self, key):
        unit_keys = DvModelConfig().part_table(key).unit_keys(key)
        unit_keys = [
            dict(unit_key, dv_model_hash=key["dv_model_hash"]) for unit_key in unit_keys
        ]
        self.insert1(dict(key, n_units=len(unit_keys)))

        self.Unit.insert(
            [dict(unit_key, response_index=i) for i, unit_key in enumerate(unit_keys)]
        )

    def responses(self, trial_idx, condition_hashes, key=None):
        if key is None:
            key, n_units = self.fetch1(dj.key, "n_units")
        else:
            key, n_units = (self & key).fetch1(dj.key, "n_units")

        dv_conf = DvModelConfig().part_table(key)
        responses = dv_conf.responses(key, trial_idx, condition_hashes)
        assert responses.shape[1] == n_units
        return responses

    def unit_keys(self, key=None):
        key = self.fetch1(dj.key) if key is None else key
        dv_conf = DvModelConfig().part_table(key)
        unit_keys = dv_conf.unit_keys(key)
        return unit_keys
