import datajoint as dj
from neuro_data.utils.config import ConfigBase
import pandas as pd
import numpy as np

experiment = dj.create_virtual_module("experiment", "pipeline_experiment")
fuse = dj.create_virtual_module("fuse", "pipeline_fuse")
meso = dj.create_virtual_module("meso", "pipeline_meso")
shared = dj.create_virtual_module("shared", "pipeline_shared")
anatomy = dj.create_virtual_module("anatomy", "pipeline_anatomy")
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
dv_nn6_architecture = dj.create_virtual_module(
    "dv_nn6_architecture", "dv_nns_v6_architecture"
)
dv_nn6_train = dj.create_virtual_module("dv_nn6_train", "dv_nns_v6_train")
dv_nn6_model = dj.create_virtual_module("dv_nn6_model", "dv_nns_v6_model")
dv_nn6_scan = dj.create_virtual_module("dv_nn6_scan", "dv_nns_v6_scan")
dv_nn6_pipe = dj.create_virtual_module("dv_nn6_pipe", "dv_nns_v6_ds_pipe")

schema = dj.schema("neurodata_static")


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
            units = (self * dv_nn6_scan.Scan.Unit & dynamic_scan).proj()
            unit_keys = units.fetch(
                *fuse.ScanSet.Unit.primary_key, as_dict=True, order_by="nn_response_index"
            )

            assert len(unit_keys) == n_units

            return unit_keys

        def responses(
            self, dynamic_scan, trial_idx, condition_hashes
        ):
            assert len(trial_idx) == len(condition_hashes)
            cond_df = pd.DataFrame({'condition_hash':condition_hashes})
            resp_key_df = pd.DataFrame((stimulus.Frame & cond_df).fetch('image_class', 'image_id', 'condition_hash', as_dict=True))
            response = dv_nn6_models() * dv_nn6_pipe.Response & dynamic_scan & self & resp_key_df
            resp_df = pd.DataFrame((
                response & resp_key_df
            ).fetch("image_class", "image_id", "response", as_dict=True))
            resp_df = cond_df.merge(resp_key_df.merge(resp_df, how='left'), how='left', validate='m:1')
            assert len(cond_df) == len(resp_df)
            return np.stack(resp_df.response.values)  # (n_images, n_units)
            