import datajoint as dj
from neuro_data.utils.config import ConfigBase
import pandas as pd
import numpy as np
from neuro_data.static_images.data_schemas import StaticScan, schema, stimulus, fuse, Preprocessing
from neuro_data.static_images import data_schemas as data

# # Archived
# dv_nn6_architecture = dj.create_virtual_module(
#     "dv_nn6_architecture", "dv_nns_v6_architecture"
# )
# dv_nn6_train = dj.create_virtual_module("dv_nn6_train", "dv_nns_v6_train")
# dv_nn6_model = dj.create_virtual_module("dv_nn6_model", "dv_nns_v6_model")
# dv_nn6_scan = dj.create_virtual_module("dv_nn6_scan", "dv_nns_v6_scan")
# dv_nn6_pipe = dj.create_virtual_module("dv_nn6_pipe", "dv_nns_v6_ds_pipe")
# dv_scan1_scan_dataset = dj.create_virtual_module(
#     "dv_scan1_scan_dataset", "dv_scans_v1_scan_dataset"
# )
# dv_scan1_scan = dj.create_virtual_module("dv_scan1_scan", "dv_scans_v1_scan")

# def dv_nn6_models():
#     keys = (
#         dv_nn6_model.ModelConfig.Scan1 * dv_nn6_scan.ScanModel * dv_nn6_scan.ModelScans
#         & "n_scans=1"
#     )
#     keys = dj.U("architecture_hash", "train_hash") * keys
#     return keys.proj(dynamic_session="session", dynamic_scan_idx="scan_idx")


# dv_nn9_architecture = dj.create_virtual_module(
#     "dv_nn9_architecture", "dv_nns_v9_architecture"
# )
# dv_nn9_train = dj.create_virtual_module("dv_nn9_train", "dv_nns_v9_train")
# dv_nn9_model = dj.create_virtual_module("dv_nn9_model", "dv_nns_v9_model")
# dv_nn9_scan = dj.create_virtual_module("dv_nn9_scan", "dv_nns_v9_scan")
# dv_nn9_resp = dj.create_virtual_module("dv_nn9_resp", "dv_nns_v9_response")
# dv_scan3_scan_dataset = dj.create_virtual_module(
#     "dv_scan3_scan_dataset", "dv_scans_v3_scan_dataset"
# )
# dv_scan3_scan = dj.create_virtual_module("dv_scan3_scan", "dv_scans_v3_scan")
# dv_stim2_stimulus = dj.create_virtual_module('dv_stim2_stimulus', 'dv_stimuli_v2_stimulus')

# dv_nn10_architecture = dj.create_virtual_module(
#     "dv_nn10_architecture", "dv_nns_v10_architecture"
# )
# dv_nn10_train = dj.create_virtual_module("dv_nn10_train", "dv_nns_v10_train")
# dv_nn10_model = dj.create_virtual_module("dv_nn10_model", "dv_nns_v10_model")
# dv_nn10_scan = dj.create_virtual_module("dv_nn10_scan", "dv_nns_v10_scan")
# dv_nn10_resp = dj.create_virtual_module("dv_nn10_resp", "dv_nns_v10_response")
# dv_scan3_scan_dataset = dj.create_virtual_module(
#     "dv_scan3_scan_dataset", "dv_scans_v3_scan_dataset"
# )
# dv_scan3_scan = dj.create_virtual_module("dv_scan3_scan", "dv_scans_v3_scan")
# dv_stim2_stimulus = dj.create_virtual_module(
#     "dv_stim2_stimulus", "dv_stimuli_v2_stimulus"
# )

foundation_stimulus = dj.create_virtual_module("stimulus", "foundation_stimulus")
nfs = dj.create_virtual_module('nfs', 'neurodata_foundation_static')
fnn = dj.create_virtual_module('fnn', 'foundation_fnn')

# class Nn10Mixin:
#     @property
#     def content(self):
#         return (
#             dj.U(*self.heading.secondary_attributes)
#             & (dv_nn10_scan.ScanModelInstance * dv_nn10_resp.ScanImageResponse)
#             & (  # preblank = 5 msec
#                 dv_nn10_resp.ImageResponseConfig().Lowpass() & "pre_duration=5"
#             )
#             & (  # all unit & all dynamic clips
#                 dv_nn10_scan.ScanConfig().Scan3()
#                 & dv_scan3_scan_dataset.UnitConfig().All()
#                 & (
#                     dv_scan3_scan_dataset.TrialConfig().TrainSet()
#                     & dv_stim2_stimulus.StimulusSetConfig()
#                     .AllDynamic()
#                     .proj(include_stimulus_set_hash="stimulus_set_hash")
#                     & dv_stim2_stimulus.StimulusSetConfig().OracleClip.proj(
#                         exclude_stimulus_set_hash="stimulus_set_hash"
#                     )
#                 )
#             )
#         )

#     def unit_keys(self, dynamic_scan):
#         dynamic_scan, n_units = (dv_nn10_scan.Scan & dynamic_scan & self).fetch1(
#             dj.key, "n_units"
#         )
#         units = (
#             self
#             * dv_nn10_scan.Scan.Unit
#             * dv_nn10_scan.ScanConfig().Scan3()
#             * (
#                 dv_scan3_scan_dataset.Preprocess
#                 * dv_scan3_scan.ResponseId.proj(..., scan_ms_delay="ms_delay")
#             )  # recover spike_method
#             & dynamic_scan
#         )
#         unit_keys = units.fetch(
#             *(fuse.ScanDone * fuse.ScanSet.Unit).primary_key,
#             as_dict=True,
#             order_by="nn_response_index"
#         )

#         assert len(unit_keys) == n_units

#         return unit_keys

#     def unique_unit_mapping(self, dynamic_scan):
#         if (
#             self
#             * dv_nn10_scan.ScanConfig.Scan3()
#             * dv_scan3_scan_dataset.UnitConfig.Unique()
#         ):

#             key = (
#                 (
#                     self
#                     * dv_nn10_scan.ScanConfig.Scan3()
#                     * dv_scan3_scan_dataset.UnitConfig.Unique()
#                 )
#                 & dynamic_scan
#             ).fetch1()  # get unique_id
#             unique_unit_key = (dv_scan3_scan.Unique() & dynamic_scan & key).fetch1(
#                 "KEY"
#             )
#             unique_unit_rel = (
#                 dv_scan3_scan.Unique.Unit
#                 * dv_scan3_scan.Unique.Neuron.proj(unique_unit_id="unit_id")
#                 & unique_unit_key
#             )
#             return (
#                 dj.U("animal_id", "session", "scan_idx", "unit_id", "unique_unit_id")
#                 & unique_unit_rel
#             )
#         elif (
#             self
#             * dv_nn10_scan.ScanConfig.Scan3()
#             * dv_scan3_scan_dataset.UnitConfig.All()
#         ):  # return a mapping from all units to themselves
#             units = self * dv_nn10_scan.Scan.Unit * dv_nn10_scan.ScanConfig().Scan3()
#             return units.proj(unique_unit_id="unit_id * 1")
#         else:
#             raise NotImplementedError(
#                 "`unique_unit_mapping` is not implemented for key {}!".format(
#                     self.fetch1()
#                 )
#             )


@schema
class DvModelConfig(ConfigBase, dj.Lookup):

    _config_type = "dv_model"

    # # Archived
    # class Nn6(dj.Part):
    #     definition = """
    #     -> master
    #     ---
    #     -> dv_nn6_scan.ScanConfig
    #     -> dv_nn6_architecture.ArchitectureConfig
    #     -> dv_nn6_train.TrainConfig
    #     -> dv_nn6_model.Instance
    #     -> dv_nn6_model.OutputConfig
    #     -> dv_nn6_pipe.ResponseConfig
    #     """

    #     @property
    #     def content(self):
    #         return dj.U(*self.heading.secondary_attributes) & (
    #             dv_nn6_models() * dv_nn6_pipe.Response
    #         )

    #     def unit_keys(self, dynamic_scan):
    #         dynamic_scan, n_units = (dv_nn6_scan.Scan & dynamic_scan & self).fetch1(
    #             dj.key, "n_units"
    #         )
    #         units = (
    #             self
    #             * dv_nn6_scan.Scan.Unit
    #             * dv_nn6_scan.ScanConfig().Scan1()
    #             * dv_scan1_scan_dataset.Preprocess
    #             * dv_scan1_scan.ResponseId
    #             & dynamic_scan
    #         )
    #         unit_keys = units.fetch(
    #             *(fuse.ScanDone * fuse.ScanSet.Unit).primary_key,
    #             as_dict=True,
    #             order_by="nn_response_index"
    #         )

    #         assert len(unit_keys) == n_units

    #         return unit_keys

    #     def responses(self, dynamic_scan, trial_idx, condition_hashes):
    #         assert len(trial_idx) == len(condition_hashes)
    #         cond_df = pd.DataFrame({"condition_hash": condition_hashes})
    #         resp_key_df = pd.DataFrame(
    #             (stimulus.Frame & cond_df).fetch(
    #                 "image_class", "image_id", "condition_hash", as_dict=True
    #             )
    #         )
    #         response = (
    #             dv_nn6_models() * dv_nn6_pipe.Response
    #             & dynamic_scan
    #             & self
    #             & resp_key_df
    #         )
    #         resp_df = pd.DataFrame(
    #             (response & resp_key_df).fetch(
    #                 "image_class", "image_id", "response", as_dict=True
    #             )
    #         )
    #         resp_df = cond_df.merge(
    #             resp_key_df.merge(resp_df, how="left"), how="left", validate="m:1"
    #         )
    #         assert len(cond_df) == len(resp_df)
    #         return np.stack(resp_df.response.values)  # (n_images, n_units)

    #     def unique_unit_mapping(self, dynamic_scan):
    #         if not (
    #             self
    #             * dv_nn6_scan.ScanConfig.Scan1()
    #             * dv_scan1_scan_dataset.UnitConfig.Unique()
    #         ):
    #             raise NotImplementedError(
    #                 "`unique_unit_mapping` is not implemented for key {}!".format(
    #                     self.fetch1()
    #                 )
    #             )
    #         key = (
    #             (
    #                 self
    #                 * dv_nn6_scan.ScanConfig.Scan1()
    #                 * dv_scan1_scan_dataset.UnitConfig.Unique()
    #             )
    #             & dynamic_scan
    #         ).fetch1()  # get unique_id
    #         unique_unit_key = (dv_scan1_scan.Unique() & dynamic_scan & key).fetch1(
    #             "KEY"
    #         )
    #         unique_unit_rel = (
    #             dv_scan1_scan.Unique.Unit
    #             * dv_scan1_scan.Unique.Neuron.proj(unique_unit_id="unit_id")
    #             & unique_unit_key
    #         )
    #         return (
    #             dj.U("animal_id", "session", "scan_idx", "unit_id", "unique_unit_id")
    #             & unique_unit_rel
    #         )

    # class Nn9(dj.Part):
    #     definition = """
    #     -> master
    #     ---
    #     -> dv_nn9_resp.ResponseDelay
    #     -> dv_nn9_resp.ImageConfig
    #     -> dv_nn9_resp.ImageResponseConfig
    #     -> dv_nn9_model.InstanceConfig
    #     -> dv_nn9_model.OutputConfig
    #     -> dv_nn9_scan.ScanConfig
    #     -> dv_nn9_scan.NnConfig
    #     """

    #     @property
    #     def content(self):
    #         return dj.U(*self.heading.secondary_attributes) & (
    #             dv_nn9_scan.ScanModelInstance * dv_nn9_resp.ScanImageResponse
    #         ) & (  # preblank = 5 msec
    #             dv_nn9_resp.ImageResponseConfig().Lowpass() & 'pre_duration=5'
    #         ) & (  # all unit & all dynamic clips
    #             dv_nn9_scan.ScanConfig().Scan3()
    #             & dv_scan3_scan_dataset.UnitConfig().All()
    #             & (
    #                 dv_scan3_scan_dataset.TrialConfig().TrainSet()
    #                 & dv_stim2_stimulus.StimulusSetConfig().AllDynamic().proj(include_stimulus_set_hash='stimulus_set_hash')
    #                 & dv_stim2_stimulus.StimulusSetConfig().OracleClip.proj(exclude_stimulus_set_hash='stimulus_set_hash')
    #             )
    #         )

    #     def unit_keys(self, dynamic_scan):
    #         dynamic_scan, n_units = (dv_nn9_scan.Scan & dynamic_scan & self).fetch1(
    #             dj.key, "n_units")
    #         units = (
    #             self
    #             * dv_nn9_scan.Scan.Unit
    #             * dv_nn9_scan.ScanConfig().Scan3()
    #             * (
    #                 dv_scan3_scan_dataset.Preprocess
    #                 * dv_scan3_scan.ResponseId.proj(..., scan_ms_delay='ms_delay')
    #             )  # recover spike_method
    #             & dynamic_scan
    #         )
    #         unit_keys = units.fetch(
    #             *(fuse.ScanDone * fuse.ScanSet.Unit).primary_key,
    #             as_dict=True,
    #             order_by="nn_response_index"
    #         )

    #         assert len(unit_keys) == n_units

    #         return unit_keys

    #     def responses(self, dynamic_scan, trial_idx, condition_hashes):
    #         assert len(trial_idx) == len(condition_hashes)
    #         cond_df = pd.DataFrame({"condition_hash": condition_hashes})
    #         resp_key_df = pd.DataFrame(
    #             (stimulus.Frame & cond_df).fetch(
    #                 "image_class", "image_id", "condition_hash", as_dict=True
    #             )
    #         )
    #         response = (
    #             dv_nn9_scan.ScanModelInstance * dv_nn9_resp.ScanImageResponse
    #             & dynamic_scan
    #             & self
    #             & resp_key_df
    #         )
    #         resp_df = pd.DataFrame(
    #             (response & resp_key_df).fetch(
    #                 "image_class", "image_id", "response", as_dict=True
    #             )
    #         )
    #         resp_df = cond_df.merge(
    #             resp_key_df.merge(resp_df, how="left"), how="left", validate="m:1"
    #         )
    #         assert len(cond_df) == len(resp_df)
    #         return np.stack(resp_df.response.values)  # (n_images, n_units)

    #     def unique_unit_mapping(self, dynamic_scan):
    #         if (
    #                 self
    #                 * dv_nn9_scan.ScanConfig.Scan3()
    #                 * dv_scan3_scan_dataset.UnitConfig.Unique()
    #             ):

    #             key = (
    #                 (
    #                     self
    #                     * dv_nn9_scan.ScanConfig.Scan3()
    #                     * dv_scan3_scan_dataset.UnitConfig.Unique()
    #                 )
    #                 & dynamic_scan
    #             ).fetch1()  # get unique_id
    #             unique_unit_key = (dv_scan3_scan.Unique() & dynamic_scan & key).fetch1(
    #                 "KEY"
    #             )
    #             unique_unit_rel = (
    #                 dv_scan3_scan.Unique.Unit
    #                 * dv_scan3_scan.Unique.Neuron.proj(unique_unit_id="unit_id")
    #                 & unique_unit_key
    #             )
    #             return (
    #                 dj.U("animal_id", "session", "scan_idx", "unit_id", "unique_unit_id")
    #                 & unique_unit_rel
    #             )
    #         elif (
    #                 self
    #                 * dv_nn9_scan.ScanConfig.Scan3()
    #                 * dv_scan3_scan_dataset.UnitConfig.All()
    #             ):  # return a mapping from all units to themselves
    #             units = (
    #                 self
    #                 * dv_nn9_scan.Scan.Unit
    #                 * dv_nn9_scan.ScanConfig().Scan3()
    #             )
    #             return units.proj(unique_unit_id='unit_id * 1')
    #         else:
    #             raise NotImplementedError(
    #                 "`unique_unit_mapping` is not implemented for key {}!".format(
    #                     self.fetch1()
    #                 )
    #             )

    # class Nn10Frame2(Nn10Mixin, dj.Part):
    #     definition = """  # process static scans with Frame2 stimuli
    #     -> master
    #     ---
    #     -> dv_nn10_resp.ResponseDelay
    #     -> dv_nn10_resp.ImageResponseConfig
    #     -> dv_nn10_model.InstanceConfig
    #     -> dv_nn10_model.OutputConfig
    #     -> dv_nn10_scan.ScanConfig
    #     -> dv_nn10_scan.NnConfig
    #     """

    #     @property
    #     def content(self):
    #         return super().content & dv_nn10_resp.ImageConfig.Aperture

    #     def responses(self, dynamic_scan, trial_idx, condition_hashes):
    #         assert len(trial_idx) == len(condition_hashes)
    #         cond_df = pd.DataFrame({"condition_hash": condition_hashes})
    #         resp_key_df = pd.DataFrame(
    #             (stimulus.Frame2 & cond_df).fetch(
    #                 "image_class",
    #                 "image_id",
    #                 "condition_hash",
    #                 "aperture_x",
    #                 "aperture_y",
    #                 "aperture_r",
    #                 "aperture_transition",
    #                 "background_value",
    #                 as_dict=True,
    #             )
    #         )
    #         response = (
    #             dv_nn10_scan.ScanModelInstance
    #             * dv_nn10_resp.ScanImageResponse
    #             * dv_nn10_resp.ImageConfig.Aperture()
    #             & dynamic_scan
    #             & self
    #             & resp_key_df
    #         )
    #         resp_df = pd.DataFrame(
    #             (response & resp_key_df).fetch(
    #                 "image_class",
    #                 "image_id",
    #                 "aperture_x",
    #                 "aperture_y",
    #                 "aperture_r",
    #                 "aperture_transition",
    #                 "background_value",
    #                 "response"
    #                 , as_dict=True
    #             )
    #         )
    #         resp_df = cond_df.merge(
    #             resp_key_df.merge(resp_df, how="left"), how="left", validate="m:1"
    #         )
    #         assert len(cond_df) == len(resp_df)
    #         return np.stack(resp_df.response.values)  # (n_images, n_units)

    # class Nn10(Nn10Mixin, dj.Part):
    #     definition = """
    #     -> master
    #     ---
    #     -> dv_nn10_resp.ResponseDelay
    #     -> dv_nn10_resp.ImageConfig
    #     -> dv_nn10_resp.ImageResponseConfig
    #     -> dv_nn10_model.InstanceConfig
    #     -> dv_nn10_model.OutputConfig
    #     -> dv_nn10_scan.ScanConfig
    #     -> dv_nn10_scan.NnConfig
    #     """

    #     @property
    #     def content(self):
    #         return super().content & dv_nn10_resp.ImageConfig.Vanilla

    #     def responses(self, dynamic_scan, trial_idx, condition_hashes):
    #         assert len(trial_idx) == len(condition_hashes)
    #         cond_df = pd.DataFrame({"condition_hash": condition_hashes})
    #         resp_key_df = pd.DataFrame(
    #             (stimulus.Frame & cond_df).fetch(
    #                 "image_class", "image_id", "condition_hash", as_dict=True
    #             )
    #         )
    #         response = (
    #             dv_nn10_scan.ScanModelInstance * dv_nn10_resp.ScanImageResponse
    #             & dynamic_scan
    #             & self
    #             & resp_key_df
    #         )
    #         resp_df = pd.DataFrame(
    #             (response & resp_key_df).fetch(
    #                 "image_class", "image_id", "response", as_dict=True
    #             )
    #         )
    #         resp_df = cond_df.merge(
    #             resp_key_df.merge(resp_df, how="left"), how="left", validate="m:1"
    #         )
    #         assert len(cond_df) == len(resp_df)
    #         return np.stack(resp_df.response.values)  # (n_images, n_units)
        
    class Foundation(dj.Part):
        definition = """
        -> master
        ---
        -> fnn.Model
        -> foundation_stimulus.FrameList
        -> Preprocessing
        """
        @property
        def content(self):
            return nfs.FoundationInputResponse
        
        def responses(self, key, trial_idx, condition_hashes):
            assert len(trial_idx) == len(condition_hashes)
            cond_df = pd.DataFrame({"condition_hash": condition_hashes})
            cond_hashes, rows = (self * nfs.FoundationInputResponse.Input & key & cond_df).fetch('condition_hash', 'row_id')
            dic = dict(zip(cond_hashes, rows))
            order = np.array([dic[cond] for cond in condition_hashes])
            responses = (self * nfs.FoundationInputResponse.ResponseBlock & key).fetch1('responses')
            return responses[order, :]
        
        def unit_keys(self, key):
            return (self * nfs.FoundationInputResponse.ResponseKeys & key).fetch(as_dict=True, order_by='col_id')


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
        from neuro_data.static_images import requests
        return super().key_source & requests.DvScanInfoRequest

    def make(self, key):
        unit_keys = DvModelConfig().part_table(key).unit_keys(key)
        unit_keys = [
            dict(unit_key, dv_model_hash=key["dv_model_hash"]) for unit_key in unit_keys
        ]
        self.insert1(dict(key, n_units=len(unit_keys)))

        self.Unit().insert(
            [dict(unit_key, response_index=i) for i, unit_key in enumerate(unit_keys)],
            ignore_extra_fields=True
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
        key = key or self.fetch1(dj.key)
        dv_conf = DvModelConfig().part_table(key)
        unit_keys = dv_conf.unit_keys(key)
        return unit_keys

    def unique_unit_mapping(self, key=None):
        """
        Returns a datajoint relation with primary keys: (animal_id, session, scan_idx, unit_id, unique_unit_id)
        The unique_unit_id is the highest ranked unit in the dyanmic scan that was detected as a duplicate to the unit_id.
        """
        key = key or self.fetch1(dj.key)
        return DvModelConfig().part_table(key).unique_unit_mapping(key)
