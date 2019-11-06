import numpy as np
import datajoint as dj


from neuro_data.static_images.data_schemas import StaticScanCandidate, StaticScan, ConditionTier, Frame, InputResponse, Eye, Treadmill, StaticMultiDataset, StaticMultiDatasetGroupAssignment


class NeuroDataPipelineManagement():
    def __init__(self):
        self.pipeline_anatomy = dj.create_virtual_module('pipeline_anatomy', 'pipeline_anatomy')
        self.pipeline_fuse = dj.create_virtual_module('pipeline_fuse', 'pipeline_fuse')
        self.pipeline_stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')

    def manually_insert_area_for_scan(self, target_scan, area):
        neuron_unit_keys = (self.pipeline_fuse.ScanSet().Unit() & target_scan).fetch('KEY')
        for neuron_unit_key in neuron_unit_keys:
            neuron_unit_key['brain_area'] = area
            self.pipeline_anatomy.AreaMembership().insert1(neuron_unit_key, allow_direct_insert=True)

    def manually_insert_layer_for_scan(self, target_scan, layer):
        neuron_unit_keys = (self.pipeline_fuse.ScanSet().Unit() & target_scan).fetch('KEY')
        for neuron_unit_key in neuron_unit_keys:
            neuron_unit_key['layer'] = layer
            self.pipeline_anatomy.LayerMembership().insert1(neuron_unit_key, allow_direct_insert=True)

    # populate functions
    def process_static_scan(self, target_scan):
        print('Running preprocessing checks for ', target_scan)

        # Check if the scan has been processed completely
        if (self.pipeline_fuse.ScanDone() & target_scan).fetch().size == 0:
            print('[Preprocessing Check]: ' + str(target_scan) + ' Scan has not been processed yet, please look into pipeline for details')
            return
        else:
            print('[Preprocessing Check]: ScanDone Check Passed')

        # Check if neurons area are labeled
        if (self.pipeline_anatomy.AreaMembership() & target_scan).fetch().size == 0:
            print('[Preprocessing Check]: ' + str(target_scan) + " AreaMembership is not populated")
            user_input = None
            while user_input not in ['y', 'n']:
                user_input = input('Should we manually insert if the area are known and all the same? [y/n/exit]')
                
                if user_input == 'y':
                    area = input('Input area to label neurons with:')
                    while area not in ['V1', 'LM', 'AL', 'RL', 'all-unknown']:
                        if area == 'exit':
                            return
                        print('Invalid Area!')
                        area = input('Input area to label neurons with:')
                        
                    self.manually_insert_area_for_scan(target_scan, area)
                elif user_input == 'n':
                    return
        else:
            print('[Preprocessing Check]: AreaMembership Check Passed')

        # Check if neuron layers are labeled
        if (self.pipeline_anatomy.LayerMembership() & target_scan).fetch().size == 0:
            print('[Preprocessing Check]: ' + str(target_scan) + " LayerMembership is not populated")

            user_input = None
            while user_input not in ['y', 'n']:
                user_input = input('Should we manually insert if the layer are known and all the same? [y/n/exit]')
                
                if user_input == 'y':
                    layer = input('Input layer to label neurons with:')
                    while layer not in ['L1', 'L2/3', 'L4']:
                        if layer == 'exit':
                            return
                        print('Invalid Layer!')
                        layer = input('Input layer to label neurons with:')
                        
                    
                    self.manually_insert_layer_for_scan(target_scan, layer)
                elif user_input == 'n':
                    return
        else:
            print('[Preprocessing Check]: LayerMembership Check Passed')

        # Check pipeline_stimulus.Sync() table
        if (self.pipeline_stimulus.Sync() & target_scan).fetch().size == 0:
            print('[Preprocessing Check]: ' + str(target_scan) + ' pipeline_stimulus.Sync() table is not processed or failed to processed')
            return
        else:
            print('[Preprocessing Check]: ScanDone Check Passed')

        # All tables requirements are met, begin neurodata dataset population
        print('[Preprocessing Check]: All table requirements passed, beginning neuro_data populating:')
        
        # Get the ScanDone primary key reference
        target_scan_done_key = (self.pipeline_fuse.ScanDone() & target_scan).fetch1('KEY')

        # Insert into StaticScanCandidate
        if (StaticScanCandidate & target_scan_done_key).fetch().size == 0:
            StaticScanCandidate.insert1(target_scan_done_key)
            print('[NeuroData.Static Populate]: Successfully inserted Scan into StaticScanCandidate')
        else:
            print('[NeuroData.Static Populate]: Scan has already been added to StaticScanCandidate')

        # Populating StaticScans
        print("[NeuroData.Static Populate]: Populating StaticScan:")
        StaticScan().populate(target_scan_done_key)

        # Populate ConditionTier
        print("[NeuroData.Static Populate]: Populating ConditionTier:")
        ConditionTier.populate(target_scan_done_key)

        # Populate Frame
        print("[NeuroData.Static Populate]: Populating Frame:")
        Frame.populate(dict(preproc_id = 0))

        # Populate InputResponse
        print("[NeuroData.Static Populate]: Populating InputResponse:")
        InputResponse().populate(target_scan_done_key, dict(preproc_id = 0))

        # Populate Eye
        print("[NeuroData.Static Populate]: Populating Eye:")
        Eye().populate(target_scan_done_key)

        # Populate Treadmill
        print("[NeuroData.Static Populate]: Populating Treadmill:")
        Treadmill().populate(target_scan_done_key)

        # Insert Scan into StaticMultiDatasetGroupAssignment with whatever is the next highest_group_id
        print("[NeuroData.Static Populate]: Inserting Scan into StaticMultiDatasetGroupAssignment with next largest group_id:")
        target_input_response_key = (InputResponse & target_scan & dict(preproc_id=0)).fetch1('KEY')
        if len(StaticMultiDatasetGroupAssignment & target_input_response_key) == 0:
            target_input_response_key['group_id'] = StaticMultiDatasetGroupAssignment().fetch('group_id').max() + 1
            target_input_response_key['description'] = 'Inserted from PipelineManagement'
            StaticMultiDatasetGroupAssignment.insert1(target_input_response_key)
        else:
            print("[NeuroData.Static Populate]:Scan is already in StaticMultiDatasetGroupAssignment, skipping")

        # Fill StaticMultiDataset
        print("[NeuroData.Static Populate]: Filling StaticMultiDataset:")
        StaticMultiDataset().fill()

        print('[PROCESSING COMPLETED FOR SCAN: ' + str(target_scan) + ']\n')