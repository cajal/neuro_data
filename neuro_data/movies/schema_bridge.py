import datajoint as dj


experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
vis = dj.create_virtual_module('vis', 'pipeline_vis')
stack = dj.create_virtual_module('stack', 'pipeline_stack')
tune = dj.create_virtual_module('tune', 'pipeline_tune')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
shared = dj.create_virtual_module('shared', 'pipeline_shared')
platinum = dj.create_virtual_module('platinum', 'pipeline_platinum')
tune = dj.create_virtual_module('tune', 'pipeline_tune')
