import datajoint as dj

anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

schema = dj.schema('neurodata_configs', locals())


@schema
class BrainAreas(dj.Lookup):
    definition = """ # group of brain areas
    brain_areas                 : varchar(256)
    """

    class BrainArea(dj.Part):
        definition = """
        -> master
        -> anatomy.Area
        """

    def fill(self):

        group_areas_0 = ('all-unknown', ('A', 'AL', 'AM', 'LI', 'LLA', 'LM', 'MAP', 'P', 'PM', 'POR', 'RL', 'V1'))
        group_areas_1 = ('V1+LM', ('V1', 'LM'))
        group_areas_2 = ('V1+LM+LI+AL+RL', ('V1', 'LM', 'LI', 'AL', 'RL'))
        group_areas_3 = ('all', ('A', 'AL', 'AM', 'LI', 'LLA', 'LM', 'MAP', 'P', 'PM', 'POR', 'RL', 'V1', 'unknown'))

        for group_areas in (group_areas_0, group_areas_1, group_areas_2, group_areas_3):
            key = dict(brain_areas=group_areas[0])
            self.insert1(key, skip_duplicates=True)
            for area in group_areas[1]:
                part_key = dict(key, brain_area=area)
                self.BrainArea.insert1(part_key, skip_duplicates=True)
