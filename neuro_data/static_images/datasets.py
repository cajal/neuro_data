from neuro_data.utils.datasets import H5ArraySet, AttributeTransformer, AttributeHandler


class StaticImageSet(H5ArraySet):
    def __init__(self, filename, *data_keys, transforms=None, cache_raw=False, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}
        self.cache_raw = cache_raw
        self.last_raw = None
        self.stats_source = stats_source if stats_source is not None else 'all'

    @property
    def n_neurons(self):
        return self[0].responses.shape[1]

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def info(self):
        return AttributeHandler('item_info', self._fid)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape


    def __repr__(self):

        return super().__repr__() + \
            ('\n\t[Stats source: {}]'.format(self.stats_source) if self.stats_source is not None else '')
