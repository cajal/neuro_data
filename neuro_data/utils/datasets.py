
from collections import namedtuple

import h5py
from torch.utils.data import Dataset


class Invertible:
    def inv(self, y):
        raise NotImplemented('Subclasses of Invertible must implement an inv method')


class H5ArraySet(Dataset):
    def __init__(self, filename, *data_keys, transforms=None):
        self._fid = h5py.File(filename, 'r')
        m = None
        for key in data_keys:
            assert key in self._fid, 'Could not find {} in file'.format(key)
            if m is None:
                m = len(self._fid[key])
            else:
                assert m == len(self._fid[key]), 'Length of datasets do not match'
        self._len = m
        self.data_keys = data_keys

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_keys)

    def __getitem__(self, item):
        x = self.data_point(*(self._fid[g][item] for g in self.data_keys))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '\n'.join(['Tensor {}: {} '.format(key, self._fid[key].shape)
                          for key in self.data_keys] + ['Transforms: ' + repr(self.transforms)])

    def transform(self, x, exclude=None):
        for tr in self.transforms:
            if exclude is None or not isinstance(tr, exclude):
                x = tr(x)
        return x

    def invert(self, x, exclude=None):
        for tr in reversed(filter(lambda tr: not isinstance(tr, exclude), self.transforms)):
            if not isinstance(tr, Invertible):
                raise TypeError('Cannot invert', tr.__class__.__name__)
            else:
                x = tr.inv(x)
        return x

    def __dir__(self):
        attrs = list(self.__dict__)
        attrs.extend(list(self._fid))
        return attrs

    def __getattr__(self, item):
        if item in self._fid:
            item = self._fid[item]
            if isinstance(item, h5py._hl.dataset.Dataset):
                item = item[()]
                if item.dtype.char == 'S':  # convert bytes to univcode
                    item = item.astype(str)
                return item
            return item
        else:
            raise AttributeError('Item {} not found in {}'.format(item, self.__class__.__name__))


class AttributeHandler:
    def __init__(self, name, h5_handle):
        assert name in h5_handle, '{} must be in {}'.format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle

    def __getattr__(self, item):
        ret = self.h5_handle[self.name][item][()]
        if ret.dtype.char == 'S':  # convert bytes to unicode
            ret = ret.astype(str)
        return ret

    def __dir__(self):
        return self.h5_handle[self.name].keys()


class AttributeTransformer(AttributeHandler):
    def __init__(self, name, h5_handle, transforms):
        super().__init__(name, h5_handle)
        self.transforms = transforms

    def __getattr__(self, item):
        ret = super().__getattr__(item)
        for tr in self.transforms:
            ret = tr.column_transform(ret)
        return ret
