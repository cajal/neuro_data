import numpy as np
import torch

from ..utils.transform import DataTransform
from ..utils.datasets import Invertible, H5ArraySet


class Normalizer(DataTransform, Invertible):
    """
    Normalizes a trial with fields: inputs, behavior, eye_position, and responses. The pair of
    behavior and eye_position can be missing. The following normalizations are applied:

    - inputs are scaled by the training std of the stats_source and centered on the mean of the movie
    - behavior is divided by the std if the std is greater than 1% of the mean std (to avoid division by 0)
    - eye_position is z-scored
    - reponses are divided by the per neuron std if the std is greater than
            1% of the mean std (to avoid division by 0)
    """

    def __init__(self, data, stats_source='all', exclude=None):
        assert isinstance(data, H5ArraySet), 'data must be a H5SequenceSet'

        self.exclude = exclude or []

        self._inputs_std = data.statistics['images/{}/mean'.format(stats_source)].value

        s = np.array(data.statistics['responses/{}/std'.format(stats_source)])

        idx = s > 0.01 * s.mean()
        self._response_precision = np.ones_like(s)
        self._response_precision[idx] = 1 / s[idx]

        transforms, itransforms = {}, {}

        # -- inputs
        transforms['images'] = lambda x: (x - x.mean()) / self._inputs_std
        itransforms['images'] = lambda x: x * self._inputs_std + x.mean()

        # -- responses
        transforms['responses'] = lambda x: x * self._response_precision
        itransforms['responses'] = lambda x: x / self._response_precision

        if 'pupil_center' in data.data_keys:
            s = np.array(data.statistics['behavior/{}/std'.format(stats_source)])
            idx = s > 0.01 * s.mean()
            self._behavior_precision = np.ones_like(s)
            self._behavior_precision[idx] = 1 / s[idx]

            s = np.array(data.statistics['pupil_center/{}/std'.format(stats_source)])
            mu = np.array(data.statistics['pupil_center/{}/mean'.format(stats_source)])
            self._eye_mean = mu
            self._eye_std = s

            # -- eye position
            transforms['pupil_center'] = lambda x: (x - self._eye_mean) / self._eye_std
            itransforms['pupil_center'] = lambda x: x * self._eye_std + self._eye_mean

            # -- behavior
            transforms['behavior'] = lambda x: x * self._behavior_precision
            itransforms['behavior'] = lambda x: x / self._behavior_precision

        self._transforms = transforms
        self._itransforms = itransforms

    def inv(self, x):
        return x.__class__(
            **{k: (self._itransforms[k](v) if not k in self.exclude else v) for k, v in zip(x._fields, x)})

    def __call__(self, x):
        return x.__class__(
            **{k: (self._transforms[k](v) if not k in self.exclude else v) for k, v in zip(x._fields, x)})


class Subsample(DataTransform):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        return x.__class__(
            **{k: getattr(x, k)[..., self.idx] if k == 'responses' else getattr(x, k) for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '(n={})'.format(len(self.idx))

    def column_transform(self, label):
        return label[self.idx]


class ToTensor(DataTransform, Invertible):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def inv(self, y):
        return y.numpy()


    def __call__(self, x):
        return x.__class__(*[torch.FloatTensor(elem).cuda() if self.cuda else torch.FloatTensor(elem) for elem in x])
