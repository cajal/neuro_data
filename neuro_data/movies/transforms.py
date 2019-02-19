from attorch.dataset import H5SequenceSet, Invertible
import numpy as np
import torch
from PIL import Image
import torchvision


class DataTransform:

    def __repr__(self):
        return self.__class__.__name__

    def column_transform(self, label):
        return label


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
        assert isinstance(data, H5SequenceSet), 'data must be a H5SequenceSet'

        exclude = self.exclude = exclude or []

        self._inputs_std = data.statistics['inputs/{}/mean'.format(stats_source)][()]

        s = np.array(data.statistics['responses/{}/std'.format(stats_source)])

        idx = s > 0.01 * s.mean()
        self._response_precision = np.ones_like(s)
        self._response_precision[idx] = 1 / s[idx]

        transforms, itransforms = {}, {}

        # -- inputs
        transforms['inputs'] = lambda x: (x - x.mean()) / self._inputs_std
        itransforms['inputs'] = lambda x: x * self._inputs_std + x.mean()

        # -- responses
        transforms['responses'] = lambda x: x * self._response_precision
        itransforms['responses'] = lambda x: x / self._response_precision

        if 'eye_position' in data.data_groups:
            s = np.array(
                data.statistics['behavior/{}/std'.format(stats_source)])
            idx = s > 0.01 * s.mean()
            self._behavior_precision = np.ones_like(s)
            self._behavior_precision[idx] = 1 / s[idx]

            s = np.array(
                data.statistics['eye_position/{}/std'.format(stats_source)])
            mu = np.array(
                data.statistics['eye_position/{}/mean'.format(stats_source)])
            self._eye_mean = mu
            self._eye_std = s

            # -- eye position
            transforms['eye_position'] = lambda x: (
                x - self._eye_mean) / self._eye_std
            itransforms['eye_position'] = lambda x: x * \
                self._eye_std + self._eye_mean

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

    def __repr__(self):
        return super().__repr__() + ('(not {})'.format(', '.join(self.exclude)) if self.exclude is not None else '')


class Subsequence(DataTransform):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, x):
        t = x.inputs.shape[1]
        i = np.random.randint(0, t - self.frames)
        return x.__class__(
            **{k: getattr(x, k)[:, i:i + self.frames, ...] if k == 'inputs' else getattr(x, k)[i:i + self.frames, ...]
               for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.frames)


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
        return x.__class__(*[torch.from_numpy(elem.astype(np.float32)).cuda()
                             if self.cuda else torch.from_numpy(elem.astype(np.float32)) for elem in x])


class ScaleInput(DataTransform):

    def __call__(self, x):
        return x.__class__(
            **{k: getattr(x, k)/255 if k == 'inputs' else getattr(x, k) for k in x._fields})


class Resize(DataTransform):

    def __init__(self, size):
        self.torchvision_resize = torchvision.transforms.Resize(size)

    def __call__(self, x):
        def resize(inp):
            resized = np.empty(
                list(inp.shape[:2]) + list(self.torchvision_resize.size))
            for frame, img in enumerate(inp[0].astype(np.uint8)):
                pil_img = Image.fromarray(img)
                resized[0, frame] = np.array(self.torchvision_resize(pil_img),
                                             dtype=np.float32)
            return resized
        return x.__class__(
            **{k: resize(getattr(x, k)) if k == 'inputs' else getattr(x, k) for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.torchvision_resize.size)


class Identity(DataTransform, Invertible):
    def __call__(self, x):
        return x

    def inv(self, y):
        return y
