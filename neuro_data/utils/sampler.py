from collections import Counter

import torch
from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BalancedSubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement, balanced by occurence of types.

    Arguments:
        indices (list): a list of indices
    """

    def configure_sampler(self, indices, types, mode='shortest'):
        self.indices = indices
        c = Counter(types[indices])
        if mode == 'longest':
            self.num_samples = max(c.values())
            self.replacement = True
        elif mode == 'shortest':
            self.num_samples = min(c.values())
            self.replacement = False

        for e, n in c.items():
            c[e] = 1 / n
        self.types = types
        self.weights = torch.DoubleTensor([c[types[i]] for i in indices])

    def __init__(self, indices, types, mode='shortest'):
        self.configure_sampler(indices, types, mode)

    def __iter__(self):
        selection = torch.multinomial(self.weights, self.num_samples, self.replacement)
        # print(Counter(self.types[self.indices[i]] for i in selection), len(selection)))
        return (self.indices[i] for i in selection)

    def __len__(self):
        return self.num_samples
