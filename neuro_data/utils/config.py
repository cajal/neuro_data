from inspect import isclass
from pprint import pformat

import datajoint as dj
from .data import key_hash, to_native
from .. import logger as log
import numpy as np
import numpy.random as rng

_INDENT = '      '

from contextlib import contextmanager

@contextmanager
def fixed_seed(seed):
    state = rng.get_state()
    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)


class ConfigBase:
    _config_type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration
        {extra_foreign} 
        ---
        {ct}_description            : varchar(2048) # description of the config
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP : timestamp      # automatic
        """.format(ct=self._config_type, cn=self.__class__.__name__,
                   extra_foreign=self._extra_foreign if hasattr(self, '_extra_foreign') else '')

    def part_table(self, key=None):
        key = {} if key is None else key
        return getattr(self & key, (self & key).fetch1('{}_type'.format(self._config_type))) & self

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        with self.connection.transaction:
            for rel in [getattr(self, member) for member in dir(self)
                        if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:
                log.info('Checking' + rel.__name__)
                for key in rel().content:
                    assert set(key.keys()) == set(rel.heading.secondary_attributes)
                    key[type_name] = rel.__name__
                    key[hash_name] = key_hash(key)

                    if not key in rel().proj():
                        self.insert1(dict(key, **{self._config_type + '_description':rel.describe(key)}),
                                     ignore_extra_fields=True, skip_duplicates=True)
                        log.info('Inserting\n' + pformat(key, indent=20))
                        rel().insert1(key,ignore_extra_fields=True)

    def parameters(self, key, selection=None):
        type_name = self._config_type + '_type'
        key = (self & key).fetch1()  # complete parameters
        part = getattr(self, key[type_name])
        ret = (self * part() & key).fetch1()
        ret = to_native(ret)
        if selection is None:
            del ret[self._config_type + '_ts']
            del ret[self._config_type + '_hash']
            return ret
        else:
            if isinstance(selection, list):
                return tuple(ret[k] for k in selection)
            else:
                return ret[selection]


    def select_hashes(self, depth=0):
        configs = [getattr(self, member) for member in dir(self) if
                   isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]
        print('\n'.join(['{}({}) {}'.format(depth * _INDENT, i, rel.__name__) for i, rel in enumerate(configs)]))

        choices = input(depth * _INDENT + 'Please select configuration [comma separated list]: ')
        choices = list(map(int, choices.split(',')))

        hashes = []
        for choice in choices:
            restriction = dict()
            rel = configs[int(choice)]()
            while restriction != '':
                old_restriction = restriction
                print(depth * _INDENT + repr(old_restriction))
                s = repr(rel & old_restriction).replace('\n', '\n' + str(depth * _INDENT))
                print(str(depth * _INDENT) + s)
                restriction = input(str(depth * _INDENT) + 'Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
        return '{}_hash'.format(self._config_type), hashes
