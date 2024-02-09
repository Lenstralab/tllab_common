import pickle
import zipfile
from contextlib import ExitStack
from functools import wraps
from io import BytesIO, StringIO
from pathlib import Path

import dill
import numpy as np
import pandas
import yaml
from bidict import bidict


class Dumper(yaml.Dumper):
    def __init__(self, path, *args, **kwargs):
        super().__init__(StringIO(), *args, **kwargs)
        self.zip_stream = zipfile.ZipFile(path, 'w')

    def close(self):
        with self.zip_stream.open('object.yml', 'w') as f:
            f.write(bytes(self.stream.getvalue(), ' utf-8'))
        self.stream.close()
        self.zip_stream.close()


def yd_register(t):
    def proxy(func):
        Dumper.add_representer(t, func)
        return func

    return proxy


@yd_register(pandas.DataFrame)
def represent_pandas(dumper: Dumper, df):
    file = f'{id(df)}.tsv'
    with dumper.zip_stream.open(file, 'w') as f:
        df.to_csv(f, sep='\t', index=False)
    return dumper.represent_scalar('!DataFrame', file)


@yd_register(np.ndarray)
def represent_numpy(dumper: Dumper, array):
    file = f'{id(array)}.npy'
    with dumper.zip_stream.open(file, 'w') as f:
        np.save(f, array)
    return dumper.represent_scalar('!ndarray', file)


@yd_register(bidict)
def represent_bidict(self, bd):
    if id(bd.inverse) in self.represented_objects:
        return self.represent_mapping('!bidict', {'dict': bd.inverse._fwdm, 'inverse': True})
    else:
        return self.represent_mapping('!bidict', {'dict': bd._fwdm, 'inverse': False})


@wraps(yaml.dump)
def zip_dump(obj, stream=None, *args, **kwargs):
    return yaml.dump(obj, stream, Dumper, *args, **kwargs)


class Loader(yaml.Loader):
    bidict_cache = {}

    def __init__(self, path):
        self.zip_stream = zipfile.ZipFile(path, 'r')
        super().__init__(self.zip_stream.open('object.yml', 'r'))

    def close(self):
        self.stream.close()
        self.zip_stream.close()


def yl_register(t):
    def proxy(func):
        Loader.add_constructor(t, func)
        return func

    return proxy


@yl_register('!DataFrame')
def construct_pandas(loader, node):
    with loader.zip_stream.open(loader.construct_python_str(node), 'r') as f:
        return pandas.read_table(f)


@yl_register('!ndarray')
def construct_numpy(loader, node):
    with loader.zip_stream.open(loader.construct_python_str(node), 'r') as f:
        return np.load(f)


@yl_register('!bidict')
def construct_bidict(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    dct, inverse = mapping['dict'], mapping['inverse']
    bdct = loader.bidict_cache.get(id(dct))
    if bdct is None:
        bdct = bidict(dct)
        loader.bidict_cache[id(dct)] = bdct
    return bdct.inverse if inverse else bdct


@wraps(yaml.load)
def zip_load(stream):
    return yaml.load(stream, Loader)


class Pickler(dill.Pickler):
    dispatch = dill.Pickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bd_dilled = []  # [id(bidict)]
        self.bd_undilled = {}  # {id(dict): bidict}


def dill_register(t):
    """decorator to register types to Pickler's :attr:`~Pickler.dispatch` table"""

    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy


def undill_bidict(dct: dict, inverse: bool, undilled: dict) -> bidict:
    """ restore bidict relationships """
    bdct = undilled.get(id(dct))
    if bdct is None:
        bdct = bidict(dct)
        undilled[id(dct)] = bdct
    return bdct.inverse if inverse else bdct


@dill_register(bidict)
def dill_bidict(pickler: Pickler, bd: bidict):
    """ pickle bidict such that relationships between bidicts is preserved upon unpickling """
    if id(bd.inverse) in pickler.bd_dilled:
        pickler.save_reduce(undill_bidict, (bd.inverse._fwdm, True, pickler.bd_undilled), obj=bd)
    else:
        pickler.bd_dilled.append(id(bd))
        pickler.save_reduce(undill_bidict, (bd._fwdm, False, pickler.bd_undilled), obj=bd)


@dill_register(pandas.DataFrame)
def dill_dataframe(pickler: Pickler, df: pandas.DataFrame):
    """ pickle dataframe as dict to ensure compatibility """
    pickler.save_reduce(pandas.DataFrame, (df.to_dict(),), obj=df)


@wraps(pickle.dump)
def pickle_dump(obj, file=None, *args, **kwargs):
    with ExitStack() as stack:
        if isinstance(file, (str, Path)):
            f = stack.enter_context(open(file, 'wb'))
        elif file is None:
            f = stack.enter_context(BytesIO())
        else:
            f = file
        Pickler(f, *args, **kwargs).dump(obj)
        if file is None:
            return f.getvalue()


@wraps(pickle.load)
def pickle_load(file):
    if isinstance(file, bytes):
        return pickle.loads(file)
    elif isinstance(file, (str, Path)):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        return pickle.load(file)
