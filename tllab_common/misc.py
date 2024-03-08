import pickle
import re
import sys
from abc import ABCMeta
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from numbers import Number
from pathlib import Path
from traceback import format_exc, print_exception
from typing import Sequence

import numpy as np
import pandas
import regex
from ruamel import yaml
from IPython import embed

from .io import pickle_dump, get_params, yaml_load


class Struct(dict):
    """ dict where the items are accessible as attributes """
    key_pattern = regex.compile(r'(^(?=\d)|\W)')

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setitem__(self, key, value):
        super().__setitem__(self.transform_key(key), value)

    def __getitem__(self, key):
        return super().__getitem__(self.transform_key(key))

    def __contains__(self, key):
        return super().__contains__(self.transform_key(key))

    def __deepcopy__(self, memodict=None):
        cls = self.__class__
        copy = cls.__new__(cls)
        copy.update(**deepcopy(super(), memodict or {}))
        return copy

    def __dir__(self):
        return self.keys()

    def __missing__(self, key):
        return None

    @classmethod
    def transform_key(cls, key):
        return cls.key_pattern.sub('_', key) if isinstance(key, str) else key

    def copy(self):
        return self.__deepcopy__()

    def update(self, *args, **kwargs):
        for arg in args:
            if hasattr(arg, 'keys'):
                for key, value in arg.items():
                    self[key] = value
            else:
                for key, value in arg:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value


dumper = yaml.SafeDumper
dumper.add_representer(Struct, dumper.represent_dict)


@dataclass
class ErrorValue:
    """ format a value and its error with equal significance
        example f"value = {ErrorValue(1.23234, 0.34463):.2g}"
    """
    value: Number
    error: Number

    def __format__(self, format_spec: str) -> str:
        notation = regex.findall(r'[efgEFG]', format_spec)
        notation = notation[0] if notation else 'f'
        value_str = f'{self.value:{format_spec}}'
        digits = regex.findall(r'\d+', format_spec)
        if digits:
            digits = int(digits[0])
        else:
            frac_part = regex.findall(r'\.(\d+)', value_str)
            digits = len(frac_part[0]) if frac_part else 0
        if notation in 'gG':
            int_part = regex.findall(r'^[-+]?(\d+)', value_str)
            if int_part:
                digits -= len(int_part[0])
            frac_part = regex.findall(r'\.(\d+)', value_str)
            if frac_part:
                zeros = regex.findall(r'^0+', frac_part[0])
                if zeros:
                    digits += len(zeros[0])
        exp = regex.findall(r'[eE]([-+]?\d+)$', value_str)
        exp = int(exp[0]) if exp else 0
        error_str = f"{round(self.error * 10 ** -exp, digits):{f'.{digits}f'}}"
        split = regex.findall(r'([^eE]+)([eE][^eE]+)', value_str)
        if split:
            return f'({split[0][0]}±{error_str}){split[0][1]}'
        else:
            return f'{value_str}±{error_str}'

    def __str__(self) -> str:
        return f'{self}'


def cfmt(string: str) -> str:
    """ format a string for color printing, see cprint """
    pattern = regex.compile(r'(?:^|[^\\])(?:\\\\)*(<)((?:(?:\\\\)*\\<|[^<])*?)(:)([^:]*?[^:\\](?:\\\\)*)(>)')
    fmt_split = regex.compile(r'(?:^|\W?)([a-zA-Z]|\d+)?')
    str_sub = regex.compile(r'(?:^|\\)((?:\\\\)*[<>])')

    def format_fmt(fmt: str) -> str:
        f = fmt_split.findall(fmt)[:3]
        color, decoration, background = f + [None] * max(0, (3 - len(f)))

        t = 'KRGYBMCWargybmcwk'
        d = {'b': 1, 'u': 4, 'r': 7}
        text = ''
        if color:
            if color.isnumeric() and 0 <= int(color) <= 255:
                text = f'\033[38;5;{color}m{text}'
            elif not color.isnumeric() and color in t:
                text = f'\033[38;5;{t.index(color)}m{text}'
        if background:
            if background.isnumeric() and 0 <= int(background) <= 255:
                text = f'\033[48;5;{background}m{text}'
            elif not background.isnumeric() and background in t:
                text = f'\033[48;5;{t.index(background)}m{text}'
        if decoration and decoration.lower() in d:
            text = f'\033[{d[decoration.lower()]}m{text}'
        return text

    while matches := pattern.findall(string, overlapped=True):
        for match in matches:
            fmt = format_fmt(match[3])
            sub_string = match[1].replace('\x1b[0m', f'\x1b[0m{fmt}')
            string = string.replace(''.join(match), f'{fmt}{sub_string}\033[0m')
    return str_sub.sub(r'\1', string)


def cprint(*args, **kwargs):
    """ print colored text
        text between <> is colored, escape using \\ to print <>
        text and color format in <> is separated using : and text color, decoration and background color are separated
        using . or any character not a letter, digit or :
        colors: 'krgybmcw' (darker if capitalized) or terminal color codes (int up to 255)
        decorations: b: bold, u: underlined, r: swap color with background color """
    print(*(cfmt(arg) for arg in args), **kwargs)


def format_list(string: str, lst: Sequence, fmt: str = None) -> str:
    """ format a list in a grammatically correct way
        example: format_list('in {channel|channels}: {}', (1, 2, 5))
            'in channels: 1, 2 and 5'
    """
    if fmt is None:
        fmt = ''
    string = string.replace('{}', '{0}')
    plurals = re.findall(r'{([^|{}]+)\|([^|{}]+)}', string)
    for i, option in enumerate(plurals, start=1):
        string = string.replace(f'{{{option[0]}|{option[1]}}}', f'{{{i}}}')
    if len(lst) == 1:
        return string.format(f'{lst[0]:{fmt}}', *[option[0] for option in plurals])
    else:
        return string.format(', '.join([f'{i:{fmt}}' for i in lst[:-1]]) + f' and {lst[-1]:{fmt}}',
                             *[option[1] for option in plurals])


class Color:
    """ deprecated: use cprint instead
        print colored text:
            print(color('Hello World!', 'r:b'))
            print(color % 'r:b' + 'Hello World! + color)
            print(f'{color("r:b")}Hello World!{color}')
        text: text to be colored/decorated
        fmt: string: 'k': black, 'r': red', 'g': green, 'y': yellow, 'b': blue, 'm': magenta, 'c': cyan, 'w': white
            'b'  text color
            '.r' background color
            ':b' decoration: 'b': bold, 'u': underline, 'r': reverse
            for colors also terminal color codes can be used

        example: >> print(color('Hello World!', 'b.208:b'))
                 << Hello world! in blue bold on orange background

        wp@tl20191122
    """

    def __init__(self, fmt=None):
        self._open = False

    def _fmt(self, fmt=None):
        if fmt is None:
            self._open = False
            return '\033[0m'

        if not isinstance(fmt, str):
            fmt = str(fmt)

        decorS = [i.group(0) for i in regex.finditer(r'(?<=:)[a-zA-Z]', fmt)]
        backcS = [i.group(0) for i in regex.finditer(r'(?<=\.)[a-zA-Z]', fmt)]
        textcS = [i.group(0) for i in regex.finditer(r'((?<=[^.:])|^)[a-zA-Z]', fmt)]
        backcN = [i.group(0) for i in regex.finditer(r'(?<=\.)\d{1,3}', fmt)]
        textcN = [i.group(0) for i in regex.finditer(r'((?<=[^.:\d])|^)\d{1,3}', fmt)]

        t = 'krgybmcw'
        d = {'b': 1, 'u': 4, 'r': 7}

        text = ''
        for i in decorS:
            if i.lower() in d:
                text = '\033[{}m{}'.format(d[i.lower()], text)
        for i in backcS:
            if i.lower() in t:
                text = '\033[48;5;{}m{}'.format(t.index(i.lower()), text)
        for i in textcS:
            if i.lower() in t:
                text = '\033[38;5;{}m{}'.format(t.index(i.lower()), text)
        for i in backcN:
            if 0 <= int(i) <= 255:
                text = '\033[48;5;{}m{}'.format(int(i), text)
        for i in textcN:
            if 0 <= int(i) <= 255:
                text = '\033[38;5;{}m{}'.format(int(i), text)
        if self._open:
            text = '\033[0m' + text
        self._open = len(decorS or backcS or textcS or backcN or textcN) > 0
        return text

    def __mod__(self, fmt):
        return self._fmt(fmt)

    def __add__(self, text):
        return self._fmt() + text

    def __radd__(self, text):
        return text + self._fmt()

    def __str__(self):
        return self._fmt()

    def __call__(self, *args):
        if len(args) == 2:
            return self._fmt(args[1]) + args[0] + self._fmt()
        else:
            return self._fmt(args[0])

    def __repr__(self):
        return self._fmt()


def ipy_debug():
    """ Enter ipython after an exception occurs any time after executing this. """
    def excepthook(etype, value, traceback):
        print_exception(etype, value, traceback)
        embed(colors='neutral')
    sys.excepthook = excepthook


def get_slice(shape, n):
    ndim = len(shape)
    if isinstance(n, type(Ellipsis)):
        n = [None] * ndim
    elif not isinstance(n, (tuple, list)):
        n = [n]
    else:
        n = list(n)
    ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
    if len(ell) > 1:
        raise IndexError('an index can only have a single ellipsis (...)')
    if len(ell):
        if len(n) > ndim:
            n.remove(Ellipsis)
        else:
            n[ell[0]] = None
            while len(n) < ndim:
                n.insert(ell[0], None)
    while len(n) < ndim:
        n.append(None)

    pad = []
    for i, (e, s) in enumerate(zip(n, shape)):
        if e is None:
            e = slice(None)
        elif isinstance(e, Number):
            e = slice(e, e + 1)
        if isinstance(e, (slice, range)):
            start = int(np.floor(0 if e.start is None else e.start))
            stop = int(np.ceil(s if e.stop is None else e.stop))
            step = round(1 if e.step is None else e.step)
            if step != 1:
                raise NotImplementedError('step sizes other than 1 are not implemented!')
            pad.append((max(0, -start) // step, max(0, stop - s) // step))
            if start < 0:
                start = 0
            elif start >= s:
                start = s
            if stop >= s:
                stop = s
            elif stop < 0:
                stop = 0
            n[i] = slice(start, stop, step)
        else:
            a = np.asarray(n[i])
            if not np.all(a[:-1] <= a[1:]):
                raise NotImplementedError('unsorted slicing arrays are not supported')
            n[i] = a[(0 <= a) * (a < s)]
            pad.append((sum(a < 0), sum(a >= s)))

    return n, pad


@dataclass
class Crop:
    """ Special crop object which never takes data from outside the array, and returns the used extent too,
        together with an image showing how much of each pixel is within the extent,
        negative indices are taken literally, they do not refer to the end of the dimension!
    """
    array: np.ndarray

    def __getitem__(self, n):
        n = get_slice(self.array.shape, n)[0]
        return np.vstack([(i.start, i.stop) for i in n]), self.array[tuple(n)]


@dataclass
class SliceKeepSize:
    """ Guarantees the size of the slice by filling with a default value,
        negative indices are taken literally, they do not refer to the end of the dimension!
    """
    array: np.ndarray
    default: Number = 0

    def __getitem__(self, n):
        n, pad = get_slice(self.array.shape, n)
        crop = self.array[tuple(n)]
        default = self.default(crop) if callable(self.default) else self.default
        return np.pad(crop, pad, constant_values=default)

    def __setitem__(self, n, value):
        n = np.vstack(n)
        idx = np.prod([(0 < i) & (i < s) for i, s in zip(n, self.array.shape)], 0) > 0
        if not isinstance(value, Number):
            value = np.asarray(value)[idx]
        if n.size:
            self.array[tuple(n[:, idx])] = value


class Data(metaclass=ABCMeta):
    params = None

    def __init__(self):
        self.stage = set()
        self.runtime = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.error = format_exc()
        self.save()

    @classmethod
    def load(cls, file):
        files = glob(str(file))
        if len(files) == 0:
            raise FileNotFoundError
        file = Path(max(files)).resolve()
        with open(file, 'rb') as f:
            new = pickle.load(f)
        new.__class__ = cls
        new.file = file
        return new

    @staticmethod
    def stage_rec(fun):
        def wrap(self, *args, **kwargs):
            res = fun(self, *args, **kwargs)
            self.stage.add(fun.__name__)
            return res

        return wrap

    def save(self, file=None):
        if file is None and hasattr(self, 'folder_out'):
            file = self.folder_out / f'{self.__class__.__name__.lower()}_{self.runtime}.pickle'
        if file is not None:
            pickle_dump(self, file)

    @classmethod
    def load_from_parameter_file(cls, parameter_file):
        parameter_file = Path(parameter_file)
        params = getParams(parameter_file.with_suffix('.yml'), required=({'paths': ('folder_out',)},))
        if Path(params['paths']['folder_out']).exists():
            pickles = [file for file in Path(params['paths']['folder_out']).iterdir()
                       if file.name.startswith(f'{cls.__name__.lower()}_') and file.suffix == '.pickle']
        else:
            pickles = None
        if not pickles:
            raise FileNotFoundError(
                f"No files matching {Path(params['paths']['folder_out']) / f'{cls.__name__.lower()}_*.pickle'}")
        return cls.load(max(pickles))

    def run(self):
        self.runtime = datetime.now().strftime("%Y%m%d_%H%M%S")

    def clean(self):
        if Path(self.params['paths']['folder_out']).exists():
            pickles = [file for file in Path(self.params['paths']['folder_out']).iterdir()
                       if file.name.startswith(f'{self.__class__.__name__.lower()}_') and file.suffix == '.pickle']
            if pickles:
                pickles.remove(max(pickles))
                for pickle in pickles:
                    pickle.unlink()

    def color(self, color_or_channel):
        return color_or_channel if isinstance(color_or_channel, str) else self.channels[color_or_channel]

    def channel(self, color_or_channel):
        return self.colors[color_or_channel] if isinstance(color_or_channel, str) else color_or_channel


def df_join(h: pandas.DataFrame) -> pandas.DataFrame:
    """ join DataFrames given by the first indices of h on the other indices
    """
    groups = h.groupby(level=0)
    n = len(groups)
    for a, (i, g) in enumerate(groups):
        if a == 0:
            df = g.droplevel(0)
        elif a == n - 1:
            return df.join(g.droplevel(0), lsuffix=f'_{j:.0f}', rsuffix=f'_{i:.0f}')
        else:
            df = df.join(g.droplevel(0), lsuffix=f'_{j:.0f}')
        j = i


color = Color()
get_config = yaml_load
getConfig = get_config
getParams = get_params
objFromDict = Struct
