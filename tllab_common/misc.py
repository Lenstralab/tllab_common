import os
import re
import yaml
import sys
import numpy as np
from traceback import print_exception
from IPython import embed
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
import roifile


class Struct(dict):
    """ dict where the items are accessible as attributes """
    key_pattern = re.compile(r'(^(?=\d)|\W)')

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

    @staticmethod
    def construct_yaml_map(y, node):
        data = Struct()
        yield data
        value = y.construct_mapping(node)
        data.update(value)


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    r'tag:yaml.org,2002:float',
    re.compile(r'''^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$''', re.X),
    list(r'-+0123456789.'))

loader.add_constructor('tag:yaml.org,2002:python/dict', Struct.construct_yaml_map)
loader.add_constructor('tag:yaml.org,2002:omap', Struct.construct_yaml_map)
loader.add_constructor('tag:yaml.org,2002:map', Struct.construct_yaml_map)


@dataclass
class ErrorValue:
    """ format a value and its error with equal significance
        example f"value = {ErrorValue(1.23234, 0.34463):.2g}"
    """
    value: Number
    error: Number

    def __format__(self, format_spec):
        notation = re.findall(r'[efgEFG]', format_spec)
        notation = notation[0] if notation else 'f'
        value_str = f'{self.value:{format_spec}}'
        digits = re.findall(r'\d+', format_spec)
        digits = int(digits[0]) if digits else 0
        if notation in 'gG':
            int_part = re.findall(r'^(\d+)', value_str)
            if int_part:
                digits -= len(int_part[0])
                zeros = re.findall(r'^0+', int_part[0])
                if zeros:
                    digits += len(zeros[0])
            frac_part = re.findall(r'.(\d+)', value_str)
            if frac_part:
                zeros = re.findall(r'^0+', frac_part[0])
                if zeros:
                    digits += len(zeros[0])
        exp = re.findall(r'[eE]([-+]?\d+)$', value_str)
        exp = int(exp[0]) if exp else 0
        error_str = f"{round(self.error * 10 ** -exp, digits):{f'.{digits}f'}}"
        split = re.findall(r'([^eE]+)([eE][^eE]+)', value_str)
        if split:
            return f'({split[0][0]}±{error_str}){split[0][1]}'
        else:
            return f'{value_str}±{error_str}'

    def __str__(self):
        return f"{self}"


def save_roi(file, coordinates, shape, columns=None, name=None):
    if columns is None:
        columns = 'xyCzT'
    coordinates = coordinates.copy()
    if '_' in columns:
        coordinates['_'] = 0
    # if we save coordinates too close to the right and bottom of the image (<1 px) the roi won't open on the image
    if not coordinates.empty:
        coordinates = coordinates.query(f'-0.5<={columns[0]}<{shape[1]-1.5} & -0.5<={columns[1]}<{shape[0]-1.5} &'
                                        f' -0.5<={columns[3]}<={shape[3]-0.5}')
    if not coordinates.empty:
        roi = roifile.ImagejRoi.frompoints(coordinates[list(columns[:2])].to_numpy().astype(float))
        roi.roitype = roifile.ROI_TYPE.POINT
        roi.options = roifile.ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        roi.counters = len(coordinates) * [0]
        roi.counter_positions = (1 + coordinates[columns[2]].to_numpy() +
                                 coordinates[columns[3]].to_numpy().round().astype(int) * shape[2] +
                                 coordinates[columns[4]].to_numpy() * shape[2] * shape[3]).astype(int)
        if name is None:
            roi.name = ''
        else:
            roi.name = name
        roi.version = 228
        roi.tofile(file)


class Color(object):
    """ print colored text:
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

        decorS = [i.group(0) for i in re.finditer(r'(?<=:)[a-zA-Z]', fmt)]
        backcS = [i.group(0) for i in re.finditer(r'(?<=\.)[a-zA-Z]', fmt)]
        textcS = [i.group(0) for i in re.finditer(r'((?<=[^.:])|^)[a-zA-Z]', fmt)]
        backcN = [i.group(0) for i in re.finditer(r'(?<=\.)\d{1,3}', fmt)]
        textcN = [i.group(0) for i in re.finditer(r'((?<=[^.:\d])|^)\d{1,3}', fmt)]

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


def get_config(file):
    """ Open a yml parameter file
    """
    with open(file, 'r') as f:
        return yaml.load(f, loader)


def get_params(parameterfile, templatefile=None, required=None):
    """ Load parameters from a parameterfile and parameters missing from that from the templatefile. Raise an error when
        parameters in required are missing. Return a dictionary with the parameters.
    """
    # recursively load more parameters from another file
    def more_params(params, file):
        if not params.get('moreParams') is None:
            if os.path.isabs(params['moreParams']):
                moreParamsFile = params['moreParams']
            else:
                moreParamsFile = os.path.join(os.path.dirname(os.path.abspath(file)), params['moreParams'])
            print(color(f'Loading more parameters from {moreParamsFile}', 'g'))
            mparams = get_config(moreParamsFile)
            more_params(mparams, file)
            for k, v in mparams.items():
                if k not in params:
                    params[k] = v

    # recursively check parameters and add defaults
    def check_params(params, template, path=''):
        for key, value in template.items():
            if key not in params and value is not None:
                print(color(f'Parameter {path}{key} missing in parameter file, adding with default value: {value}.',
                            'r'))
                params[key] = value
            elif isinstance(value, dict):
                check_params(params[key], value, f'{path}{key}.')

    def check_required(params, required):
        if required is not None:
            for p in required:
                if isinstance(p, dict):
                    for key, value in p.items():
                        check_required(params[key], value)
                else:
                    if p not in params:
                        raise Exception(f'Parameter {p} not given in parameter file.')

    params = get_config(parameterfile)
    more_params(params, parameterfile)
    check_required(params, required)

    if templatefile is not None:
        check_params(params, get_config(templatefile))
    return params


def convertParamFile2YML(file):
    """ Convert a py parameter file into a yml file
    """
    with open(file, 'r') as f:
        lines = f.read(-1)
    with open(re.sub(r'\.py$', '.yml', file), 'w') as f:
        for line in lines.splitlines():
            if not re.match(r'^import', line):
                line = re.sub(r'(?<!#)\s*=\s*', ': ', line)
                line = re.sub(r'(?<!#);', '', line)
                f.write(line+'\n')


def ipy_debug():
    """ Enter ipython after an exception occurs any time after executing this. """
    def excepthook(etype, value, traceback):
        print_exception(etype, value, traceback)
        embed(colors='neutral')
    sys.excepthook = excepthook


def get_slice(array, n):
    if isinstance(n, slice):
        n = (n,)
    if isinstance(n, type(Ellipsis)):
        n = (None,) * array.ndim
    if isinstance(n, Number):
        n = (slice(n),)
    n = list(n)
    ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
    if len(ell) > 1:
        raise IndexError('an index can only have a single ellipsis (...)')
    if len(ell):
        if len(n) > array.ndim:
            n.remove(Ellipsis)
        else:
            n[ell[0]] = None
            while len(n) < array.ndim:
                n.insert(ell[0], None)
    while len(n) < array.ndim:
        n.append(None)

    pad = []
    for i, (e, s) in enumerate(zip(n, array.shape)):
        if e is None:
            e = slice(None)
        elif isinstance(e, Number):
            e = slice(e, e)
        start, stop, step = int(np.floor(e.start or 0)), int(np.ceil(e.stop or s)), round(e.step or 1)
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
    return n, pad


@dataclass
class Crop:
    """ Special crop object which never takes data from outside the array, and returns the used extent too,
        together with an image showing how much of each pixel is within the extent,
        negative indices are taken literally, they do not refer to the end of the dimension!
    """
    array: np.ndarray

    def __getitem__(self, n):
        n = get_slice(self.array, n)[0]
        return np.vstack([(i.start, i.stop) for i in n]), self.array[tuple(n)]


@dataclass
class SliceKeepSize:
    """ Guarantees the size of the slice by filling with a default value,
        negative indices are taken literally, they do not refer to the end of the dimension!
    """
    array: np.ndarray
    default: Number = 0

    def __getitem__(self, n):
        n, pad = get_slice(self.array, n)
        return np.pad(self.array[tuple(n)], pad, constant_values=self.default)

    def __setitem__(self, n, value):
        n = np.vstack(n)
        idx = np.prod([(0 < i) & (i < s) for i, s in zip(n, self.array.shape)], 0) > 0
        if not isinstance(value, Number):
            value = np.asarray(value)[idx]
        if n.size:
            self.array[tuple(n[:, idx])] = value


color = Color()
getConfig = get_config
getParams = get_params
objFromDict = Struct
