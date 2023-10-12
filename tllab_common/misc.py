import os
import regex
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

    @staticmethod
    def construct_yaml_map(y, node):
        data = Struct()
        yield data
        value = y.construct_mapping(node)
        data.update(value)


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    r'tag:yaml.org,2002:float',
    regex.compile(r'''^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$''', regex.X),
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
        notation = regex.findall(r'[efgEFG]', format_spec)
        notation = notation[0] if notation else 'f'
        value_str = f'{self.value:{format_spec}}'
        digits = regex.findall(r'\d+', format_spec)
        digits = int(digits[0]) if digits else 0
        if notation in 'gG':
            int_part = regex.findall(r'^(\d+)', value_str)
            if int_part:
                digits -= len(int_part[0])
                zeros = regex.findall(r'^0+', int_part[0])
                if zeros:
                    digits += len(zeros[0])
            frac_part = regex.findall(r'.(\d+)', value_str)
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


def cfmt(string):
    """ format a string for color printing, see cprint """
    pattern = regex.compile(r'(?:^|[^\\])(?:\\\\)*(<)((?:(?:\\\\)*\\<|[^<])*?)(:)([^:]*?[^:\\](?:\\\\)*)(>)')
    fmt_split = regex.compile(r'(?:^|\W?)([a-zA-Z]|\d+)?')
    str_sub = regex.compile(r'(?:^|\\)((?:\\\\)*[<>])')

    def format_fmt(fmt):
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
        text between <> is colored, escape using \ to print <>
        text and color format in <> is separated using : and text color, decoration and background color are separated
        using . or any character not a letter, digit or :
        colors: 'krgybmcw' (darker if capitalized) or terminal color codes (int up to 255)
        decorations: b: bold, u: underlined, r: swap color with background color """
    print(*(cfmt(arg) for arg in args), **kwargs)


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
        more_parameters = params['more_parameters'] or params['more_params'] or params['moreParams']
        if not more_parameters is None:
            if os.path.isabs(more_parameters):
                moreParamsFile = more_parameters
            else:
                moreParamsFile = os.path.join(os.path.dirname(os.path.abspath(file)), more_parameters)
            cprint(f'<Loading more parameters from <{moreParamsFile}:.b>:g>')
            mparams = get_config(moreParamsFile)
            more_params(mparams, file)
            for k, v in mparams.items():
                if k not in params:
                    params[k] = v

    # recursively check parameters and add defaults
    def check_params(params, template, path=''):
        for key, value in template.items():
            if key not in params and value is not None:
                cprint(f'<Parameter <{path}{key}:.b> missing in parameter file, adding with default value: {value}.:r>')
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


color = Color()
getConfig = get_config
getParams = get_params
objFromDict = Struct
