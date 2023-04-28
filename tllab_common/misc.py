import os
import re
import yaml
import sys
from traceback import print_exception
from IPython import embed
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
import roifile


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


class color_class(object):
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

color = color_class()


def getConfig(file):
    """ Open a yml parameter file
    """
    with open(file, 'r') as f:
        return yaml.load(f, loader)


def getParams(parameterfile, templatefile=None, required=None):
    """ Load parameters from a parameterfile and parameters missing from that from the templatefile. Raise an error when
        parameters in required are missing. Return a dictionary with the parameters.
    """
    params = getConfig(parameterfile)

    # recursively load more parameters from another file
    def more_params(params, file):
        if not params.get('moreParams') == none():
            if os.path.isabs(params['moreParams']):
                moreParamsFile = params['moreParams']
            else:
                moreParamsFile = os.path.join(os.path.dirname(os.path.abspath(file)), params['moreParams'])
            print(color(f'Loading more parameters from {moreParamsFile}', 'g'))
            mparams = getConfig(moreParamsFile)
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

    # recursively convert string nones to type None
    def convert_none(params):
        for key, value in params.items():
            if value == none():
                params[key] = None
            elif isinstance(value, dict):
                convert_none(value)

    convert_none(params)
    more_params(params, parameterfile)

    if required is not None:
        for p in required:
            if p not in params:
                raise Exception(f'Parameter {p} not given in parameter file.')

    if templatefile is not None:
        check_params(params, getConfig(templatefile))
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


class objFromDict(dict):
    """ Usage: objFromDict(**dictionary).
        Print gives the list of attributes.
    """
    def __init__(self, **entries):
        super(objFromDict, self).__init__()
        for key, value in entries.items():
            key = key.replace('-', '_').replace('*', '_').replace('+', '_').replace('/', '_')
            self[key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __repr__(self):
        return '** {} attr. --> '.format(self.__class__.__name__)+', '.join(filter((lambda s: (s[:2]+s[-2:]) != '____'),
                                                                                   self.keys()))

    def copy(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        copy = cls.__new__(cls)
        copy.update(**deepcopy(super(objFromDict, self), memodict))
        return copy

    def __dir__(self):
        return self.keys()


class none():
    """ special class to check if an object is some variation of none
    """
    def __eq__(self, other):
        if isinstance(other, none):
            return True
        if other is None:
            return True
        if hasattr(other, 'lower') and other.lower() == 'none':
            return True
        return False


def ipy_debug():
    """ Enter ipython after an exception occurs any time after executing this. """
    def excepthook(etype, value, traceback):
        print_exception(etype, value, traceback)
        embed(colors='neutral')
    sys.excepthook = excepthook
