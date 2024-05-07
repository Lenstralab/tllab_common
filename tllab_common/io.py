import pickle
from contextlib import ExitStack
from functools import wraps
from io import BytesIO
from pathlib import Path
from re import sub
from typing import IO, Any, Callable, Hashable, Iterator, Optional, Sequence, Type

import dill
import pandas
import roifile
from bidict import bidict
from ruamel import yaml


class Pickler(dill.Pickler):
    dispatch = dill.Pickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bd_dilled = []  # [id(bidict)]
        self.bd_undilled = {}  # {id(dict): bidict}


def dill_register(t: Type) -> Callable:
    """decorator to register types to Pickler's :attr:`~Pickler.dispatch` table"""

    def proxy(func: Callable) -> Callable:
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
        pickler.save_reduce(undill_bidict, (bd.inverse._fwdm, True, pickler.bd_undilled), obj=bd)  # noqa
    else:
        pickler.bd_dilled.append(id(bd))
        pickler.save_reduce(undill_bidict, (bd._fwdm, False, pickler.bd_undilled), obj=bd)  # noqa


@dill_register(pandas.DataFrame)
def dill_dataframe(pickler: Pickler, df: pandas.DataFrame):
    """ pickle dataframe as dict to ensure compatibility """
    pickler.save_reduce(pandas.DataFrame, (df.to_dict(),), obj=df)


@wraps(pickle.dump)
def pickle_dump(obj, file: Optional[IO] = None, *args, **kwargs) -> Optional[str]:
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
def pickle_load(file: [bytes, str, Path, IO]) -> Any:
    if isinstance(file, bytes):
        return pickle.loads(file)
    elif isinstance(file, (str, Path)):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        return pickle.load(file)


class CommentedDefaultMap(yaml.CommentedMap):
    def __missing__(self, key: Hashable) -> Any:
        return None

    def __repr__(self) -> str:
        return dict.__repr__(self)


class RoundTripConstructor(yaml.RoundTripConstructor):
    pass


def construct_yaml_map(loader, node: Any) -> Iterator[CommentedDefaultMap]:
    data = CommentedDefaultMap()
    data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)  # noqa
    yield data
    loader.construct_mapping(node, data, deep=True)
    loader.set_collection_style(data, node)


RoundTripConstructor.add_constructor('tag:yaml.org,2002:map', construct_yaml_map)


class RoundTripRepresenter(yaml.RoundTripRepresenter):
    pass


RoundTripRepresenter.add_representer(CommentedDefaultMap, RoundTripRepresenter.represent_dict)


@wraps(yaml.load)
def yaml_load(stream: [str, bytes, Path, IO]) -> Any:
    with ExitStack() as stack:
        y = yaml.YAML()
        y.Constructor = RoundTripConstructor
        try:
            if isinstance(stream, (str, bytes, Path)):
                stream = stack.enter_context(open(stream, 'r'))
        except (FileNotFoundError, OSError):
            pass
        return y.load(stream)


@wraps(yaml.dump)
def yaml_dump(data: Any, stream: Optional[IO] = None) -> Optional[str]:
    with ExitStack() as stack:
        y = yaml.YAML()
        y.Representer = RoundTripRepresenter
        if isinstance(stream, (str, bytes, Path)):
            stream = stack.enter_context(open(stream, 'w'))
        return y.dump(data, stream)


def get_params(parameter_file: [str, Path], template_file: [str, Path] = None,
               required: Sequence[dict] = None) -> CommentedDefaultMap:
    """ Load parameters from a parameterfile and parameters missing from that from the templatefile. Raise an error when
        parameters in required are missing. Return a dictionary with the parameters.
    """

    from .misc import cprint

    parameter_file = Path(parameter_file)

    def yaml_load_and_format(file: Path) -> CommentedDefaultMap:
        with open(file) as f:
            return yaml_load(sub(r'{{\s*(.+)\s*}}', r'{\1}', f.read()).format(
                name=parameter_file.stem, folder=str(parameter_file.parent), suffix=parameter_file.suffix))

    def more_params(parameters: dict, file: [str, Path]) -> None:
        """ recursively load more parameters from another file """
        file = Path(file)
        more_parameters_file = parameters['more_parameters'] or parameters['more_params'] or parameters['moreParams']
        if more_parameters_file is not None:
            more_parameters_file = Path(more_parameters_file)
            if not more_parameters_file.is_absolute():
                more_parameters_file = Path(file).absolute().parent / more_parameters_file
            cprint(f'<Loading more parameters from <{more_parameters_file}:.b>:g>')
            more_parameters = yaml_load_and_format(more_parameters_file)
            more_params(more_parameters, file)

            def add_items(sub_params, item):
                for k, v in item.items():
                    if k not in sub_params:
                        sub_params[k] = v
                    elif isinstance(v, dict):
                        add_items(sub_params[k], v)

            add_items(parameters, more_parameters)

    def check_params(parameters: dict, template: dict, path: str = '') -> None:
        """ recursively check parameters and add defaults """
        for key, value in template.items():
            if key not in parameters and value is not None:
                cprint(f'<Parameter <{path}{key}:.b> missing in parameter file, adding with default value: {value}.:r>')
                parameters[key] = value
            elif isinstance(value, dict):
                check_params(parameters[key], value, f'{path}{key}.')

    def check_required(parameters: dict, required: Sequence[dict]) -> None:  # noqa
        if required is not None:
            for p in required:
                if isinstance(p, dict):
                    for key, value in p.items():
                        check_required(parameters[key], value)
                else:
                    if p not in parameters:
                        raise Exception(f'Parameter {p} not given in parameter file.')

    params = yaml_load_and_format(parameter_file)
    more_params(params, parameter_file)
    check_required(params, required)

    if template_file is not None:
        check_params(params, yaml_load(template_file))
    return params


def save_roi(file: [str, Path], coordinates: pandas.DataFrame, shape: tuple, columns: Optional[Sequence[str]] = None,
             name: Optional[str] = None):
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
