import re
import yaml
from copy import deepcopy


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def color(text, fmt):
    """ print colored text: print(color('Hello World!', 'r:b'))
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

    if not isinstance(fmt, str):
        fmt = str(fmt)

    decorS = [i.group(0) for i in re.finditer('(?<=\:)[a-zA-Z]', fmt)]
    backcS = [i.group(0) for i in re.finditer('(?<=\.)[a-zA-Z]', fmt)]
    textcS = [i.group(0) for i in re.finditer('((?<=[^\.\:])|^)[a-zA-Z]', fmt)]
    backcN = [i.group(0) for i in re.finditer('(?<=\.)\d{1,3}', fmt)]
    textcN = [i.group(0) for i in re.finditer('((?<=[^\.\:\d])|^)\d{1,3}', fmt)]

    t = ('k', 'r', 'g', 'y', 'b', 'm', 'c', 'w')
    d = {'b': 1, 'u': 4, 'r': 7}

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

    return text + '\033[0m'


def getConfig(file):
    """ Open a yml parameter file
    """
    with open(file, 'r') as f:
        return yaml.load(f, loader)


def convertParamFile2YML(file):
    """ Convert a py parameter file into a yml file
    """
    with open(file, 'r') as f:
        lines = f.read(-1)
    with open(re.sub('\.py$', '.yml', file), 'w') as f:
        for line in lines.splitlines():
            if not re.match('^import', line):
                line = re.sub('(?<!#)\s*=\s*', ': ', line)
                line = re.sub('(?<!#);', '', line)
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