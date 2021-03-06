# -*- coding: utf-8 -*-

import os
import re
import inspect
import json
import untangle
import pandas
import tifffile
import czifile
import yaml
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
from itertools import product
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from functools import cached_property
from parfor import parfor
from tiffwrite import IJTiffFile
from tllab_common.transforms import Transform, Transforms
from numbers import Number
from argparse import ArgumentParser

try:
    import javabridge
    import bioformats
    java = True
except ImportError:
    java = False


class jvm:
    """ There can be only one java virtual machine per python process, so this is a singleton class to manage the jvm.
    """
    _instance = None
    vm_started = False
    vm_killed = False

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def start_vm(self):
        if java:
            if not self.vm_started and not self.vm_killed:
                javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
                outputstream = javabridge.make_instance('java/io/ByteArrayOutputStream', "()V")
                printstream = javabridge.make_instance('java/io/PrintStream', "(Ljava/io/OutputStream;)V", outputstream)
                javabridge.static_call('Ljava/lang/System;', "setOut", "(Ljava/io/PrintStream;)V", printstream)
                javabridge.static_call('Ljava/lang/System;', "setErr", "(Ljava/io/PrintStream;)V", printstream)
                self.vm_started = True
                log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
                log4j.enableLogging()
                log4j.setRootLevel("ERROR")

            if self.vm_killed:
                raise Exception('The JVM was killed before, and cannot be restarted in this Python process.')
        else:
            raise ImportError('python-bioformats and or python-javabridge are not installed')

    def kill_vm(self):
        javabridge.kill_vm()
        self.vm_started = False
        self.vm_killed = True


class ImTransformsExtra(Transforms):
    def coords(self, array, colums=None):
        if isinstance(array, pandas.DataFrame):
            return pandas.concat([self(int(row['C']), int(row['T'])).coords(row, colums)
                                  for _, row in array.iterrows()], axis=1).T
        elif isinstance(array, pandas.Series):
            return self(int(array['C']), int(array['T'])).coords(array, colums)
        else:
            raise TypeError('Not a pandas DataFrame or Series.')


class ImTransforms(ImTransformsExtra):
    """ Transforms class with methods to calculate channel transforms from bead files etc.
    """
    def __init__(self, path, cyllens, tracks=None, detectors=None, file=None, transforms=None):
        super().__init__()
        self.cyllens = cyllens
        self.tracks = tracks
        self.detectors = detectors
        if transforms is None:
            # TODO: check this
            if re.search(r'^Pos\d+', os.path.basename(path.rstrip(os.path.sep))):
                self.path = os.path.dirname(os.path.dirname(path))
            else:
                self.path = os.path.dirname(path)
            if file is not None:
                if isinstance(file, str) and file.lower().endswith('.yml'):
                    self.ymlpath = file
                    self.beadfile = None
                else:
                    self.ymlpath = os.path.join(self.path, 'transform.yml')
                    self.beadfile = file
            else:
                self.ymlpath = os.path.join(self.path, 'transform.yml')
                self.beadfile = None
            self.tifpath = self.ymlpath[:-3] + 'tif'
            try:
                self.load(self.ymlpath)
            except Exception:
                print('No transform file found, trying to generate one.')
                if not self.files:
                    raise FileNotFoundError('No bead files found to calculate the transform from.')
                self.calculate_transforms()
                self.save(self.ymlpath)
                self.save_transform_tiff()
                print(f'Saving transform in {self.ymlpath}.')
                print(f'Please check the transform in {self.tifpath}.')
        else:  # load from dict transforms
            self.path = path
            self.beadfile = file
            for key, value in transforms.items():
                self[tuple([int(i) for i in key.split(':')])] = Transform(value)

    @cached_property
    def files(self):
        try:
            if self.beadfile is None:
                files = self.get_bead_files()
            else:
                files = self.beadfile
            if isinstance(files, str):
                files = (files,)
            return files
        except Exception:
            return ()

    def __reduce__(self):
        return self.__class__, (self.path, self.cyllens, self.tracks, self.detectors, self.files, self.asdict())

    def __call__(self, channel, time=None, tracks=None, detectors=None):
        tracks = tracks or self.tracks
        detectors = detectors or self.detectors
        return super().__call__(channel, time, tracks, detectors)

    def get_bead_files(self):
        files = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().startswith('beads')
                        and not f.lower().endswith('.pdf')])
        if not files:
            raise Exception('No bead file found!')
        Files = []
        for file in files:
            try:
                if os.path.isdir(file):
                    file = os.path.join(file, 'Pos0')
                with imread(file) as im:  # check for errors opening the file
                    pass
                Files.append(file)
            except Exception:
                continue
        if not Files:
            raise Exception('No bead file found!')
        return Files

    def calculate_transform(self, file):
        """ When no channel is not transformed by a cylindrical lens, assume that the image is scaled by a factor 1.162
            in the horizontal direction
        """
        with imread(file) as im:
            ims = [im.max(c) for c in range(im.shape[2])]
            goodch = [c for c in range(im.shape[2]) if not im.isnoise(im.max(c))]
            untransformed = [c for c in range(im.shape[2]) if self.cyllens[im.detector[c]].lower() == 'none']

            good_and_untrans = sorted(set(goodch) & set(untransformed))
            if good_and_untrans:
                masterch = good_and_untrans[0]
            else:
                masterch = goodch[0]
            print(f'{untransformed = }, {masterch = }, {goodch = }')
            C = Transform()
            if not np.any(good_and_untrans):
                M = C.matrix
                M[0, 0] = 0.86
                C.matrix = M
            Tr = Transforms()
            for c in tqdm(goodch):
                if c == masterch:
                    Tr[im.track[c], im.detector[c]] = C
                else:
                    Tr[im.track[c], im.detector[c]] = Transform(ims[masterch], ims[c]) * C
        return Tr

    def calculate_transforms(self):
        Tq = [self.calculate_transform(file) for file in self.files]
        for key in set([key for t in Tq for key in t.keys()]):
            T = [t[key] for t in Tq if key in t]
            if len(T) == 1:
                self[key] = T[0]
            else:
                self[key] = Transform()
                self[key].parameters = np.mean([t.parameters for t in T], 0)
                self[key].dparameters = (np.std([t.parameters for t in T], 0) / np.sqrt(len(T))).tolist()

    def save_transform_tiff(self):
        C = 0
        for file in self.files:
            with imread(file) as im:
                C = max(C, im.shape[2])
        with IJTiffFile(self.tifpath, (C, 1, len(self.files))) as tif:
            for t, file in enumerate(self.files):
                with imread(file) as im:
                    with imread(file, transform=True) as jm:
                        for c in range(im.shape[2]):
                            tif.save(np.hstack((im.max(c), jm.max(c))), c, 0, t)


class ImShiftTransforms(ImTransformsExtra):
    """ Class to handle drift in xy. The image this is applied to must have a channeltransform already, which is then
        replaced by this class.
    """
    def __init__(self, im, shifts=None):
        """ im:                     Calculate shifts from channel-transformed images
            im, t x 2 array         Sets shifts from array, one row per frame
            im, dict {frame: shift} Sets shifts from dict, each key is a frame number from where a new shift is applied
            im, file                Loads shifts from a saved file
        """
        super().__init__()
        with (imread(im, transform=True, drift=False) if isinstance(im, str)
                                                      else im.new(transform=True, drift=False)) as im:
            self.impath = im.path
            self.path = os.path.splitext(self.impath)[0] + '_shifts.txt'
            self.tracks, self.detectors, self.files = im.track, im.detector, im.beadfile
            if shifts is not None:
                if isinstance(shifts, np.ndarray):
                    self.shifts = shifts
                    self.shifts2transforms(im)
                elif isinstance(shifts, dict):
                    self.shifts = np.zeros((im.shape[4], 2))
                    for k in sorted(shifts.keys()):
                        self.shifts[k:] = shifts[k]
                    self.shifts2transforms(im)
                elif isinstance(shifts, str):
                    self.load(im, shifts)
            elif os.path.exists(self.path):
                self.load(im, self.path)
            else:
                self.calulate_shifts(im)
                self.save()

    def __call__(self, channel, time, tracks=None, detectors=None):
        tracks = tracks or self.tracks
        detectors = detectors or self.detectors
        track, detector = tracks[channel], detectors[channel]
        if (track, detector, time) in self:
            return self[track, detector, time]
        elif (0, detector, time) in self:
            return self[0, detector, time]
        else:
            return Transform()

    def __reduce__(self):
        return self.__class__, (self.impath, self.shifts)

    def load(self, im, file):
        self.shifts = np.loadtxt(file)
        self.shifts2transforms(im)

    def save(self, file=None):
        self.path = file or self.path
        np.savetxt(self.path, self.shifts)

    def calulate_shifts0(self, im):
        """ Calculate shifts relative to the first frame """
        im0 = im[:, 0, 0].squeeze().transpose(2, 0, 1)

        @parfor(range(1, im.shape[4]), (im, im0), desc='Calculating image shifts.')
        def fun(t, im, im0):
            return Transform(im0, im[:, 0, t].squeeze().transpose(2, 0, 1), 'translation')
        transforms = [Transform()] + fun
        self.shifts = np.array([t.parameters[4:] for t in transforms])
        self.setTransforms(transforms, im.transform)

    def calulate_shifts(self, im):
        """ Calculate shifts relative to the previous frame """
        @parfor(range(1, im.shape[4]), (im,), desc='Calculating image shifts.')
        def fun(t, im):
            return Transform(im[:, 0, t-1].squeeze().transpose(2, 0, 1), im[:, 0, t].squeeze().transpose(2, 0, 1),
                             'translation')
        transforms = [Transform()] + fun
        self.shifts = np.cumsum([t.parameters[4:] for t in transforms])
        self.setTransforms(transforms, im.transform)

    def shifts2transforms(self, im):
        self.setTransforms([Transform(np.array(((1, 0, s[0]), (0, 1, s[1]), (0, 0, 1))))
                            for s in self.shifts], im.transform)

    def setTransforms(self, shift_transforms, channel_transforms):
        for key, value in channel_transforms.items():
            for t, T in enumerate(shift_transforms):
                self[key[0], key[1], t] = T * channel_transforms[key]


class deque_dict(OrderedDict):
    def __init__(self, maxlen=None, *args, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __truncate__(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(False)

    def __setitem__(self, *args, **kwargs):
        super().__setitem__(*args, **kwargs)
        self.__truncate__()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.__truncate__()


def tolist(item):
    if isinstance(item, xmldata):
        return [item]
    elif hasattr(item, 'items'):
        return item
    elif isinstance(item, str):
        return [item]
    try:
        iter(item)
        return list(item)
    except TypeError:
        return list((item,))


class xmldata(OrderedDict):
    def __init__(self, elem):
        super(xmldata, self).__init__()
        if elem:
            if isinstance(elem, dict):
                self.update(elem)
            else:
                self.update(xmldata._todict(elem)[1])

    def re_search(self, reg, default=None, *args, **kwargs):
        return tolist(xmldata._output(xmldata._search(self, reg, True, default, *args, **kwargs)[1]))

    def search(self, key, default=None):
        return tolist(xmldata._output(xmldata._search(self, key, False, default)[1]))

    def re_search_all(self, reg, *args, **kwargs):
        K, V = xmldata._search_all(self, reg, True, *args, **kwargs)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    def search_all(self, key):
        K, V = xmldata._search_all(self, key, False)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    @staticmethod
    def _search(d, key, regex=False, default=None, *args, **kwargs):
        if isinstance(key, (list, tuple)):
            if len(key) == 1:
                key = key[0]
            else:
                for v in xmldata._search_all(d, key[0], regex, *args, **kwargs)[1]:
                    found, value = xmldata._search(v, key[1:], regex, default, *args, **kwargs)
                    if found:
                        return True, value
                return False, default

        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        return True, v
                    elif isinstance(v, dict):
                        found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                        if found:
                            return True, value
                    elif isinstance(v, (list, tuple)):
                        for w in v:
                            found, value = xmldata._search(w, key, regex, default, *args, **kwargs)
                            if found:
                                return True, value
                else:
                    found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                    if found:
                        return True, value
        return False, default

    @staticmethod
    def _search_all(d, key, regex=False, *args, **kwargs):
        K = []
        V = []
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        K.append(k)
                        V.append(v)
                    elif isinstance(v, dict):
                        q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                        K.extend([str(k) + '|' + i for i in q])
                        V.extend(w)
                    elif isinstance(v, (list, tuple)):
                        for j, val in enumerate(v):
                            q, w = xmldata._search_all(val, key, regex, *args, **kwargs)
                            K.extend([str(k) + '|' + str(j) + '|' + i for i in q])
                            V.extend(w)
                else:
                    q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                    K.extend([str(k) + '|' + i for i in q])
                    V.extend(w)
        return K, V

    @staticmethod
    def _enumdict(d):
        d2 = {}
        for k, v in d.items():
            idx = [int(i) for i in re.findall(r'(?<=:)\d+$', k)]
            if idx:
                key = re.findall(r'^.*(?=:\d+$)', k)[0]
                if key not in d2:
                    d2[key] = {}
                d2[key][idx[0]] = d['{}:{}'.format(key, idx[0])]
            else:
                d2[k] = v
        rec = False
        for k, v in d2.items():
            if [int(i) for i in re.findall(r'(?<=:)\d+$', k)]:
                rec = True
                break
        if rec:
            return xmldata._enumdict(d2)
        else:
            return d2

    @staticmethod
    def _unique_children(l):
        if l:
            keys, values = zip(*l)
            d = {}
            for k in set(keys):
                value = [v for m, v in zip(keys, values) if k == m]
                if len(value) == 1:
                    d[k] = value[0]
                else:
                    d[k] = value
            return d
        else:
            return {}

    @staticmethod
    def _todict(elem):
        d = {}
        if hasattr(elem, 'Key') and hasattr(elem, 'Value'):
            name = elem.Key.cdata
            d = elem.Value.cdata
            return name, d

        if hasattr(elem, '_attributes') and elem._attributes is not None and 'ID' in elem._attributes:
            name = elem._attributes['ID']
            elem._attributes.pop('ID')
        elif hasattr(elem, '_name'):
            name = elem._name
        else:
            name = 'none'

        if name == 'Value':
            if hasattr(elem, 'children') and len(elem.children):
                return xmldata._todict(elem.children[0])

        if hasattr(elem, 'children'):
            children = [xmldata._todict(child) for child in elem.children]
            children = xmldata._unique_children(children)
            if children:
                d = OrderedDict(d, **children)
        if hasattr(elem, '_attributes'):
            children = elem._attributes
            if children:
                d = OrderedDict(d, **children)
        if not len(d.keys()) and hasattr(elem, 'cdata'):
            return name, elem.cdata

        return name, xmldata._enumdict(d)

    @staticmethod
    def _output(s):
        if isinstance(s, dict):
            return xmldata(s)
        elif isinstance(s, (tuple, list)):
            return [xmldata._output(i) for i in s]
        elif not isinstance(s, str):
            return s
        elif len(s) > 1 and s[0] == '[' and s[-1] == ']':
            return [xmldata._output(i) for i in s[1:-1].split(', ')]
        elif re.search(r'^[-+]?\d+$', s):
            return int(s)
        elif re.search(r'^[-+]?\d?\d*\.?\d+([eE][-+]?\d+)?$', s):
            return float(s)
        elif s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return xmldata(value) if isinstance(value, dict) else value


class imread(metaclass=ABCMeta):
    """ class to read image files, while taking good care of important metadata,
            currently optimized for .czi files, but can open anything that bioformats can handle
        path: path to the image file
        optional:
        series: in case multiple experiments are saved in one file, like in .lif files
        transform: automatically correct warping between channels, need transforms.py among others
        drift: automatically correct for drift, only works if transform is not None or False
        beadfile: image file(s) with beads which can be used for correcting warp
        dtype: datatype to be used when returning frames
        meta: define metadata, used for pickle-ing

        NOTE: run imread.kill_vm() at the end of your script/program, otherwise python might not terminate

        modify images on the fly with a decorator function:
            define a function which takes an instance of this object, one image frame,
            and the coordinates c, z, t as arguments, and one image frame as return
            >> imread.frame_decorator = fun
            then use imread as usually

        Examples:
            >> im = imread('/DATA/lenstra_lab/w.pomp/data/20190913/01_YTL639_JF646_DefiniteFocus.czi')
            >> im
             << shows summary
            >> im.shape
             << (256, 256, 2, 1, 600)
            >> plt.imshow(im(1, 0, 100))
             << plots frame at position c=1, z=0, t=100 (python type indexing), note: round brackets; always 2d array
                with 1 frame
            >> data = im[:,:,0,0,:25]
             << retrieves 5d numpy array containing first 25 frames at c=0, z=0, note: square brackets; always 5d array
            >> plt.imshow(im.max(0, None, 0))
             << plots max-z projection at c=0, t=0
            >> len(im)
             << total number of frames
            >> im.pxsize
             << 0.09708737864077668 image-plane pixel size in um
            >> im.laserwavelengths
             << [642, 488]
            >> im.laserpowers
             << [0.02, 0.0005] in %

            See __init__ and other functions for more ideas.

        Subclassing:
            Subclass this class to add more file types. A subclass should always have at least the following methods:
                staticmethod _can_open(path): returns True when the subclass can open the image in path
                __metadata__(self): pulls some metadata from the file and do other format specific things, it needs to
                                    define a few properties, like shape, etc.
                __frame__(self, c, z, t): this should return a single frame at channel c, slice z and time t
                optional close(self): close the file in a proper way
                optional field priority: subclasses with lower priority will be tried first, default = 99
                Any other method can be overridden as needed
        wp@tl2019-2021
    """
    priority = 99

    @staticmethod
    @abstractmethod
    def _can_open(path):  # Override this method, and return true when the subclass can open the file
        return False

    @abstractmethod
    def __metadata__(self):
        return

    @abstractmethod
    def __frame__(self, c, z, t):  # Override this, return the frame at c, z, t
        return np.random.randint(0, 255, self.shape[:2])

    def close(self):
        return

    def __new__(cls, path, *args, **kwargs):
        if cls is not imread:
            return super().__new__(cls)
        if len(cls.__subclasses__()) == 0:
            raise Exception('Restart python kernel please!')
        if isinstance(path, imread):
            path = path.path
        for subclass in sorted(cls.__subclasses__(), key=lambda subclass: subclass.priority):
            if subclass._can_open(path):
                return super().__new__(subclass)

    def __init__(self, path, series=0, transform=False, drift=False, beadfile=None, sigma=None, dtype=None, meta=None):
        if isinstance(path, str):
            self.path = os.path.abspath(path)
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.acquisitiondate = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime('%y-%m-%dT%H:%M:%S')
        else:
            self.path = path  # ndarray
        self.transform = transform
        self.drift = drift
        self.beadfile = beadfile
        self.dtype = dtype
        self.shape = (0, 0, 0, 0, 0)
        self.series = series
        self.meta = meta
        self.pxsize = 1e-1
        self.settimeinterval = 0
        self.pxsizecam = 0
        self.magnification = 0
        self.exposuretime = (0,)
        self.deltaz = 1
        self.pcf = (1, 1)
        self.laserwavelengths = [[]]
        self.laserpowers = [[]]
        self.powermode = 'normal'
        self.optovar = (1,)
        self.binning = 1
        self.collimator = (1,)
        self.tirfangle = (0,)
        self.gain = (100, 100)
        self.objective = 'unknown'
        self.filter = 'unknown'
        self.NA = 1
        self.cyllens = ['None', 'None']
        self.duolink = 'None'
        self.detector = [0, 1]
        self.track = [0]
        self.metadata = {}
        self.cache = deque_dict(16)
        self._frame_decorator = None
        self.frameoffset = (self.shape[0] / 2, self.shape[1] / 2)  # how far apart the centers of frame and sensor are

        self.__metadata__()

        if not hasattr(self, 'cnamelist'):
            self.cnamelist = 'abcdefghijklmnopqrstuvwxyz'[:self.shape[2]]

        if self.meta is not None:
            for key, item in self.meta.items():
                self.__dict__[key] = item

        m = self.extrametadata
        if m is not None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                if 'FeedbackChannels' in m:
                    self.feedback = m['FeedbackChannels']
                else:
                    self.feedback = m['FeedbackChannel']
            except Exception:
                self.cyllens = ['None', 'None']
                self.duolink = 'None'
                self.feedback = []
        try:
            self.cyllenschannels = np.where([self.cyllens[self.detector[c]].lower() != 'none'
                                             for c in range(self.shape[2])])[0].tolist()
        except Exception:
            pass
        self.set_transform()
        try:
            s = int(re.findall(r'_(\d{3})_', self.duolink)[0])
        except Exception:
            s = 561
        if sigma is None:
            try:
                sigma = []
                for t, d in zip(self.track, self.detector):
                    l = np.array(self.laserwavelengths[t]) + 22
                    sigma.append([l[l > s].max(initial=0), l[l < s].max(initial=0)][d])
                sigma = np.array(sigma)
                sigma[sigma == 0] = 600
                sigma /= 2 * self.NA * self.pxsize * 1000
                self.sigma = sigma.tolist()
            except Exception:
                self.sigma = [2] * self.shape[2]
        else:
            self.sigma = sigma
        if 1.5 < self.NA:
            self.immersionN = 1.661
        elif 1.3 < self.NA < 1.5:
            self.immersionN = 1.518
        elif 1 < self.NA < 1.3:
            self.immersionN = 1.33
        else:
            self.immersionN = 1

    @cached_property
    def timeseries(self):
        return self.shape[4] > 1

    @cached_property
    def zstack(self):
        return self.shape[3] > 1

    def set_transform(self):
        # handle transforms
        if self.transform is False or self.transform is None:
            self.transform = None
        else:
            if isinstance(self.transform, Transforms):
                self.transform = self.transform
            else:
                if isinstance(self.transform, str):
                    self.transform = ImTransforms(self.path, self.cyllens, self.track, self.detector, self.transform)
                else:
                    self.transform = ImTransforms(self.path, self.cyllens, self.track, self.detector, self.beadfile)
                if self.drift is True:
                    self.transform = ImShiftTransforms(self)
                elif not (self.drift is False or self.drift is None):
                    self.transform = ImShiftTransforms(self, self.drift)
            self.transform.adapt(self.frameoffset, self.shape)
            self.beadfile = self.transform.files

    def __framet__(self, c, z, t):
        return self.transform_frame(self.__frame__(c, z, t), c, t)

    def new(self, **kwargs):
        c, a = self.__reduce__()
        new_kwargs = {key: value for key, value in zip(inspect.getfullargspec(c).args[1:], a)}
        for key, value in kwargs.items():
            new_kwargs[key] = value
        return c(**new_kwargs)

    @staticmethod
    def getConfig(file):
        """ Open a yml parameter file
        """
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
        with open(file, 'r') as f:
            return yaml.load(f, loader)

    @staticmethod
    def kill_vm():
        jvm().kill_vm()

    @property
    def frame_decorator(self):
        return self._frame_decorator

    @frame_decorator.setter
    def frame_decorator(self, decorator):
        self._frame_decorator = decorator
        self.cache = deque_dict(self.cache.maxlen)

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            res = self(self.index)
            self.index += 1
            return res

    def __repr__(self):
        """ gives a helpful summary of the recorded experiment
        """
        s = [100 * '#']
        s.append('path/filename: {}'.format(self.path))
        s.append('shape (xyczt): {} x {} x {} x {} x {}'.format(*self.shape))
        s.append('pixelsize:     {:.2f} nm'.format(self.pxsize * 1000))
        if self.zstack:
            s.append('z-interval:    {:.2f} nm'.format(self.deltaz * 1000))
        s.append('Exposuretime:  ' + ('{:.2f} ' * len(self.exposuretime)).format(
            *(np.array(self.exposuretime) * 1000)) + 'ms')
        if self.timeseries:
            if self.timeval and np.diff(self.timeval).shape[0]:
                s.append('t-interval:    {:.3f} ?? {:.3f} s'.format(
                    np.diff(self.timeval).mean(), np.diff(self.timeval).std()))
            else:
                s.append('t-interval:    {:.2f} s'.format(self.settimeinterval))
        s.append('binning:       {}x{}'.format(self.binning, self.binning))
        s.append('laser colors:  ' + ' | '.join([' & '.join(len(l)*('{:.0f}',)).format(*l)
                                                 for l in self.laserwavelengths]) + ' nm')
        s.append('laser powers:  ' + ' | '.join([' & '.join(len(l)*('{}',)).format(*[100 * i for i in l])
                                                 for l in self.laserpowers]) + ' %')
        s.append('objective:     {}'.format(self.objective))
        s.append('magnification: {}x'.format(self.magnification))
        s.append('optovar:      ' + (' {}' * len(self.optovar)).format(*self.optovar) + 'x')
        s.append('filterset:     {}'.format(self.filter))
        s.append('powermode:     {}'.format(self.powermode))
        s.append('collimator:   ' + (' {}' * len(self.collimator)).format(*self.collimator))
        s.append('TIRF angle:   ' + (' {:.2f}??' * len(self.tirfangle)).format(*self.tirfangle))
        s.append('gain:         ' + (' {:.0f}' * len(self.gain)).format(*self.gain))
        s.append('pcf:          ' + (' {:.2f}' * len(self.pcf)).format(*self.pcf))
        return '\n'.join(s)

    def __str__(self):
        return self.path

    def __len__(self):
        return self.shape[2] * self.shape[3] * self.shape[4]

    def __call__(self, *n):
        """ returns single 2D frame
            im(n):     index linearly in czt order
            im(c,z):   return im(c,z,t=0)
            im(c,z,t): return im(c,z,t)
        """
        if len(n) == 1:
            n = self.get_channel(n[0])
            c = int(n % self.shape[2])
            z = int((n // self.shape[2]) % self.shape[3])
            t = int((n // (self.shape[2] * self.shape[3])) % self.shape[4])
            return self.frame(c, z, t)
        else:
            return self.frame(*[int(i) for i in n])

    def __getitem__(self, n):
        """ returns sliced 5D block
            im[n]:     index linearly in czt order
            im[c,z]:   return im(c,z,t=0)
            im[c,z,t]: return im(c,z,t)
            RESULT IS ALWAYS 5D!
        """
        if isinstance(n, slice):
            n = (n,)
        if isinstance(n, type(Ellipsis)):
            n = (None,) * 5
        if isinstance(n, Number):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            n = (c, z, t)
        n = list(n)

        ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
        if len(ell) > 1:
            raise IndexError("an index can only have a single ellipsis (...)")
        if len(ell):
            if len(n) > 5:
                n.remove(Ellipsis)
            else:
                n[ell[0]] = None
                while len(n) not in (3, 5):
                    n.insert(ell[0], None)
        while len(n) not in (3, 5):
            n.append(0)
        while len(n) < 5:
            n.insert(0, None)

        for i, e in enumerate(n):
            if e is None:
                e = slice(None)
            if isinstance(e, (slice, range)):
                a = [e.start, e.stop, e.step]
                if a[0] is None:
                    a[0] = 0
                if a[1] is None:
                    a[1] = self.shape[i]
                if a[2] is None:
                    a[2] = 1
                for j in range(2):
                    if a[j] < 0:
                        a[j] %= self.shape[i]
                n[i] = np.arange(*a, dtype=int)
        return self.block(*[np.array(i, int) for i in n])

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if hasattr(self, 'close'):
            self.close()

    def __reduce__(self):
        return self.__class__, (self.path, self.series, self.transform, self.drift, self.beadfile, self.sigma,
                                self.dtype)

    def czt(self, n):
        """ returns indices c, z, t used when calling im(n)
        """
        if not isinstance(n, tuple):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return c, z, t
        n = list(n)
        if len(n) == 2 or len(n) == 4:
            n.append(slice(0, -1, 1))
        if len(n) == 3:
            n = list(n)
            for i, e in enumerate(n):
                if isinstance(e, slice):
                    a = [e.start, e.stop, e.step]
                    if a[0] is None:
                        a[0] = 0
                    if a[1] is None:
                        a[1] = -1
                    if a[2] is None:
                        a[2] = 1
                    for j in range(2):
                        if a[j] < 0:
                            a[j] %= self.shape[2 + i]
                            a[j] += 1
                    n[i] = np.arange(*a)
            n = [np.array(i) for i in n]
            return tuple(n)
        if len(n) == 5:
            return tuple(n[2:5])

    def czt2n(self, c, z, t):
        return c + z * self.shape[2] + t * self.shape[2] * self.shape[3]

    def transform_frame(self, frame, c, t=0):
        if self.transform is None:
            return frame
        else:
            return self.transform(c, t, self.track, self.detector).frame(frame)

    def get_czt(self, c, z, t):
        czt = []
        for i, n in enumerate((c, z, t), 2):
            if n is None:
                czt.append(list(range(self.shape[i])))
            elif isinstance(n, range):
                if n.stop < 0:
                    stop = n.stop % self.shape[i]
                elif n.stop > self.shape[i]:
                    stop = self.shape[i]
                else:
                    stop = n.stop
                czt.append(list(range(n.start % self.shape[i], stop, n.step)))
            elif isinstance(n, Number):
                czt.append([n % self.shape[i]])
            else:
                czt.append([k % self.shape[i] for k in n])
        return [self.get_channel(c) for c in czt[0]], *czt[1:]

    def _stats(self, fun, c=None, z=None, t=None, ffun=None):
        """ fun = np.min, np.max, np.sum or their nan varieties """
        c, z, t = self.get_czt(c, z, t)
        if fun in (np.min, np.nanmin):
            val = np.inf
        elif fun in (np.max, np.nanmax):
            val = -np.inf
        else:
            val = 0
        if ffun is None:
            ffun = lambda im: im
        T = np.full(self.shape[:2], val, self.dtype)
        for ic in c:
            m = np.full(self.shape[:2], val, self.dtype)
            if isinstance(self.transform, ImShiftTransforms):
                for it in t:
                    n = np.full(self.shape[:2], val, self.dtype)
                    for iz in z:
                        n = fun((n, ffun(self.__frame__(ic, iz, it))), 0)
                    m = self.transform_frame(n, ic, it)
            else:
                for it, iz in product(t, z):
                    m = fun((m, ffun(self.__frame__(ic, iz, it))), 0)
                if isinstance(self.transform, ImTransforms):
                    m = self.transform_frame(m, ic, 0)
            T = fun((T, m), 0)
        return T

    def sum(self, c=None, z=None, t=None):
        return self._stats(np.sum, c, z, t)

    def nansum(self, c=None, z=None, t=None):
        return self._stats(np.nansum, c, z, t)

    def min(self, c=None, z=None, t=None):
        return self._stats(np.min, c, z, t).astype(self.dtype)

    def nanmin(self, c=None, z=None, t=None):
        return self._stats(np.nanmin, c, z, t).astype(self.dtype)

    def max(self, c=None, z=None, t=None):
        return self._stats(np.max, c, z, t).astype(self.dtype)

    def nanmax(self, c=None, z=None, t=None):
        return self._stats(np.nanmax, c, z, t).astype(self.dtype)

    def mean(self, c=None, z=None, t=None):
        return self.sum(c, z, t) / np.prod([len(i) for i in self.get_czt(c, z, t)])

    def nanmean(self, c=None, z=None, t=None):
        return self.nansum(c, z, t) / (np.prod([len(i) for i in self.get_czt(c, z, t)])
                                       - self._stats(np.sum, c, z, t, lambda im: np.isnan(im)))

    def isnoise(self, c=None, z=None, t=None):
        """ True is volume c, z, t only has noise
        """
        if isinstance(c, np.ndarray) and c.ndim >= 2:
            a = c
        else:
            c, z, t = [slice(-1) if i is None else i for i in (c, z, t)]
            a = self[c, z, t].squeeze()
        F = np.fft.fftn(a)
        S = np.fft.fftshift(np.fft.ifftn(F * F.conj()).real / np.sum(a ** 2))
        return -np.log(1 - S[tuple([0] * S.ndim)]) > 5

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    def get_channel(self, channel_name):
        if not isinstance(channel_name, str):
            return channel_name
        else:
            c = [i for i, c in enumerate(self.cnamelist) if c.lower().startswith(channel_name.lower())]
            assert len(c) > 0, 'Channel {} not found in {}'.format(c, self.cnamelist)
            assert len(c) < 2, 'Channel {} not unique in {}'.format(c, self.cnamelist)
            return c[0]

    def frame(self, c=0, z=0, t=0):
        """ returns single 2D frame
        """
        c = self.get_channel(c)
        c %= self.shape[2]
        z %= self.shape[3]
        t %= self.shape[4]

        # cache last n (default 16) frames in memory for speed (~250x faster)
        if (c, z, t) in self.cache:
            self.cache.move_to_end((c, z, t))
            f = self.cache[(c, z, t)]
        else:
            f = self.__framet__(c, z, t)
            if self.frame_decorator is not None:
                f = self.frame_decorator(self, f, c, z, t)
            self.cache[(c, z, t)] = f
        if self.dtype is not None:
            return f.copy().astype(self.dtype)
        else:
            return f.copy()

    def data(self, c=0, z=0, t=0):
        """ returns 3D stack of frames
        """
        c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1) for i, e in enumerate((c, z, t), 2)]
        return np.dstack([self.frame(ci, zi, ti) for ci, zi, ti in product(c, z, t)])

    def block(self, x=None, y=None, c=None, z=None, t=None):
        """ returns 5D block of frames
        """
        x, y, c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1)
                         for i, e in enumerate((x, y, c, z, t))]
        d = np.full((len(x), len(y), len(c), len(z), len(t)), np.nan)
        for (ci, cj), (zi, zj), (ti, tj) in product(enumerate(c), enumerate(z), enumerate(t)):
            d[:, :, ci, zi, ti] = self.frame(cj, zj, tj)[x][:, y]
        return d

    @cached_property
    def timeval(self):
        if hasattr(self, 'metadata') and isinstance(self.metadata, xmldata):
            image = self.metadata.search('Image')
            if (isinstance(image, dict) and self.series in image) or (isinstance(image, list) and len(image)):
                image = xmldata(image[0])
            return sorted(np.unique(image.search_all('DeltaT').values()))[:self.shape[4]]
        else:
            return (np.arange(self.shape[4]) * self.settimeinterval).tolist()

    @cached_property
    def timeinterval(self):
        return float(np.diff(self.timeval).mean()) if len(self.timeval) > 1 else 1

    @cached_property
    def piezoval(self):
        """ gives the height of the piezo and focus motor, only available when CylLensGUI was used
        """
        def upack(idx):
            time = list()
            val = list()
            if len(idx) == 0:
                return time, val
            for i in idx:
                time.append(int(re.search(r'\d+', n[i]).group(0)))
                val.append(w[i])
            return zip(*sorted(zip(time, val)))

        # Maybe the values are stored in the metadata
        n = self.metadata.search('LsmTag|Name')[0]
        w = self.metadata.search('LsmTag')[0]
        if not n is None:
            # n = self.metadata['LsmTag|Name'][1:-1].split(', ')
            # w = str2float(self.metadata['LsmTag'][1:-1].split(', '))

            pidx = np.where([re.search(r'^Piezo\s\d+$', x) is not None for x in n])[0]
            sidx = np.where([re.search(r'^Zstage\s\d+$', x) is not None for x in n])[0]

            ptime, pval = upack(pidx)
            stime, sval = upack(sidx)

        # Or maybe in an extra '.pzl' file
        else:
            m = self.extrametadata
            if m is not None and 'p' in m:
                q = np.array(m['p'])
                if not len(q.shape):
                    q = np.zeros((1, 3))

                ptime = [int(i) for i in q[:, 0]]
                pval = [float(i) for i in q[:, 1]]
                sval = [float(i) for i in q[:, 2]]

            else:
                ptime = []
                pval = []
                sval = []

        df = pandas.DataFrame(columns=['frame', 'piezoZ', 'stageZ'])
        df['frame'] = ptime
        df['piezoZ'] = pval
        df['stageZ'] = np.array(sval) - np.array(pval) - \
                       self.metadata.re_search(r'AcquisitionModeSetup\|ReferenceZ', 0)[0] * 1e6

        # remove duplicates
        df = df[~df.duplicated('frame', 'last')]
        return df

    @cached_property
    def extrametadata(self):
        if isinstance(self.path, str) and len(self.path) > 3:
            if os.path.isfile(self.path[:-3] + 'pzl2'):
                pname = self.path[:-3] + 'pzl2'
            elif os.path.isfile(self.path[:-3] + 'pzl'):
                pname = self.path[:-3] + 'pzl'
            else:
                return
            try:
                return self.getConfig(pname)
            except Exception:
                return
        return

    def save_as_tiff(self, fname=None, c=None, z=None, t=None, split=False, bar=True, pixel_type='uint16'):
        """ saves the image as a tiff-file
            split: split channels into different files
        """
        if fname is None:
            if isinstance(self.path, str):
                fname = self.path[:-3] + 'tif'
            else:
                raise Exception('No filename given.')
        elif not fname[-3:] == 'tif':
            fname += '.tif'
        if split:
            for i in range(self.shape[2]):
                if self.timeseries:
                    self.save_as_tiff(fname[:-3] + 'C{:01d}.tif'.format(i), i, 0, None, False, bar, pixel_type)
                else:
                    self.save_as_tiff(fname[:-3] + 'C{:01d}.tif'.format(i), i, None, 0, False, bar, pixel_type)
        else:
            n = [c, z, t]
            for i in range(len(n)):
                if n[i] is None:
                    n[i] = range(self.shape[i + 2])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            shape = [len(i) for i in n]
            at_least_one = False
            with IJTiffFile(fname, shape, pixel_type, pxsize=self.pxsize, deltaz=self.deltaz) as tif:
                for i, m in tqdm(zip(product(*[range(s) for s in shape]), product(*n)),
                                 total=np.prod(shape), desc='Saving tiff', disable=not bar):
                    if np.any(self(*m)) or not at_least_one:
                        tif.save(self(*m), *i)
                        at_least_one = True

    @cached_property
    def summary(self):
        return self.__repr__()


class cziread(imread):
    priority = 0

    @staticmethod
    def _can_open(path):
        return isinstance(path, str) and path.endswith('.czi')

    def __metadata__(self):
        # TODO: make sure frame function still works when a subblock has data from more than one frame
        self.reader = czifile.CziFile(self.path)
        # self.reader.asarray()
        self.shape = tuple([self.reader.shape[self.reader.axes.index(directory_entry)] for directory_entry in 'XYCZT'])

        filedict = {}
        for directory_entry in self.reader.filtered_subblock_directory:
            idx = self.get_index(directory_entry, self.reader.start)
            for c in range(*idx[self.reader.axes.index('C')]):
                for z in range(*idx[self.reader.axes.index('Z')]):
                    for t in range(*idx[self.reader.axes.index('T')]):
                        if (c, z, t) in filedict:
                            filedict[(c, z, t)].append(directory_entry)
                        else:
                            filedict[(c, z, t)] = [directory_entry]
        self.filedict = filedict
        self.metadata = xmldata(untangle.parse(self.reader.metadata()))

        image = [i for i in self.metadata.search_all('Image').values() if i]
        if len(image) and self.series in image[0]:
            image = xmldata(image[0][self.series])
        else:
            image = self.metadata

        pxsize = image.search('ScalingX')[0]
        if pxsize is not None:
            self.pxsize = pxsize * 1e6
        if self.zstack:
            deltaz = image.search('ScalingZ')[0]
            if deltaz is not None:
                self.deltaz = deltaz * 1e6

        self.title = self.metadata.re_search(('Information', 'Document', 'Name'), self.title)[0]
        self.acquisitiondate = self.metadata.re_search(('Information', 'Document', 'CreationDate'),
                                                       self.acquisitiondate)[0]
        self.exposuretime = self.metadata.re_search(('TrackSetup', 'CameraIntegrationTime'), self.exposuretime)
        if self.timeseries:
            self.settimeinterval = self.metadata.re_search(('Interval', 'TimeSpan', 'Value'),
                                                           self.settimeinterval * 1e3)[0] / 1000
            if not self.settimeinterval:
                self.settimeinterval = self.exposuretime[0]
        self.pxsizecam = self.metadata.re_search(('AcquisitionModeSetup', 'PixelPeriod'), self.pxsizecam)
        self.magnification = self.metadata.re_search('NominalMagnification', self.magnification)[0]
        attenuators = self.metadata.search_all('Attenuator')
        self.laserwavelengths = [[1e9 * float(i['Wavelength']) for i in tolist(attenuator)]
                                 for attenuator in attenuators.values()]
        self.laserpowers = [[float(i['Transmission']) for i in tolist(attenuator)]
                            for attenuator in attenuators.values()]
        self.collimator = self.metadata.re_search(('Collimator', 'Position'))
        detector = self.metadata.search(('Instrument', 'Detector'))
        self.gain = [int(i.get('AmplificationGain', 1)) for i in detector]
        self.powermode = self.metadata.re_search(('TrackSetup', 'FWFOVPosition'))[0]
        optovar = self.metadata.re_search(('TrackSetup', 'TubeLensPosition'), '1x')
        self.optovar = []
        for o in optovar:
            a = re.search(r'\d?\d*[,.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))
        self.pcf = [2 ** self.metadata.re_search(('Image', 'ComponentBitCount'), 14)[0] / float(i)
                    for i in self.metadata.re_search(('Channel', 'PhotonConversionFactor'), 1)]
        self.binning = self.metadata.re_search(('AcquisitionModeSetup', 'CameraBinning'), 1)[0]
        self.objective = self.metadata.re_search(('AcquisitionModeSetup', 'Objective'))[0]
        self.NA = self.metadata.re_search(('Instrument', 'Objective', 'LensNA'))[0]
        self.filter = self.metadata.re_search(('TrackSetup', 'BeamSplitter', 'Filter'))[0]
        self.tirfangle = [50 * i for i in self.metadata.re_search(('TrackSetup', 'TirfAngle'), 0)]
        self.frameoffset = [self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetX'))[0],
                            self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetY'))[0]]
        self.cnamelist = [c['DetectorSettings']['Detector']['Id'] for c in
                          self.metadata['ImageDocument']['Metadata']['Information'].search('Channel')]
        try:
            self.track, self.detector = zip(*[[int(i) for i in re.findall(r'\d', c)] for c in self.cnamelist])
        except ValueError:
            self.track = tuple(range(len(self.cnamelist)))
            self.detector = (0,) * len(self.cnamelist)

    def __frame__(self, c=0, z=0, t=0):
        f = np.zeros(self.shape[:2], self.dtype)
        for directory_entry in self.filedict[(c, z, t)]:
            subblock = directory_entry.data_segment()
            tile = subblock.data(resize=True, order=0)
            index = [slice(i - j, i - j + k) for i, j, k in
                     zip(directory_entry.start, self.reader.start, tile.shape)]
            index = tuple([index[self.reader.axes.index(i)] for i in 'XY'])
            f[index] = tile.squeeze()
        return f

    def close(self):
        self.reader.close()

    @staticmethod
    def get_index(directory_entry, start):
        return [(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, directory_entry.shape)]

    @cached_property
    def timeval(self):
        tval = np.unique(list(filter(lambda x: x.attachment_entry.filename.startswith('TimeStamp'),
                                     self.reader.attachments()))[0].data())
        return sorted(tval[tval > 0])[:self.shape[4]]


class seqread(imread):
    priority = 10

    @staticmethod
    def _can_open(path):
        return isinstance(path, str) and os.path.splitext(path)[1] == ''

    def __metadata__(self):
        filelist = sorted([file for file in os.listdir(self.path) if re.search(r'^img_\d{3,}.*\d{3,}.*\.tif$', file)])

        try:
            with tifffile.TiffFile(os.path.join(self.path, filelist[0])) as tif:
                self.metadata = xmldata({key: yaml.safe_load(value)
                                         for key, value in tif.pages[0].tags[50839].value.items()})
        except Exception:  # fallback
            with open(os.path.join(self.path, 'metadata.txt'), 'r') as metadatafile:
                self.metadata = xmldata(json.loads(metadatafile.read()))

        # compare channel names from metadata with filenames
        cnamelist = self.metadata.search('ChNames')
        cnamelist = [c for c in cnamelist if any([c in f for f in filelist])]

        self.filedict = {}
        maxc = 0
        maxz = 0
        maxt = 0
        for file in filelist:
            T = re.search(r'(?<=img_)\d{3,}', file)
            Z = re.search(r'\d{3,}(?=\.tif$)', file)
            C = file[T.end() + 1:Z.start() - 1]
            t = int(T.group(0))
            z = int(Z.group(0))
            if C in cnamelist:
                c = cnamelist.index(C)
            else:
                c = len(cnamelist)
                cnamelist.append(C)

            self.filedict[(c, z, t)] = file
            if c > maxc:
                maxc = c
            if z > maxz:
                maxz = z
            if t > maxt:
                maxt = t
        self.cnamelist = [str(cname) for cname in cnamelist]

        X = self.metadata.search('Width')[0]
        Y = self.metadata.search('Height')[0]
        self.shape = (int(X), int(Y), maxc + 1, maxz + 1, maxt + 1)

        self.pxsize = self.metadata.re_search(r'(?i)pixelsize_?um', 0)[0]
        if self.zstack:
            self.deltaz = self.metadata.re_search(r'(?i)z-step_?um', 0)[0]
        if self.timeseries:
            self.settimeinterval = self.metadata.re_search(r'(?i)interval_?ms', 0)[0] / 1000
        if 'Hamamatsu' in self.metadata.search('Core-Camera', '')[0]:
            self.pxsizecam = 6.5
        self.title = self.metadata.search('Prefix')[0]
        self.acquisitiondate = self.metadata.search('Time')[0]
        self.exposuretime = [i / 1000 for i in self.metadata.search('Exposure-ms')]
        self.objective = self.metadata.search('ZeissObjectiveTurret-Label')[0]
        self.optovar = []
        for o in self.metadata.search('ZeissOptovar-Label'):
            a = re.search(r'\d?\d*[,.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))
        if self.pxsize == 0:
            self.magnification = int(re.findall(r'(\d+)x', self.objective)[0]) * self.optovar[0]
            self.pxsize = self.pxsizecam / self.magnification
        else:
            self.magnification = self.pxsizecam / self.pxsize
        self.pcf = self.shape[2] * self.metadata.re_search(r'(?i)conversion\sfactor\scoeff', 1)
        self.filter = self.metadata.search('ZeissReflectorTurret-Label', self.filter)[0]
        self.track = [0] * self.shape[2]
        self.detector = list(range(self.shape[2]))

    def __frame__(self, c=0, z=0, t=0):
        return tifffile.imread(os.path.join(self.path, self.filedict[(c, z, t)]))


class bfread(imread):
    """ This class is used as a last resort, when we don't have another way to open the file. We don't like it because
        it requires the java vm.
    """
    priority = 99  # panic and open with BioFormats

    @staticmethod
    def _can_open(path):
        return True

    def __metadata__(self):
        jvm().start_vm()  # We need java for this :(
        self.key = np.random.randint(1e9)
        self.reader = bioformats.get_image_reader(self.key, self.path)
        omexml = bioformats.get_omexml_metadata(self.path)
        self.metadata = xmldata(untangle.parse(omexml))

        s = self.reader.rdr.getSeriesCount()
        if self.series >= s:
            print('Series {} does not exist.'.format(self.series))
        self.reader.rdr.setSeries(self.series)

        X = self.reader.rdr.getSizeX()
        Y = self.reader.rdr.getSizeY()
        C = self.reader.rdr.getSizeC()
        Z = self.reader.rdr.getSizeZ()
        T = self.reader.rdr.getSizeT()
        self.shape = (X, Y, C, Z, T)

        image = list(self.metadata.search_all('Image').values())
        if len(image) and self.series in image[0]:
            image = xmldata(image[0][self.series])
        else:
            image = self.metadata

        unit = lambda u: 10 ** {'nm': 9, '??m': 6, 'um': 6, 'mm': 3, 'm': 0}[u]

        pxsizeunit = image.search('PhysicalSizeXUnit')[0]
        pxsize = image.search('PhysicalSizeX')[0]
        if pxsize is not None:
            self.pxsize = pxsize / unit(pxsizeunit) * 1e6

        if self.zstack:
            deltazunit = image.search('PhysicalSizeZUnit')[0]
            deltaz = image.search('PhysicalSizeZ')[0]
            if deltaz is not None:
                self.deltaz = deltaz / unit(deltazunit) * 1e6

        if not isinstance(self, bfread):
            self.title = self.metadata.search('Name')[0]

        if self.path.endswith('.lif'):
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.exposuretime = self.metadata.re_search(r'WideFieldChannelInfo\|ExposureTime', self.exposuretime)
            if self.timeseries:
                self.settimeinterval = \
                    self.metadata.re_search(r'ATLCameraSettingDefinition\|CycleTime', self.settimeinterval * 1e3)[
                        0] / 1000
                if not self.settimeinterval:
                    self.settimeinterval = self.exposuretime[0]
            self.pxsizecam = self.metadata.re_search(r'ATLCameraSettingDefinition\|TheoCamSensorPixelSizeX',
                                                     self.pxsizecam)
            self.objective = self.metadata.re_search(r'ATLCameraSettingDefinition\|ObjectiveName', 'none')[0]
            self.magnification = \
                self.metadata.re_search(r'ATLCameraSettingDefinition\|Magnification', self.magnification)[0]
        elif self.path.endswith('.ims'):
            self.magnification = self.metadata.search('LensPower', 100)[0]
            self.NA = self.metadata.search('NumericalAperture', 1.47)[0]
            self.title = self.metadata.search('Name', self.title)
            self.binning = self.metadata.search('BinningX', 1)[0]

    def __frame__(self, *args):
        frame = self.reader.read(*args, rescale=False).astype('float')
        if frame.ndim == 3:
            return frame[..., args[0]]
        else:
            return frame

    def close(self):
        bioformats.release_image_reader(self.key)


class ndread(imread):
    priority = 20

    @staticmethod
    def _can_open(path):
        return isinstance(path, np.ndarray) and path.ndim in (2, 3, 5)

    def __metadata__(self):
        assert isinstance(self.path, np.ndarray), 'Not a numpy array'
        if np.ndim(self.path) == 5:
            self.shape = self.path.shape
        elif np.ndim(self.path) == 3:
            self.shape = (self.path.shape[:2]) + (1, 1, self.path.shape[2])
        elif np.ndim(self.path) == 2:
            self.shape = self.path.shape + (1, 1, 1)
        self.title = 'numpy array'
        self.acquisitiondate = 'now'

    def __frame__(self, c, z, t):
        if self.path.ndim == 5:
            return self.path[:, :, c, z, t]
        elif self.path.ndim == 3:
            return self.path[:, :, t]
        else:
            return self.path

    def __repr__(self):
        path, self.path = self.path, 'numpy array'
        lines, self.path = super().__repr__(), path
        return lines

    def __str__(self):
        return self.path.__str__()

    @property
    def frame_decorator(self):
        return self._frame_decorator

    @frame_decorator.setter
    def frame_decorator(self, decorator):
        if 'origcache' not in self:
            self.origcache = self.cache
        if decorator is None:
            self.cache = self.origcache
        else:
            for k, v in self.origcache.items():
                self.cache[k] = decorator(self, v, *k)


class tiffread(imread):
    priority = 0

    @staticmethod
    def _can_open(path):
        if isinstance(path, str) and (path.endswith('.tif') or path.endswith('.tiff')):
            with tifffile.TiffFile(path) as tif:
                return tif.is_imagej
        else:
            return False

    def __metadata__(self):
        self.tif = tifffile.TiffFile(self.path)
        self.metadata = self.tif.imagej_metadata
        P = self.tif.pages[0]
        self.pndim = P.ndim
        X = P.imagelength
        Y = P.imagewidth
        if self.pndim == 3:
            C = P.samplesperpixel
            self.transpose = [i for i in [P.axes.find(j) for j in 'SYX'] if i >= 0]
            T = self.metadata.get('frames', 1)  # // C
        else:
            C = self.metadata.get('channels', 1)
            T = self.metadata.get('frames', 1)
        Z = self.metadata.get('slices', 1)
        self.shape = (X, Y, C, Z, T)
        # TODO: more metadata

    def __frame__(self, c, z, t):
        if self.pndim == 3:
            return np.transpose(self.tif.asarray(z + t * self.shape[3]), self.transpose)[c]
        else:
            return self.tif.asarray(c + z * self.shape[2] + t * self.shape[2] * self.shape[3])

    def close(self):
        self.tif.close()


def main():
    parser = ArgumentParser(description='Display info and save as tif')
    parser.add_argument('file', help='image_file')
    parser.add_argument('out', help='path to tif out', type=str, default=None, nargs='?')
    parser.add_argument('-r', '--register', help='register channels', action='store_true')
    parser.add_argument('-c', '--channel', help='channel', type=int, default=None)
    parser.add_argument('-z', '--zslice', help='z-slice', type=int, default=None)
    parser.add_argument('-t', '--time', help='time', type=int, default=None)
    parser.add_argument('-s', '--split', help='split channels', action='store_true')
    parser.add_argument('-f', '--force', help='force overwrite', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.file):
        with imread(args.file, transform=args.register) as im:
            print(im.summary)
            if args.out:
                out = os.path.abspath(args.out)
                if not os.path.exists(os.path.dirname(out)):
                    os.makedirs(os.path.dirname(out))
                if os.path.exists(out) and not args.force:
                    print('File {} exists already, add the -f flag if you want to overwrite it.'.format(args.out))
                else:
                    im.save_as_tiff(out, args.channel, args.zslice, args.time, args.split)
    else:
        print('File does not exist.')
