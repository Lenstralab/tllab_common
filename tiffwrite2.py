import sys
import struct
import numpy as np
from io import BytesIO
from multiprocessing import Pool, Process, Queue, Event, cpu_count, Value, queues
from time import sleep
from tqdm.auto import tqdm
import tifffile
import colorcet
from itertools import product


def get_colormap(colormap, dtype='int8', byteorder='<'):
    colormap = getattr(colorcet, colormap)
    colormap[0] = '#ffffff'
    colormap[-1] = '#000000'
    colormap = 65535 * np.array([[int(''.join(i), 16) for i in zip(*[iter(s[1:])] * 2)] for s in colormap]) // 255
    if np.dtype(dtype).itemsize == 2:
        colormap = np.tile(colormap, 256).reshape((-1, 3))
    return b''.join([struct.pack(byteorder + 'H', c) for c in colormap.T.flatten()])


def tiffwrite(file, data, axes='TZCXY', bar=False, colormap=None):
    """ Saves an image using the bigtiff format, openable in ImageJ as hyperstack
        Uses multiple processes to quickly compress as good as possible
        file: filename of the new tiff file
        data: 2 to 5D array
        axes: order of dimensions in data, default: TZCXY for 5D, ZCXY for 4D, CXY for 3D, XY for 2D
        wp@tl20200214
    """
    
    axes = axes[-np.ndim(data):].upper()
    if not axes == 'CZTXY':
        T = [axes.find(i) for i in 'CZTXY']
        E = [i for i, j in enumerate(T) if j < 0]
        T = [i for i in T if i >= 0]
        data = np.transpose(data, T)
        for e in E:
            data = np.expand_dims(data, e)

    shape = data.shape[:3]
    with IJTiffWriter(file, shape, data.dtype, colormap) as f:
        at_least_one = False
        for n in tqdm(product(*[range(i) for i in shape]), total=np.prod(shape), desc='Saving tiff', disable=not bar):
            if np.any(data[n]) or not at_least_one:
                f.save(data[n], *n)
                at_least_one = True


def makeheader(shape, byteorder='<', bigtiff=True, colormap=None, dtype=None):
    """ Makes a bytesarray with the header for a tiff file
        ZT: shape ZxT
        wp@tl20200214
        returns:
            header + software + description + colormap tags
            lengths of those parts
    """
    software = b'tiffwrite_tllab_NKI'
    if colormap is None:
        description = \
            'ImageJ=1.11a\nimages={}\nslices={}\nframes={}\nhyperstack=true\nmode=grayscale\nloop=false\n'.\
            format(np.prod(shape[1:]), *shape[1:])
    else:
        description = \
            'ImageJ=1.11a\nimages={}\nchannels={}\nslices={}\nframes={}\nhyperstack=true\nmode=grayscale\nloop=false\n'.\
            format(np.prod(shape), *shape)
    try:
        description = bytes(description, 'ascii')  # python 3
    except:
        pass

    colormap = b'' if colormap is None else get_colormap(colormap, dtype)

    if len(software) % 2:
        software += b'\x00'
    if len(description) % 2:
        description += b'\x00'

    if byteorder == '<':
        header = b'II'
    else:
        header = b'MM'
    if bigtiff:
        offset = 16
        header += struct.pack(byteorder+'H', 43)
        header += struct.pack(byteorder+'H', 8)
        header += struct.pack(byteorder+'H', 0)
        header += struct.pack(byteorder+'Q', offset + len(software) + len(description))
    else:
        offset = 8
        header += struct.pack(byteorder+'H', 42)
        header += struct.pack(byteorder+'I', offset + len(software) + len(description))
    return header + software + description + colormap, (offset, len(software), len(description), len(colormap))


def readheader(b):
    b.seek(0)
    byteorder = {b'II': '<', b'MM': '>'}[b.read(2)]
    bigtiff = {42: False, 43: True}[struct.unpack(byteorder + 'H', b.read(2))[0]]

    if bigtiff:
        tagsize = 20
        tagnoformat = 'Q'
        offsetsize = struct.unpack(byteorder + 'H', b.read(2))[0]
        offsetformat = {8: 'Q', 16: '2Q'}[offsetsize]
        assert struct.unpack(byteorder + 'H', b.read(2))[0] == 0, 'Not a TIFF-file'
        offset = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]
    else:
        tagsize = 12
        tagnoformat = 'H'
        offsetformat = 'I'
        offsetsize = 4
        offset = struct.unpack(byteorder + offsetformat, b.read(offsetsize))[0]
    return byteorder, bigtiff, tagsize, tagnoformat, offsetformat, offsetsize, offset


def readifd(b):
    """ Reads the first IFD of the tiff file in the file handle b
        wp@tl20200214
    """
    byteorder, bigtiff, tagsize, tagnoformat, offsetformat, offsetsize, offset = readheader(b)
        
    b.seek(offset)
    nTags = struct.unpack(byteorder+tagnoformat, b.read(struct.calcsize(tagnoformat)))[0]
    assert nTags < 4096, 'Too many tags'
    addr = []
    addroffset = []

    length = 8 if bigtiff else 2
    length += nTags * tagsize + offsetsize

    tags = {}
    for i in range(nTags):
        pos = offset+struct.calcsize(tagnoformat)+tagsize*i
        b.seek(pos)

        code, tp = struct.unpack(byteorder+'HH', b.read(4))
        count = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]
        
        dtype = {1: 'H', 2: 's', 3: 'H', 4: 'I', 5: '2I', 16: 'Q', 17: 'U', 18: 'I'}[tp]
        dtypelen = struct.calcsize(dtype)

        toolong = struct.calcsize(dtype) * count > offsetsize
        if toolong:
            addr.append(b.tell()-offset)
            caddr = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]
            addroffset.append(caddr-offset)
            cp = b.tell()
            b.seek(caddr)

        if tp == 1:
            value = b.read(count)
        elif tp == 2:
            value = b.read(count).decode('ascii').rstrip('\x00')
        elif tp == 5:
            value = [struct.unpack(byteorder + dtype, b.read(dtypelen)) for _ in range(count)]
        else:
            value = [struct.unpack(byteorder+dtype, b.read(dtypelen))[0] for _ in range(count)]

        if toolong:
            b.seek(cp)

        tags[code] = (tp, value, None)

    b.seek(offset)
    return tags


def getchunks(frame):
    with BytesIO(frame) as b:
        tags = readifd(b)
        stripoffsets = tags[273][1]
        stripbytecounts = tags[279][1]
        chunks = []
        for o, c in zip(stripoffsets, stripbytecounts):
            b.seek(o)
            chunks.append(b.read(c))
    return stripbytecounts, tags, chunks


def fmt_err(exc_info):
    t, m, tb = exc_info
    while tb.tb_next:
        tb = tb.tb_next
    return 'line {}: {}'.format(tb.tb_lineno, m)


def writer(file, shape, byteorder, bigtiff, Qo, V, W, E, colormap=None, dtype=None):
    """ Writes a tiff file, writer function for IJTiffWriter
        file:      filename of the new tiff file
        shape:     shape (CZT) of the data to be written
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qo:        Queue from which to take the compressed frames for writing
        V:         Value; 1 when more frames need to be written, 0 when writer can finish
        W:         Value in which writer will log how many frames are written
        colormap:  array with 2^bitspersample x 3 values RGB
        wp@tl20200214
    """
    spp = shape[0] if colormap is None else 1  # samples/pixel
    nframes = np.prod(shape[1:]) if colormap is None else np.prod(shape)
    error = False

    offsetformat, offsetsize, tagnoformat, tagsize = fmt(bigtiff)
    strips = {}
    tags = {}
    N = []

    def frn(n):
        if colormap is None:
            return n[1] + n[2] * shape[1], n[0]
        else:
            return n[0] + n[1] * shape[0] + n[2] * shape[0] * shape[1], 0

    def addframe(frame, n):
        framenr, channel = frn(n)
        stripbytecounts, tags[framenr], chunks = getchunks(frame)
        stripbyteoffsets = []
        for c in chunks:
            if fh.tell() % 2:
                fh.write(b'\x00')
            stripbyteoffsets.append(fh.tell())
            fh.write(c)  # write the data now, ifds later
        strips[(framenr, channel)] = (stripbyteoffsets, stripbytecounts)
        W.value += 1
        N.append(n)
        return framenr, channel

    with open(file, 'wb') as fh:
        # lengths in bytes of: header, software, desc, colormap
        header, lengths = makeheader(shape, byteorder, bigtiff, colormap, dtype)
        fh.write(header)
        fminmax = np.tile((np.inf, -np.inf), (shape[0], 1))
        while not V.is_set() and not error:  # take frames from queue and write to file
            try:
                frame, n, fmin, fmax = Qo.get(True, 0.02)
                fminmax[n[0]] = min(fminmax[n[0]][0], fmin), max(fminmax[n[0]][1], fmax)
                addframe(frame, n)
            except queues.Empty:
                continue
            except Exception:
                E.put(fmt_err(sys.exc_info()))
                error = True

        if dtype.kind == 'i':
            dmin, dmax = np.iinfo(dtype).min, np.iinfo(dtype).max
        else:
            dmin, dmax = 0, 65535
        fminmax[np.isposinf(fminmax)] = dmin
        fminmax[np.isneginf(fminmax)] = dmax
        for i in range(fminmax.shape[0]):
            if fminmax[i][0] == fminmax[i][1]:
                fminmax[i] = dmin, dmax

        if len(N) < np.prod(shape):  # add empty frames if needed
            empty_frame = None
            for n in product(*[range(i) for i in shape]):
                if not n in N:
                    framenr, channel = frn(n)
                    if empty_frame is None:
                        tag = tags[framenr] if framenr in tags.keys() else tags[list(tags.keys())[-1]]
                        frame = IJTiffFrame(np.zeros(tag[257][1] + tag[256][1], dtype), byteorder, bigtiff)
                        empty_frame = addframe(frame, n)
                    else:
                        strips[(framenr, channel)] = strips[empty_frame]
                        if not framenr in tags.keys():
                            tags[framenr] = tags[empty_frame[0]]

        if not error:
            offset_addr = lengths[0] - offsetsize

            # unfortunately, ImageJ doesn't read this from bigTiff, maybe we'll figure out how to force IJ in the future
            for tag in tifffile.tifffile.imagej_metadata_tag(
                    {'Ranges': tuple(fminmax.flatten().astype(int))}, byteorder):
                tags[0][tag[0]] = ({50839: 1, 50838: 4}[tag[0]], tag[3], None)

            for framenr in range(nframes):
                stripbyteoffsets, stripbytecounts = zip(*[strips[(framenr, channel)] for channel in range(spp)])
                stripbyteoffsets = sum(stripbyteoffsets, [])
                stripbytecounts = sum(stripbytecounts, [])
                tp, value, _ = tags[framenr][258]
                tags[framenr][258] = (tp, spp * value, None)
                tags[framenr][270] = (2, None, (lengths[2], sum(lengths[:2])))  # description
                tags[framenr][273] = (16, stripbyteoffsets, None)
                tags[framenr][277] = (3, [spp], None)
                tags[framenr][279] = (16, stripbytecounts, None)
                tags[framenr][284] = (3, [2], None)
                tags[framenr][305] = (2, None, (lengths[1], lengths[0]))  # software
                if not colormap is None:
                    tags[framenr][320] = (3, None, (lengths[3] // 2, sum(lengths[:3])))
                    tags[framenr][262] = (3, [3], None)

                # write offset to this ifd in the previous one
                if fh.tell() % 2:
                    fh.write(b'\x00')
                offset = fh.tell()
                fh.seek(offset_addr)
                fh.write(struct.pack(byteorder + offsetformat, offset))

                # write ifd
                fh.seek(offset)
                fh.write(struct.pack(byteorder + tagnoformat, len(tags[framenr])))
                a = [addtag(fh, code, *tags[framenr][code]) for code in sorted(tags[framenr].keys())]
                offset_addr = fh.tell()
                fh.seek(tagsize, 1)
                for i in [j for j in a if j is not None]:
                    addtagdata(fh, byteorder, bigtiff, *i)
            fh.write(struct.pack(byteorder + tagnoformat, 0))


def fmt(bigtiff=True):
    # offsetformat, offsetsize, tagnoformat, tagsize
    return (('I', 4, 'H', 8), ('Q', 8, 'Q', 20))[bigtiff]


def addtagdata(b, byteorder, bigtiff, addr, bvalue):
    offsetformat, offsetsize, tagnoformat, tagsize = fmt(bigtiff)
    if b.tell() % 2:
        b.write(b'\x00')
    tagoffset = b.tell()
    b.write(bvalue)
    b.seek(addr)
    b.write(struct.pack(byteorder + offsetformat, tagoffset))
    b.seek(0, 2)


def IJTiffFrame(frame, byteorder, bigtiff):
    with BytesIO() as framedata:
        with tifffile.TiffWriter(framedata, bigtiff, byteorder) as t:
            t.save(frame, compress=9, contiguous=True)
        return framedata.getvalue()

        
def compressor(byteorder, bigtiff, Qi, Qo, V, E):
    """ Compresses tiff frames
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qi:        Queue from which new frames which need to be compressed are taken
        Qo:        Queue where compressed frames are stored
        V:         Value; 1 when more frames need to be compressed, 0 when compressor can finish
    """
    try:
        while not V.is_set():
            try:
                frame, n = Qi.get(True, 0.02)
                if isinstance(frame, tuple):
                    fun, args, kwargs = frame[:3]
                    frame = fun(*args, **kwargs)
                Qo.put((IJTiffFrame(frame, byteorder, bigtiff), n,
                        np.nanmin(np.nanmin(frame.flatten()[frame.flatten()>0])), np.nanmax(frame)))
            except queues.Empty:
                continue
    except Exception:
        E.put(fmt_err(sys.exc_info()))


class IJTiffWriterMulti():
    def __init__(self, files, shapes, dtype='uint16', colormap=None):
        nP = np.clip(int(cpu_count()/2/len(shapes)), 1, 6)
        self.tiffs = {file: IJTiffWriter(file, shape, dtype, colormap, nP) for file, shape in zip(files, shapes) if file}

    def save(self, file, frame, *n):
        if file:
            if file in self.tiffs:
                self.tiffs[file].save(frame, *n)
            else:
                print('Not a file in our care: '.format(file))

    def close(self):
        for tiff in self.tiffs.values():
            tiff.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class IJTiffWriter():
    """ Class for writing ImageJ big tiff files using good compression and multiprocessing to compress quickly
        Usage:
            with IJTiffWriter(file, shape) as t:
                t.save(frame, c, z, t)
                
        file:     filename of the new tiff file
        shape:    shape (CZT) of the data to be written
        frame:    2D numpy array with data

        optional:
        dtype:    coerce to datatype before saving, (u)int8, (u)int16, float32
        colormap: colormap name from colorcet package
        nP:       number of compressor workers to use

        c, z, t: color, z, time coordinates of the frame
        
    """
    def __init__(self, file, shape, dtype='uint16', colormap=None, nP=None):
        self.file = file
        assert len(shape)==3, 'please specify all c, z, t for the shape'
        self.shape = shape  #CZT
        self.bigtiff = True  #normal tiff also possible, but should be opened by bioformats in ImageJ
        self.byteorder = '<'
        self.frames = []
        self.nP = nP or min(int(cpu_count()/6), np.prod(shape))
        self.dshape = (256, 256)
        self.dtype = np.dtype(dtype)
        assert self.dtype.char in 'BbHhf', 'datatype not supported'
        self.errors = []
        self.Qi = Queue(10*self.nP)
        self.Qo = Queue(10*self.nP)
        self.E  = Queue()
        self.V  = Event()
        self.W  = Value('i', 0)
        self.colormap = colormap
        self.Compressor = Pool(self.nP, compressor, (self.byteorder, self.bigtiff, self.Qi, self.Qo, self.V, self.E))
        self.Writer = Process(target=writer, args=(file, shape, self.byteorder, self.bigtiff,
                                                   self.Qo, self.V, self.W, self.E, self.colormap, self.dtype))
        self.Writer.start()

        
    def save(self, frame, *n):
        assert len(n) == 3, 'please specify all c, z, t'
        assert n not in self.frames, 'frame {} {} {} is present already'.format(*n)
        assert all([0 <= i < s for i, s in zip(n, self.shape)]),\
            'frame {} {} {} is outside shape {} {} {}'.format(*(n + self.shape))
        if not self.E.empty():
            raise Exception(self.E.get())
        if isinstance(frame, tuple):
            #fun, args, kwargs, dshape = frame
            self.dshape = frame[3]
        else:
            self.dshape = frame.shape
            assert frame.ndim==2, 'data must be 2 dimensional'
            if not self.dtype is None:
                frame = frame.astype(self.dtype)
        self.frames.append(n)
        self.Qi.put((frame, n))


    def close(self):
        if self.W.value<len(self.frames):
            with tqdm(total=len(self.frames), leave=False, desc='Finishing writing frames',
                      disable=(len(self.frames)-self.W.value)<100) as bar:
                while self.W.value<len(self.frames):
                    if not self.E.empty():
                        print(self.E.get())
                        break
                    bar.n = self.W.value
                    bar.refresh()
                    sleep(0.02)
                bar.n = len(self.frames)
                bar.refresh()

        self.V.set()
        while not self.Qi.empty():
            self.Qi.get()
        self.Qi.close()
        self.Qi.join_thread()
        while not self.Qo.empty():
            self.Qo.get()
        self.Qo.close()
        self.Qo.join_thread()
        while not self.E.empty():
            self.errors.append(self.E.get())
        self.E.close()
        self.Compressor.close()
        self.Compressor.join()
        self.Writer.join(5)
        if self.Writer.is_alive():
            self.Writer.terminate()
            self.Writer.join(5)
            if self.Writer.is_alive():
                print('Writer process won''t close.')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


def addtag(b, code, ttype, value=None, addr=None, byteorder='<', offsetformat='Q', offsetsize=8, tagsize=20):
    if not value is None:
        count = len(value)
    else:
        count, addr = addr
    dtype = {1: 'H', 2: 's', 3: 'H', 4: 'I', 5: '2I', 16: 'Q', 17: 'U', 18: 'I'}[ttype]
    offset = b.tell()
    b.write(struct.pack(byteorder + 'HH', code, ttype))
    b.write(struct.pack(byteorder + offsetformat, count))
    a = None
    if not value is None:
        if isinstance(value, bytes):
            bvalue = value
        elif isinstance(value, str):
            bvalue = value.encode('ascii')
        elif ttype==5:
            bvalue = b''.join([struct.pack(byteorder + dtype, *v) for v in value])
        else:
            bvalue = b''.join([struct.pack(byteorder + dtype, v) for v in value])
        if count * struct.calcsize(dtype) <= offsetsize:
            b.write(bvalue)
        else:
            a = (b.tell(), bvalue)
    if not addr is None:
        b.write(struct.pack(byteorder + offsetformat, addr))
    b.seek(offset + tagsize)
    return a
