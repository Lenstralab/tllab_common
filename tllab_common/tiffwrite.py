import struct
import numpy as np
from io import BytesIO
from tifffile import TiffWriter
from tifffile.tifffile import imagej_description
from multiprocessing import Pool, Process, Queue, Event, cpu_count, Value, util, queues
from time import sleep
from tqdm.auto import tqdm

def tiffwrite(file, data, axes='TZCXY'):
    """ Saves an image using the bigtiff format, openable in ImageJ as hyperstack
        Uses multiple processes to quickly compress as good as possible
        file: filename of the new tiff file
        data: 2 to 5D array
        axes: order of dimensions in data, default: TCZXY for 5D, CZXY for 4D, ZXY for 3D, XY for 2D
        wp@tl20200214
    """
    
    axes = axes[-np.ndim(data):].upper()
    if not axes=='TZCXY':
        T = [axes.find(i) for i in 'TZCXY']
        E = [i for i,j in enumerate(T) if j<0]
        T = [i for i in T if i>=0]
        data = np.transpose(data, T)
        for e in E:
            data = np.expand_dims(data, e)            

    shape = data.shape[:3]
    with IJTiffWriter(file, shape[::-1]) as f:
        with tqdm(total=np.prod(shape), desc='Saving tiff') as bar:
            for t in range(shape[0]):
                for z in range(shape[1]):
                    for c in range(shape[2]):
                        f.save(data[t, z, c], c, z, t)
                        bar.update()

def makeheader(byteorder='<', bigtiff=False):
    """ Makes a bytesarray with the header for a tiff file
        wp@tl20200214
    """
    with BytesIO() as b:
        if byteorder=='<':
            b.write(b'II')
        else:
            b.write(b'MM')
        if bigtiff:
            b.write(struct.pack(byteorder+'H', 43))
            b.write(struct.pack(byteorder+'H', 8))
            b.write(struct.pack(byteorder+'H', 0))
            b.write(struct.pack(byteorder+'Q', 16))
        else:
            b.write(struct.pack(byteorder+'H', 42))
            b.write(struct.pack(byteorder+'I', 8))
        header = b.getvalue()
    return header

def readifd(b):
    """ Reads the first IFD of the tiff file in the file handle b
        wp@tl20200214
    """
    b.seek(0)
    byteorder = {b'II': '<', b'MM': '>'}[b.read(2)]
    bigtiff = {42: False, 43: True}[struct.unpack(byteorder+'H', b.read(2))[0]]
    
    if bigtiff:
        tagsize = 20
        tagnoformat = 'Q'
        offsetsize = struct.unpack(byteorder+'H', b.read(2))[0]
        offsetformat = {8: 'Q', 16: '2Q'}[offsetsize]
        assert struct.unpack(byteorder+'H', b.read(2))[0]==0, 'Not a TIFF-file'
        offset = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]
    else:
        tagsize = 12
        tagnoformat = 'H'
        offsetformat = 'I'
        offsetsize = 4
        offset = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]        
        
    b.seek(offset)
    nTags = struct.unpack(byteorder+tagnoformat, b.read(struct.calcsize(tagnoformat)))[0]
    assert nTags<4096, 'Too many tags'
    addr = []
    addroffset = []
    dataoffsetoffset = 0
    dataoffsettype = 'H'
    
    for i in range(nTags):
        pos = offset+struct.calcsize(tagnoformat)+tagsize*i
        b.seek(pos)

        code, tp = struct.unpack(byteorder+'HH', b.read(4))
        count = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]
        
        dtype = {1: 'H', 2: 's', 3: 'H', 4: 'I', 5: '2I', 16: 'Q', 17: 'U', 18: 'I'}[tp]
        dtypelen = struct.calcsize(dtype)
        
        if struct.calcsize(dtype) * count > offsetsize:
            addr.append(b.tell()-offset)
            caddr = struct.unpack(byteorder+offsetformat, b.read(offsetsize))[0]
            addroffset.append(caddr-offset)
            cp = b.tell()
            
            b.seek(caddr)
            if tp==1:
                value = b.read(count)
            elif tp==2:
                value = b.read(count).decode('ascii').rstrip('\x00')
            elif tp==5:
                value = [struct.unpack(byteorder+dtype, b.read(4))[0]/struct.unpack(byteorder+'I', b.read(4))[0] for i in range(count)]
            else:
                value = [struct.unpack(byteorder+dtype, b.read(dtypelen))[0] for i in range(count)]
            b.seek(cp)
            if len(value)==1:
                value = value[0]
        else:
            caddr = b.tell()
            value = [struct.unpack(byteorder+dtype, b.read(dtypelen))[0] for i in range(count)]
            
        if code==273:
            dataoffsetoffset = [caddr-offset+i*dtypelen for i in range(count)]
            dataoffsettype = dtype
            dataoffset = [v-offset for v in value]
    
    nifdoffset = struct.calcsize(tagnoformat)+tagsize*nTags
        
    b.seek(offset)    
    dataifd = b.read()
        
    return dataifd, dataoffset, dataoffsetoffset, dataoffsettype, addr, addroffset, nifdoffset

def writer(file, shape, byteorder, bigtiff, Qo, V, W, E):
    """ Writes a tiff file, writer function for IJTiffWriter
        file:      filename of the new tiff file
        shape:     shape (CZT) of the data to be written
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qo:        Queue from which to take the compressed frames for writing
        V:         Value; 1 when more frames need to be written, 0 when writer can finish
        W:         Value in which writer will log how many frames are written
        wp@tl20200214
    """
    error = False
    try:
        if bigtiff:
            offsetformat = 'Q'
            tagnoformat = 'Q'
        else:
            offsetformat = 'I'
            tagnoformat = 'H'

        what = {}
        where = {}
        for i in range(np.prod(shape)+1):
            what[i] = None
            where[i] = None
        where[0] = 4+4*bigtiff
        what[np.prod(shape)] = 0

        with open(file, 'wb') as fh:
            fh.write(makeheader(byteorder, bigtiff))
            while not V.is_set() and not error:
                try:
                    frame, framenr = Qo.get(True, 0.02)
                    with BytesIO(frame) as b:
                        data, dataoffset, dataoffsetoffset, dataoffsettype, addr, addroffset, nifd = readifd(b)
                    offset = fh.tell()

                    if offset%2:
                        fh.write(b'\x00')
                        offset += 1

                    what[framenr[0]] = offset
                    where[framenr[-1]+1] = nifd+offset

                    for f in framenr[1:]:
                        what[f] = -1
                    for f in framenr[:-1]:
                        where[f+1] = -1

                    with BytesIO(data) as b:
                        for i, j in zip(addroffset, addr):
                            b.seek(j)
                            b.write(struct.pack(byteorder+offsetformat, i+offset))

                        for i, j in zip(dataoffset, dataoffsetoffset):
                            b.seek(j)
                            b.write(struct.pack(byteorder+dataoffsettype, i+offset))
                        fh.write(b.getvalue())

                    W.value += len(framenr)
                except queues.Empty:
                    continue
                except Exception as e:
                    E.put(e)
                    error = True

            if not error:
                what[W.value+1] = 0
                for i in range(np.prod(shape)+1):
                    if not what[i] is None and not where[i] is None:
                        if what[i]>=0 and where[i]>=0:
                            fh.seek(where[i])
                            fh.write(struct.pack(byteorder+tagnoformat, what[i]))
                    else:
                        print('Warning: frame {} is missing!'.format(i))
    except Exception as e:
        E.put(e)
        
def compressor(shape, byteorder, bigtiff, Qi, Qo, V, E):
    """ Compresses tiff frames
        shape:     shape (CZT) of the data to be written
        byteorder: byteorder of the file to be written, '<' or '>'
        bigtiff:   False: file will be normal tiff, True: file will be bigtiff
        Qi:        Queue from which new frames which need to be compressed are taken
        Qo:        Queue where compressed frames are stored
        V:         Value; 1 when more frames need to be compressed, 0 when compressor can finish
    """
    try:
        shape = shape[::-1]
        while not V.is_set():
            try:
                frame, framenr = Qi.get(True, 0.02)
                if isinstance(frame, tuple):
                    fun, args, kwargs = frame[:3]
                    frame = fun(*args, **kwargs)
                framedata = BytesIO()
                with IJTiffFrame(framedata, shape, bigtiff, byteorder, False, frame.ndim==3) as t:
                    t.save(frame, compress=9, contiguous=True)
                Qo.put((framedata.getvalue(), framenr))
            except queues.Empty:
                continue
    except Exception as e:
        E.put(e)
        
class IJTiffWriterMulti():
    def __init__(self, files, shapes):
        nP = np.clip(int(cpu_count()/2/len(shapes)), 1, 6)
        self.tiffs = {file: IJTiffWriter(file, shape, nP) for file, shape in zip(files, shapes) if file}

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
                
        file:    filename of the new tiff file
        shape:   shape (CZT) of the data to be written
        frame:   2D numpy array with data, supported: uint16, uint8
                   or: 3D array for RGB, 3rd dimension has the color, its size must be 2, 3 or 4 and it must be uint8
        c, z, t: color, z, time coordinates of the frame
        
    """
    def __init__(self, file, shape, nP=None):
        self.file = file
        self.shape = shape #CZT
        self.bigtiff = True #normal tiff also possible, but should be opened by bioformats in ImageJ
        self.byteorder = '<'
        self.frames = []
        self.nP = nP or min(int(cpu_count()/6), np.prod(shape))
        self.dshape = (256, 256)
        self.dtype = 'uint16'
        
        self.Qi = Queue(10*self.nP)
        self.Qo = Queue(10*self.nP)
        self.E  = Queue()
        self.V  = Event()
        self.W  = Value('i', 0)
        self.Compressor = Pool(self.nP, compressor, (shape, self.byteorder, self.bigtiff,
                                                     self.Qi, self.Qo, self.V, self.E))
        self.Writer = Process(target=writer, args=(file, shape, self.byteorder, self.bigtiff,
                                                   self.Qo, self.V, self.W, self.E))
        self.Writer.start()
        
        if self.bigtiff:
            self.offsetformat = 'Q'
            self.tagnoformat = 'Q'
            self.nifdoffset = 8
        else:
            self.offsetformat = 'I'
            self.tagnoformat = 'H'
            self.nifdoffset = 4
        
    def save(self, frame, *n):
        if not self.E.empty():
            raise Exception(self.E.get())
        if isinstance(frame, tuple):
            #fun, args, kwargs, dshape = frame
            self.dshape = frame[3]
            ndim = len(self.dshape)
        else:
            self.dshape = frame.shape
            self.dtype = frame.dtype
            assert self.dtype.char in 'BHhf', 'datatype not supported'
            #RGB maybe?
            if frame.ndim==3:
                assert frame.shape[2] in (1,2,3,4), 'RGB frame must have color as 3rd dimension'
                if frame.shape[2]==1:
                    frame = frame.squeeze()
                if frame.shape[2]==2:
                    frame = np.dstack((frame, np.zeros(frame.shape[:2], frame.dtype)))
                if frame.ndim==3:
                    assert frame.dtype=='B', 'RGB frame can only be uint8'
            else:
                assert frame.ndim==2, 'Frame must either have 2 or 3 dimensions'
            ndim = frame.ndim
        
        if len(n)==1:
            framenr = n[0]
        elif len(n)==2:
            framenr = n[0]*self.shape[0] + n[1]*self.shape[0]*self.shape[1]
        else:    
            framenr = n[0] + n[1]*self.shape[0] + n[2]*self.shape[0]*self.shape[1]
        if ndim==3:
            framenr = [framenr+i for i in range(self.dshape[2])]
        else:
            framenr = [framenr]
        
        self.frames.extend(framenr)
        self.Qi.put((frame, framenr))

    def close(self):
        if len(self.frames)<np.prod(self.shape):
            with tqdm(total=np.prod(self.shape), leave=False, desc='Adding empty frames') as bar:
                bar.n = len(self.frames)
                for i in range(np.prod(self.shape)):
                    if not self.E.empty():
                        print(self.E.get())
                        break
                    if not i in self.frames:
                        self.save(np.zeros(self.dshape, self.dtype), i)
                        bar.update()

        if self.W.value<len(self.frames):
            with tqdm(total=len(self.frames), leave=True, desc='Finishing writing frames', disable=(len(self.frames)-self.W.value)<100) as bar:
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

class IJTiffFrame(TiffWriter):
    """ Subclassing tifffile.TiffWriter to obtain single compressed tiff frames compatible with ImageJ
    """
    def __init__(self, file, shape, bigtiff=True, byteorder=None, append=False, rgb=False):
        self.shape = shape
        super(IJTiffFrame, self).__init__(file, bigtiff, byteorder, append, False)
        self._descriptionoffsets = []
        self._precompressbuffer = []
        self._imagej = True
        self._buffer = []
        self._rgb = rgb

    def _write_image_description(self):
        """Write metadata to ImageDescription tag."""

        if not self._datashape or self._descriptionoffset <= 0: # change wrt tifffile._write_image_description:
            return                                              # do not bail out if not time-series (wtf?)

        colormapped = self._colormap is not None
        if hasattr(self, '_storedshape'):
            isrgb = self._storedshape[-1] in (3, 4)
        else:
            isrgb = self._shape[-1] in (3, 4)
        description = imagej_description(self._datashape, isrgb, colormapped, **self._metadata)

        # rewrite description and its length to file
        description = description.encode()
        description = description[:self._descriptionlen]
        pos = self._fh.tell()
        self._fh.seek(self._descriptionoffset)
        self._fh.write(description)
        self._fh.seek(self._descriptionlenoffset)
        self._fh.write(struct.pack(self._byteorder + self._offsetformat, len(description)))
        self._fh.seek(pos)
        self._descriptionoffset = 0
        self._descriptionlenoffset = 0
        self._descriptionlen = 0

    def save(self, data, *args, **kwargs):
        super(IJTiffFrame, self).save(data, *args, **kwargs)
        self._descriptionoffsets.append((self._descriptionoffset, self._descriptionlenoffset, self._descriptionlen))
        
    def close(self):
        """Write remaining pages and close file handle."""
        if not self._truncate:
            self._write_remaining_pages()

        if hasattr(self, '_storedshape'):
            self._datashape = self.shape[:3] + self._storedshape[3:5]
        else:
            self._datashape = self.shape[:3] + self._shape[3:5]
        if self._rgb:
            self._datashape = list(self._datashape)
            self._datashape += (np.clip(self.shape[2], 3, 4),)
            self._datashape[2] = 1
            self._datashape = tuple(self._datashape)
        
        for descriptionoffset in self._descriptionoffsets:
            self._descriptionoffset, self._descriptionlenoffset, self._descriptionlent = descriptionoffset
            self._write_image_description()
            
        self._fh.close()