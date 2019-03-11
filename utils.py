import numpy as np
import os, logging
from skimage import io, exposure
import tifffile


counter_int = 5


class DCache:
    # TODO: AUGMENT IT SUCH THAT IT WORKS FOR MULTIPLE

    def __init__(self, size=20, thres=3, buffer=False):
        """
        :param size: int, size of the dampening cache
        :param thres: float, threshold for valid data caching, ignore signal if |x - miu_x| > thres * std
        :param buffer: boolean, for whether keeping a dynamic buffer
        """
        self.size = size
        self.thres = thres
        self.counter = 0

        if buffer:
            self.cache = []
        else:
            self.avg = 0
            self.std = 0

    def __len__(self):
        return self.size

    def add(self, signal):
        if signal != 0:
            if self.counter < self.size:
                print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
                self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
                diff2 = (signal - self.avg) ** 2
                self.std = (diff2 + self.std * self.counter) / (self.counter+1)

            elif signal - self.avg < np.sqrt(self.std * self.thres):
                print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
                self.avg = (self.avg * (self.size - 1) + signal) / self.size
                diff2 = (signal - self.avg) ** 2
                self.std = (diff2 + self.std * (self.size - 1)) / self.size
            self.counter += 1

    def get_val(self):
        return self.avg


class MemmapHandler:

    def __init__(self, outpath, dims, fnamemm='temp_mem.mmap', fnametif='mergetif{}.tif', batch_tif=None,
                 dtype=np.int16, order='F'):
        """
        :param outpath: output path
        :param dims: In the order of (totlen, data_shape)
        :param order: 'F'
        """
        self.fnamemm = os.path.join(outpath, fnamemm)
        self.mem = np.memmap(self.fnamemm, mode='w+', dtype=dtype, shape=dims, order=order)
        self.counter = 0
        self.totlen = dims[0]
        self.fshape = dims[1:]
        self.batch = self.totlen // batch_tif
        if self.batch is not None:
            self.fnametif = os.path.join(outpath, fnametif)
            self.lastsave = 0

    def add_data(self, data, high_contrast=False):
        """change the length of data everytime there is a modification  if counter + 1 is integer multiple of batch
        or is totlen"""
        target_shape = data.shape
        if np.prod(target_shape) != np.prod(self.fshape):
            raise RuntimeError("data shape {} is not aligned with required shape {}".format(target_shape, self.fshape))
        self.mem[self.counter] = exposure.rescale_intensity(data.reshape(self.fshape))\
            if self.batch and high_contrast else data.reshape(self.fshape)
        self.counter += 1
        if self.batch is not None:
            if self.counter == self.totlen or self.counter % self.batch == 0:
                ncount = (self.counter-1) // self.batch
                nexttif = self.fnametif.format(ncount)
                io.imsave(nexttif, self.mem[self.lastsave:self.counter], plugin='tifffile')
                logging.info("Created {}".format(nexttif))
                self.lastsave = self.counter

    def close(self, delete_mem=True):
        self.mem.flush()
        if delete_mem and os.path.exists(self.fnamemm):
            os.remove(self.fnamemm)
        if self.batch is not None:
            return [self.fnametif.format(i) for i in range((self.lastsave-1) // self.batch + 1)]
        del self.mem


def get_tif_size(fname):
    tif = tifffile.TiffFile(fname)
    T = len(tif.pages)
    dims = tif.pages[0].shape
    tif.close()
    return T, dims


def second_to_hmt(sec):
    h, sec = sec // 3600, sec % 3600
    m, sec = sec // 60, sec % 60
    return h, m, sec


def recursively_check_dtype(dic, context=()):
    for k, v in dic.items():
        if isinstance(k, tuple):
            print(k, context)
        if isinstance(v, dict):
            recursively_check_dtype(dic, (context, k))
        elif isinstance(v, tuple):
            print("Found:", k, v, context)



