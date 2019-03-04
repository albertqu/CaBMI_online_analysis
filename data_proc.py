import tifffile
import os
import numpy as np
from skimage import io
from tifffile import TiffWriter


def split_to_series(filename, destF, seq=(0,), ns='online_{}.tiff', numplanes=6):
    """
    Taking filename and split the big tiff into frames
    :param filename: filename of the big tiff, unsplitted
    :param destF: destination folder format, e.g. ROOT/series_{}
    :param seq: sequence of planes needed to record
    :param ns: namescheme of the tiffs saved
    :param numplanes: number of planes
    """
    ims = tifffile.TiffFile(filename)
    series_len = len(ims.pages)
    dur = series_len // numplanes
    print(dur)
    for i in range(dur):
        for j in range(numplanes):
            if j in seq:
                dest = destF.format(j)
                if not os.path.exists(dest):
                    os.mkdir(dest)
                io.imsave(os.path.join(dest, ns.format(i)),
                          ims.asarray(i * numplanes + j), plugin='tifffile')


def merge_tiffs(fls, outpath, num=1, tifn='bigtif{}.tif'):
    # Takes in a list of single tiff fls and save them in memmap
    tifn = os.path.join(outpath, tifn)
    imgs = tifffile.TiffSequence(fls)
    totlen = imgs.shape[0]
    #dims = imgs.imread(imgs.files[0]).shape
    chunklen = totlen // num
    fnames = []
    for j in range(num):
        fname = tifn.format(num)
        with TiffWriter(fname, bigtiff=True) as tif:
            fnames.append(fname)
            for i in range(chunklen):
                pointer = i + chunklen * j
                if pointer >= totlen:
                    break
                else:
                    tif.save(imgs.imread(imgs.files[pointer]), compress=6)
    return fnames


