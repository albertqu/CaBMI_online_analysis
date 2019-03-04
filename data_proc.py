import tifffile
import os
from skimage import io


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

