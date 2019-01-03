try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

# Base Utils
from IPython.display import display, clear_output
import glob
import logging
import numpy as np
import os, time
import scipy
import cv2
from .utils import counter_int, second_to_hmt

# Caiman Modules
import caiman as cm
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman import mmapping
from caiman.source_extraction import cnmf
# EXPR
from caiman.source_extraction.cnmf.initialization import imblur, initialize_components, hals
from caiman.source_extraction.cnmf.utilities import update_order, get_file_size, peak_local_max
from scipy.sparse import coo_matrix, csc_matrix, spdiags
import numbers
# END EXPR


def base_prepare(folder, bfg, cnm, view=False):
    # TODO: Takes in folder, bfg (baseline flag) and cnm object
    # TODO: This version is suitable for a large tif baseline; next step add ability to handle discrete imgs
    bp0 = time.time() # SET Breakpoint
    bp1 = bp0
    counter = 0
    while True:
        time.sleep(0.1) # Let thread sleep for certain intervals to
        bp2 = time.time()
        if (bp2 - bp1) > counter_int:
            print("Time Since Start: {} minutes", counter_int)  # MORE DETAILED TIMER LATER
            counter += 1
            bp1 = bp2
        for f in os.listdir(folder):
            # When baseline found, terminate the querying and start initial processing
            if f.find(bfg) != -1:
                fnames = [os.path.join(folder, f)]
                cnm.params.change_params({"fnames": fnames})
                break

    cnm.initialize_online()
    dur = time.time() - bp0
    ha, ma, sa = second_to_hmt(bp2-bp0)
    hd, md, sd = second_to_hmt(dur)
    print("BaseLine Data Acquisition Time: {}H:{}M:{}S\n".format(ha, ma, sa) +
          "Total Processing Time: {}H:{}M:{}S".format(hd, md, sd))

    if view:
        Cn = cm.load(cnm.params.get("online", "fnames")[0], subindices=slice(0, 500)).local_correlations(
            swap_dim=False)
        cnm.estimates.plot_contours(img=Cn, display_numbers=False)
        # %% view components
        cnm.estimates.view_components(img=Cn)

    # TODO: ADD CODE TO PROCESS THE REMAINING DATA IN BASELINE


def online_process(folder, ns, ns_start, cnm, view=False):
    # TODO: ADD A LENGTH OF EXPR
    fls = cnm.params.get('data', 'fnames')
    init_batch = cnm.params.get('online', 'init_batch')
    #epochs = cnm.params.get('online', 'epochs') TODO: NO EPOCH In online
    #cnm.initialize_online()
    extra_files = len(fls) - 1   # TODO: ONLINE VERSION MAY DIFFER
    init_files = 1
    t = init_batch
    #cnm.Ab_epoch = []  TODO: NO EPOCH IN ONLINE
    t_online = []
    cnm.comp_upd = []
    cnm.t_shapes = []
    cnm.t_detect = []
    cnm.t_motion = []
    max_shifts_online = cnm.params.get('online', 'max_shifts_online')
    if extra_files == 0:  # check whether there are any additional files
        process_files = fls[:init_files]  # end processing at this file
        init_batc_iter = [init_batch]  # place where to start
    else:
        process_files = fls[:init_files + extra_files]  # additional files
        # where to start reading at each file
        init_batc_iter = [init_batch] + [0] * extra_files
    if cnm.params.get('online', 'save_online_movie'):
        fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
        out = cv2.VideoWriter(cnm.params.get('online', 'movie_name_online'),
                              fourcc, 30.0, tuple([int(2 * x) for x in cnm.params.get('data', 'dims')]))

        for file_count, ffll in enumerate(process_files):
            print('Now processing file ' + ffll)
            # %% file_count, ffll and init_batc taking in
            Y_ = cm.load(ffll, subindices=slice(init_batc_iter[file_count], None, None))

            old_comps = cnm.N  # number of existing components
            for frame_count, frame in enumerate(Y_):  # process each file
                uno_proc(frame, frame_count, cnm, t, old_comps, t_online, out, max_shifts_online)

        #cnm.Ab_epoch.append(cnm.estimates.Ab.copy()) TODO: NO EPOCH IN ONLINE
    if cnm.params.get('online', 'normalize'):
        cnm.estimates.Ab /= 1. / cnm.img_norm.reshape(-1, order='F')[:, np.newaxis]
        cnm.estimates.Ab = csc_matrix(cnm.estimates.Ab)
    cnm.estimates.A, cnm.estimates.b = cnm.estimates.Ab[:, cnm.params.get('init', 'nb'):], cnm.estimates.Ab[:,
                                                                                               :cnm.params.get('init',
                                                                                                                'nb')].toarray()
    cnm.estimates.C, cnm.estimates.f = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, :], cnm.estimates.C_on[
                                                                                                                  :cnm.params.get(
                                                                                                                      'init',
                                                                                                                      'nb'),
                                                                                                                :]
    noisyC = cnm.estimates.noisyC[cnm.params.get('init', 'nb'):cnm.M, :]
    cnm.estimates.YrA = noisyC - cnm.estimates.C
    cnm.estimates.bl = [osi.b for osi in cnm.estimates.OASISinstances] if hasattr(
        cnm, 'OASISinstances') else [0] * cnm.estimates.C.shape[0]
    if cnm.params.get('online', 'save_online_movie'):
        out.release()
    if cnm.params.get('online', 'show_movie'):
        cv2.destroyAllWindows()
    cnm.t_online = t_online
    cnm.estimates.C_on = cnm.estimates.C_on[:cnm.M]
    cnm.estimates.noisyC = cnm.estimates.noisyC[:cnm.M]

    return cnm


def uno_proc(frame, frame_count, cnm, t, old_comps, t_online, out, max_shifts_online):
    t_frame_start = time()
    if np.isnan(np.sum(frame)):
        raise Exception('Frame ' + str(frame_count) +
                        ' contains NaN')
    if t % 100 == 0:
        print('Epoch: ' + str(iter + 1) + '. ' + str(t) +
              ' frames have beeen processed in total. ' +
              str(cnm.N - old_comps) +
              ' new components were added. Total # of components is '
              + str(cnm.estimates.Ab.shape[-1] - cnm.params.get('init', 'nb')))
        old_comps = cnm.N

    frame_ = frame.copy().astype(np.float32)
    if cnm.params.get('online', 'ds_factor') > 1:
        frame_ = cv2.resize(frame_, cnm.img_norm.shape[::-1])

    if cnm.params.get('online', 'normalize'):
        frame_ -= cnm.img_min  # make data non-negative
    t_mot = time()
    if cnm.params.get('online', 'motion_correct'):  # motion correct
        templ = cnm.estimates.Ab.dot(
            cnm.estimates.C_on[:cnm.M, t - 1]).reshape(cnm.params.get('data', 'dims'),
                                                       order='F') * cnm.img_norm
        if cnm.params.get('motion', 'pw_rigid'):
            frame_cor1, shift = motion_correct_iteration_fast(
                frame_, templ, max_shifts_online, max_shifts_online)
            frame_cor, shift = tile_and_correct(frame_, templ, cnm.params.motion['strides'],
                                                cnm.params.motion['overlaps'],
                                                cnm.params.motion['max_shifts'], newoverlaps=None,
                                                newstrides=None, upsample_factor_grid=4,
                                                upsample_factor_fft=10, show_movie=False,
                                                max_deviation_rigid=cnm.params.motion[
                                                    'max_deviation_rigid'],
                                                add_to_movie=0, shifts_opencv=True, gSig_filt=None,
                                                use_cuda=False, border_nan='copy')[:2]
        else:
            frame_cor, shift = motion_correct_iteration_fast(
                frame_, templ, max_shifts_online, max_shifts_online)
        cnm.estimates.shifts.append(shift)
    else:
        templ = None
        frame_cor = frame_
    cnm.t_motion.append(time() - t_mot)

    if cnm.params.get('online', 'normalize'):
        frame_cor = frame_cor / cnm.img_norm
    cnm.fit_next(t, frame_cor.reshape(-1, order='F'))
    if cnm.params.get('online', 'show_movie'):
        cnm.t = t
        vid_frame = cnm.create_frame(frame_cor)
        if cnm.params.get('online', 'save_online_movie'):
            out.write(vid_frame)
            for rp in range(len(cnm.estimates.ind_new) * 2):
                out.write(vid_frame)

        cv2.imshow('frame', vid_frame)
        for rp in range(len(cnm.estimates.ind_new) * 2):
            cv2.imshow('frame', vid_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, None
    t += 1
    t_online.append(time() - t_frame_start)
    return t, old_comps


# %%
def main():
    pass

    # IMPORTANT DATA PATHS
    data_root = "/Users/albertqu/Documents/7.Research/BMI"
    data0, data1 = os.path.join(data_root, "data0"), os.path.join(data_root, "data1")
    logfile = os.path.join(data_root, "online.log")
    bas_tif, bigtif, petitif = 'base.tif', 'bigtif.tif', 'petitif.tif'
    base_flag, ext, frame_ns = "base", 'tif', "{}.tif"
    folder = data1

    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename=logfile,
                        level=logging.INFO)


    # PARAMETER SETTING
    fr = 15  # frame rate (Hz)
    decay_time = 0.5  # approximate length of transient event in seconds
    gSig = (3, 3)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 1   # minimum SNR for accepting new components
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(10.).astype('int')  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 0.9  # soace correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 200  # number of frames for initialization (presumably from the first file)
    K = 2  # initial number of components
    epochs = 2  # number of passes over the data
    show_movie = False # show the movie as the data gets processed

    params_dict = {'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

# %% fit online

    cnm = cnmf.online_cnmf.OnACID(params=opts)
    base_prepare(folder, base_flag, cnm)


# %% plot contours

    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(cnm.params.get("online", "fnames")[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

# %% view components
    cnm.estimates.view_components(img=Cn)



def initialize_online(cnm):
    fls = cnm.params.get('data', 'fnames')
    opts = cnm.params.get_group('online')
    Y = cm.load(fls[0], subindices=slice(0, opts['init_batch'],
             None)).astype(np.float32)
    print(Y.shape, cnm.params.get('online', 'init_batch'), cnm.params.get('online', 'minibatch_shape'))
    ds_factor = np.maximum(opts['ds_factor'], 1)
    if ds_factor > 1:
        Y = Y.resize(1./ds_factor, 1./ds_factor)
    mc_flag = cnm.params.get('online', 'motion_correct')
    cnm.estimates.shifts = []  # store motion shifts here
    cnm.estimates.time_new_comp = []
    if mc_flag:
        max_shifts_online = cnm.params.get('online', 'max_shifts_online')
        mc = Y.motion_correct(max_shifts_online, max_shifts_online)
        Y = mc[0].astype(np.float32)
        cnm.estimates.shifts.extend(mc[1])

    img_min = Y.min()

    if cnm.params.get('online', 'normalize'):
        Y -= img_min
    img_norm = np.std(Y, axis=0)
    img_norm += np.median(img_norm)  # normalize data to equalize the FOV
    print('Size frame:' + str(img_norm.shape))
    if cnm.params.get('online', 'normalize'):
        Y = Y/img_norm[None, :, :]
    if opts['show_movie']:
        cnm.bnd_Y = np.percentile(Y,(0.001,100-0.001))
    _, d1, d2 = Y.shape
    print(Y.shape)
    Yr = Y.to_2D().T        # convert data into 2D array
    print(Yr.shape, cnm.params.get('online', 'init_batch'))
    cnm.img_min = img_min
    cnm.img_norm = img_norm
    if cnm.params.get('online', 'init_method') == 'bare':
        cnm.estimates.A, cnm.estimates.b, cnm.estimates.C, cnm.estimates.f, cnm.estimates.YrA = bare_initialization(
                Y.transpose(1, 2, 0), gnb=cnm.params.get('init', 'nb'), k=cnm.params.get('init', 'K'),
                gSig=cnm.params.get('init', 'gSig'), return_object=False)
        cnm.estimates.S = np.zeros_like(cnm.estimates.C)
        nr = cnm.estimates.C.shape[0]
        cnm.estimates.g = np.array([-np.poly([0.9] * max(cnm.params.get('preprocess', 'p'), 1))[1:]
                           for gg in np.ones(nr)])
        cnm.estimates.bl = np.zeros(nr)
        cnm.estimates.c1 = np.zeros(nr)
        cnm.estimates.neurons_sn = np.std(cnm.estimates.YrA, axis=-1)
        cnm.estimates.lam = np.zeros(nr)
    elif cnm.params.get('online', 'init_method') == 'cnmf':
        cnm = cnmf.CNMF(n_processes=1, params=cnm.params)
        cnm.estimates.shifts = cnm.estimates.shifts
        if cnm.params.get('patch', 'rf') is None:
            cnm.dview = None
            cnm.fit(np.array(Y))
            cnm.estimates = cnm.estimates
#                cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f,\
#                cnm.estimates.S, cnm.estimates.YrA = cnm.estimates.A, cnm.estimates.C, cnm.estimates.b,\
#                cnm.estimates.f, cnm.estimates.S, cnm.estimates.YrA

        else:
            f_new = mmapping.save_memmap(fls[:1], base_name='Yr', order='C',
                                         slices=[slice(0, opts['init_batch']), None, None])
            Yrm, dims_, T_ = mmapping.load_memmap(f_new)
            Y = np.reshape(Yrm.T, [T_] + list(dims_), order='F')
            cnm.fit(Y)
            cnm.estimates = cnm.estimates
#                cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f,\
#                cnm.estimates.S, cnm.estimates.YrA = cnm.estimates.A, cnm.estimates.C, cnm.estimates.b,\
#                cnm.estimates.f, cnm.estimates.S, cnm.estimates.YrA
            if cnm.params.get('online', 'normalize'):
                cnm.estimates.A /= cnm.img_norm.reshape(-1, order='F')[:, np.newaxis]
                cnm.estimates.b /= cnm.img_norm.reshape(-1, order='F')[:, np.newaxis]
                cnm.estimates.A = csc_matrix(cnm.estimates.A)

    elif cnm.params.get('online', 'init_method') == 'seeded':
        cnm.estimates.A, cnm.estimates.b, cnm.estimates.C, cnm.estimates.f, cnm.estimates.YrA = seeded_initialization(
                Y.transpose(1, 2, 0), cnm.estimates.A, gnb=cnm.params.get('init', 'nb'), k=cnm.params.get('init', 'k'),
                gSig=cnm.params.get('init', 'gSig'), return_object=False)
        cnm.estimates.S = np.zeros_like(cnm.estimates.C)
        nr = cnm.estimates.C.shape[0]
        cnm.estimates.g = np.array([-np.poly([0.9] * max(cnm.params.get('preprocess', 'p'), 1))[1:]
                           for gg in np.ones(nr)])
        cnm.estimates.bl = np.zeros(nr)
        cnm.estimates.c1 = np.zeros(nr)
        cnm.estimates.neurons_sn = np.std(cnm.estimates.YrA, axis=-1)
        cnm.estimates.lam = np.zeros(nr)
    else:
        raise Exception('Unknown initialization method!')
    dims, Ts = get_file_size(fls)
    dims = Y.shape[1:]
    cnm.params.set('data', {'dims': dims})
    T1 = np.array(Ts).sum()*cnm.params.get('online', 'epochs')
    print("EN FIN:",Yr.shape, cnm.params.get('online', 'init_batch'))
    cnm._prepare_object(Yr, T1)
    if opts['show_movie']:
        cnm.bnd_AC = np.percentile(cnm.estimates.A.dot(cnm.estimates.C),
                                    (0.001, 100-0.005))
        cnm.bnd_BG = np.percentile(cnm.estimates.b.dot(cnm.estimates.f),
                                    (0.001, 100-0.001))
    return cnm


def bare_initialization(Y, init_batch=1000, k=1, method_init='greedy_roi', gnb=1,
                        gSig=[5, 5], motion_flag=False, p=1,
                        return_object=True, **kwargs):
    """
    Quick and dirty initialization for OnACID, bypassing CNMF entirely
    Args:
        Y               movie object or np.array
                        matrix of data

        init_batch      int
                        number of frames to process

        method_init     string
                        initialization method

        k               int
                        number of components to find

        gnb             int
                        number of background components

        gSig            [int,int]
                        half-size of component

        motion_flag     bool
                        also perform motion correction

    Output:
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """

    if Y.ndim == 4:  # 3D data
        Y = Y[:, :, :, :init_batch]
    else:
        Y = Y[:, :, :init_batch]

    Ain, Cin, b_in, f_in, center = initialize_components(
        Y, K=k, gSig=gSig, nb=gnb, method_init=method_init)
    Ain = coo_matrix(Ain)
    b_in = np.array(b_in)
    Yr = np.reshape(Y, (Ain.shape[0], Y.shape[-1]), order='F')
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = spdiags(old_div(1., nA), 0, nr, nr) * \
        (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = spdiags(old_div(1., nA), 0, nr, nr) * (Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    if return_object:
        cnm_init = cm.source_extraction.cnmf.cnmf.CNMF(2, k=k, gSig=gSig, Ain=Ain, Cin=Cin, b_in=np.array(
            b_in), f_in=f_in, method_init=method_init, p=p, gnb=gnb, **kwargs)

        cnm_init.estimates.A, cnm_init.estimates.C, cnm_init.estimates.b, cnm_init.estimates.f, cnm_init.estimates.S,\
            cnm_init.estimates.YrA = Ain, Cin, b_in, f_in, np.maximum(np.atleast_2d(Cin), 0), YrA

        #cnm_init.g = np.array([-np.poly([0.9]*max(p,1))[1:] for gg in np.ones(k)])
        cnm_init.estimates.g = np.array([-np.poly([0.9, 0.5][:max(1, p)])[1:]
                               for gg in np.ones(k)])
        cnm_init.estimates.bl = np.zeros(k)
        cnm_init.estimates.c1 = np.zeros(k)
        cnm_init.estimates.neurons_sn = np.std(YrA, axis=-1)
        cnm_init.estimates.lam = np.zeros(k)
        cnm_init.dims = Y.shape[:-1]
        cnm_init.params.set('online', {'init_batch': init_batch})

        return cnm_init
    else:
        return Ain, np.array(b_in), Cin, f_in, YrA


def old_div(a, b):
    """
    Equivalent to ``a / b`` on Python 2 without ``from __future__ import
    division``.

    TODO: generalize this to other objects (like arrays etc.)
    """
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return a // b
    else:
        return a / b


if __name__ == "__main__":
    main()
