__author__ = 'Albert Qu'
__acknowledgement__ = 'A Python toolbox for large scale Calcium Imaging data Analysis and behavioral analysis (CaImAn)'

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

# Base Utils
import logging
import random
import numpy as np
import os, time, copy
import matplotlib.pyplot as plt
import cv2, tifffile
from scipy.sparse import csc_matrix
from utils import counter_int, second_to_hmt, MemmapHandler, get_tif_size
from BMI_acq import set_up_bmi, baselineSimulation
from datetime import datetime
import SETTINGS


# Caiman Modules
import caiman as cm
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.source_extraction import cnmf


def cnm_init(dur_base, dur_online, fs=10):
    """Function that initializes the CNM
        Args:
            dur_base: type: int
                (seconds) duration of baseline activities
                At least 15 mins
            dur_online: type: int
                (seconds) duration of online activities,
                has to be integer multiples of dur_base

        Returns:
            OnAcid Object featuring a C_on with length
            [dur_base+dur_online].
        """

    # IMPORTANT DATA PATHS
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename=logfile,
                        level=logging.INFO)

    # PARAMETER SETTING
    fr = 80  # frame rate (Hz) TODO: CHANGED TO 10 SEE IF IT AFFECTS
    decay_time = 0.5  # approximate length of transient event in seconds
    gSig = (4, 4)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 2.5  # minimum SNR for accepting new components
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))  # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(10./ds_factor).astype('int')  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 0.9  # soace correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 500  # number of frames for initialization (presumably from the first file)
    K = 2  # initial number of components
    epochs = int(np.ceil(dur_online / dur_base)) + 1  # number of passes over the data
    exprLen = epochs * dur_base
    if exprLen > dur_online + dur_base:
        logging.warning("Experiment length must be integer multiples of baseline. Therefore, the length of online "
                        "period is rounded up to {}".format(int(exprLen) - int(dur_base)))
    show_movie = False  # show the movie as the data gets processed
    print("Frame rate: {}".format(fr))
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
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.base = dur_base * fs
    return cnm


def base_prepare(folder, bfg, cnm, view=False, debug=False, batch_tif=10, **kwargs):
    # TODO: Takes in folder, bfg (baseline flag) and cnm objec
    """Implements the caiman online pipeline initialization procedure based on
        baseline activity.

        Args:

            bfg: 1) string: one baseline tif
                2) tuple: (namescheme(e.g. online_{}.tiff), start, end(exclusive))

            cnm: OnAcid
                the OnACID object being used

            query_rate: float
                used to calculate q_intv (query interval), which is
                query_rate * (1 / fr)
                fr: frame_rate in 'self.params.data'
                query interval is the sleep time between two directory
                check of new frames to save computation

            view: boolean
                flag for plotting the dynamic plots of the temporal
                components

            init_batch: int
                number of frames to be processed during initialization

            epochs: int
                number of passes over the data

            motion_correct: bool
                flag for performing motion correction

            kwargs: dict
                additional parameters used to modify self.params.online']
                see options.['online'] for details

        Returns:
            cnm: OnAcid Object
            field updated:
                Ts: int
                    length of the total experiment
                base: int
                    length of the baseline file
                baseline: ndarray
                    array with dimension number_of_baseline components
                base_proc: tuple
                    data acquisition time, baseline process time
            """
    bp0 = time.time()  # SET Breakpoint
    bp1 = bp0
    counter = 0
    fnames = []
    memH = None
    if isinstance(bfg, tuple):
        bflag, start, end = bfg
        bfile = None
    else:
        bfile, start, end = bfg, 0, 1
        bflag = None
    # TODO: Define bfile outside of the loop, and only string format when found new bfile
    while start < end:
        bp2 = time.time()
        if (bp2 - bp1) > counter_int:
            counter += 1
            print("Time Since Start: {}h:{}m:{}s".format(*second_to_hmt(counter * counter_int)))
            if debug:
                print("Last File {}".format(bfile))
            bp1 = bp2
            # When baseline found, terminate the querying and start initial processing
        if bflag:
            bfile = bflag.format(start)
        fname = os.path.join(folder, bfile)
        if os.path.exists(fname):  # IF TiffFile pages length is 1, drop in memmap, with given duration
            if debug:
                logging.info("Found {}".format(fname))
            if memH is None: # You don't want to know what is inside
                T, dims = get_tif_size(fname)
                if T == 1:
                    if 'outpath' in kwargs:
                        outpath = kwargs['outpath']
                    else:
                        outpath = folder
                    memH = MemmapHandler(outpath, (cnm.base,)+dims, batch_tif=batch_tif)
                else:
                    memH = 0
            if memH:
                memH.add_data(tifffile.imread(fname))
            else:
                fnames.append(fname)
            start += 1
    if memH:
        fnames = memH.close()
    cnm.params.change_params({"fnames": fnames})
    print("Starting Initializing Online")
    cnm.initialize_online()
    print(cnm.estimates.C_on.shape[-1])
    fls = cnm.params.get('data', 'fnames')
    init_batch = cnm.params.get('online', 'init_batch')
    out = None
    extra_files = len(fls) - 1
    init_files = 1
    t = init_batch
    t_online = []
    cnm.comp_upd = []
    cnm.t_shapes = []
    cnm.t_detect = []
    cnm.t_motion = []
    cnm.dtypes_min_max = [] # TODO: validation step, delete it afterwards

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
            t, old_comps = uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online)
            # temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, t-1] # TODO: HERE FEEDS TO BMI
    dur0 = bp2-bp0
    dur = time.time() - bp0
    ha, ma, sa = second_to_hmt(dur0)
    hd, md, sd = second_to_hmt(dur)
    print("BaseLine Data Acquisition Time: {}H:{}M:{}S\n".format(ha, ma, sa) +
          "Total Processing Time: {}H:{}M:{}S".format(hd, md, sd))
    # Other CNM params useful for the online pipeline
    cnm.base = t
    cnm.Ts = cnm.estimates.C_on.shape[-1]
    cnm.baseline = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, :cnm.base]
    cnm.first_run = True
    cnm.t_online = t_online
    cnm.base_proc = [dur0, dur]
    if view:
        Cn = cm.load(cnm.params.get("data", "fnames")[0]).local_correlations(
            swap_dim=False)
        cnm.estimates.plot_contours(img=Cn, display_numbers=False)
        # %% view components
        cnm.estimates.view_components(img=Cn)

    # TODO: ADD CODE TO PROCESS THE REMAINING DATA IN BASELINE
    return cnm


def online_process(folder, ns, ns_start, cnm, query_rate=0, view=False, timeout=10):
    # TODO: ADD ABILITY TO CHANGE C_ON LENGTH ONLINE
    """Implements the caiman online algorithm on baseline and on the online frames.
        The taken in live as they populate the [folder].
        Caiman online is initialized using the seeded or bare initialization
        methods.


        Args:

            folder: str
                the folder to which the frames would be populating

            ns:  str
                the naming scheme of image frames, e.g. "frame_{0}.tif",
                "{0}" is necessary for formating reasons

            ns_start: int
                the starting frame number in the naming scheme, e.g.
                ns_start would be 0 if the first frame would be called
                "frame_0.tif"

            cnm: OnAcid
                the OnACID object being used

            query_rate: float
                used to calculate q_intv (query interval), which is
                query_rate * (1 / fr)
                fr: frame_rate in 'self.params.data'
                query interval is the sleep time between two directory
                check of new frames to save computation

            view: boolean
                flag for plotting the dynamic plots of the temporal
                components

            motion_correct: bool
                flag for performing motion correction

            kwargs: dict
                additional parameters used to modify self.params.online']
                see options.['online'] for details

        Returns:
            cnm: OnAcid Object
            field updated:
                Ts: int
                    length of the total experiment
            """
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    if not cnm.first_run:
        # TODO: LATER MODIFY TO ALLOW MULTIPLE RUNS, YET NOT NECESSARY FOR EXPR
        #cnm.baseline = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, :cnm.base]
        #cnm.estimates.C_on[:, cnm.base:] = 0
        logging.error("Expr already interrupted, please reinitialize the experiment.")
        return cnm
    else:
        cnm.first_run = False

    # %% Parameter initialization for online process
    ns_counter = ns_start
    old_comps = cnm.N
    exp_files = int(cnm.Ts - cnm.base)
    fullns = os.path.join(folder, ns)  # Full image name scheme
    q_intv = query_rate * 1.0 / cnm.params.get('data', 'fr')  # query interval
    out = None
    t = cnm.base
    # cnm.Ab_epoch = []  TODO: NO EPOCH IN ONLINE
    t_online = copy.deepcopy(cnm.t_online)

    t_wait = []
    t_bmi = []
    max_shifts_online = cnm.params.get('online', 'max_shifts_online')
    if cnm.params.get('online', 'save_online_movie'):
        fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
        out = cv2.VideoWriter(cnm.params.get('online', 'movie_name_online'),
                              fourcc, 30.0, tuple([int(2 * x) for x in cnm.params.get('data', 'dims')]))
    # %% online file processing starts
    durstart = time.time()
    print('--- Now processing online files ---')
    # TODO: DETERMINE CASES IN WHICH EXPERIMENT WOULD TERMINATE, THEN HANDLE ACCORDINGLY, NOW ONLY Keyboard Interrupt
    waitstart = time.time()
    try:
        while ns_counter < exp_files+ns_start:
            target = fullns.format(ns_counter)
            if os.path.exists(target):
                t_wait.append(time.time() - waitstart)
                frame = cm.load(target)
                t, old_comps = uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online)
                bmistart = time.time()
                ns_counter += 1
                temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, t - cnm.params.get('bmi', 'dynamicLen'): t]
                cnm.feed_to_bmi(temp)
                t_bmi.append(time.time()-bmistart)
                waitstart = time.time()
            else:
                if time.time() - waitstart > timeout:
                    logging.error("File Search Timeout!")
                    break
                print("Waiting for new file, {}".format(target))
                cnm.ns = ns_counter

    except KeyboardInterrupt as e:
        print("Keyboard Interrupted!")
        print("Warning: {}".format(e.args))
        print("Only processed {} frames".format(t))


    # %% final processing
    if cnm.params.get('online', 'normalize'):
        cnm.estimates.Ab /= 1. / cnm.img_norm.reshape(-1, order='F')[:, np.newaxis]
        cnm.estimates.Ab = csc_matrix(cnm.estimates.Ab)
    cnm.estimates.A, cnm.estimates.b = cnm.estimates.Ab[:, cnm.params.get('init', 'nb'):].toarray(), \
                                       cnm.estimates.Ab[:, :cnm.params.get('init', 'nb')].toarray()
    cnm.estimates.C, cnm.estimates.f = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, :], \
                                       cnm.estimates.C_on[:cnm.params.get('init','nb'),:]
    noisyC = cnm.estimates.noisyC[cnm.params.get('init', 'nb'):cnm.M, :]
    cnm.estimates.YrA = noisyC - cnm.estimates.C
    cnm.estimates.bl = [osi.b for osi in cnm.estimates.OASISinstances] if hasattr(
        cnm, 'OASISinstances') else [0] * cnm.estimates.C.shape[0]
    if cnm.params.get('online', 'save_online_movie'):
        out.release()
    if cnm.params.get('online', 'show_movie'):
        cv2.destroyAllWindows()
    cnm.t_online = t_online
    logging.error("Expr ended, please close the object by calling [close_online]")
    cnm.t_full = time.time() - durstart
    cnm.t_wait = t_wait
    cnm.t_bmi = t_bmi
    return cnm


def uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online):
    cnm.dtypes_min_max.append((frame.dtype, np.min(frame), np.max(frame)))
    t_frame_start = time.time()
    if np.isnan(np.sum(frame)):
        raise Exception('Frame ' + str(t) + ' contains NaN')
    if t % 100 == 0:
        print(str(t) + ' frames have been processed in total. ' + str(cnm.N - old_comps) +
              ' new components were added. Total # of components is '
              + str(cnm.estimates.Ab.shape[-1] - cnm.params.get('init', 'nb')))
        old_comps = cnm.N

    frame_ = frame.copy().astype(np.float32)
    if cnm.params.get('online', 'ds_factor') > 1:
        frame_ = cv2.resize(frame_, cnm.img_norm.shape[::-1])

    if cnm.params.get('online', 'normalize'):
        frame_ -= cnm.img_min  # make data non-negative
    t_mot = time.time()
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
    cnm.t_motion.append(time.time() - t_mot)

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
    t_online.append(time.time() - t_frame_start)
    return t, old_comps


def vis_neuron(sigs, seq=None, save=False):
    if seq is None:
        seq = np.arange(len(sigs))
    elif isinstance(seq, tuple):
        low, high = seq
        if low is None:
            low = 0
        if high is None:
            high = len(sigs)
        seq = np.arange(low, high)
    seqlen = len(seq)
    goodfac = max([i for i in range(1, int(np.sqrt(seqlen))+1) if seqlen % i == 0])
    complement = seqlen // goodfac
    best_col, best_row = min(goodfac, complement), max(goodfac, complement)
    fig, axes = plt.subplots(nrows=best_row, ncols=best_col, figsize=(20, 10))
    rowlen = axes.shape[1]
    for i in range(len(axes)):
        for j, ax in enumerate(axes[i]):
            ax.plot(sigs[i*rowlen+j])
            ax.set_title(seq[0] + i*rowlen+j)
    if save:
        fig.savefig('neurons_{}_from{}to{}.png'.format(sigs.shape, seq[0], seq[-1]))


def random_test():
    # Test If setting random seed would make the process deterministic, TEST NEGATIVE
    data_root = "/Users/albertqu/Documents/7.Research/BMI"
    data0, data1 = os.path.join(data_root, "data0"), os.path.join(data_root, "data1")
    fullseries = os.path.join(data_root, 'full_series2')
    logfile = os.path.join(data_root, "online.log")
    bas_tif, bigtif, petitif = 'base2.tif', 'bigtif.tif', 'petitif.tif'
    base_flag, ext, frame_ns = "base_2_d1_256_d2_256_d3_1_order_F_frames_9000_.mmap", 'tif', "online_{}.tiff"
    basedir = os.path.join(data_root, 'full_data')
    #randSeed = 10
    for randSeed in [i * 10 for i in range(1, 4)]:
        random.seed(randSeed)
        np.random.seed(randSeed)
        cnm = cnm_init(15 * 60, 15 * 60 * 4)
        cnm = base_prepare(basedir, bas_tif, cnm)
        # visualize_neurons(cnm.baseline)
        E1, E2 = [0, 1], [2, 3]
        pc, T1 = baselineSimulation(cnm, E1, E2)
        print("Recommending T1: {}, with {}% correct.".format(T1, pc))
        T1 = -0.77
        set_up_bmi(cnm, E1, E2, T1)
        online_process(fullseries, frame_ns, 0, cnm)
        np.save(os.path.join(basedir, 'online_seed_{}_2.npy'.format(randSeed)), cnm.estimates.C)

    # %% plot contours

    # logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    # Cn = cm.load(cnm.params.get("data", "fnames")[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    # cnm.estimates.plot_contours(img=Cn, display_numbers=False)

    # %% view components
    # cnm.estimates.view_components(img=Cn)
    return cnm


def close_online(cnm, data_root, folder, del_mergetif=True, **kwargs):
    # TODO: Merge all files to one bigtiff?
    log = os.path.join(data_root, 'analysis_data{}online_log'.format(SETTINGS.PATH_SPLITTER))
    dt = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    try:
        del cnm.feed_to_bmi
    except:
        pass
    if 'dims' in cnm.__dict__:
        del cnm.dims
    if 'saveopt' in kwargs:
        savefil = 'onacid_{}_{}_t={}.hdf5'.format(folder.split(SETTINGS.PATH_SPLITTER)[-1], kwargs['saveopt'], dt)
    else:
        savefil = 'onacid_{}_{}.hdf5'.format(folder.split(SETTINGS.PATH_SPLITTER)[-1], dt)
    if del_mergetif:
        for f in cnm.params.get('data', 'fnames'):
            if f.find("mergetif") != -1 and os.path.exists(f):
                    os.remove(f)
    cnm.save(os.path.join(log, savefil))
    print("Object successfully closed!")


def cnm_benchmark(cnm, data_root, folder, **kwargs):
    performance = os.path.join(data_root, 'analysis_data{}onacid_performance'.format(SETTINGS.PATH_SPLITTER))
    import h5py
    #fp = h5py.File(os.path.join(performance, 'onacid_{}_seed{}.hdf5'.format(folder.split('/')[-2], randseed)),
    # mode='a')
    if 'saveopt' in kwargs:
        savefil = 'onacid_{}_{}.hdf5'.format(folder.split(SETTINGS.PATH_SPLITTER)[-1],  kwargs['saveopt'])
    else:
        savefil = 'onacid_{}.hdf5'.format(folder.split(SETTINGS.PATH_SPLITTER)[-1])
    fp = h5py.File(os.path.join(performance, savefil), mode='a')
    fp.create_dataset('comp_upd', data=cnm.comp_upd)
    fp.create_dataset('t_online', data=cnm.t_online)
    fp.create_dataset('t_shapes', data=cnm.t_shapes)
    fp.create_dataset('t_detect', data=cnm.t_detect)
    fp.create_dataset('t_motion', data=cnm.t_motion)
    fp.create_dataset('t_bmi', data=cnm.t_bmi)
    fp.create_dataset('t_wait', data=cnm.t_wait)
    fp.attrs['t_full'] = cnm.t_full
    fp.create_dataset('C', data=cnm.estimates.C)
    fp.create_dataset('A', data=cnm.estimates.A)
    fp.close()


def analysis_time_contrast(t1, opt1, t2, opt2, savepath, draw_thres=True):
    low1, high1 = min(min(t1), min(t2)) * 0.8, max(max(t1), max(t2)) * 1.1
    linear1 = np.linspace(low1, high1)
    olier = len(np.where(t2 > t1)[0])
    fracless = np.round(olier / len(t2), 3)
    plt.subplot(121)
    plt.scatter(t1, t2, s=30, alpha=0.8, c=np.arange(len(t1)), cmap="coolwarm")
    if draw_thres:
        plt.plot(linear1, linear1, color='#FFD700')
    plt.xlabel("t1_{} (s)".format(opt1))
    plt.ylabel('t2_{} (s)'.format(opt2))
    cbar = plt.colorbar()
    cbar.set_label("Frame #")
    plt.xlim((low1, high1))
    plt.ylim((low1, high1))
    plt.subplot(122)
    sum1 = np.cumsum(t1)
    sum2 = np.cumsum(t2)
    low2, high2 = min(sum1[0], sum2[0]) * 0.8, max(sum1[-1], sum2[-1]) * 1.1
    linear2 = np.linspace(low2, high2)
    plt.plot(sum1, sum2)
    plt.plot(linear2, linear2)
    plt.xlabel("t1_{}".format(opt1))
    plt.ylabel('t2_{}'.format(opt2))
    plt.xlim((low2, high2))
    plt.ylim((low2, high2))
    plt.legend(['contrast', "y=x"])
    plt.suptitle('# points t2 > t1: {}, fraction:{}'.format(olier, fracless))
    plt.savefig(os.path.join(savepath, "{}_VS_{}.png".format(opt1, opt2)))


def full_vs_t_thres(obj, t_thres, opt, savepath, online_dur, only_proc=False):
    if only_proc:
        t_all = obj.t_online
    else:
        t_all = np.array(obj.t_online[-online_dur:]) + np.array(obj.t_bmi) + np.array(obj.t_wait)
    tt_arr = np.full_like(t_all, t_thres)
    analysis_time_contrast(tt_arr, 't_thres={}'.format(t_thres), t_all, opt, savepath)


def ensemble_test(cnm, E1, E2, outpath=None, saveopt=None):
    all_E = E1+E2
    ensemble = cnm.estimates.C[all_E]
    ensemble_online = ensemble[:, cnm.base:]
    seqlen = ensemble.shape[0]
    goodfac = max([i for i in range(1, int(np.sqrt(seqlen)) + 1) if seqlen % i == 0])
    complement = seqlen // goodfac
    best_col, best_row = min(goodfac, complement), max(goodfac, complement)
    fig, axes = plt.subplots(nrows=best_row, ncols=best_col, figsize=(20, 10))
    rowlen = axes.shape[1]
    for i in range(len(axes)):
        for j, ax in enumerate(axes[i]):
            target = i * rowlen + j
            sigs = np.vstack((ensemble_online[target], cnm.raw_sig[target], cnm.base_vals[target],
                               cnm.raw_sig[target]-cnm.base_vals[target])).T
            ax.plot(sigs)
            ax.legend(['Online', 'Smoothed', 'Base_vals', 'Filtered'])
            ax.set_title('Neuron {}'.format(all_E[target]))
    if saveopt is not None and outpath is not None:
        fig.savefig(os.path.join(outpath, "Ensemble Neuron With Filters {}".format(saveopt)))



# %%
def demo():
    pass
    out_root = SETTINGS.out_root
    data_root = SETTINGS.data_root  # TODO: make the directories easy for change
    merge_outpath = os.path.join(out_root, SETTINGS.merge)
    data0, data1 = os.path.join(data_root, "data0"), os.path.join(data_root, "data1")
    fullseries = os.path.join(data_root, SETTINGS.series)
    logfile = os.path.join(out_root, "online.log")
    ext, frame_ns = 'tif', "online_{}.tiff"
    basedir = os.path.join(data_root, SETTINGS.basedir)
    base_dur, online_dur = 15 * 60, 15 * 60 * 4
    fs = 10 # DATA Framerate
    base_flag = 'online_{}.tiff', 0, base_dur * fs
    cnm = cnm_init(base_dur, online_dur, fs)
    cnm = base_prepare(basedir, base_flag, cnm, batch_tif=10, outpath=merge_outpath)
    #visualize_neurons(cnm.baseline)
    E1, E2 = [0, 1], [2, 3]
    pc, T1 = baselineSimulation(cnm, E1, E2)
    print("Recommending T1: {}, with {}% correct.".format(T1, pc))
    set_up_bmi(cnm, E1, E2, T1)
    online_process(fullseries, frame_ns, 0, cnm)
    opt = SETTINGS.OPT
    cnm_benchmark(cnm, out_root, fullseries, saveopt=opt)
    close_online(cnm, out_root, fullseries, saveopt=opt)
    os.system("""osascript -e 'display notification "Job Done" with title "{}"'""".format('pipeline'))
    full_vs_t_thres(cnm, 0.075, opt, os.path.join(out_root, 'plots'), online_dur * fs)
    # TODO: ADD FUNCTION TO SAVE THE INDIVIDUAL TIFFS?


# %% plot contours

    #logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    #Cn = cm.load(cnm.params.get("data", "fnames")[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    #cnm.estimates.plot_contours(img=Cn, display_numbers=False)

# %% view components
    #cnm.estimates.view_components(img=Cn)
    return cnm


if __name__ == "__main__":
    demo()


