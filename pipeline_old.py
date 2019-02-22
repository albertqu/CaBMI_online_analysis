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
from .BMI_acq import set_up_bmi

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
    eflag = False
    while not eflag:
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
                eflag = True
                break
    print("Starting Initializing Online")
    cnm.initialize_online()
    dur0 = bp2-bp0
    dur = time.time() - bp0
    ha, ma, sa = second_to_hmt(dur0)
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
    return dur0, dur


def online_process(folder, ns, ns_start, cnm, query_rate=0, view=False):
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
            """
    # TODO: ADD A LENGTH OF EXPR
    fls = cnm.params.get('data', 'fnames')
    init_batch = cnm.params.get('online', 'init_batch')
    epochs = cnm.params.get('online', 'epochs')
    out = None
    #cnm.initialize_online()
    # TODO: First finish processing the remaining files in baseline (fls[0]),
    # TODO: THEN start to listen and query for all the new files
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
            t, old_comps = uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online)
            #temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, t-1] # TODO: HERE FEEDS TO BMI
    cnm.base = t
    # %% SETTING UP BMI
    #feed_to_bmi = set_up_bmi(cnm)
    # %% online file processing starts
    print('--- Now processing online files ---')
    ns_counter = ns_start
    cnm.Ts = cnm.estimates.C_on.shape[-1]
    exp_files = int(cnm.Ts * (epochs - 1) / epochs)
    fullns = os.path.join(folder, ns) # Full image name scheme
    q_intv = query_rate * 1.0 / cnm.params.get('data', 'fr')  # query interval
    # TODO: DETERMINE CASES IN WHICH EXPERIMENT WOULD TERMINATE, THEN HANDLE ACCORDINGLY, NOW ONLY Keyboard Interrupt
    try:
        while ns_counter < exp_files+ns_start:
            target = fullns.format(ns_counter)
            if os.path.exists(target):
                frame = cm.load(target)
                t, old_comps = uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online)
                ns_counter += 1
                # FEEDING DATA TO BMI
                #temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M, t - cnm.params.get('bmi', 'dynamicLen'): t]
                #feed_to_bmi(temp)
            else:
                print("Waiting for new file, {}".format(target))
                time.sleep(q_intv)

    except KeyboardInterrupt as e:
        print("Keyboard Interrupted!")
        print("Warning: {}".format(e.args))
        print("Only processed {} frames".format(t))

    # %% final processing
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


def uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online):
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


# %%
def main():
    pass

    # IMPORTANT DATA PATHS
    data_root = "/Users/albertqu/Documents/7.Research/BMI"
    data0, data1 = os.path.join(data_root, "data0"), os.path.join(data_root, "data1")
    logfile = os.path.join(data_root, "online.log")
    bas_tif, bigtif, petitif = 'base.tif', 'bigtif.tif', 'petitif.tif'
    base_flag, ext, frame_ns = "base.tif", 'tif', "{}.tiff"
    folder = data1

    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename=logfile,
                        level=logging.INFO)


    # PARAMETER SETTING
    fr = 40  # frame rate (Hz)
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
    init_batch = 500  # number of frames for initialization (presumably from the first file)
    K = 2  # initial number of components
    epochs = 9  # number of passes over the data
    show_movie = False # show the movie as the data gets processed
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

# %% fit online

    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.base_acq, cnm.init_proc = base_prepare(folder, base_flag, cnm)
    online_process(data1, frame_ns, 1000, cnm)


# %% plot contours

    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(cnm.params.get("data", "fnames")[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

# %% view components
    cnm.estimates.view_components(img=Cn)

    return cnm


if __name__ == "__main__":
    main()


