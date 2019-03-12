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
import scipy
import matplotlib.pyplot as plt
import cv2
from data_proc import merge_tiffs
import SETTINGS
from scipy.sparse import csc_matrix
from utils import counter_int, second_to_hmt
from BMI_acq_old import set_up_bmi, baselineSimulation
from pipeline import cnm_init, base_prepare, close_online, vis_neuron, analysis_time_contrast

# Caiman Modules
import caiman as cm
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.source_extraction import cnmf


def online_process(folder, ns, ns_start, cnm, query_rate=0, view=False, timeout=10):
    # TODO: ADD ABILITY TO CHANGE C_ON LENGTH ONLINE
    """Implements the caiman online algorithm on baseline and on the online frames.
        The taken in live as they populate the [folder].
        Caiman online is initialized using the seeded or bare initialization
        methods.

        Greedy approach no longer has t_shapes and t_detect


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
        logging.error("Expr already interrupted, please reinitialize the experiment.")
        #return cnm
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
    print(cnm.estimates.YrA.shape, cnm.estimates.A.shape, cnm.estimates.C_on.shape)
    cnm.estimates.C_on = cnm.estimates.C_on[:cnm.M, :]
    # cnm.Ab_epoch = []  TODO: NO EPOCH IN ONLINE
    t_online = copy.deepcopy(cnm.t_online)
    cnm.t_greedy = [0 for _ in range(len(t_online))]
    t_wait = []
    t_bmi = []
    max_shifts_online = cnm.params.get('online', 'max_shifts_online')
    if cnm.params.get('online', 'save_online_movie'):
        fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
        out = cv2.VideoWriter(cnm.params.get('online', 'movie_name_online'),
                              fourcc, 30.0, tuple([int(2 * x) for x in
                                                    cnm.params.get('data', 'dims')]))
    # %% online file processing starts
    durstart = time.time()
    print('--- Now processing online files ---')
    # TODO: DETERMINE CASES IN WHICH EXPERIMENT WOULD TERMINATE, THEN HANDLE ACCORDINGLY, NOW ONLY Keyboard Interrupt
    waitstart = time.time()
    try:
        while ns_counter < exp_files + ns_start:
            target = fullns.format(ns_counter)
            if os.path.exists(target):
                t_wait.append(time.time() - waitstart)
                frame = cm.load(target)
                t, old_comps = uno_proc(frame, cnm, t, old_comps, t_online, out,
                                        max_shifts_online, cnm.estimates.Ab.T)
                bmistart = time.time()
                ns_counter += 1
                temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M,
                       t - cnm.params.get('bmi', 'dynamicLen'): t]
                cnm.feed_to_bmi(temp)
                t_bmi.append(time.time() - bmistart)
                waitstart = time.time()
            else:
                if time.time() - waitstart > timeout:
                    logging.error("File Search Timeout!")
                    break
                print("Waiting for new file, {}".format(target))
                cnm.ns = ns_counter
                time.sleep(q_intv)

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
                                       cnm.estimates.C_on[:cnm.params.get('init', 'nb'), :]
    noisyC = cnm.estimates.noisyC[cnm.params.get('init', 'nb'):cnm.M, :]
    cnm.estimates.YrA = noisyC - cnm.estimates.C
    cnm.estimates.bl = [osi.b for osi in cnm.estimates.OASISinstances] if hasattr(
        cnm, 'OASISinstances') else [0] * cnm.estimates.C.shape[0]
    if cnm.params.get('online', 'save_online_movie'):
        out.release()
    if cnm.params.get('online', 'show_movie'):
        cv2.destroyAllWindows()
    cnm.t_online = np.array(t_online)
    logging.error("Expr ended, please close the object by calling [close_online]")
    cnm.t_full = time.time() - durstart
    cnm.t_wait = t_wait
    cnm.t_bmi = t_bmi
    cnm.t_other = cnm.t_online - np.array(cnm.t_motion) - np.array(cnm.t_greedy)
    return cnm


def uno_proc(frame, cnm, t, old_comps, t_online, out, max_shifts_online, AT=None):
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
    frm = frame_cor.reshape(-1, order='F')

    if AT is not None:
        dotstart = time.time()
        cnm.estimates.C_on[:, t] = AT @ frm
        greedydur = time.time() - dotstart
        cnm.t_greedy.append(greedydur)
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


def online_process_comb(folder, ns, ns_start, cnm, query_rate=0, view=False, timeout=10):
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
        logging.error("Expr already interrupted, please reinitialize the experiment.")
        #return cnm
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
    #cnm.estimates.C_on = cnm.estimates.C_on[:cnm.M, :]
    # cnm.Ab_epoch = []  TODO: NO EPOCH IN ONLINE
    t_online = copy.deepcopy(cnm.t_online)
    cnm.t_greedy = [0 for _ in range(len(t_online))]
    t_wait = []
    t_bmi = []
    Adummy, Cdummy = cnm.estimates.Ab.T.copy(), np.copy(cnm.estimates.C_on[:cnm.M, :])
    max_shifts_online = cnm.params.get('online', 'max_shifts_online')
    if cnm.params.get('online', 'save_online_movie'):
        fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
        out = cv2.VideoWriter(cnm.params.get('online', 'movie_name_online'),
                              fourcc, 30.0, tuple([int(2 * x) for x in
                                                    cnm.params.get('data', 'dims')]))
    # %% online file processing starts
    durstart = time.time()
    print('--- Now processing online files ---')
    # TODO: DETERMINE CASES IN WHICH EXPERIMENT WOULD TERMINATE, THEN HANDLE ACCORDINGLY, NOW ONLY Keyboard Interrupt
    waitstart = time.time()
    try:
        while ns_counter < exp_files + ns_start:
            target = fullns.format(ns_counter)
            if os.path.exists(target):
                t_wait.append(time.time() - waitstart)
                frame = cm.load(target)
                t, old_comps = uno_proc_comb(frame, cnm, t, old_comps, t_online, out,
                                        max_shifts_online, Adummy, Cdummy)
                bmistart = time.time()
                ns_counter += 1
                temp = cnm.estimates.C_on[cnm.params.get('init', 'nb'):cnm.M,
                       t - cnm.params.get('bmi', 'dynamicLen'): t]
                cnm.feed_to_bmi(temp)
                t_bmi.append(time.time() - bmistart)
                waitstart = time.time()
            else:
                if time.time() - waitstart > timeout:
                    logging.error("File Search Timeout!")
                    break
                print("Waiting for new file, {}".format(target))
                cnm.ns = ns_counter
                time.sleep(q_intv)

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
                                       cnm.estimates.C_on[:cnm.params.get('init', 'nb'), :]
    noisyC = cnm.estimates.noisyC[cnm.params.get('init', 'nb'):cnm.M, :]
    cnm.estimates.YrA = noisyC - cnm.estimates.C
    cnm.estimates.bl = [osi.b for osi in cnm.estimates.OASISinstances] if hasattr(
        cnm, 'OASISinstances') else [0] * cnm.estimates.C.shape[0]
    if cnm.params.get('online', 'save_online_movie'):
        out.release()
    if cnm.params.get('online', 'show_movie'):
        cv2.destroyAllWindows()
    cnm.t_online = np.array(t_online) - np.array(cnm.t_greedy)
    logging.error("Expr ended, please close the object by calling [close_online]")
    cnm.t_full = time.time() - durstart
    cnm.t_wait = t_wait
    cnm.t_bmi = t_bmi
    cnm.t_proc = cnm.t_online - np.array(cnm.t_motion) - \
                 np.array(cnm.t_detect) - np.array(cnm.t_shapes)
    cnm.estimates.C_inf = Cdummy
    cnm.estimaetes.A_init = Adummy.toarray()
    return cnm


def uno_proc_comb(frame, cnm, t, old_comps, t_online, out, max_shifts_online, AT=None, CS=None):
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
    frm = frame_cor.reshape(-1, order='F')

    if AT is not None and CS is not None:
        dotstart = time.time()
        CS[:, t] = AT @ frm
        greedydur = time.time() - dotstart
        cnm.t_greedy.append(greedydur)
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


def cnm_benchmark_compare(cnm, data_root, folder, **kwargs):
    consistency = os.path.join(data_root, 'analysis_data/onacid_consistency')
    performance = os.path.join(data_root, 'analysis_data/onacid_performance')
    import h5py
    # fp = h5py.File(os.path.join(performance, 'onacid_{}_seed{}.hdf5'.format(folder.split('/')[-2], randseed)),
    # mode='a')
    if 'saveopt' in kwargs:
        savefil = 'onacid_{}_{}.hdf5'.format(folder.split('/')[-1], kwargs['saveopt'])
    else:
        savefil = 'onacid_{}.hdf5'.format(folder.split('/')[-1])
    fp = h5py.File(os.path.join(performance, savefil), mode='a')
    fp.create_dataset('comp_upd', data=cnm.comp_upd)
    fp.create_dataset('t_online', data=cnm.t_online)
    fp.create_dataset('t_shapes', data=cnm.t_shapes)
    fp.create_dataset('t_detect', data=cnm.t_detect)
    fp.create_dataset('t_motion', data=cnm.t_motion)
    fp.create_dataset('t_bmi', data=cnm.t_bmi)
    fp.create_dataset('t_wait', data=cnm.t_wait)
    fp.create_dataset('t_greedy', data=cnm.t_greedy)
    fp.create_dataset('t_proc', data=cnm.t_proc)
    fp.attrs['t_full'] = cnm.t_full
    fp.create_dataset('C', data=cnm.estimates.C)
    fp.create_dataset('A', data=cnm.estimates.A)
    fp.create_dataset('C_on', data=cnm.estimates.C_on)
    fp.create_dataset('Ab', data=cnm.estimates.Ab.toarray())
    fp.create_dataset('C_inf', data=cnm.estimates.C_inf)
    fp.create_dataset('A_base', data=cnm.estimates.A_init)
    fp.close()


def cnm_benchmark(cnm, data_root, folder, **kwargs):
    """
    :param cnm:
    :param data_root: ROOT path where analysis data will be saved
    :param folder:
    :param kwargs:
    :return:
    """
    layer1 = os.path.join(data_root, 'analysis_data')
    if not os.path.exists(layer1):
        os.mkdir(layer1)

    consistency = os.path.join(data_root, 'analysis_data/onacid_consistency')
    performance = os.path.join(data_root, 'analysis_data/onacid_performance')
    if not os.path.exists(performance):
        os.mkdir(performance)

    import h5py
    # fp = h5py.File(os.path.join(performance, 'onacid_{}_seed{}.hdf5'.format(folder.split('/')[-2], randseed)),
    # mode='a')
    if 'saveopt' in kwargs:
        savefil = 'onacid_{}_{}.hdf5'.format(folder.split('/')[-1], kwargs['saveopt'])
    else:
        savefil = 'onacid_{}.hdf5'.format(folder.split('/')[-1])
    fp = h5py.File(os.path.join(performance, savefil), mode='a')
    fp.create_dataset('comp_upd', data=cnm.comp_upd)
    fp.create_dataset('t_online', data=cnm.t_online)
    fp.create_dataset('t_shapes', data=cnm.t_shapes)
    fp.create_dataset('t_detect', data=cnm.t_detect)
    fp.create_dataset('t_motion', data=cnm.t_motion)
    fp.create_dataset('t_bmi', data=cnm.t_bmi)
    fp.create_dataset('t_wait', data=cnm.t_wait)
    fp.create_dataset('t_greedy', data=cnm.t_greedy)
    fp.attrs['t_full'] = cnm.t_full
    fp.create_dataset('C', data=cnm.estimates.C)
    fp.create_dataset('A', data=cnm.estimates.A)
    fp.create_dataset('C_on', data=cnm.estimates.C_on)
    fp.create_dataset('Ab', data=cnm.estimates.Ab.toarray())
    fp.close()

# %%
def demo():
    pass
    data_root = "/Users/albertqu/Documents/7.Research/BMI"
    fullseries = os.path.join(data_root, 'full_series0')
    logfile = os.path.join(data_root, "online.log")
    base_flag, ext, frame_ns = "online_", 'tif', "online_{}.tiff"
    basedir = os.path.join(data_root, 'basedir0')
    # randSeed = 10
    # random.seed(randSeed)
    # np.random.seed(randSeed)
    cnm = cnm_init(15 * 60, 15 * 60 * 4)
    cnm = base_prepare(basedir, base_flag, cnm)
    # visualize_neurons(cnm.baseline)
    E1, E2 = [0, 1], [2, 3]
    pc, T1 = baselineSimulation(cnm, E1, E2)
    print("Recommending T1: {}, with {}% correct.".format(T1, pc))
    set_up_bmi(cnm, E1, E2, T1)
    online_process(fullseries, frame_ns, 0, cnm)
    # np.save(os.path.join(basedir, 'online_seed_{}.npy'.format(randSeed)), cnm.estimates.C)
    cnm_benchmark(cnm, data_root, fullseries, saveopt="greedy_fr=40")
    close_online(cnm, data_root, fullseries, saveopt="greedy_fr=40")
    # TODO: ADD FUNCTION TO SAVE THE INDIVIDUAL TIFFS?
    return cnm


if __name__ == "__main__":
    demo()


