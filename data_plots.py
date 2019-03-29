__author__ = "Albert Qu"

import matplotlib.pyplot as plt
import numpy as np
import os, h5py


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


################################################
############## Benchmark Plots #################
################################################
def benchmark_full_vs_t_thres(hfile, t_thres, opt, savepath, online_dur, only_proc=False):
    with h5py.File(hfile, 'r') as f:
        if only_proc:
            t_all = f['t_online']
        else:
            t_all = np.array(f['t_online'][-online_dur:]) + np.array(f['t_bmi']) + np.array(f['t_wait'])
        tt_arr = np.full_like(t_all, t_thres)
        analysis_time_contrast(tt_arr, 't_thres={}'.format(t_thres), t_all, opt, savepath)
