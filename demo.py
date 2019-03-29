__author__ = 'Albert Qu'

import numpy as np
from utils import DCache
import matplotlib.pyplot as plt
import h5py
import os

def func():
    train = []
    bf = 10
    train = [np.random.randint(10) for i in range(10)]
    base = np.mean(train)
    avgs = [base]
    sigs = [base]
    for i in range(100):
        train.append(np.random.randint(10))
        avgs.append(np.mean(train[-10:]))
        sigs.append((sigs[-1] * (bf-1) + avgs[-1]) / bf)
    plt.figure()
    plt.plot(avgs, 'b-', sigs, 'r-', train[-100:], 'm-')
    plt.show()
    return train, avgs, sigs


def func2(ftr=lambda x, i: x[i]):
    """ This Function simulates a exponential decay data with noise and random spike. The goal is to obtain the
    exponential baseline with a filter.
    @ Ref: https://scipy-cookbook.readthedocs.io/items/robust_regression.html
    """

    kern = lambda x: 1 + np.e ** (-0.005*x)
    length = 200 # Duration of the experiment
    samp_rate = 30 # 30 Hz
    xlen = int(length*samp_rate)
    tline = np.linspace(0, length, xlen)
    signal = kern(tline)
    randstate = 0
    noise = 0.1
    rdm = np.random.RandomState(randstate)
    noisysig = signal + noise * rdm.randn(xlen)
    cache = np.copy(noisysig)
    wspike = 31 # ODD
    halfw = wspike // 2
    peak = 2
    coeff = peak / (wspike // 2) ** 2
    spikesf = lambda x, c: peak - coeff * (x - c) ** 2
    spike_prob = 0.2
    seg = int(1.5 * wspike)
    for i in range(xlen // seg):
        if rdm.rand() < spike_prob:
            start = seg * i
            randi = rdm.randint(start, start+seg)
            #randi = rdm.randint(start+halfw, start+seg-halfw)
            spikerange = np.arange(randi - wspike // 2, randi + wspike // 2 + 1)
            spike = spikesf(spikerange, randi)
            assert np.max(spike) == peak, "Here {} {}".format(spike, spikerange)
            noisysig[spikerange] += spike
    test_filter(ftr, noisysig)


def test_filter(ftr, raw):
    # Takes in row vector data
    xlen = len(raw)
    filtered = np.empty(xlen)
    for i in range(xlen):
        filtered[i] = ftr(raw, i)
    plt.figure()
    plt.plot(raw)
    plt.plot(filtered)
    plt.legend(['raw', 'filtered'])
    plt.show()


def std_filter():
    dc = DCache(20, 2)

    def fil(sigs, i):
        dc.add(sigs[i])
        #print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc


def ori_filter():
    dc = DCache(20, 10000)

    def fil(sigs, i):
        dc.add(sigs[i])
        # print(sigs[i], dc.get_val())
        return dc.get_val()

    return fil


def online_test():
    data_root = "/Users/albertqu/Documents/7.Research/BMI"
    performance = os.path.join(data_root, 'analysis_data/onacid_performance')
    ip = h5py.File(os.path.join(performance, 'onacid_fullseries2_1.hdf5'), 'r')
    test_filter(std_filter()[0], ip['C'])


def exp_test(data, rate=0):
    ftr, dc = std_filter()
    sigs = np.copy(data)

    kern = lambda x: 1 + np.e ** (-rate * x)
    valids = np.where(sigs != 0)[0]
    exp_base = kern(valids)
    sigs[valids] += exp_base
    test_filter(ftr, sigs)


def filter_sig(sig, ftr):
    xlen = len(sig)
    filtered = np.empty(xlen)
    for i in range(xlen):
        filtered[i] = ftr(sig, i)
    return filtered


def vis_diff_dist_pca(m1, m2, cov1, cov2, n=100):
    d1 = np.random.multivariate_normal(m1, cov1, size=n)
    d2 = np.random.multivariate_normal(m2, cov2, size=n)
    X = np.vstack((d1, d2))
    print(X.shape)
    mu = np.mean(X, axis=0)
    dmX = X - mu
    cov = dmX.T @ dmX / n
    c = ['r' if i < n else 'b' for i in range(2 * n)]
    w, v = np.linalg.eig(cov)
    x1, x2 = X[:, 0], X[:, 1]
    plt.figure(figsize=(20, 10))
    plt.scatter(x1, x2, c=c)
    arr1 = plt.arrow(mu[0], mu[1], w[0] * v[0, 0], w[0] * v[1, 0], width=0.1,
                    length_includes_head=True, color='#FFD700')
    arr2 = plt.arrow(mu[0], mu[1], w[1] * v[0, 1], w[1] * v[1, 1], width=0.1,
                    length_includes_head=True, color='#B87333')
    plt.legend([arr1, arr2], ['eigvec 1', 'eigvec 2'])
    plt.show()

if __name__ == '__main__':
    func2(ftr=std_filter()[0])


