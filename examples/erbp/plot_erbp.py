import pickle
import numpy as np
import matplotlib.pylab as plt
import experimentTools as ext
from matplotlib import gridspec
from pyNSATlib import nsat_reader
# from experimentLib import *

if __name__ == '__main__':
    # idir = "005__05-04-2017/"
    idir = "Results/016__22-09-2017/"
    cfg = ext.load(idir+'cfg_train.pkl')

    nsat_stats = np.cumsum(pickle.load(file('Results/016__22-09-2017/stats_nsat.pkl','r')),axis=0)
    stats_h = nsat_stats[:,:100].sum(axis=1)
    stats_o = nsat_stats[:,100:110].sum(axis=1)
    stats_e = nsat_stats[:,110:].sum(axis=1)
    weights = nsat_reader.read_synaptic_weights(cfg[0], idir+'/weights.pkl', idir +'/ptr.pkl')
    w0 = weights[:794, 794:894, 0].flatten()
    w1 = weights[793:893, 894:(894+10), 0].flatten()
    w2 = weights[904:, 794:894, 1].flatten()
    W = [w0,w1,w2]
    epo, err = np.array(pickle.load(file(idir+'pip.pkl', 'r'))).T
    print(epo)
    # w = cfg.core_cfgs[0].W[:794, 794:894, 1]
    # np.append(w, cfg.core_cfgs[0].W[:])

    epo_tf, acc_tf, macs_tf = pickle.load(file('/home/eneftci/Projects/simulations/tensorflow/Results/002__21-09-2017/acc.pkl','r')).T
    epo_tf2, acc_tf2, macs_tf2 = pickle.load(file('/home/eneftci/Projects/simulations/tensorflow/Results/003__21-09-2017/acc.pkl','r')).T

    fig = plt.figure(figsize=(15, 4))
    plt.gcf().set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 2, 2, 2], height_ratios=[2])
    fig.subplots_adjust(wspace=0.3, hspace=0.2, top= .82, bottom=.16, left=.06, right = .98)
    # ax = fig.add_subplot(121, aspect=1)
    ax = plt.subplot(gs[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['bottom'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(a) eRBP Network',
            fontsize=12,
            weight='bold')
    
    ax = plt.subplot(gs[1])
    ax.semilogx(epo_tf, 100-acc_tf*100, 'bx-', lw=3, label = 'GPU 784-100-10')
    ax.semilogx(epo_tf2, 100-acc_tf2*100, 'rx-', lw=3, label = 'GPU 784-30-10')
    ax.semilogx(epo, 100-err, 'ko-', lw=3, markerfacecolor='none', label = 'NSAT 784-100-10')
    print(epo.shape, err.shape)
    ticks = ax.get_xticks()
    ticks = [int(i) for i in ticks]
    ticks = [2,15,30,100,300]
    ax.set_xticks(ticks)
    # ax.set_xlim([1, 36])
    ax.set_xticklabels(ticks, fontsize=14, weight='bold')
    ax.set_ylim([0, 10])
    ticks = ax.get_yticks()
    ticks = [int(i) for i in ticks]
    ax.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax.axhline(4, c='r', lw=2, ls='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel("Training Epochs", fontsize=14, weight='bold')
    ax.set_ylabel("Test Set Error (%)", fontsize=14, weight='bold')
    ax.set_title('(b) MNIST Test Error',
            fontsize=12,
            weight='bold')
    ax.xaxis.set_tick_params(width=2, length=7, direction='out')
    ax.yaxis.set_tick_params(width=2, length=7, direction='out')
    plt.legend(loc=3)

    # ax = fig.add_subplot(122)
    ax = plt.subplot(gs[3])
    # ax.hist(w.flatten(), bins=range(-128, 127), alpha=0.7, color='k')
    ax.hist(np.concatenate([w0,w1]), bins=range(-128, 127), alpha=1.0, color='k', edgecolor='black', zorder=0)
    ax.set_xlim([-150, 150])
    ax.set_xticks([-128, 0, 127])
    ticks = ax.get_yticks()
    for i in ticks:
        ax.axhline(i, c='k', alpha=0.3, zorder=10)
    ticks = [int(i / 100) for i in ticks]
    ax.set_yticklabels(ticks, fontsize=13, alpha=1.0, weight='bold')
    ticks = ax.get_xticks()
    ax.set_xticklabels(ticks, fontsize=13, alpha=1.0, weight='bold')
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_visible(False)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_color(None)
    # ax.spines['bottom'].set_position(('outward', 10))
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title('(d) Synaptic Weights Distribution',
            fontsize=12,
            weight='bold')
    ax.set_ylabel('# Counts (x100)',
            fontsize=12,
            weight='bold',
            alpha=1.0)
    ax.set_xlabel('Synaptic Weight (8bit)',
            fontsize=12,
            weight='bold',
            alpha=1.0)

    synops = 193632073 * 100 * np.arange(30) + stats_h*10 + stats_o*20 + stats_e*110
    ax = plt.subplot(gs[2])
    ax.semilogx(macs_tf ,100-acc_tf*100   , 'bx', lw=3, label = 'GPU 784-100-10')
    ax.semilogx(macs_tf2,100-acc_tf2*100  , 'rx', lw=3, label = 'GPU 784-30-10')
    ax.semilogx(synops  ,100-err          , 'ko', lw=3, markerfacecolor='none', label = 'NSAT 784-100-10')
    print(epo.shape, err.shape)
    ticks = ax.get_yticks()
    ticks = [int(i) for i in ticks]
    ticks = range(0,10)
    ax.set_yticks(ticks)
    ax.set_ylim([0, 10])
    ax.set_yticklabels(ticks, fontsize=14, weight='bold')
    #ticks = ax.get_yticks()
    #ticks = [float(i) for i in ticks]
    #ax.set_yticklabels(ticks, fontsize=14, weight='bold')
    #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')
    plt.xticks(ax.get_xticks(), fontsize=14, weight='bold')
    ax.set_ylabel("Test Set Error (%)", fontsize=14, weight='bold')
    ax.set_xlabel("# Ops (SynOps or MACs)", fontsize=14, weight='bold')
    ax.set_title('(c) Computational Efficiency', fontsize=12,
            weight='bold')
    ax.yaxis.set_tick_params(width=2, length=7, direction='out')
    ax.xaxis.set_tick_params(width=2, length=7, direction='out')
    ax.set_xlim([1e7, 5e12])

    plt.savefig('Figure08.png', axis='tight', dpi=500)
    #193632073
    plt.show()
