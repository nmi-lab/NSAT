#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 17:44:18 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def SimSpikingStimulus(rates=[5, 10], t_start=1000, t_stop=4000):
    n = np.shape(rates)[0]
    SL = pyST.SpikeList(id_list=list(range(n)))
    for i in range(n):
        SL[i] = pyST.STCreate.regular_generator(rates[i],
                                                t_start=t_start,
                                                t_stop=t_stop,
                                                jitter=False)
        # SL[i] = pyST.STCreate.poisson_generator(rates[i], t_start, t_stop)
    return SL


def h(x, sigma=1):
    return np.exp(-(x)**2/(2*sigma**2))


def lateral_connectivity(n):
    x = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x[i, j] = np.abs(i - j)
    return 190 * h(x, 2) - 120 * h(x, 1)


def build_synaptic_w(m, n, k):
    units = m + n
    w = np.zeros((units, units, k), dtype='i')
    cw = np.zeros((units, units, k), dtype='i')

    # input units
    w[0, 3, 0] = 20
    w[1, 4, 0] = 20
    w[2, 5, 0] = 20

    # excitatory units
    w[3, 6, 1] = 15
    w[4, 6, 1] = 15
    w[5, 6, 1] = 15

    w[4, 3, 1] = 10
    w[3, 4, 1] = 10
    w[4, 5, 1] = 10
    w[5, 4, 1] = 10

    # inhibitory units
    w[6, 3, 1] = -35
    w[6, 4, 1] = -35
    w[6, 5, 1] = -35

    cw[w != 0] = True
    return w.astype('i'), cw.astype('i')


def pretty_fig(spks, states, t_stop=1000):
    spikes = np.zeros((4, t_stop))
    spks = spks.astype('i')

    for i in range(4):
        idx = spks[0, spks[1, :] == i]
        if len(idx) > 0:
            spikes[i, idx] = 1

    spikes[spikes == 0] = np.nan

    K = 5000
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=5, rowspan=1)
    for i in range(4):
        if i == 3:
            col = (0.192, 0.647, 0.796)
            al = 1
        else:
            col = (0.847, 0, 0.329)
            al = 1 - i*0.2
        ax1.plot(spikes[i, :K]+i, '|', c=col, ms=20, mew=3, alpha=al)

    ax1.set_ylim([0, 5])
    ax1.set_xticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.set_ylabel("Neuron", fontsize=22, weight='bold')
    ax1.set_yticklabels(ax1.get_yticks().astype('i'),
                        fontsize=18,
                        weight='bold')

    for i in range(1, 5):
        if i == 1:
            col = (0.192, 0.647, 0.796)
            al = 1
        else:
            col = (0.847, 0, 0.329)
            al = 0.2 + i*0.2
        ax2 = plt.subplot2grid((5, 5), (i, 0), colspan=5, rowspan=1)
        ax2.plot(states[:K, 4-i, 0], c=col, alpha=al)
        ax2.set_ylim([-500, 200])
        ax2.set_yticklabels(ax2.get_yticks().astype('i'),
                            fontsize=18,
                            weight='bold')
        ax2.set_ylabel("V", fontsize=18, weight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        if i != 4:
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xticks([])
        else:
            ax2.set_xlabel("Time (ticks)", fontsize=18, weight='bold')
            ax2.set_xticklabels(ax2.get_xticks().astype('i'),
                                fontsize=18,
                                weight='bold')


if __name__ == "__main__":
    sim_ticks = 60000           # Simulation ticks
    N_CORES = 1                 # Number of cores
    N_NEURONS = [4]             # Number of neurons per core
    N_INPUTS = [3]              # Number of inputs per core
    N_STATES = [4]              # Number of states per core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total number of units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Configuration class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    # Transition matrix group 0
    cfg.core_cfgs[0].A[0] = [[-7,  OFF,  OFF, OFF],
                             [0, -5, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Transition matrix group 1
    cfg.core_cfgs[0].A[1] = [[-7, OFF, OFF, OFF],
                             [0, -7,  OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix group 0
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [+1, -1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Sign matrix group 1
    cfg.core_cfgs[0].sA[1] = cfg.core_cfgs[0].sA[0].copy()

    # Refractory period group 0
    cfg.core_cfgs[0].t_ref[0] = 40
    # Refractory period group 1
    cfg.core_cfgs[0].t_ref[1] = 30
    # Bias group 0
    cfg.core_cfgs[0].b[0] = [0, 0, 0, 0]
    # Bias group 1
    cfg.core_cfgs[0].b[1] = [0, 0, 0, 0]
    # Threshold group 0
    cfg.core_cfgs[0].Xth[0] = 100
    # Threshold group 1
    cfg.core_cfgs[0].Xth[1] = 80
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([0, 0, 0, 0], 'int')
    # Reset value group 0
    cfg.core_cfgs[0].Xreset[0] = [0, MAX, MAX, MAX]
    # Reset value group 1
    cfg.core_cfgs[0].Xreset[1] = cfg.core_cfgs[0].Xreset[0].copy()
    # Turn reset on group 0
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')
    # Turn reset on group 1
    cfg.core_cfgs[0].XresetOn[1] = np.array([True, False, False, False],
                                            'bool')

    # Global modulator state group 0
    cfg.core_cfgs[0].modstate[0] = 3
    # Global modulator state group 0
    cfg.core_cfgs[0].modstate[1] = 3

    # Parameters groups mapping function
    nmap = np.array([0, 0, 0, 1], dtype='int')
    cfg.core_cfgs[0].nmap = nmap

    # Synaptic weights and adjacent matrix
    W, CW = build_synaptic_w(N_INPUTS[0], N_NEURONS[0], N_STATES[0])

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Set external events
    rates = [60, 30, 30]
    t_start, t_stop = 1, 35000
    SL = SimSpikingStimulus(rates, t_start, t_stop)
    ext_evts_data = nsat.exportAER(SL)
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_wta')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
##    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
##                                             prefix='test_wta')
##    intel_fpga_writer.write()
##    intel_fpga_writer.write_globals()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]

    in_apikelist = SL
    out_spikelist = nsat.importAER(c_nsat_reader.read_c_nsat_raw_events()[0],
                                   sim_ticks=sim_ticks,
                                   id_list=[0])

    spks = out_spikelist.convert("[times,ids]")
    spks = np.vstack([spks[0], spks[1]]).astype('int')
    pretty_fig(spks, states_core0, t_stop=t_stop)
    plt.show()
