#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Mon Dec 12 10:36:44 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i],
                                                    t_start=1,
                                                    t_stop=ticks)
    return nsat.exportAER(SL)


if __name__ == '__main__':
    sim_ticks = 500             # Simulation ticks
    N_CORES = 1                 # Number of cores
    N_NEURONS = [2]             # Number of neurons per core
    N_INPUTS = [1]              # Number of inputs per core
    N_STATES = [4]              # Number of states per core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Configuration NSAT class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 monitor_weights=True,
                                 plasticity_en=[True],
                                 ben_clock=True)

    # Transition matrix group 0
    cfg.core_cfgs[0].A[0] = [[-5, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Transition matrix group 1
    cfg.core_cfgs[0].A[1] = [[-2,  OFF,  OFF, OFF],
                             [0,   -5,  OFF, OFF],
                             [OFF, OFF, 0, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix group 0
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Sign matrix group 1
    cfg.core_cfgs[0].sA[1] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100
    cfg.core_cfgs[0].Xth[1] = 175
    # Refractory period
    cfg.core_cfgs[0].t_ref[1] = 0
    cfg.core_cfgs[0].t_ref[0] = 120
    # Bias
    cfg.core_cfgs[0].b[0] = [5, 0, 0, 0]
    cfg.core_cfgs[0].b[1] = [5, 0, 5, 0]
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([0, 0, 0, 0], 'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = [0, MAX, MAX, MAX]
    cfg.core_cfgs[0].Xreset[1] = [0, MAX, MAX, MAX]
    # Turn on reset
    cfg.core_cfgs[0].XresetOn[0] = [True, False, False, False]
    cfg.core_cfgs[0].XresetOn[1] = [True, False, False, False]

    # Enable plasticity per state
    cfg.core_cfgs[0].plastic[1] = True
    # Enable STDP per state group 0
    cfg.core_cfgs[0].stdp_en[0] = True
    # Enable STDP per state group 1
    cfg.core_cfgs[0].stdp_en[1] = True
    # Global modulator state group 1
    cfg.core_cfgs[0].modstate[1] = 2

    # Set NSAT parameters mapping function
    nmap = np.zeros((N_NEURONS[0],), dtype='int')
    nmap[0] = 0
    nmap[1] = 1
    cfg.core_cfgs[0].nmap = nmap

    # Set learning parameters mapping function
    lrnmap = np.zeros((nsat.N_GROUPS, N_STATES[0]), dtype='int')
    lrnmap[nmap[0], 1] = 0
    lrnmap[nmap[1], 1] = 1
    cfg.core_cfgs[0].lrnmap = lrnmap

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 1, 0] = 40
    W[1, 2, 1] = 50

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, 1, 0] = True
    CW[1, 2, 1] = True

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Build external events (spikes)
    freqs = [5]
    ext_evts_data = RegularSpikingStimulus(freqs, sim_ticks)
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_two_neurons_stdp')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
##    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
##                                             prefix='test_two_neurons_stdp')
##    intel_fpga_writer.write()
##    intel_fpga_writer.write_globals()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]
    wt = c_nsat_reader.read_c_nsat_weights_evo(2)[0]
    out_spikelist = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events+'_core_0.dat'),
                                   sim_ticks=sim_ticks, id_list=[0])

    # Plot the results
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(4, 2, 1)
    ax.plot(states_core0[:-1, 0, 0], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$V_m$')

    ax = fig.add_subplot(4, 2, 2)
    ax.plot(states_core0[:-1, 1, 0], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$V_m$')

    ax = fig.add_subplot(4, 2, 3)
    ax.plot(states_core0[:-1, 0, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')

    ax = fig.add_subplot(4, 2, 4)
    ax.plot(states_core0[:-1, 1, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')

    ax = fig.add_subplot(4, 2, 5)
    ax.plot(states_core0[:-1, 0, 3], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')

    ax = fig.add_subplot(4, 2, 6)
    ax.plot(states_core0[:-1, 1, 1], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')

    ax = fig.add_subplot(4, 2, 7)
    ax.plot(wt[:, 0, 1], 'r', lw=3)
    ax.set_ylabel('$w$')

    ax = fig.add_subplot(4, 2, 8)
    ax.plot(wt[:, 1, 1], 'r', lw=3)
    ax.set_ylabel('$w$')

    fig = plt.figure(figsize=(10, 5))
    for i in out_spikelist[0].spike_times:
        plt.plot(wt[:, 1, 1], 'r', lw=3)
        plt.axvline(i, color='k', lw=1)
    for i in out_spikelist[1].spike_times:
        plt.axvline(i, color='b', lw=1)
    plt.show()
