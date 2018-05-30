#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Mon Dec 12 10:19:58 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def SimSpikingStimulus(t_sim=None):
    SL = pyST.SpikeList(id_list=[0, 1, 2, 3])
    spk_train0 = [63, 120]
    spk_train1 = [10, 75, 140]
    spk_train2 = [15, 80, 135]
    spk_train3 = [20, 130]
    SL[0] = pyST.SpikeTrain(spk_train0, t_start=1)
    SL[1] = pyST.SpikeTrain(spk_train1, t_start=1)
    SL[2] = pyST.SpikeTrain(spk_train2, t_start=1)
    SL[3] = pyST.SpikeTrain(spk_train3, t_start=1)
    return SL


if __name__ == '__main__':
    sim_ticks = 200                # Simulation ticks
    N_CORES = 1                     # Number of cores
    N_NEURONS = [4]                 # Number of neurons per core (list)
    N_INPUTS = [4]                 # Number of inputs per core (list)
    N_STATES = [4]                  # Number of states per core (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total inputs

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Main class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 tstdpmax=[64],
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 monitor_weights=True,
                                 monitor_weights_final=True,
                                 plasticity_en=[True],
                                 ben_clock=True)

    # Transition matrix
    cfg.core_cfgs[0].A[0] = [[-1, OFF, OFF, OFF],
                             [0, -1, OFF, OFF],
                             [OFF, OFF, 0, OFF],
                             [OFF, OFF, OFF, OFF]]
    cfg.core_cfgs[0].A[1] = cfg.core_cfgs[0].A[0].copy()

    # Sign matrix
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, 1]]
    cfg.core_cfgs[0].sA[1] = cfg.core_cfgs[0].sA[0].copy()

    # Refractory period
    cfg.core_cfgs[0].t_ref[0] = 0
    cfg.core_cfgs[0].t_ref[1] = 2
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100
    cfg.core_cfgs[0].Xth[1] = 100
    # Bias
    cfg.core_cfgs[0].b[0] = [30, 0, 5, 0]
    cfg.core_cfgs[0].b[1] = [30, 0, 5, 0]
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([0, 0, 0, 0], 'int')
    cfg.core_cfgs[0].Xinit[1] = np.array([0, 0, 0, 0], 'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = [0, MAX, MAX, MAX]
    cfg.core_cfgs[0].Xreset[1] = [0, MAX, MAX, MAX]
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = [True, False, False, False]
    cfg.core_cfgs[0].XresetOn[1] = [True, False, False, False]

    # Enable plasticity at states
    cfg.core_cfgs[0].plastic[0] = True
    cfg.core_cfgs[0].plastic[1] = False
    # Enable STDP
    cfg.core_cfgs[0].stdp_en[0] = True
    cfg.core_cfgs[0].stdp_en[1] = False
    # Global modulator state
    cfg.core_cfgs[0].modstate[0] = 2
    cfg.core_cfgs[0].modstate[1] = 2

    # Parameters for the STDP kernel function
    rNLRN_GROUPS = list(range(nsat.N_LRNGROUPS))
    cfg.core_cfgs[0].tstdp = np.array([64 for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].tca = np.array([[16, 36] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].hica = np.array([[1, 0, -1] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].sica = np.array([[1, 1, 1] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].tac = np.array([[-16, -36] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].hiac = np.array([[1, 0, -1] for _ in rNLRN_GROUPS],
                                     'int')
    cfg.core_cfgs[0].siac = np.array([[-1, -1, -1] for _ in rNLRN_GROUPS],
                                     'int')

    # Set parameters mapping
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')
    cfg.core_cfgs[0].nmap[0] = 0
    cfg.core_cfgs[0].nmap[1] = 1
    cfg.core_cfgs[0].nmap[2] = 1
    cfg.core_cfgs[0].nmap[3] = 1

    # Learning parameters groups mapping
    lrnmap = np.zeros((nsat.N_GROUPS, N_STATES[0]), dtype='int')
    lrnmap[0, :] = 1
    lrnmap[0, 1] = 0
    lrnmap[1, 2] = 0
    lrnmap[1, 0] = 0
    # lrnmap[:] = 0
    cfg.core_cfgs[0].lrnmap = lrnmap

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 4, 1] = 50
    W[1, 5, 1] = 50
    W[2, 6, 1] = 50
    W[3, 7, 1] = 50

    W[4, 5, 1] = 5
    W[4, 6, 1] = 5
    W[4, 7, 1] = 5

    # Adjacent matrix
    CW = W.astype('bool')

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate spike events (external events)
    # spk_train = np.array([0, 1, 2, 50, 60, 70,  100, 110, 120])
    SL = SimSpikingStimulus(t_sim=sim_ticks)
    ext_evts_data = nsat.exportAER(SL)
    cfg.set_ext_events(ext_evts_data)

    # Write the C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_stdp')
    c_nsat_writer.write()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)

    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]

    # w0 = c_nsat_reader.read_synaptic_weights_history(post=0)[0][:, 0, 1]
    w0 = c_nsat_reader.read_synaptic_weights_history(post=5)[0][:, 4, 1]
    w1 = c_nsat_reader.read_synaptic_weights_history(post=6)[0][:, 4, 1]
    w2 = c_nsat_reader.read_synaptic_weights_history(post=7)[0][:, 4, 1]
    np.save('w0', w0)
    np.save('w1', w1)
    np.save('w2', w2)
    # w = c_nsat_reader.read_synaptic_weights()

    out_spikelist = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events+'_core_0.dat'),
                                   sim_ticks=sim_ticks)
    np.save('spk0', out_spikelist[0].spike_times)
    np.save('spk1', out_spikelist[1].spike_times)
    np.save('spk2', out_spikelist[2].spike_times)
    np.save('spk3', out_spikelist[3].spike_times)

    # Plot the results
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(5, 2, 1)
    ax.plot(states_core0[:-1, 0, 0], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$V_m$')

    ax = fig.add_subplot(5, 2, 2)
    ax.plot(states_core0[:-1, 1, 0], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$V_m$')

    ax = fig.add_subplot(5, 2, 3)
    ax.plot(states_core0[:-1, 0, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')

    ax = fig.add_subplot(5, 2, 4)
    ax.plot(states_core0[:-1, 1, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')

    ax = fig.add_subplot(5, 2, 5)
    ax.plot(states_core0[:-1, 0, 2], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')

    ax = fig.add_subplot(5, 2, 6)
    ax.plot(states_core0[:-1, 1, 2], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')

    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(4, 1, 1)
        for i in out_spikelist[0].spike_times:
            plt.axvline(i, color='r', lw=1, zorder=0)
        ax.set_ylabel('Spikes')
        ax.set_xlim([0, sim_ticks])

        ax = fig.add_subplot(4, 1, 2)
        plt.step(w0, 'r', lw=2, zorder=10, where='post')
        for i in out_spikelist[1].spike_times:
            plt.axvline(i, color='k', lw=1, zorder=0)
        ax.set_ylabel('Spikes')
        ax.set_xlim([0, sim_ticks])

        ax = fig.add_subplot(4, 1, 3)
        plt.step(w1, 'r', lw=2, zorder=10, where='post')
        for i in out_spikelist[2].spike_times:
            plt.axvline(i, color='k', lw=1, zorder=0)
        ax.set_xlim([0, sim_ticks])

        ax = fig.add_subplot(4, 1, 4)
        plt.step(w2, 'r', lw=2, zorder=10, where='post')
        for i in out_spikelist[3].spike_times:
            plt.axvline(i, color='k', lw=1, zorder=0)
        ax.set_xlim([0, sim_ticks])
    plt.show()
