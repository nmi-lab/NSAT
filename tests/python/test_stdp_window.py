#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Ting-Shuo Chou
#
# Creation Date : 20-11-2017
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
from corr_spike_trains import correlated_spikes
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time

# Globals
sim_ticks = 1000                # Simulation ticks
SL = None
spk0 = None

def SimSpikingStimulus(rates, t_sim=None):
    m = np.shape(rates)[0]
    n = int(m / 2)

    C1 = (np.ones((n, n)) + np.random.uniform(0, 1, (n, n)) * 2)
    np.fill_diagonal(C1, rates[:n])
    C1 = np.maximum(C1, C1.T)

    cor_spk1 = correlated_spikes(C1, rates, n)
    cor_spk1.cox_process(time=t_sim)
    spk1 = cor_spk1.extract_pyNCS_list()
    tmp1 = nsat.importAER(spk1)

    SL = pyST.SpikeList(id_list=list(range(m)))
    for i in range(m):
        if i < n:
            SL[i] = tmp1[i]
        if i >= n:
            SL[i] = pyST.STCreate.poisson_generator(rates[i], t_stop=t_sim)
    return SL


def PeriodicPrePostSpikingStimulus(freqs, diff, ticks=1000):
    SL = pyST.SpikeList(id_list=list(range(2)))
    base_phase = 100.0
    shift_phase = base_phase + diff

    SL[0] = pyST.STCreate.regular_generator(freqs,
                                            t_start=0,
                                            phase=base_phase,
                                            jitter=False,
                                            t_stop=ticks,
                                            array=False)
    SL[1] = pyST.STCreate.regular_generator(freqs,
                                            t_start=0,
                                            phase=shift_phase,
                                            jitter=False,
                                            t_stop=ticks,
                                            array=False)
    return SL


def setup():
    global SL, spk0
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    N_CORES = 1                     # Number of cores
    N_NEURONS = [1]                 # Number of neurons per core (list)
    N_INPUTS = [2]                 # Number of inputs per core (list)
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
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 monitor_weights=True,
                                 w_check=False,
                                 tstdpmax=[50],
                                 monitor_weights_final=True,
                                 plasticity_en=[True],
                                 ben_clock=True)

    # Transition matrix
    cfg.core_cfgs[0].A[0] = [[-1, OFF, OFF, OFF],
                             [0, -2, OFF, OFF],
                             [0, OFF, -2, OFF],
                             [OFF, OFF, OFF, 0]]

    # Sign matrix
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Refractory period
    cfg.core_cfgs[0].t_ref[0] = 20
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 25
    # Bias
    cfg.core_cfgs[0].b[0] = [0, 0, 0, 1]
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([0, 0, 0, 0], 'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = [0, MAX, MAX, MAX]
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = [True, False, False, False]

    # Enable plasticity at states
    cfg.core_cfgs[0].plastic[0] = True
    cfg.core_cfgs[0].plastic[1] = False
    # Enable STDP
    cfg.core_cfgs[0].stdp_en[0] = True
    cfg.core_cfgs[0].stdp_en[1] = False
    # Global modulator state
    cfg.core_cfgs[0].modstate[0] = 3

    # Parameters for the STDP kernel function
    rNLRN_GROUPS = list(range(8))
    cfg.core_cfgs[0].tstdp = np.array([30 for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].tca = np.array([[6, 15] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].hica = np.array([[2, 1, 0] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].sica = np.array([[1, 1, 1] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].tac = np.array([[6, 15] for _ in rNLRN_GROUPS], 'int')
    cfg.core_cfgs[0].hiac = np.array([[-2, -1, 0] for _ in rNLRN_GROUPS],
                                     'int')
    cfg.core_cfgs[0].siac = np.array([[-1, -1, -1] for _ in rNLRN_GROUPS],
                                     'int')

    # Set parameters mapping
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    lrnmap = np.zeros((nsat.N_GROUPS, N_STATES[0]), dtype='int')
    lrnmap[0, 1] = 0
    lrnmap[0, 2] = 1
    cfg.core_cfgs[0].lrnmap = lrnmap

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, N_INPUTS[0], 2] = 100
    W[1, N_INPUTS[0], 1] = 0
    print(W)

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, N_INPUTS[0], 2] = True
    CW[1, N_INPUTS[0], 1] = True

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate spike events (external events)
    # rates = np.random.randint(5, 20, (N_INPUTS,), dtype='i')
    # rates = np.hstack([np.random.randint(5, 10, (N_INPUTS[0]//2,), 'int'),
    #                   np.random.randint(10, 20, (N_INPUTS[0]//2,), 'int')])

    # Build external spikes
    freqs = 5
    SL = PeriodicPrePostSpikingStimulus(freqs, -5, sim_ticks)
    import copy
    spk0 = copy.deepcopy(SL)
    ext_evts_data = nsat.exportAER(SL)
    cfg.set_ext_events(ext_evts_data)

    # Write the C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_stdp')
    c_nsat_writer.write()

    # Write Intel FPGA hex parameters files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_stdp')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()

    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
 

def run():
    # Call the C NSAT
    global spk0
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(nsat.fnames.pickled)
    nsat.run_c_nsat()

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fnames)
    #ww = np.array(c_nsat_reader.read_c_nsat_synaptic_weights()[0])
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]
    # wt = c_nsat_reader.read_c_nsat_syn_evo()[0][0]
    wt, pids = c_nsat_reader.read_synaptic_weights_history()
    wt = wt[0]

    spk = nsat.importAER(c_nsat_reader.read_events(0),
                         sim_ticks=sim_ticks)
    spk.raster_plot()

    print((SL[0].spike_times))
    print((SL[1].spike_times))
    print((spk[0].spike_times))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ww[:N_INPUTS[0], N_INPUTS[0], 1], 'k.')

    # Plot the results
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(4, 2, 1)
    ax.plot(states_core0[:-1, 0, 0], 'b', lw=3, label='$x_m$')
    for i in spk[0].spike_times:
        plt.axvline(i, color='k', lw=1)
    ax.set_ylabel('$V_m$')

    # ax = fig.add_subplot(4, 2, 2)
    # ax.plot(states_core0[:-1, 1, 0], 'b', lw=3, label='$x_m$')
    # ax.set_ylabel('$V_m$')

    ax = fig.add_subplot(4, 2, 3)
    ax.plot(states_core0[:-1, 0, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')

    # ax = fig.add_subplot(4, 2, 4)
    # ax.plot(states_core0[:-1, 1, 1], 'b', lw=3, label='$I_{syn}$')
    # ax.set_ylabel('$I_{syn}$')

    ax = fig.add_subplot(4, 2, 5)
    ax.plot(states_core0[:-1, 0, 2], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')

    # ax = fig.add_subplot(4, 2, 6)
    # ax.plot(states_core0[:-1, 1, 1], 'b', lw=3, label='$x_m$')
    # ax.set_ylabel('$x_m$')

    print(wt.shape)
    ax = fig.add_subplot(4, 2, 7)
    # ax.plot(wt[wt[:, 0, :] == 0, 0], wt[wt[:, 1, :] == 0, 4], 'r', lw=3)
    ax.plot(wt[:, 0, 0], wt[:, 1, 0], 'r', lw=3)
    for i in spk[0].spike_times:
        plt.axvline(i, color='k', lw=1)
    for i in spk0[1].spike_times:
        plt.axvline(i, color='r', lw=1)
    ax.set_ylabel('$w$')

    ax = fig.add_subplot(4, 2, 8)
    # ax.plot(wt[wt[:, 1, :] == 1, 0], wt[wt[:, 1, :] == 1, 4], 'r', lw=3)
    for i in spk[0].spike_times:
        plt.axvline(i, color='k', lw=1)
    for i in spk0[1].spike_times:
        plt.axvline(i, color='r', lw=1)
    ax.set_ylabel('$w$')

    # ax = fig.add_subplot(4, 2, 8)
    # ax.plot(wt[:, 1, 1], 'r', lw=3)
    # ax.set_ylabel('$w$')

    # fig = plt.figure(figsize=(10, 5))
    # for i in spk[0].spike_times:
    #    plt.plot(wt[:, 1, 1], 'r', lw=3)
    #    plt.axvline(i, color='k', lw=1)

    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
    setup()
    run()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
 