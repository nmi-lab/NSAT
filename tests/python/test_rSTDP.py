#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : 19-08-2016
# Last Modified : Fri 30 Dec 2016 03:49:06 PM PST
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['savefig.dpi'] = 800.
matplotlib.rcParams['font.size'] = 25.0
matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
matplotlib.rcParams['axes.formatter.limits'] = [-10, 10]
matplotlib.rcParams['figure.subplot.bottom'] = .2

# Globals
sim_ticks = 1000                # Simulation time
core = 0 
SL = None

def SimSpikingStimulus(stim, time=1000, t_sim=None):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean data/poisson,
               scaled by *poisson*.
    '''
    pyST.STCreate.seed(130)

    n = np.shape(stim)[0]
    SL = pyST.SpikeList(id_list=list(range(n)))
    for i in range(n):
        SL[i] = pyST.STCreate.poisson_generator(stim[i], t_start=1, t_stop=t_sim)
    return SL


def setup():
    global SL
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    np.random.seed(100)             # Numpy RNG seed
    pyST.STCreate.seed(130)         # PyNCS RNG seed
    N_CORES = 1                     # Number of cores
    N_NEURONS = [100]                 # Number of neurons per core
    N_INPUTS = [101]                  # Number of inputs per core
    N_STATES = [4]                  # Number of states pare core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN
    NLRN_GROUPS = 8
    N_GROUPS = 8

    # Main class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 monitor_weights=True,
                                 plasticity_en=np.array([True], 'bool'),
                                 ben_clock=True)

    cfg.core_cfgs[core].sigma[0] = [0,0,10,0]

    # Transition matrix group 0
    cfg.core_cfgs[core].A[0] = [[-5, OFF, OFF, OFF],
                             [3, -5, OFF, OFF],
                             [OFF, OFF, -6, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Transition matrix group 1
    cfg.core_cfgs[core].A[1] = [[OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix group 0
    cfg.core_cfgs[core].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Threshold
    cfg.core_cfgs[core].Xth[0] = 25000
    # Refractory period
    cfg.core_cfgs[core].t_ref[0] = 40
    # Bias
    cfg.core_cfgs[core].b[0] = [0,  0, 0, 0]
    # Initial conditions
    cfg.core_cfgs[core].Xinit = np.array([[0, 0, 0, 0] for _ in
                                      range(N_NEURONS[0])])
    # Reset value
    cfg.core_cfgs[core].Xreset[0] = [0, MAX, MAX, MAX]
    # Turn reset on
    cfg.core_cfgs[core].XresetOn[0] = [True, False, False, False]

    # Turn plasticity per state on
    cfg.core_cfgs[core].plastic[0] = True
    # Turn stdp per state on
    cfg.core_cfgs[core].stdp_en[0] = True
    cfg.core_cfgs[core].is_stdp_exp_on[0] = True

    # Global modulator state
    cfg.core_cfgs[core].modstate[0] = 2

    # Parameters groups mapping function
    cfg.core_cfgs[core].nmap = np.zeros((N_NEURONS[0],), dtype='int')
    lrnmap = 1+np.zeros((N_GROUPS, N_STATES[0]), dtype='int')
    lrnmap[0, 1] = 0
    cfg.core_cfgs[core].lrnmap = lrnmap

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[core]], 'int')
    W[0:100, N_INPUTS[core]:, 1] = np.eye(N_NEURONS[core])*100
    W[100, N_INPUTS[core]:, 2] = 100

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0:100, N_INPUTS[core]:, 1] = np.eye(N_NEURONS[core])
    CW[100, N_INPUTS[core]:, 2] = 1

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[core].wgt_table = wgt_table
    cfg.core_cfgs[core].ptr_table = ptr_table

    # Learning STDP parameters
    cfg.core_cfgs[core].tca  = [[24, 48] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].hica = [[-3, -5, -6] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].sica = [[1, 1, 1] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].slca = [[16, 16, 16] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].tac  = [[-32, -64] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].hiac = [[-6, -8, -9] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].siac = [[-1, -1, -1] for _ in range(NLRN_GROUPS)]
    cfg.core_cfgs[core].slac = [[16, 16, 16] for _ in range(NLRN_GROUPS)]

    # Prepare external stimulus (spikes events)
    stim = [50]*N_NEURONS[core]
    SL = SimSpikingStimulus(stim, t_sim=sim_ticks)
    SLr = pyST.SpikeList(id_list=[100])
    SLr[100] = pyST.STCreate.inh_poisson_generator(rate=np.array([0., 100., 0.]),
                                                   t=np.array([0, 400, 600]),
                                                   t_stop=sim_ticks)
    SL = pyST.merge_spikelists(SL, SLr)
    ext_evts_data = nsat.exportAER(SL)
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary file
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_rSTDP')
    c_nsat_writer.write()
    
    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    return c_nsat_writer.fname


def run(fnames):
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(fnames.pickled)
    nsat.run_c_nsat(fnames)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, fnames)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[core][0], states[core][1]

    wt, pids = c_nsat_reader.read_synaptic_weights_history(post=[130, 150, 120])
    wt, pids = wt[0], pids[0]
    in_spikelist = SL
    out_spikelist = nsat.importAER(c_nsat_reader.read_events(0),
                                   sim_ticks=sim_ticks,
                                   id_list=[0])

    plt.matplotlib.rcParams['figure.subplot.bottom'] = .1
    plt.matplotlib.rcParams['figure.subplot.left'] = .2
    plt.matplotlib.rcParams['figure.subplot.right'] = .98
    plt.matplotlib.rcParams['figure.subplot.hspace'] = .1
    plt.matplotlib.rcParams['figure.subplot.top'] = 1.0
    # Plot the results
    fig = plt.figure(figsize=(14, 10))

    i = 4
    ax1 = fig.add_subplot(5, 1, i)
    for t in SL[100].spike_times:
        plt.axvline(t, color='g', alpha=.4)
    ax1.plot(states_core0[:-1, 0, 2], 'b', lw=3, label='$x_m$')
    ax1.set_ylabel('$x_m$')
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.axhline(0, color='b', alpha=.5, linewidth=3)
    plt.locator_params(axis='y', nbins=4)

    i = 5
    ax = fig.add_subplot(5, 1, i, sharex=ax1)
    for t in SL[0].spike_times:
        plt.axvline(t, color='k', alpha=.4)
    ax.plot(wt[:, 19, 1], 'r', lw=3)
    # ax.imshow(wt[:, :, 1], aspect='auto', interpolation='nearest')
    ax.set_ylabel('$w$')
    ax.set_xlabel('Time Step')
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.locator_params(axis='y', nbins=4)

    i = 2
    ax = fig.add_subplot(5, 1, i, sharex=ax1)
    ax.plot(states_core0[:-1, 0, 0], 'b', lw=3, label='$V_m$')
    ax.set_ylabel('$V_m$')
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.locator_params(axis='y', nbins=4)

    i = 3
    ax = fig.add_subplot(5, 1, i, sharex=ax1)
    ax.plot(states_core0[:-1, 0, 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.locator_params(axis='y', nbins=4)
    for t in np.ceil(SL[0].spike_times):
        plt.axvline(t, color='k', alpha=.4)

    ax1 = fig.add_subplot(5, 1, 1, sharex=ax1)
    out_spikelist.id_slice([0]).raster_plot(display=ax1, kwargs={'color': 'b'})
    out_spikelist.id_slice(list(range(1, 30))).raster_plot(display=ax1,
                                                           kwargs={'color': 'k'})
    ax1.set_xlabel('')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.tight_layout()
    
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
    filenames = setup()
    run(filenames)
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
 