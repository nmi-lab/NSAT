#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri 30 Dec 2016 03:48:11 PM PST
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import timeit

import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['savefig.dpi'] = 800.
matplotlib.rcParams['font.size'] = 25.0
matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
matplotlib.rcParams['axes.formatter.limits'] = [-10, 10]
matplotlib.rcParams['figure.subplot.bottom'] = .2

sim_ticks = 1000        # Number of simulation ticks
SL = None 

def SimSpikingStimulus(stim, time=1000, t_sim=None):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean data/poisson,
               scaled by *poisson*.
    '''

    n = np.shape(stim)[0]
    SL = pyST.SpikeList(id_list=list(range(n)))
    for i in range(n):
        SL[i] = pyST.STCreate.poisson_generator(stim[i], t_stop=t_sim)
    return SL


def setup():
    global SL
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))

    N_CORES = 1             # Number of cores
    N_NEURONS = [1]         # Number of neurons per core (list)
    N_INPUTS = [1]          # Number of inputs per core (list)
    N_STATES = [4]          # Number of states per core (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units

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
                                 monitor_weights=True,
                                 monitor_spikes=True,
                                 plasticity_en=[True],
                                 ben_clock=True)

    # Transition matrix A (parameters group 0)
    cfg.core_cfgs[0].A[0] = [[-5, OFF, OFF, OFF],
                             [2, -5,  OFF, OFF],
                             [OFF, OFF, -7, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Transition matrix A (parameters group 1)
    cfg.core_cfgs[0].A[1] = [[OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix A (parameters group 0)
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Bias
    cfg.core_cfgs[0].b[0] = [0,  0, 0, 0]
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 25000
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([[0, 0, 0, 0] for _
                                         in range(N_NEURONS[0])],
                                         'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, MAX, MAX, MAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # STDP kernel height of acausal part
    cfg.core_cfgs[0].hiac = [[-1, 4, 0] for _ in range(8)]

    cfg.core_cfgs[0].plastic[0] = True          # Plastic states group 0
    cfg.core_cfgs[0].stdp_en[0] = False         # STDP enabled for group 0
    cfg.core_cfgs[0].sigma[0] = [0, 0, 5, 0]    # Additive noise
    cfg.core_cfgs[0].modstate[0] = 2            # Global modulator state
    cfg.core_cfgs[0].t_ref[0] = 40              # Refractory period

    # Mapping between neurons and parameter groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 1, 1] = 100

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, 1, 1] = 1

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate external event spikes (firing rate 50Hz)
    stim = [50]
    SL = SimSpikingStimulus(stim, t_sim=sim_ticks)
    ext_evts_data = nsat.exportAER(SL)

    # Set external events
    cfg.set_ext_events(ext_evts_data)

    # Write the C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_modstate')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_modstate')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()
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
    states_core0 = states[0][1]

    # wt = c_nsat_reader.read_c_nsat_weights_evo(0)[:, 1, 1]
    wt, pids = c_nsat_reader.read_synaptic_weights_history(post=[0])
    in_spikelist = SL
    out_spikelist = nsat.importAER(nsat.read_from_file(fnames.events+'_core_0.dat'),
                                   sim_ticks=sim_ticks,
                                   id_list=[0])

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    i = 1
    ax = fig.add_subplot(4, 1, i)
    ax.plot(states_core0[:-1, 0, i - 1], 'b', lw=3, label='$V_m$')
    ax.set_ylabel('$V_m$')
    ax.set_xticks([])
    plt.locator_params(axis='y', nbins=4)
    i = 2
    ax = fig.add_subplot(4, 1, i)
    ax.plot(states_core0[:-1, 0, i - 1], 'b', lw=3, label='$I_{syn}$')
    ax.set_ylabel('$I_{syn}$')
    ax.set_xticks([])
    plt.locator_params(axis='y', nbins=4)
    for t in SL[0].spike_times:
        plt.axvline(t, color='k')

    i = 4
    ax = fig.add_subplot(4, 1, i)
    for t in SL[0].spike_times:
        plt.axvline(t, color='k')
    ax.plot(states_core0[:-1, 0, 2], 'b', lw=3, label='$x_m$')
    ax.set_ylabel('$x_m$')
    plt.axhline(0, color='b', alpha=.5, linewidth=3)
    plt.locator_params(axis='y', nbins=4)
    i = 3

    ax = fig.add_subplot(4, 1, i)
    for t in SL[0].spike_times:
        plt.axvline(t, color='k')
    ax.plot(wt[0][:, 1, 1], 'r', lw=3)
    ax.set_ylabel('$w$')
    ax.set_xticks([])
    plt.locator_params(axis='y', nbins=4)

    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = timeit.default_timer()
    
    fname = setup()
    run(fname)
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], timeit.default_timer()-start_t))
 