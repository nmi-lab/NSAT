#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Mon Dec 12 09:56:22 PST 2016
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

def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i],
                                                    t_start=1,
                                                    t_stop=ticks)
    return SL


if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    np.random.seed(30)          # Numpy random number generator seed
    sim_ticks = 5000            # Total simulation time
    N_CORES = 2                 # Number of cores
    N_NEURONS = [10, 10]        # Number of neurons per core (list)
    N_INPUTS = [10, 10]         # Number of inputs per core (list)
    N_STATES = [4, 4]           # Number of states per core (list)
    N_UNITS = [sum(i) for i in zip(N_INPUTS, N_NEURONS)]

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
                                 ben_clock=True)

    # Loop over the cores and set the parameters
    for i in range(N_CORES):
        # Transition matrix A
        cfg.core_cfgs[i].A[0] = [[-1,  OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF]]

        # Sign matrix sA
        cfg.core_cfgs[i].sA[0] = [[-1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1]]

        # Bias
        cfg.core_cfgs[i].b[0] = np.array([30, 0, 0, 0], dtype='int')
        # Threshold
        cfg.core_cfgs[i].Xth[0] = 100
        # Reset value
        cfg.core_cfgs[i].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
        # Turn reset on per state
        cfg.core_cfgs[i].XresetOn[0] = np.array([True, False, False, False],
                                                'bool')

    for i in range(N_CORES):
        # Mapping NSAT parameters (per core)
        cfg.core_cfgs[i].nmap = np.zeros((N_NEURONS[0],), dtype='int')

        if i == 0:
            # Assigning synaptic strength to each core
            W = np.zeros([N_UNITS[i], N_UNITS[i], 4], 'int')
            W[0, 10, 0] = 50
            W[1, 11, 0] = 50
            W[2, 12, 0] = 50
            W[3, 13, 0] = 50
            W[4, 14, 0] = 50
            W[5, 15, 0] = 50
            W[6, 16, 0] = 50
            W[7, 17, 0] = 50
            W[8, 18, 0] = 50
            W[9, 19, 0] = 50

            # Assigning adjacent matrix to each core
            CW = np.zeros(W.shape, dtype='int')
            CW[0, 10, 0] = 1
            CW[1, 11, 0] = 1
            CW[2, 12, 0] = 1
            CW[3, 13, 0] = 1
            CW[4, 14, 0] = 1
            CW[5, 15, 0] = 1
            CW[6, 16, 0] = 1
            CW[7, 17, 0] = 1
            CW[8, 18, 0] = 1
            CW[9, 19, 0] = 1

            wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
            np.set_printoptions(threshold=np.nan)
            cfg.core_cfgs[i].wgt_table = wgt_table
            cfg.core_cfgs[i].ptr_table = ptr_table
        if i == 1:
            W = np.zeros([N_UNITS[i], N_UNITS[i], 4], 'int')
            CW = np.zeros(W.shape, dtype='int')
            W[1, 11, 0] = 50
            W[5, 15, 0] = 50
            CW[1, 11, 0] = 1
            CW[5, 15, 0] = 1
            wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
            np.set_printoptions(threshold=np.nan)
            cfg.core_cfgs[i].wgt_table = wgt_table
            cfg.core_cfgs[i].ptr_table = ptr_table

    # Connect core 0 neuron 9 with core 1 neurons 1 and 5
    cfg.set_L1_connectivity({(0, 9): ((1, 1), (1, 5))})

    # Generate external events firing rates from 5 to 50 inc by 5
    freqs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    SL = []
    for i in range(N_CORES-1):
        SL.append(RegularSpikingStimulus(freqs, sim_ticks))
    ext_evts_data = nsat.exportAER(SL)

    # Set external events
    cfg.set_ext_events(ext_evts_data)

    # Generate all the CNSAT parameters files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_external_evts')
    c_nsat_writer.write()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]
    states_core1 = states[1][1]

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        ax = fig.add_subplot(10, 1, i)
        ax.plot(states_core0[:500, i, 0], 'b', lw=3)
        ax.set_ylim([0, 110])

    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        ax = fig.add_subplot(10, 1, i)
        ax.plot(states_core1[:500, i, 0], 'b', lw=3)
        ax.set_ylim([0, 110])

    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))