#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Thu 15 Sep 2016 10:52:04 AM PDT
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pylab as plt
import pyNSATlib as nsat

if __name__ == '__main__':
    # sim_ticks = 10000             # Simulation time
    sim_ticks = 1000             # Simulation time
    N_CORES = 5                 # Number of cores
    N_NEURONS = [300, 300, 300, 300, 300]      # Number of neurons per core
    N_INPUTS = [100, 100, 100, 100, 100]     # Number of inputes per core
    N_STATES = [4, 4, 4, 4, 4]              # Number of states per core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    for i in range(N_CORES):
        # Transition matrix
        cfg.core_cfgs[i].A[0] = [[-1,  OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF]]

        # Sign matrix
        cfg.core_cfgs[i].sA[0] = [[-1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1]]

        # Bias
        cfg.core_cfgs[i].b[0] = np.array([50, 0, 0, 0], dtype='int')
        # Threshold
        cfg.core_cfgs[i].Xth[0] = 100
        # Reset value
        cfg.core_cfgs[i].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
        # Turn reset on
        cfg.core_cfgs[i].XresetOn[0] = np.array([True, False, False, False],
                                                'bool')

        # Mapping function between neurons and NSAT parameters groups
        cfg.core_cfgs[i].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    cfg.set_L1_connectivity(({(0, 102): ((1, 1), (1, 50), (1, 73)),
                              (1, 103): ((1, 15), (1, 95), (1, 7)),
                              (2, 105): ((1, 89), (3, 65), (3, 56)),
                              (3, 143): ((2, 21), (4, 45), (4, 33)),
                              (4, 113): ((1, 5), (1, 50), (1, 7)),
                              (2, 133): ((3, 41), (3, 5), (4, 77)),
                              (4, 123): ((3, 19), (1, 75), (2, 57)),
                              (0, 104): ((2, 3),)}))

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_reset')
    c_nsat_writer.write()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 5):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(states_core0[:-1, 0, i-1], 'b', lw=3)
    plt.show()
