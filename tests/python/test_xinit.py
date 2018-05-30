#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 17:00:34 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import sys
import numpy as np
import pyNSATlib as nsat
import matplotlib.pylab as plt


if __name__ == '__main__':
    sim_ticks = 100         # Simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [1]         # Number of neurons per core (list)
    N_INPUTS = [0]          # Number of inputs per core (list)
    N_STATES = [4]          # Number of states (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units

    # Basic constants
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

    # Transistion matrix A
    cfg.core_cfgs[0].A[0] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix sA
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([50, 0, 0, 0], dtype='int')
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # Tested value
    cfg.core_cfgs[0].Xinit[0] = np.zeros((N_NEURONS[0], N_STATES[0]), 'int')
    if len(sys.argv) != 2:
        print('Missing argument! Default value is used [xinit=0]!')
        cfg.core_cfgs[0].Xinit[0, 0] = 0
        cfg.core_cfgs[0].Xinit[0, 1] = 0
        cfg.core_cfgs[0].Xinit[0, 2] = 0
        cfg.core_cfgs[0].Xinit[0, 3] = 0
    else:
        tmp = int(sys.argv[1])
        print(tmp)
        cfg.core_cfgs[0].Xinit[0, 0] = tmp
        cfg.core_cfgs[0].Xinit[0, 1] = 0
        cfg.core_cfgs[0].Xinit[0, 2] = 0
        cfg.core_cfgs[0].Xinit[0, 3] = 0

    # Write C NSAT parameters files
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_initial_cond')
    c_nsat_writer.write()

    # Write Intel FPGA parameters files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_initial_cond')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()

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
