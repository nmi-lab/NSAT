#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 16:47:17 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# -----------------------------------------------------------------------------
import numpy as np
import pyNSATlib as nsat
import matplotlib.pylab as plt

if __name__ == '__main__':
    sim_ticks = 100         # Simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [8]         # Number of neurons
    N_INPUTS = [0]          # Number of inputs
    N_STATES = [4]          # Number of states
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total number of units

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

    # Transition matrix per group
    cfg.core_cfgs[0].A[0, 0, 0] = -1
    cfg.core_cfgs[0].A[1, 0, 0] = -2
    cfg.core_cfgs[0].A[2, 0, 0] = -3
    cfg.core_cfgs[0].A[3, 0, 0] = -4
    cfg.core_cfgs[0].A[4, 0, 0] = -5
    cfg.core_cfgs[0].A[5, 0, 0] = -12
    cfg.core_cfgs[0].A[6, 0, 0] = -7
    cfg.core_cfgs[0].A[7, 0, 0] = -3

    # Sign matrix per group
    cfg.core_cfgs[0].sA[0, 0, 0] = -1
    cfg.core_cfgs[0].sA[1, 0, 0] = -1
    cfg.core_cfgs[0].sA[2, 0, 0] = -1
    cfg.core_cfgs[0].sA[3, 0, 0] = -1
    cfg.core_cfgs[0].sA[4, 0, 0] = -1
    cfg.core_cfgs[0].sA[5, 0, 0] = -1
    cfg.core_cfgs[0].sA[6, 0, 0] = -1
    cfg.core_cfgs[0].sA[7, 0, 0] = -1

    # Bias matrix per group
    cfg.core_cfgs[0].b[0, 0] = 50
    cfg.core_cfgs[0].b[1, 0] = 80
    cfg.core_cfgs[0].b[2, 0] = 50
    cfg.core_cfgs[0].b[3, 0] = 20
    cfg.core_cfgs[0].b[4, 0] = 50
    cfg.core_cfgs[0].b[5, 0] = 60
    cfg.core_cfgs[0].b[6, 0] = 50
    cfg.core_cfgs[0].b[7, 0] = 5

    # Threshold
    cfg.core_cfgs[0].Xth = np.ones((nsat.N_GROUPS, )) * 100
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # Mapping between neurons and NSAT parameters groups
    nmap = np.arange(N_NEURONS[0], dtype='int')
    cfg.core_cfgs[0].nmap = nmap

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_params_groups')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_parameters_groups')
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
    for i in range(1, 8):
        ax = fig.add_subplot(8, 1, i)
        ax.plot(states_core0[:-1, i, 0], 'b', lw=3)
    plt.show()
