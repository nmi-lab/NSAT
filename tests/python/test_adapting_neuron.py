#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : 19-08-2016
# Last Modified : Fri 30 Dec 2016 03:48:40 PM PST
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
import pyNSATlib as nsat
import matplotlib.pylab as plt

if __name__ == '__main__':
    sim_ticks = 5000        # Simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [2]         # Number of neurons per core (list)
    N_INPUTS = [0]          # Number of inputs per core (list)
    N_STATES = [4]          # Number of states per core (list)

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

    # Transition matrix A
    cfg.core_cfgs[0].A[0] = [[-6,  OFF, OFF, OFF],
                             [0, -11, OFF, OFF],
                             [0, OFF, -8, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix sA
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [-1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([600, 0, 0, 1], dtype='int')
    # Threshold
    cfg.core_cfgs[0].Xth[0] = XMAX
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')
    # Global modulator state (e.g. Dopamine)
    cfg.core_cfgs[0].modstate[0] = 3

    # Synaptic weights
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 0, 1] = 115
    W[0, 1, 2] = 125
    W[1, 1, 1] = 115
    W[1, 0, 2] = 125
    cfg.core_cfgs[0].W = W

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, 0, 1] = 1
    CW[0, 1, 2] = 1
    CW[1, 1, 1] = 1
    CW[1, 0, 2] = 1
    cfg.core_cfgs[0].CW = CW

    # Mapping between neurons and NSAT parameters groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_adapting')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_adapting')
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
