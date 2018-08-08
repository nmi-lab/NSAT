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
import numpy as np
import matplotlib.pylab as plt
import pyNSATlib as nsat
import time
import os


def setup():
    sim_ticks = 500         # Total simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [51]        # Number of neurons per core
    N_INPUTS = [0]          # Number of inputs per core
    N_STATES = [4]          # Number of states per neuron per core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Configuration class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 plasticity_en=np.array([True], 'bool'),
                                 ben_clock=True)

    # Parameters group 0
    # Transition matrix A
    cfg.core_cfgs[0].A[0] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix A
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

    # Parameters group 0
    # Transition matrix
    cfg.core_cfgs[0].A[1] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix
    cfg.core_cfgs[0].sA[1] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Bias
    cfg.core_cfgs[0].b[1] = np.array([0, 0, 0, 0], dtype='int')
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')
    # Enable plasticity per state
    cfg.core_cfgs[0].plastic[1] = True
    # Enable STDP per state
    cfg.core_cfgs[0].stdp_en[1] = True

    # Mapping function between NSAT parameters groups and neurons
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')
    cfg.core_cfgs[0].nmap[0] = 1

    # Mapping function between learning parameters groups and neurons
    lrnmap = np.zeros((nsat.N_GROUPS, N_STATES[0]), dtype='int')
    lrnmap[1] = 1
    cfg.core_cfgs[0].lrnmap = lrnmap

    # Synaptic strength matrix
    W = np.zeros((N_UNITS, N_UNITS, N_STATES[0]))
    W[1:, 0, 0] = 1
    cfg.core_cfgs[0].W = W

    # Adjacent matrix
    CW = np.zeros((N_UNITS, N_UNITS, N_STATES[0]))
    CW[1:, 0, 0] = 1
    cfg.core_cfgs[0].CW = CW

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_accumulator')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_accumulator')
#    intel_fpga_writer.write()
#
#    # Call the C NSAT
#    print("Running C NSAT!")
#    nsat.run_c_nsat(nsat.fname)
#
#    # Load the results (read binary files)
#    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fname)
#    states = c_nsat_reader.read_c_nsat_states()
#    time_core0, states_core0 = states[0][0], states[0][1]
#
#    # Plot the results
#    fig = plt.figure(figsize=(10, 10))
#    for i in range(1, 5):
#        ax = fig.add_subplot(4, 1, i)
#        ax.plot(states_core0[:-1, 0, i-1], 'b', lw=3)
#
#     import os
#     plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
#     plt.close()
    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))


if __name__ == '__main__':
    print('Begin %s:main()' %
          (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()

    setup()
#      run(filenames)
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(
        os.path.basename(__file__))[0], time.perf_counter() - start_t))
