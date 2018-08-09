#!/usr/bin/env python
#  ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 16:06:33 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
#  ---------------------------------------------------------------------------
import numpy as np
import pyNSATlib as nsat
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import timeit

def setup():
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    sim_ticks = 500         # Simulation ticks
    NUM_CORES = 1           # Number of cores
    N_INPUTS = [0]          # Number of inputes per core
    N_NEURONS = [1]         # Number of neurons per core
    N_STATES = [4]          # Number of states per core per neuron

    # Class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=NUM_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    # For every core set the parameters
    for i in range(NUM_CORES):
        # Transition matrix
        cfg.core_cfgs[i].A[0] = [[-4, -8, OFF, OFF],
                                 [OFF, -7, OFF, OFF],
                                 [0, OFF, -2, OFF],
                                 [0, OFF, OFF, -6]]
        # Sign matrix
        cfg.core_cfgs[i].sA[0] = [[-1, 1, 1, 1],
                                  [1, -1, 1, 1],
                                  [1, 1, -1, 1],
                                  [1, 1, 1, -1]]
        # Adaptive threshold enabled
        cfg.core_cfgs[i].flagXth[0] = True
        # Threshold
        cfg.core_cfgs[i].Xth[0] = MAX
        # Bias
        cfg.core_cfgs[i].b[0] = np.array([-250, -10, 0, 0], 'int')
        # Initial conditions
        cfg.core_cfgs[i].Xinit[0] = np.array([-7000, -5000, 100, 10], 'int')
        # Reset value
        cfg.core_cfgs[i].Xreset[0] = np.array([-7000, -6000, 0, 0], 'int')
        # Turn on reset
        cfg.core_cfgs[i].XresetOn[0] = np.array([True, False, True, False],
                                                'bool')

    # For every core set the synaptic strengths
    for i in range(NUM_CORES):
        # Synaptic strength
        N_UNITS = N_INPUTS[i] + N_NEURONS[i]
        W = np.zeros([N_UNITS, N_UNITS, 4], 'int')
        W[0, 0, 1] = 5
        W[0, 0, 3] = 0

        # Adjacent matrix
        CW = np.zeros(W.shape, dtype='int')
        CW[0, 0, 1] = 1
        CW[0, 0, 3] = 1

        wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
        np.set_printoptions(threshold=np.nan)
        cfg.core_cfgs[i].wgt_table = wgt_table
        cfg.core_cfgs[i].ptr_table = ptr_table

        # NSAT parameters mapping
        cfg.core_cfgs[i].nmap = np.zeros((N_NEURONS[i],), dtype='int')

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_mn_neuron')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    # intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#    #                                          prefix='test_mn_neuron')
#    # intel_fpga_writer.write()
#    # intel_fpga_writer.write_globals()
    cfg.core_cfgs[0].latex_print_parameters(1)
    
    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
 

def run():
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(nsat.fnames.pickled)
    nsat.run_c_nsat()

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fnames)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]

    # Plot the results
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 5):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(states_core0[:-1, 0, i-1], 'b', lw=3)
    
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = timeit.default_timer()
    
    setup()
    run()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], timeit.default_timer()-start_t))
 