#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 09:22:36 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import sys
import numpy as np
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time

def setup():
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    sim_ticks = 100         # Simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [2]         # Number of neurons per core (list)
    N_INPUTS = [0]          # Number of input units per core (list)
    N_STATES = [4]          # Number of states per core (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Instance of main class
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    # Define transition matrices
    cfg.core_cfgs[0].A[0] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]
    cfg.core_cfgs[0].A[1] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrices
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]
    cfg.core_cfgs[0].sA[1] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([50, 0, 0, 0], dtype='int')
    cfg.core_cfgs[0].b[1] = np.array([0, 0, 0, 0], dtype='int')
    # Thresholds
    cfg.core_cfgs[0].Xth[0] = 100
    cfg.core_cfgs[0].Xth[1] = 100
    # Set reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    cfg.core_cfgs[0].Xreset[1] = np.array([0, MAX, MAX, MAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')
    cfg.core_cfgs[0].XresetOn[1] = np.array([True, False, False, False],
                                            'bool')
    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 1, 0] = 50

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, 1, 0] = 1

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Tested value
    if len(sys.argv) != 2:
        print('Missing argument! Default value is used[prob=0]!')
        cfg.core_cfgs[0].prob_syn[1] = np.array([0, 0, 0, 0], dtype='int')
    else:
        p = int(sys.argv[1])
        cfg.core_cfgs[0].prob_syn[1] = np.array([p, p, p, p], 'int')

    # NSAT parameters and neurons mapping function
    cfg.core_cfgs[0].nmap = np.array([0, 1], dtype='int')

    # Write C NSAT parameter files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_prob')
    c_nsat_writer.write()

    # Write Intel FPGA parameters files
#    # intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#    #                                         prefix='test_prob')
#    # intel_fpga_writer.write()
#    # intel_fpga_writer.write_globals()
    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
 

def run():
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(nsat.fnames.pickled)
    nsat.run_c_nsat()

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fnames)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 5):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(states_core0[:-1, 0, i-1], 'b', lw=3)
        ax.plot(states_core0[:-1, 1, i-1], 'r', lw=3)
        # ax.plot(states_core0[:-1, 0, i-1], 'r--', lw=1, alpha=0.4)
    
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
    setup()
    run()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
 