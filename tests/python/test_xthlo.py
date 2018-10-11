#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri Dec  9 15:33:07 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import sys
import numpy as np
import pyNSATlib as nsat
import matplotlib.pylab as plt
import os
import timeit

def setup():
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    sim_ticks = 100             # Simulation time
    N_CORES = 1                 # Number of cores
    N_NEURONS = [1]             # Number of neurons per core (list)
    N_INPUTS = [0]              # Number of inputs per core (list)
    N_STATES = [4]              # Number of states per core (list)
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
                                 ben_clock=True)

    # Transition matrix A
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
    # Upper boundary
    cfg.core_cfgs[0].Xthup[0] = np.array([XMAX, XMAX, XMAX, XMAX], 'int')
    # Lower boundary
    cfg.core_cfgs[0].Xthlo[0] = np.ones(4, 'int') * XMIN
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # Tested value
    if len(sys.argv) != 2:
        print('Missing argument! Default value is used[Xthlo=-32767]!')
        cfg.core_cfgs[0].Xreset[0] = np.array([XMIN, XMIN, XMIN, XMIN], 'int')
        cfg.core_cfgs[0].Xthlo[0] = np.array([XMIN, XMAX, XMAX, XMAX], 'int')
    else:
        cfg.core_cfgs[0].Xreset[0] = np.array([XMIN, XMIN, XMIN, XMIN], 'int')
        cfg.core_cfgs[0].Xthlo[0] = np.array([int(sys.argv[1]), XMAX, XMAX, XMAX],
                                             'int')

    # NSAT parameters groups mapping to neurons
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Write C NSAT parameters files (binary)
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_lower_boundary')
    c_nsat_writer.write()

    # Write Intel FPGA parameters files (hex)
#    # intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#    #                                         prefix='test_lower_boundary')
#    # intel_fpga_writer.write()
#    # intel_fpga_writer.write_globals()

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
    time_core0, states_core0 = states[0][0], states[0][1]

    # Plot the results
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
    
    fname = setup()
    run(fname)
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], timeit.default_timer()-start_t))
 