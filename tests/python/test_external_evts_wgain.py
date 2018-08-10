#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Wed 02 Nov 2016 11:54:27 AM PDT
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

def read_states_hex(fname):
    states = []
    with open(fname, 'r') as f:
        for line in f:
            for word in line.split():
                states.append(int(word, 16))
    return np.array(states, dtype='int').reshape(5001, 10, 8)


def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i],
                                                    t_start=1,
                                                    t_stop=ticks,
                                                    jitter=False)
    return nsat.exportAER(SL)


def setup():
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    np.random.seed(30)          # Numpy RNG seed
    sim_ticks = 5000            # Simulation ticks
    N_CORES = 1                 # Number of cores
    N_NEURONS = [10]            # Number of neurons
    N_INPUTS = [10]             # Number of inputs
    N_STATES = [4]              # Number of states
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

    # Transition matrix
    cfg.core_cfgs[0].A[0] = [[-1,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([30, 0, 0, 0], dtype='int')
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # Synaptic strengths gain
    cfg.core_cfgs[0].Wgain[0] = 1

    # Mapping neurons and NSAT parameters groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Synaptic strengths matrix
    W = np.zeros([N_UNITS, N_UNITS, 4], 'int')
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

    # Adjacent matrix
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
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate external events
    freqs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ext_evts_data = RegularSpikingStimulus(freqs, sim_ticks)

    # Set external events
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_ext_evts_wgain')
    c_nsat_writer.write()

    # Write Intel FPGA hex parameters files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_ext_evts_wgain')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()
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
    for i in range(1, 10):
        ax = fig.add_subplot(10, 1, i)
        ax.plot(states_core0[:200, i, 0], 'b', lw=3)
        ax.set_ylim([0, 110])
    
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = timeit.default_timer()
    
    setup()
    run()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], timeit.default_timer()-start_t))
 