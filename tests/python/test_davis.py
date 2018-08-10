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
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time


def build_davis_file(fname, num_ticks=1000):
    from struct import pack
    np.random.seed(100)

    t = np.arange(1, num_ticks).astype('i')
    core = np.zeros((t.shape[0], ), dtype='i')
    core[t.shape[0]//2:] = 1
    evts = np.ones((t.shape[0], ), 'i')
    neuron = np.array([0, 1, 2, 3], 'i')

    with open(fname, 'wb') as f:
        for i in range(t.shape[0]):
            f.write(pack('i', t[i]))
            f.write(pack('i', evts[i]))
            for j in range(neuron.shape[0]):
                f.write(pack('i', core[i]))
                f.write(pack('i', neuron[j]))


def setup():
    print('Begin %s:setup()' %
          (os.path.splitext(os.path.basename(__file__))[0]))

    sim_ticks = 5000            # Simulation time
    N_CORES = 2                 # Number of cores
    N_NEURONS = [1, 1]          # Number of neurons per core
    N_INPUTS = [5, 5]           # Number of inputes per core
    N_STATES = [2, 2]           # Number of states per core

    # Constants
    XMAX = nsat.XMAX
    OFF = -16

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
        cfg.core_cfgs[i].A[0] = [[-1,  1],
                                 [-1, OFF]]

        # Sign matrix
        cfg.core_cfgs[i].sA[0] = [[-1, 1],
                                  [-1, 1]]

        # Bias
        cfg.core_cfgs[i].b[0] = np.array([50, 0], dtype='int')
        # Threshold
        cfg.core_cfgs[i].Xth[0] = 100
        # Reset value
        cfg.core_cfgs[i].Xreset[0] = np.array([0, XMAX], 'int')
        # Turn reset on
        cfg.core_cfgs[i].XresetOn[0] = np.array([True, False], 'bool')

        W = np.zeros((6, 6, 2), dtype='i')
        W[0, 5, 0] = 10
        W[1, 5, 0] = 5
        W[2, 5, 0] = 5
        W[3, 5, 0] = 5
        W[3, 5, 0] = 5

        CW = np.zeros(W.shape, 'i')
        CW[0, 5, 0] = 1
        CW[1, 5, 0] = 1
        CW[2, 5, 0] = 1
        CW[3, 5, 0] = 1
        CW[3, 5, 0] = 1

        wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
        np.set_printoptions(threshold=np.nan)
        cfg.core_cfgs[i].wgt_table = wgt_table
        cfg.core_cfgs[i].ptr_table = ptr_table

        # Mapping function between neurons and NSAT parameters groups
        cfg.core_cfgs[i].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    cfg.ext_evts = True

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_davis')
    c_nsat_writer.write()
    build_davis_file("/tmp/test_davis_davis_events", num_ticks=sim_ticks)

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
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 5):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(states_core0[:-1, 0, i - 1], 'b', lw=3)

    plt.savefig('/tmp/%s.png' %
                (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))


if __name__ == '__main__':
    print('Begin %s:main()' %
          (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()

    setup()
    run()

    print("End %s:main() , running time: %f seconds" % (os.path.splitext(
        os.path.basename(__file__))[0], time.perf_counter() - start_t))
