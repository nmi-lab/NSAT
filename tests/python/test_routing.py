#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Mon Dec 12 09:56:22 PST 2016
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
import time


def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i],
                                                    t_start=1,
                                                    t_stop=ticks)
    return SL


def setup():
    print('Begin %s:setup()' %
          (os.path.splitext(os.path.basename(__file__))[0]))

    np.random.seed(30)
    sim_ticks = 5000
    N_CORES = 2
    N_NEURONS = [100, 100]
    N_INPUTS = [100, 100]
    N_STATES = [4, 4]
    N_UNITS = [sum(i) for i in zip(N_INPUTS, N_NEURONS)]

    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    for i in range(N_CORES):
        cfg.core_cfgs[i].A[0] = [[-1,  OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF],
                                 [OFF, OFF, OFF, OFF]]

        cfg.core_cfgs[i].sA[0] = [[-1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1]]

        # if i == 1:
        #     cfg.core_cfgs[i].A[0] = [[-2,  OFF, OFF, OFF],
        #                              [OFF, OFF, OFF, OFF],
        #                              [OFF, OFF, OFF, OFF],
        #                              [OFF, OFF, OFF, OFF]]

        cfg.core_cfgs[i].b[0] = np.array([30, 0, 0, 0], dtype='int')
        cfg.core_cfgs[i].Xth[0] = 100
        cfg.core_cfgs[i].Xthup[0] = np.array([XMAX, XMAX, XMAX, XMAX], 'int')
        cfg.core_cfgs[i].Xthlo[0] = np.ones(4, 'int') * XMIN
        cfg.core_cfgs[i].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
        cfg.core_cfgs[i].XresetOn[0] = np.array([True, False, False, False],
                                                'bool')

    for i in range(N_CORES):
        # Mapping NSAT parameters (per core)
        cfg.core_cfgs[i].nmap = np.zeros((N_NEURONS[0],), dtype='int')

        # Assigning synaptic strength to each core
        W = np.zeros([N_UNITS[i], N_UNITS[i], 4], 'int')
        tmp = np.zeros((N_INPUTS[i], N_NEURONS[i]))
        np.fill_diagonal(tmp, 50)
        W[:N_INPUTS[i], N_INPUTS[i]:, 0] = tmp

        # Assigning adjacent matrix to each core
        CW = np.zeros(W.shape, dtype='bool')
        CW[:N_INPUTS[i], N_INPUTS[i]:, 0] = tmp.astype('bool')

        wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
        cfg.core_cfgs[i].wgt_table = wgt_table
        cfg.core_cfgs[i].ptr_table = ptr_table

    cfg.set_L1_connectivity(({(0, 102): ((1, 1), (1, 50), (1, 73)),
                              (0, 103): ((1, 15), (1, 95), (1, 7)),
                              (0, 105): ((1, 89), (1, 65), (1, 56)),
                              (0, 143): ((1, 21), (1, 45), (1, 33)),
                              (0, 113): ((1, 5), (1, 50), (1, 7)),
                              (0, 133): ((1, 41), (1, 5), (1, 77)),
                              (0, 123): ((1, 19), (1, 75), (1, 57)),
                              (0, 104): ((1, 3),)}))

    # Generate external events
    # freqs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    freqs = np.arange(5, 1000, 10)
    SL = []
    for i in range(1):
        SL.append(RegularSpikingStimulus(freqs, sim_ticks))
    ext_evts_data = nsat.exportAER(SL)

    # Set external events
    cfg.set_ext_events(ext_evts_data)

    # Generate all the CNSAT parameters files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_routing')
    c_nsat_writer.write()
    c_nsat_writer.write_L1connectivity()

    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))


def run():
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(nsat.fnames.pickled)
    nsat.run_c_nsat()

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fnames)
    states = c_nsat_reader.read_states()
    states_core0 = states[0][1]
    states_core1 = states[1][1]

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        ax = fig.add_subplot(10, 1, i)
        ax.plot(states_core0[:500, i, 0], 'b', lw=3)
        ax.set_ylim([0, 110])
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        ax = fig.add_subplot(10, 1, i)
        ax.plot(states_core1[:500, i, 0], 'r', lw=3)
        ax.set_ylim([0, 110])

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
