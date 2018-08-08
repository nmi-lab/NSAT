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
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time

sim_ticks = 50000              # Simulation time
SL = None


def RegularSpikingStimulus(freqs, ticks=1000):
    global SL
    pyST.STCreate.seed(100)
    m = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(m)))

    tm = 500
    for i in range(m):
        # SL[i] = pyST.STCreate.regular_generator(freqs[i],
        #                                         t_start=1,
        #                                         t_stop=ticks)
        if i != (m - 1):
            t_start = tm
            t_stop = tm + 2000
            SL[i] = pyST.STCreate.poisson_generator(freqs[i],
                                                    t_start,
                                                    t_stop)
            tm += 2000
        SL[m - 2] = SL[m - 3]
        SL[m - 1] = SL[m - 3]

    return SL


def setup():
    print('Begin %s:setup()' %
          (os.path.splitext(os.path.basename(__file__))[0]))

    pyST.STCreate.seed(100)
    N_CORES = 1                 # Number of cores
    N_NEURONS = [17]            # Number of neurons per core
    N_INPUTS = [16]             # Number of inputes per core
    N_STATES = [8]              # Number of states per core
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    # Transition matrix group 0
    cfg.core_cfgs[0].A[0] = [[-3, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [2, -5, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, -7, -5, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, 0, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF]]

    # Sign matrix group 0
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1, 1, 1, 1, 1],
                              [1, -1, 1, 1, 1, 1, 1, 1],
                              [1, 1, -1, -1, 1, 1, 1, 1],
                              [1, 1, 1, -1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]]

    # Transition matrix group 1
    cfg.core_cfgs[0].A[1] = [[0, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [0, -10, OFF, OFF, OFF, OFF, OFF, OFF],
                             [0, OFF, -8, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF]]

    # Sign matrix group 1
    cfg.core_cfgs[0].sA[1] = [[-1, 1, 1, 1, 1, 1, 1, 1],
                              [+1, -1, 1, 1, 1, 1, 1, 1],
                              [-1, 1, -1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]]

    # Transition matrix group 2
    cfg.core_cfgs[0].A[2] = [[0, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [-3, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, -13, -6, OFF, OFF, OFF, OFF, OFF],
                             [OFF, -15, OFF, -8, OFF, OFF, OFF, OFF],
                             [0, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF, -7, -8, OFF, OFF],
                             [OFF, OFF, OFF, OFF, -7, OFF, -3, OFF],
                             [OFF, OFF, OFF, OFF, OFF, OFF, OFF, OFF]]

    # Sign matrix group 2
    cfg.core_cfgs[0].sA[2] = [[-1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, -1, 1, 1, 1, 1, 1],
                              [1, -1, 1, -1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, -1, 1, 1],
                              [1, 1, 1, 1, -1, 1, -1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([-12000, 0, -7, 0, 0, 0, 0, 0], 'int')
    cfg.core_cfgs[0].b[1] = np.array([-40, 0, 0, 0, 0, 0, 0, 0], dtype='int')
    cfg.core_cfgs[0].b[2] = np.array([-40, 0, 0, 0, 0, 0, 0, 0], dtype='int')
    # Threshold
    cfg.core_cfgs[0].t_ref[0] = 0
    # cfg.core_cfgs[0].Xth[0] = 30
    # Spike increment value
    cfg.core_cfgs[0].XspikeIncrVal[1] = np.array([-1000] + [0] * 7, 'int')
    # Additive noise variance
    cfg.core_cfgs[0].sigma[0] = np.array([15000] + [0] * 7, 'int')
    # refractory period
    cfg.core_cfgs[0].t_ref[0] = 0
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0] + [XMAX] * 7, 'int')

    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True] + [False] * 7, 'bool')
    cfg.core_cfgs[0].XresetOn[1] = np.array([False] * 8, 'bool')
    cfg.core_cfgs[0].XresetOn[2] = np.array([False] * 8, 'bool')

    # Mapping function between neurons and NSAT parameters groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')
    cfg.core_cfgs[0].nmap[-2] = 1
    cfg.core_cfgs[0].nmap[-1] = 2

    # Synaptic strength
    from scipy.linalg import toeplitz
    col = np.zeros((N_INPUTS[0] - 1,))
    col[0] = 1
    row = np.zeros((N_NEURONS[0] - 2,))
    row[0:3] = np.array([1, 2, 1])
    T = toeplitz(col, row)
    W = np.zeros((N_UNITS, N_UNITS, N_STATES[0]), 'i')
    W[:N_INPUTS[0] - 1, N_INPUTS[0]:-2, 1] = T
    W[N_INPUTS[0]:, N_UNITS - 2, 1] = 100
    W[N_INPUTS[0]:, N_UNITS - 2, 2] = 100
    W[N_INPUTS[0]:, N_UNITS - 1, 2] = 1
    W[N_INPUTS[0]:, N_UNITS - 1, 3] = 1
    W[N_INPUTS[0] - 1, N_UNITS - 1, 5] = 100
    W[N_INPUTS[0] - 1, N_UNITS - 1, 6] = 100

    CW = W.astype('bool')

    # np.set_printoptions(threshold=np.nan)
    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate external events firing rates from 5 to 50 inc by 5
    freqs = np.random.randint(300, 400, (N_INPUTS[0],))
    SL = RegularSpikingStimulus(freqs, sim_ticks)
    # SL.raster_plot()
    ext_evts_data = nsat.exportAER(SL)

    # Set external events
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_td')
    c_nsat_writer.write()

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
    # np.save('states', states_core0)

    in_spikelist = SL
    ifname = nsat.fnames.events + '_core_0.dat'
    out_spikelist = nsat.importAER(nsat.read_from_file(ifname),
                                   sim_ticks=sim_ticks,
                                   id_list=[0])

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 9):
        ax = fig.add_subplot(8, 1, i)
        ax.plot(states_core0[:-1, 15, i - 1], 'b', lw=1.5)

    # fig = plt.figure(figsize=(10, 10))
    # for i in range(1, 9):
    #     ax = fig.add_subplot(8, 1, i)
    #     ax.plot(states_core0[:-1, 16, i-1], 'b', lw=1.5)

    # out_spikelist.raster_plot()

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
