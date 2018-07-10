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

def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i],
                                                    t_start=1,
                                                    t_stop=ticks)
    return nsat.exportAER(SL)


def PoissonSpikingStimulus(rates, n_inputs=2):
    SL = pyST.SpikeList(id_list=list(range(n_inputs)))
    for i in range(n_inputs):
        # Tracking
        # if i > 0 and i < 20:
        #     t_start = 0
        #     t_stop = 500
        #     rate = 50
        # elif i > 20 and i < 40:
        #     t_start = 500
        #     t_stop = 1000
        #     rate = 50
        # elif i > 40 and i < 60:
        #     t_start = 1000
        #     t_stop = 1500
        #     rate = 50
        # elif i > 60 and i < 80:
        #     t_start = 1500
        #     t_stop = 2000
        #     rate = 50
        # elif i > 80 and i < 100:
        #     t_start = 2000
        #     t_stop = 2500
        #     rate = 50
        # else:
        #     continue
        # SL[i] = pyST.STCreate.poisson_generator(rate,
        #                                         t_start,
        #                                         t_stop)

        t_start = 100
        t_stop = 500
        rate = 1
        if (i > 40 and i < 60):
            t_start = 100
            t_stop = 500
            rate = 35
        if (i < 40 or i > 60):
            t_start = 100
            t_stop = 500
            rate = 10
        SL[i] = pyST.STCreate.poisson_generator(rate,
                                                t_start,
                                                t_stop)

        # Selection
        # if (i > 20 and i < 40) or (i > 70 and i < 90):
        #     t_start = 100
        #     t_stop = 500
        #     rate = 50
        #     SL[i] = pyST.STCreate.poisson_generator(rate,
        #                                             t_start,
        #                                             t_stop)
    return nsat.exportAER(SL), SL


def g(x, mean=0, sigma=0.1):
    return np.exp(-(x - mean)**2 / (2 * sigma**2))


def kernel(size, amp=(2, 1), sigma=(0.1, 1.0)):
    w = np.zeros((size, size))

    c = np.linspace(-0.5, 0.5, size)
    x = np.linspace(-0.5, 0.5, size)
    for i in range(size):
        w[i] = (amp[0] * g(x, c[i], sigma[0]) -
                amp[1] * g(x, c[i], sigma[1]))
#                amp[1] * g(x, c[i]-0.1, sigma[1]))  # for tracking

    q = 0.5 * np.floor(w / 0.5 + 0.5)
    return q.astype('i')


def h(x, sigma):
    scale = 1.0 / (np.sqrt(sigma * np.pi))
    return scale * np.exp(-x**2 / sigma)


def kernel_(size, amp=(5, 5), sigma=(1.0/28.0, 1.0/20.0)):
    w = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            tmp = np.abs(i - j) / float(size)
            w[i, j] = (amp[0] * h(tmp, sigma[0]) -
                       amp[1] * h(tmp, sigma[1]))
    return w.astype('int')


if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    sim_ticks = 2500
    N_CORES = 1
    N_NEURONS = [100]
    N_INPUTS = [100]
    N_STATES = [4]
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]

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
                                 monitor_spikes=True,
                                 ben_clock=True)

    cfg.core_cfgs[0].A[0] = [[-2,  OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF],
                             [OFF, OFF, OFF, OFF]]

    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    cfg.core_cfgs[0].b[0] = np.array([-5, 0, 0, 0], dtype='int')
    cfg.core_cfgs[0].Xth[0] = 20
    cfg.core_cfgs[0].Xthup[0] = np.array([XMAX, XMAX, XMAX, XMAX], 'int')
    cfg.core_cfgs[0].Xthlo[0] = np.ones(4, 'int') * XMIN
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    cfg.core_cfgs[0].XresetOn[0] = np.array([False, False, False, False],
                                            'bool')

    # Synaptic weights
    # Bump
    tmp_w = kernel(N_NEURONS[0], amp=(7, 3), sigma=(0.1, 0.35))
    # Selection
    # tmp_w = kernel(N_NEURONS, amp=(6, 3), sigma=(0.1, 1.0))
    # Tracking
    # tmp_w = kernel(N_NEURONS, amp=(6, 3), sigma=(0.1, 1.0))
    # tmp_w = kernel_(N_NEURONS, (3, 4), (0.01, 0.03))

    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    # W[:N_INPUTS, N_INPUTS+40:N_INPUTS+60, 0] = np.ones((20,)) * 30
    np.fill_diagonal(W[:N_INPUTS[0], N_INPUTS[0]:, 0], 30)
    W[N_INPUTS[0]:, N_INPUTS[0]:, 0] = tmp_w

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    # CW[:N_INPUTS, N_INPUTS+40:N_INPUTS+60, 0] = 1
    np.fill_diagonal(CW[:N_INPUTS[0], N_INPUTS[0]:, 0], 1)
    CW[N_INPUTS[0]:, N_INPUTS[0]:, 0] = 1

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Parameters groups mapping function
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Set external events
    rates = [20]*N_INPUTS[0]
    ext_evts_data, sl = PoissonSpikingStimulus(rates, n_inputs=N_INPUTS[0])
    events_i = sl.convert()
    mat = np.zeros((sim_ticks, N_INPUTS[0]))
    mat[events_i[0].astype('i'), events_i[1].astype('i')] = 1

    # ext_evts_data = None
    # f = [i for i in np.random.randint(0, 15, (N_INPUTS,))]
    # ext_evts_data = RegularSpikingStimulus(f, 300)
    cfg.set_ext_events(ext_evts_data)

    cfg.core_cfgs[0].latex_print_parameters(1)

    # Write C NSAT parameters files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_nf')
    c_nsat_writer.write()

    # Write FPGA NSAT parameters files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_nf')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()

    # Call the C NSAT
    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]

    plt.figure()
    for i in range(1, 5):
        plt.subplot(4, 1, i)
        plt.plot(states_core0[:, :, i-1])

    plt.figure()
    S = np.maximum(states_core0[:, :, 0], 0)
    plt.imshow(S, interpolation='spline36', cmap=plt.get_cmap('gray'),#plt.cm.gray,
               aspect='auto', origin='lower')

    spks = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events+'_core_0.dat'),
                          sim_ticks=sim_ticks,
                          id_list=list(range(N_NEURONS[0])))
    
    raster = spks.raster_plot()
    raster.savefig('/tmp/%s_raster.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    raster.close()
    
    # Plot the results
#     events = spks.convert()
#     mat = np.zeros((sim_ticks, N_NEURONS[0]))
#     print(mat.shape)
#     print(len(events[0]))
#     print(len(events[1]))
#     mat[events[0].astype('i'), events[1].astype('i')] = 1
    


    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))