#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Mon Dec 12 10:19:58 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
from corr_spike_trains import correlated_spikes
#import corr_spike_trains
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import time

sim_ticks = 5000                # Simulation ticks


def SimSpikingStimulus(rates, t_sim=None):
    m = np.shape(rates)[0]
    n = int(m / 2)

    C1 = (np.ones((n, n)) + np.random.uniform(0, 1, (n, n)) * 2)
    np.fill_diagonal(C1, rates[:n])
    C1 = np.maximum(C1, C1.T)

    cor_spk1 = correlated_spikes(C1, rates, n)
    cor_spk1.cox_process(time=t_sim)
    spk1 = cor_spk1.extract_pyNCS_list()
    tmp1 = nsat.importAER(spk1)

    SL = pyST.SpikeList(id_list=list(range(m)))
    for i in range(m):
        if i < n:
            SL[i] = tmp1[i]
        if i >= n:
            SL[i] = pyST.STCreate.poisson_generator(rates[i], t_stop=t_sim)
    return SL


def setup():
    global SL
    print('Begin %s:setup()' %
          (os.path.splitext(os.path.basename(__file__))[0]))

    N_CORES = 1                     # Number of cores
    N_NEURONS = [1]                 # Number of neurons per core (list)
    N_INPUTS = [1000]                 # Number of inputs per core (list)
    N_STATES = [4]                  # Number of states per core (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]    # Total inputs

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
                                 monitor_spikes=True,
                                 w_check=False,
                                 tstdpmax=[100],
                                 monitor_weights_final=True,
                                 plasticity_en=[True],
                                 ben_clock=True)

    # Transition matrix
    cfg.core_cfgs[0].A[0] = [[-1, OFF, OFF, OFF],
                             [0, -2, OFF, OFF],
                             [OFF, OFF, 0, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Refractory period
    cfg.core_cfgs[0].t_ref[0] = 20
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 25
    # Bias
    cfg.core_cfgs[0].b[0] = [0, 0, 1, 0]
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([0, 0, 0, 0], 'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = [0, MAX, MAX, MAX]
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = [True, False, False, False]

    # Enable plasticity at states
    cfg.core_cfgs[0].plastic[0] = True
    # Enable STDP
    cfg.core_cfgs[0].stdp_en[0] = True
    # Global modulator state
    cfg.core_cfgs[0].modstate[0] = 2

    # Parameters for the STDP kernel function
    # cfg.core_cfgs[0].tstdp = np.array([32 for _ in rNLRN_GROUPS], 'int')
    # cfg.core_cfgs[0].tca = np.array([[6, 15] for _ in rNLRN_GROUPS], 'int')
    # cfg.core_cfgs[0].hica = np.array([[2, 1, 0] for _ in rNLRN_GROUPS], 'int')
    # cfg.core_cfgs[0].sica = np.array([[1, 1, 1] for _ in rNLRN_GROUPS], 'int')
    # cfg.core_cfgs[0].tac = np.array([[6, 15] for _ in rNLRN_GROUPS], 'int')
    # cfg.core_cfgs[0].hiac = np.array([[-16, -16, 16] for _ in rNLRN_GROUPS],
    #                                  'int')
    # cfg.core_cfgs[0].siac = np.array([[-1, -1, -1] for _ in rNLRN_GROUPS],
    #                                  'int')

    # Set parameters mapping
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Synaptic weights
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[:N_INPUTS[0], N_INPUTS[0], 1] = 50

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[:N_INPUTS[0], N_INPUTS[0], 1] = True

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Generate spike events (external events)
    # rates = np.random.randint(5, 20, (N_INPUTS,), dtype='i')
    rates = np.hstack([np.random.randint(5, 10, (N_INPUTS[0] // 2,), 'int'),
                       np.random.randint(10, 20, (N_INPUTS[0] // 2,), 'int')])
    SL = SimSpikingStimulus(rates, t_sim=sim_ticks)
    ext_evts_data = nsat.exportAER(SL)
    cfg.set_ext_events(ext_evts_data)

    # Write the C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_stdp')
    c_nsat_writer.write()

    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))


def run():
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(nsat.fnames.pickled)
    nsat.run_c_nsat()

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, nsat.fnames)
    ww = np.array(c_nsat_reader.read_c_nsat_synaptic_weights()[0])

    spk = nsat.importAER(c_nsat_reader.read_events(0),
                         sim_ticks=sim_ticks)
    spk.raster_plot()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ww[:cfg.core_cfgs[0].n_inputs, cfg.core_cfgs[0].n_inputs, 1], 'k.')

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
