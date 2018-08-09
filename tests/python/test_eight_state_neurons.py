#!/usr/bin/env python
#  ---------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : Fri 20 Jan 2017 10:23:17 PM PST
#
# Copyright : (c)
# Licence : GPLv2
#  ---------------------------------------------------------------------------
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import os
import timeit

sim_ticks = 500                 # Total simulation time

def RegularSpikingStimulus(freqs, ticks=1000):
    N_NEURONS = np.shape(freqs)[0]
    SL = pyST.SpikeList(id_list=list(range(N_NEURONS)))
    for i in range(N_NEURONS):
        f = freqs[i]
        if f > 0:
            SL[i] = pyST.STCreate.regular_generator(freqs[i], t_stop=ticks)
    return nsat.exportAER(SL)


def setup():
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    pyST.STCreate.seed(130)         # pyNCSre random number generator
    np.random.seed(30)              # Numpy random number generator

    N_CORES = 1                     # Number of cores
    N_NEURONS = [1]                 # Number of neurons per core (list)
    N_INPUTS = [1]                  # Number of inputs per core (list)
    N_STATES = [8]                  # Number of states per core (list)
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units

    # Number of states (8)
    rN_STATES = list(range(N_STATES[0]))
    # Number of parameters groups
    rN_GROUPS = list(range(nsat.N_GROUPS))

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Main class isntance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 ben_clock=True)

    # Transition matrix A
    cfg.core_cfgs[0].A[0] = np.array([[-1, OFF, OFF, OFF, OFF, OFF, OFF, OFF],
                                      [-3, -1,  OFF, OFF, OFF, OFF, OFF, OFF],
                                      [OFF, -2,  -1, OFF, OFF, OFF, OFF, OFF],
                                      [OFF, OFF, -1, -1, OFF, OFF,  OFF, OFF],
                                      [OFF, OFF, OFF, -1,  -1, OFF,  OFF, OFF],
                                      [OFF, OFF, OFF, OFF, -2, -1,   OFF, OFF],
                                      [OFF, OFF, OFF, OFF, OFF, -3, -1,  OFF],
                                      [OFF, OFF, OFF, OFF, OFF, OFF, 0, -1]],
                                     'int')

    # Sign matrix sA
    cfg.core_cfgs[0].sA[0] = np.array([[-1, 1, 1, 1, 1, 1, 1, 1],
                                       [1, -1, 1, 1, 1, 1, 1, 1],
                                       [1, 1, -1, 1, 1, 1, 1, 1],
                                       [1, 1, 1, -1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, -1, 1, 1, 1],
                                       [1, 1, 1, 1, 1, -1, 1, 1],
                                       [1, 1, 1, 1, 1, 1, -1, 1],
                                       [1, 1, 1, 1, 1, 1, 1, -1]], 'int')

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0], 'int')
    # Initiali conditions
    cfg.core_cfgs[0].Xinit[0] = np.array([[0]*8 for _ in range(N_NEURONS[0])],
                                         'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False, False,
                                             False, False, False], 'bool')
    # Synaptic probability
    cfg.core_cfgs[0].prob = np.ones(8, dtype='int') * 15
    # Threshold
    cfg.core_cfgs[0].Xth[0] = 100

    # Mapping between neurons and NSAT parameters groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Sunaptic weights
    W = np.zeros([N_UNITS, N_UNITS, 8], dtype='int')
    W[0, 1, 7] = 50

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='bool')
    CW[0, 1, 7] = True

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Set external events (spikes)
    freqs = [15]
    ext_evts_data = RegularSpikingStimulus(freqs, ticks=sim_ticks)
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='test_eight_states_neuron')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_eight_states_neuron')
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
    time_core0, states_core0 = states[0][0], states[0][1]

    out_spikelist = nsat.importAER(nsat.read_from_file(nsat.fnames.events+'_core_0.dat'),
                                   sim_ticks=sim_ticks,
                                   id_list=[0])

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 9):
        ax = fig.add_subplot(8, 1, i)
        ax.plot(states_core0[:, 0, i-1], 'b', lw=2)
        
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = timeit.default_timer()
    
    setup()
    run()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], timeit.default_timer()-start_t))
    