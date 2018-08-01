#!/usr/bin/env python
# ----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_bs.py
# Purpose: eCD learning of bars abd stripes with NSAT
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Fri Dec  9 18:01:18 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ----------------------------------------------------------------------------
#
# Update Thu Feb 3

import sys
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
from scipy.optimize import curve_fit
import os
import time

sim_ticks = 25000            # Simulation time

def SimSpikingStimulus(stim, t=1000, t_sim=None):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean
    data/poisson, scaled by *poisson*.
    '''
    n = np.shape(stim)[0]
    SL = pyST.SpikeList(id_list=list(range(n)))
    for i in range(n):
        SL[i] = pyST.STCreate.regular_generator(stim[i], t_stop=t_sim)
    return SL


def setup():
    global SL
    print('Begin %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    N_CORES = 1                 # Number of cores
    N_test = 2                  # Number of tests
    Nv = 100                    # Visible neurons
    Nh = 100                    # Hidden neurons
    N_NEURONS = [Nh]            # Total number of inputes per core
    N_INPUTS = [Nv]             # Total number of inputs per core
    N_STATES = [4]              # Number of states per core
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

    # Transition matrix
    cfg.core_cfgs[0].A[0] = [[-3, OFF, OFF, OFF],
                             [2, -5, OFF, OFF],
                             [OFF, OFF, -7, -5],
                             [OFF, OFF, OFF, 0]]

    # Sign matrix
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, -1, 1, 1],
                              [1, 1, -1, -1],
                              [1, 1, 1, -1]]

    # Bias
    cfg.core_cfgs[0].b[0] = [-12000, 0, -7, 0]
    # Refractory period
    cfg.core_cfgs[0].t_ref[0] = 40
    # Initial conditions
    cfg.core_cfgs[0].Xinit[0] = np.zeros((N_STATES[0],), 'int')
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, MAX, MAX, MAX], 'int')
    # Turn on reset
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')

    # Spike increment value
    cfg.core_cfgs[0].XspikeIncrVal[1] = np.array([-1000, 0, 0, 0], 'int')
    # Additive noise variance
    cfg.core_cfgs[0].sigma[0] = np.array([15000, 0, 0, 0], 'int')

    # Set parameters
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Set weight matrix
    extW = np.zeros([N_INPUTS[0], N_NEURONS[0], N_STATES[0]])
    extCW = np.zeros([N_INPUTS[0], N_NEURONS[0], N_STATES[0]])
    extW[:Nv, :Nh, 1] = np.eye(N_INPUTS[0]) * 127
    extCW[:Nv, :Nh, 1] = np.eye(N_INPUTS[0]).astype('bool')

    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[:N_INPUTS[0], N_INPUTS[0]:] = extW

    # Set adjacent matrix
    CW = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    CW[:N_INPUTS[0], N_INPUTS[0]:] = extCW

    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    cfg.core_cfgs[0].wgt_table = wgt_table
    cfg.core_cfgs[0].ptr_table = ptr_table

    # Set external events
    stim = np.linspace(1, 1000, N_NEURONS[0])
    ext_evts_data = nsat.exportAER(SimSpikingStimulus(stim,
                                                      sim_ticks,
                                                      t_sim=sim_ticks))
    cfg.set_ext_events(ext_evts_data)

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_sigmoid_ext')
    c_nsat_writer.write()
    
    print('End %s:setup()' % (os.path.splitext(os.path.basename(__file__))[0]))
    return c_nsat_writer.fname


def run(fnames):
    # Call the C NSAT
    print('Begin %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    cfg = nsat.ConfigurationNSAT.readfileb(fnames.pickled)
    nsat.run_c_nsat(fnames)

    print("Plotting data")
    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, fnames)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]

    # pip = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events+'_core_0.dat'),
    #                      sim_ticks=sim_ticks,
    #                      id_list=range(N_NEURONS[0])).time_slice(1000, sim_ticks-1000).mean_rates()
    pip = nsat.importAER(c_nsat_reader.read_events(0),
                         sim_ticks=sim_ticks,
                         id_list=list(range(cfg.core_cfgs[0].n_neurons))).time_slice(1000, sim_ticks-1000).mean_rates()
    x = np.arange(pip.shape[0])

    from scipy.optimize import curve_fit
    # define "to-fit" function
    def sigmoid(x, a, b, c):
        return a / (1 + np.exp(-b * x / a) * (a - c) / a)

    # Fit data to function sigmoid
    popt, pcov = curve_fit(sigmoid, x, pip)
    print(("Sigmoid's parameters: a = {}, b = {}, c = {}".format(popt[0],
                                                                popt[1],
                                                                popt[2])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, pip, 'k', lw=2)
    ax.plot(x, sigmoid(x, popt[0], popt[1], popt[2]), 'r--', lw=2)
    
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
       
if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
    filenames = setup()
    run(filenames)
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
 