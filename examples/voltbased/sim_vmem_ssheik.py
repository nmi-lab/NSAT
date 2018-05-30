#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_bs.py
# Purpose: Vmem based pattern recognition
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Thu 22 Sep 2016 10:13:08 AM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#
# Update Thu Feb 3

import numpy as np
from pylab import *
import time, sys, copy
from pyNCS import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
from gestures import genEmbeddedSpikePattern
#from mniststrokes import genEmbeddedSpikePattern
from mylib import *

def SimSpikingStimulus(N_INPUTS, **kwargs):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean data/poisson, scaled by *poisson*.
    '''
    # Generate input spike pattern
    #ids,spkt,tp,p = genEmbeddedSpikePattern(N=N_INPUTS, 
    #                                        rate=kwargs['f'], 
    #                                        t=kwargs['t_sim'], 
    #                                        pf=kwargs['pf'], 
    #                                        pd=kwargs['tp'], 
    #                                        target= 'mixed',
    #                                        NP=kwargs['NP'],
    #                                        jitter=kwargs['jitter'],
    #                                        target_patterns=['random',
    #                                                         'random',
    #                                                         'random',
    #                                                         'random']
    #                                       )
    ids, spkt, tp, p, pds = genEmbeddedSpikePattern(N=N_INPUTS,
                                               t=kwargs['t_sim'],
                                               pf=kwargs['pf']*kwargs['NP'],
                                               pd=50,
                                               rate = kwargs['f'],
                                               target=range(kwargs['NP']))

    # Input spike pattern
    stimSpk = np.concatenate([[ids],[spkt]]).T
    stimSpk = stimSpk.astype(int)
    SL = nsat.build_SpikeList(spkt, ids)

    return SL, tp, p



if __name__ == '__main__':
    import yaml
    # Load parameters from file
    with open('/home/gdetorak/NMI-lab/HiAER-NSAT/examples/Sheik_etal16/parameters_mnist.yaml') as f:
        pdict = yaml.load(f)

    ## Adjust parameters for learning optimally
    #pdict.update(genParamsfor(**pdict))

    testParameters(**pdict)

    Nv = int(pdict['N_INP'])
    Nl = 0
    Nh = int(pdict['N_NEURONS'])
    Np = 0
    Ng1 = 0
    Ng2 = 0

    N_NEURONS = Nh + Ng1 + Ng2 + Np

    n_mult = 1
    sim_ticks = n_mult*int(pdict['t_sim'])
    N_INPUTS = Nv + Nl
    N_STATES = 8
    N_CORES = 1
    N_UNITS = N_NEURONS + N_INPUTS

    N_GROUPS = 8
    N_LRNGROUPS = 8

    rN_STATES = range(N_STATES)
    rN_GROUPS = range(N_GROUPS)
    rN_LRNGROUPS = range(N_LRNGROUPS)

    MAX = nsat.MAX
    MIN = nsat.MIN
    OFF = nsat.OFF

    ###################### Stimulus Creation ##################################
    SL_train,tp, p = SimSpikingStimulus(N_INPUTS, **pdict)
    ext_evts_data = nsat.exportAER(SL_train)
    ###########################################################################

    # Configuration class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=[N_INPUTS],
                                 N_NEURONS=[N_NEURONS],
                                 N_STATES=[N_STATES],
                                 monitor_states=True,
                                 monitor_spikes=True,
                                 monitor_weights_final=True,
                                 plasticity_en=[True],
                                 ben_clock=True)
    d = cfg.core_cfgs[0]

    # Parameters groups mapping function
    d.nmap = np.zeros(N_NEURONS, dtype='int')
    d.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
    d.lrnmap[:,  0] = 0

    ################### NSAT Parameter configuration ##########################

    d.Xinit          = np.array([[0 for _ in rN_STATES] for _ in range(N_NEURONS)])
    d.t_ref           = np.array( [0 for _ in rN_GROUPS])
    d.prob_syn        = np.array( [[15 for _ in rN_STATES ] for _ in rN_GROUPS])
    d.sigma           = np.array( [[0 for _ in rN_STATES] for _ in rN_GROUPS] )
    d.modstate        = np.array( [1 for _ in rN_GROUPS])
    d.A               = np.array( [ [[OFF]*N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
    d.sA              = np.array( [ [[-1]*N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
    d.b               = np.array( [ [0 for _ in rN_STATES] for _ in rN_GROUPS])

    d.XresetOn        = np.array([[True]+[False for _ in rN_STATES[:-1]]]*N_GROUPS)
    d.Xreset          = np.array( [[0]+[MAX for _ in rN_STATES[:-1]] for _ in rN_GROUPS])
    d.XspikeResetVal  = np.array( [[0 for _ in rN_STATES] for _ in rN_GROUPS])
    d.XspikeIncrVal   = np.array( [[0 for _ in rN_STATES] for _ in rN_GROUPS])
    d.Xth             = np.array( [MAX for _ in rN_GROUPS])
    d.Xthlo           = np.array( [[MIN for _ in rN_STATES] for _ in rN_GROUPS])
    d.Xthup           = np.array( [[MAX for _ in rN_STATES] for _ in rN_GROUPS])
    d.flagXth         = np.array([False for _ in rN_GROUPS])
    d.plastic         = np.array([False for _ in rN_LRNGROUPS])
    d.plasticity_en   = False
    d.stdp_en         = np.array([False for _ in rN_LRNGROUPS])
    d.Wgain           = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
    #d.tstdpmax        = np.array( [64 for _ in rN_GROUPS])
    d.tca             = [[16,36] for _ in rN_LRNGROUPS]
    d.hica            = [[2, 0, -2] for _ in rN_LRNGROUPS]
    d.sica            = [[1, 1, 1] for _ in rN_LRNGROUPS]
    d.tac             = [[16,36] for _ in rN_LRNGROUPS]
    d.hiac            = [[0, -2, -4] for _ in rN_LRNGROUPS]
    d.siac            = [[-1, -1, -1] for _ in rN_LRNGROUPS]
    d.t_ref[0] = int(pdict['t_ref']);


    ###########################################################################

    ################### NSAT Parameter configuration ##########################

    sV = 0
    sL = sV + Nv

    sH = 0
    sP = sH + Nh
    sg1 = sP + Np
    sg2 = sg1 + Ng1

    #np.random.seed(10)
    N_UNITS = N_INPUTS + N_NEURONS
    W = np.zeros([N_UNITS, N_UNITS, N_STATES], 'int')
    CW = np.zeros([N_UNITS, N_UNITS, N_STATES], 'int')
    syn_ids_rec = np.arange(N_UNITS, dtype='int')

    # External weights
    extW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
    extCW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
    Wvh = np.random.uniform(0,100, size=[Nv, Nh]).astype('int')
    extW[sV:sV+Nv, sH:sH+Nh, 0] = Wvh
    extCW[sV:sV+Nv, sH:sH+Nh, 0] = True
    W[:N_INPUTS,N_INPUTS:] = extW
    CW[:N_INPUTS,N_INPUTS:] = extCW

    # Recurrent connections for WTA
    Wvv = 5.0
    Winh = -150.0
    if pdict['NP'] > 1:
        N_BLOCK = int(N_NEURONS/pdict['NP'])
        # Inhibitory connections
        W[N_INPUTS:,N_INPUTS:,4] = Winh
        # Excitatory connections
        for i in range(pdict['NP']):
            W[N_INPUTS+i*N_BLOCK:N_INPUTS+(i+1)*N_BLOCK,
              N_INPUTS+i*N_BLOCK:N_INPUTS+(i+1)*N_BLOCK,
              4] = Wvv
        CW[N_INPUTS:,N_INPUTS:, 4] = True


    W = np.array(W)
    W_init = W.copy()


    ############################## Neuron dynamics ################################
    np.fill_diagonal(d.A[0], 0)
    d.A[0,:5,0] = [tauToA(pdict['tau_m']), OFF, OFF, OFF, -5] # Membrane potential
    d.A[0,:5,1] = [0, 0, OFF, OFF, OFF] # Thresholded state of membrane potential
    d.A[0,:5,2] = [OFF, OFF, tauToA(pdict['tau_r']), OFF, OFF] # Calcium
    d.A[0,:5,3] = [OFF, 0, -9, 0, OFF] # Weight state
    d.A[0,:5,4] = [OFF, OFF, OFF, OFF, tauToA(pdict['tau_e'])] # Static synapsea


    d.sA[0,:5,:5] =[[-1, 1, 1, 1, 1],
                  [ 1,-1, 1, 1, 1],
                  [ 1, 1,-1,-1, 1],
                  [ 1, 1, 1,-1, 1],
                  [ 1, 1, 1, 1,-1]]

    # the first term here hiac[0][0] controls the learning rate (exponentially)
    d.hiac=[[1, 4, 0] for _ in range(N_LRNGROUPS)]

    d.b[0,:4] = [0, -int(pdict['V_th_stdp']), 0, 4];
    d.Xreset[0,:4] = [0, MAX, MAX, MAX];
    d.Xth[0] = int(pdict['vt'])
    d.Xthup[0,:4] =   [MAX,  pdict['Ap'], MAX, MAX]
    d.Xthlo[0,:4] =   [0, -1*int(pdict['Am']), MIN, MIN]

    d.spk_rec_mon = np.arange(N_NEURONS, dtype='int')
    d.plasticity_en = True
    d.plastic[0] = True
    d.stdp_en[0] = False

    d.XspikeIncrVal[0,:4] = [0, 0, 1024, 0];
    d.modstate[0] = 3;

    # Synaptic weights and adjacent matrix
    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    d.wgt_table = wgt_table
    d.ptr_table = ptr_table

    ###########################################################################
    cfg.set_ext_events(ext_evts_data)

    print("Generating parameters files!")
    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='vmem_ssheik')
    c_nsat_writer.write()

    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]

    fW = c_nsat_reader.read_c_nsat_weights()


    # Load results
    # Output spikes
    test_spikelist = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events+'_core_0.dat'),
                               sim_ticks = sim_ticks,
                               id_list = range(N_NEURONS))


    # Plot data
    from matplotlib.pyplot import cm

    figure(figsize=(8,4))
    gs = gridspec.GridSpec(4+5, 1)
    ax1 = plt.subplot(gs[0:3,0])
    colors_test=cm.rainbow(np.linspace(0,1,pdict['NP']))
    for i in range(pdict['NP']):
            ax1.axvspan(-1000, -1000+pdict['tp'], alpha=0.3, color=colors_test[i])
            ax1.legend(range(pdict['NP']))
    # Plot highlighting input patterns
    for t,i in zip(tp,p):
            ax1.axvspan(t, t+pdict['tp'], alpha=0.2, color=colors_test[i])
    #for t in tp:
    #    ax1.fill_between([t,t+pdict['tp']],[0,0],[N_INPUTS,N_INPUTS],
    #                     linewidth=0,alpha=0.5,facecolor='b')

    ax2 = plt.subplot(gs[3:4,0], sharex=ax1)

    SL_train.raster_plot(display = ax1, kwargs={'marker':'.','color':'k'})
    test_spikelist.raster_plot(display = ax2, kwargs={'marker':'.','color':'k'})

    # Plot all states
    for i in range(5):
        ax = plt.subplot(gs[i+4:i+5,0], sharex=ax1)
        ax.plot(states_core0[:, 0, i], lw=2)
    plt.show()
