#!/bin/python
# Purpose: eCD learning of bars abd stripes with NSAT
#

import numpy as np
from pylab import *
from gestures import *
import time
import sys
import copy
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def SimSpikingStimulus(N_INPUTS, **kwargs):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean data/poisson, scaled by *poisson*.
    '''
    ids, spkt, tp = genCoincidencePattern(N=N_INPUTS,
                                          t=kwargs['t_sim'],
                                          pf=kwargs['pf'],
                                          rate=kwargs['f'],
                                          Nf_co=0.1,
                                          jitter=kwargs['jitter']
                                          )

    # Input spike pattern
    stimSpk = np.concatenate([[ids], [spkt]]).T
    stimSpk = stimSpk.astype(int)
    SL = nsat.build_SpikeList(spkt, ids)
    add = SL.id_list().astype('i')
    SL_shuffled = pyST.SpikeList(id_list=range(max(add)))
    # g = np.arange(110)
    np.random.shuffle(add)
    for i, k in enumerate(add):
        SL_shuffled[i] = SL[k]
    return SL_shuffled, tp


def genParamsfor(NP=4, pf=5.0, f=20.0, tp=40.0, **kwargs):
    '''
    Generate parameters for number of input patterns N
    '''
    # Parameters for 1 pattern
    ap = kwargs['Ap']  # 0.0025
    am = kwargs['Am']  # 0.0003
    tfr = kwargs['tfr']
    params = {
        # Vmem learning
        'Ap': ap,
        'Am': am,
        'V_th_stdp': 19.0,
        # Homeostasis
        'tfr': tfr,
        'heta': am,  # (am*1.3)/tfr,
        'tau_r': 1000.0 / (pf),
        #'R_th_lo': 0.5,
        #'R_th_hi': 2.0,
        # Input pattern dependent parameters
        'N_NEURONS': 20 * NP,
    }
    return params


def testParameters(**kwargs):
    '''
    Tests for various conditions for learning stability
    '''
    tn = int(1000.0 / kwargs['pf']) - kwargs['tp']
    try:
        # Check for homeostasis
        assert kwargs['Am'] <= kwargs['tfr'] * \
            kwargs['heta'], 'Homeostasis ineffective'
        # Check for negative weight drift
        assert kwargs['tp'] * kwargs['Ap'] <= 2 * tn * \
            kwargs['Am'], 'No negative weight drift'
        # Rate of learning more than rate of forgetting
        assert kwargs['Ap'] > (tn / 1000.0) * kwargs['f'] * \
            kwargs['Am'], 'Potentiation not retained'
        print('All is well that ends well!')
    except (KeyError, AssertionError) as e:
        print e
        Am_min = kwargs['tp'] * (kwargs['Ap']) / tn / 2.0
        Am_max = kwargs['Ap'] * 1000.0 / tn / kwargs['f']
        print('Am {2} should be in the range ({0}, {1})'.format(
            Am_min, Am_max, kwargs['Am']))


def tauToA(tau=20.0, dt=1.0):
    '''
    Given time constant in ms, return the corresponding A in the 1-2**A
    formulation
    '''
    A = np.log2(dt / tau)
    return int(np.round(A))


if __name__ == '__main__':
    import yaml
    # Load parameters from file
    with open('./examples/voltbased/parameters_mnist.yaml') as f:
        pdict = yaml.load(f)

    pdict['N_INP'] = 100
    pdict['N_NEURONS'] = 5
    pdict['t_sim'] = 5000
    pdict['NP'] = 1

    # Adjust parameters for learning optimally
    # pdict.update(genParamsfor(**pdict))

    testParameters(**pdict)

    Nv = int(pdict['N_INP'])
    Nl = 0
    Nh = int(pdict['N_NEURONS'])
    Np = 0
    Ng1 = 0
    Ng2 = 0

    N_NEURONS = Nh + Ng1 + Ng2 + Np

    n_mult = 1
    sim_ticks = n_mult * int(pdict['t_sim'])
    N_CORES = 1
    N_INPUTS = Nv + Nl
    N_STATES = 8
    N_UNITS = N_NEURONS + N_INPUTS

    N_GROUPS = 8
    N_LRNGROUPS = 8

    rN_STATES = range(N_STATES)
    rN_GROUPS = range(N_GROUPS)
    rN_LRNGROUPS = range(N_LRNGROUPS)

    #MAX = 2**15-1
    #MIN =-2**15
    #OFF = -16
    OFF = nsat.OFF
    MAX = nsat.MAX
    MIN = nsat.MIN

    ###################### Stimulus Creation ##################################
    SL_train, tp = SimSpikingStimulus(N_INPUTS, **pdict)
    SL_train.save('ext_evts')
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

    d.Xinit = np.array([[0 for _ in rN_STATES] for _ in range(N_NEURONS)])
    d.t_ref = np.array([0 for _ in rN_GROUPS])
    d.prob_syn = np.array([[15 for _ in rN_STATES] for _ in rN_GROUPS])
    d.sigma = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
    d.modstate = np.array([1 for _ in rN_GROUPS])
    d.A = np.array([[[OFF] * N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
    d.sA = np.array([[[-1] * N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
    d.b = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])

    d.XresetOn = np.array(
        [[True] + [False for _ in rN_STATES[:-1]]] * N_GROUPS)
    d.Xreset = np.array([[0] + [MAX for _ in rN_STATES[:-1]]
                         for _ in rN_GROUPS])
    d.XspikeResetVal = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
    d.XspikeIncrVal = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
    d.Xth = np.array([MAX for _ in rN_GROUPS])
    d.Xthlo = np.array([[MIN for _ in rN_STATES] for _ in rN_GROUPS])
    d.Xthup = np.array([[MAX for _ in rN_STATES] for _ in rN_GROUPS])
    d.flagXth = np.array([False for _ in rN_GROUPS])
    d.plastic = np.array([False for _ in rN_LRNGROUPS])
    d.plasticity_en = False
    d.stdp_en = np.array([False for _ in rN_LRNGROUPS])
    d.Wgain = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
    #tstdpmax        = np.array( [64 for _ in rN_GROUPS])
    d.tca = np.array([[16, 36] for _ in rN_LRNGROUPS])
    d.hica = np.array([[2, 0, -2] for _ in rN_LRNGROUPS])
    d.sica = np.array([[1, 1, 1] for _ in rN_LRNGROUPS])
    d.tac = np.array([[16, 36] for _ in rN_LRNGROUPS])
    d.hiac = np.array([[-6, -2, -4] for _ in rN_LRNGROUPS])
    d.siac = np.array([[-1, -1, -1] for _ in rN_LRNGROUPS])
    d.t_ref[0] = int(pdict['t_ref'])

    #########################Connectivity#####################################

    sV = 0
    sL = sV + Nv

    sH = 0
    sP = sH + Nh
    sg1 = sP + Np
    sg2 = sg1 + Ng1

    # np.random.seed(10)
    N_UNITS = N_INPUTS + N_NEURONS
    W = np.zeros([N_UNITS, N_UNITS, N_STATES], 'int')
    CW = np.zeros([N_UNITS, N_UNITS, N_STATES], 'int')
    syn_ids_rec = np.arange(N_UNITS, dtype='int')

    # External weights
    extW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
    extCW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
    Wvh = np.random.uniform(0, 100, size=[Nv, Nh]).astype('int')
    extW[sV:sV + Nv, sH:sH + Nh, 0] = Wvh
    extCW[sV:sV + Nv, sH:sH + Nh, 0] = True
    W[:N_INPUTS, N_INPUTS:] = extW
    CW[:N_INPUTS, N_INPUTS:] = extCW

    # Recurrent connections for WTA
    Wvv = 40.0
    Winh = -80.0
    if pdict['NP'] > 1:
        N_BLOCK = int(N_NEURONS / pdict['NP'])
        # Inhibitory connections
        W[N_INPUTS:, N_INPUTS:, 4] = Winh
        # Excitatory connections
        for i in range(pdict['NP']):
            W[N_INPUTS + i * N_BLOCK:N_INPUTS + (i + 1) * N_BLOCK,
              N_INPUTS + i * N_BLOCK:N_INPUTS + (i + 1) * N_BLOCK,
              4] = Wvv
        CW[N_INPUTS:, N_INPUTS:, 4] = True

    W = np.array(W)
    W_init = W.copy()
    print W[N_INPUTS:, N_INPUTS:]
    print CW[N_INPUTS:, N_INPUTS:]

    ############################## Neuron dynamics ###########################
    np.fill_diagonal(d.A[0], 0)
    d.A[0, :5, 0] = [tauToA(pdict['tau_m']), OFF, OFF,
                     OFF, -5]  # Membrane potential
    # Thresholded state of membrane potential
    d.A[0, :5, 1] = [0, 0, OFF, OFF, OFF]
    d.A[0, :5, 2] = [OFF, OFF, tauToA(pdict['tau_r']), OFF, OFF]  # Calcium
    d.A[0, :5, 3] = [OFF, 0, -9, 0, OFF]  # Weight state
    d.A[0, :5, 4] = [OFF, OFF, OFF, OFF, tauToA(
        pdict['tau_e'])]  # Static synapsea
    print d.A[0]

    d.sA[0, :5, :5] = [[-1, 1, 1, 1, 1],
                       [1, -1, 1, 1, 1],
                       [1, 1, -1, -1, 1],
                       [1, 1, 1, -1, 1],
                       [1, 1, 1, 1, -1]]
    print d.sA[0]

    # the first term here hiac[0][0] controls the learning rate (exponentially)
    d.hiac = [[1, 4, 0] for _ in range(N_LRNGROUPS)]

    d.b[0, :4] = [0, -int(pdict['V_th_stdp']), 0, 5]
    print d.b[0]
    d.Xreset[0, :4] = [0, MAX, MAX, MAX]
    d.Xth[0] = int(pdict['vt'])
    d.Xthup[0, :4] = [MAX,  pdict['Ap'], MAX, MAX]
    d.Xthlo[0, :4] = [0, -1 * int(pdict['Am']), MIN, MIN]

    d.plasticity_en = True
    d.plastic[0] = True
    d.stdp_en[0] = False

    d.XspikeIncrVal[0, :4] = [0, 0, 1024, 0]
    d.modstate[0] = 3

    # Synaptic weights and adjacent matrix
    wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
    np.set_printoptions(threshold=np.nan)
    d.wgt_table = wgt_table
    d.ptr_table = ptr_table

    ###########################################################################
    cfg.set_ext_events(ext_evts_data)

    print("Generating parameters files for training")
    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp',
                                      prefix='vmem_coincidence')
    c_nsat_writer.write()

    cfg.core_cfgs[0].latex_print_parameters(1)

    print("Running C NSAT!")
    nsat.run_c_nsat(c_nsat_writer.fname)

    c_nsat_reader = nsat.C_NSATReader(cfg, c_nsat_writer.fname)
    states = c_nsat_reader.read_c_nsat_states()
    states_core0 = states[0][1]
    np.save('states_vmem_', states_core0)

    fW = c_nsat_reader.read_c_nsat_synaptic_weights()

    # Load results
    # Output spikes
    # test_spikelist = nsat.importAER(nsat.read_from_file(c_nsat_writer.fname.events + '_core_0.dat'),
    #                                 sim_ticks=sim_ticks,
    #                                 id_list=range(N_NEURONS))

    test_spikelist = nsat.importAER(c_nsat_reader.read_c_nsat_raw_events()[0],
                                    sim_ticks=sim_ticks,
                                    id_list=range(N_NEURONS))

    test_spikelist.save('nsat_spikes')

    # # Plot data
    figure(figsize=(8, 4))
    gs = gridspec.GridSpec(4 + 5, 1)
    ax1 = plt.subplot(gs[0:3, 0])
    np.save('times_vmem_', tp)
    for t in tp:
        ax1.fill_between([t - 25, t + pdict['tp'] - 25], [0, 0], [N_INPUTS, N_INPUTS],
                         linewidth=0, alpha=0.5, facecolor='b')

    ax2 = plt.subplot(gs[3:4, 0], sharex=ax1)

    SL_train.raster_plot(display=ax1, kwargs={'marker': '|', 'color': 'k'})
    test_spikelist.raster_plot(
        display=ax2, kwargs={'marker': '|', 'color': 'k'})

    # Plot all states
    for i in range(5):
        ax = plt.subplot(gs[i + 4:i + 5, 0], sharex=ax1)
        ax.plot(states_core0[:, 1, i + 1], lw=2)
    plt.show()
