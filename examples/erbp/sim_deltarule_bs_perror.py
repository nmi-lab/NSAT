#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_bs.py
# Purpose: eCD learning of bars abd stripes with NSAT
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Fri 13 Jan 2017 09:54:37 AM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#
# Update Thu Feb 3

import numpy as np
from pylab import *
import time, sys, copy
from bars_stripes import *
from pyNCS import pyST
from pyNSATlib import *
from ml_funcs import data_preprocess

def SimSpikingStimulus(stim, time = 1000, t_sim = None):
        '''
        Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
        *poisson*: integer, output is a poisson process with mean data/poisson, scaled by *poisson*.
        '''
	n = np.shape(stim)[1]
        SL = pyST.SpikeList(id_list = range(n))
        for i in range(n):
            if np.any(stim[:,i]>0):
                SL[i] = pyST.STCreate.inh_poisson_generator(stim[:,i], range(0,len(stim)*time,time), t_stop=t_sim, refractory = 40)
        return SL

def pretty_state_plot(n_inputs, states, inSL, outSL, pre, post, state=None, wt = None):
    '''
    *pre*: address to plot in inSL
    *post*: address to plot in outSL

    '''
    
    # Plot the results
    fig = plt.figure(figsize=(10, 10))

    i=4
    ax1 = fig.add_subplot(4, 1, i)
    for t in inSL[pre].spike_times:
        plt.axvline(t, color='k')
    ax1.plot(states[:-1, post-n_inputs, 2], 'b', lw=3, label = '$x_m$')
    ax1.set_ylabel('$x_m$')
    plt.axhline(0,color = 'b', alpha=.5, linewidth=3)
    plt.locator_params(axis='y',nbins=4)

    ax2 = fig.add_subplot(4, 1, 3, sharex=ax1)
    if wt is not None:
        wt = wt[:,post,state]
        for t in inSL[pre].spike_times:
            plt.axvline(t, color='k')
        ax2.plot(wt, 'r', lw=3)
        ax2.set_ylabel('$w$')
        plt.locator_params(axis='y',nbins=4)

    i=1
    ax3 = fig.add_subplot(4, 1, i, sharex=ax1)
    for t in outSL[post-n_inputs].spike_times:
        plt.axvline(t, color='k')
    ax3.plot(states[:-1, post-n_inputs, i-1], 'b', lw=3, label = '$V_m$')
    ax3.set_ylabel('$V_m$')
    plt.locator_params(axis='y',nbins=4)
    i=2
    ax4 = fig.add_subplot(4, 1, i, sharex=ax1)
    ax4.plot(states[:-1, post-n_inputs, i-1], 'b', lw=3, label = '$I_{syn}$')
    ax4.set_ylabel('$I_{syn}$')
    plt.locator_params(axis='y',nbins=4)
    for t in np.ceil(inSL[pre].spike_times):
        plt.axvline(t, color='k')
    return ax1,ax2,ax3,ax4

pyST.STCreate.seed(130)
np.random.seed(100)

N_test=range(32)

Nv = 16
Nl = 2
Nh = 20
Np = 2
Ng1 = 0
Ng2 = Np
Ng3 = Np

#Convenience variables
sV = 0
sL = sV + Nv
sH = 0 
sP = sH + Nh
sg1 = sP + Np
sg2 = sg1 + Ng1
sg3 = sg2 + Ng2

initv = 180
inith = 4
input_rate_mult = 25
n_mult = 10
train_duration = 1500
test_duration = 3000

N_NEURONS = Nh + Ng1 + Ng2 + Np + Ng3
N_INPUTS = Nv + Nl
N_STATES = 4
N_UNITS = N_NEURONS + N_INPUTS

spk_rec_mon = range(N_NEURONS)

prefix = '../data/'
exp_name = 'erbp_bs'
exp_name_test = 'erbp_bs_test'


print("################## Loading Data ###########################################")
data_train, targets_train, _= bs_loader_npy(dset='train', prefix = 'data/')
data_classify, targets_classify,  _ = bs_loader_npy(dset='train', prefix = 'data/')
idx = range(32)*n_mult
np.random.shuffle(idx)
data_train = data_train[idx,:]

stim_train = np.zeros([data_train.shape[0], N_INPUTS])
stim_train[:,:data_train.shape[1]] = data_train

sim_ticks_train = len(stim_train)*train_duration

stim_test = np.zeros([data_classify.shape[0], N_INPUTS])
stim_test[:,:data_classify.shape[1]] = data_classify

sim_ticks_test = len(stim_test[N_test])*test_duration

###################### Stimulus Creation ##################################
SL_train = SimSpikingStimulus(input_rate_mult*stim_train      +1e-4, train_duration, t_sim = sim_ticks_train)
ext_evts_data_train = exportAER(SL_train)

SL_test = SimSpikingStimulus(input_rate_mult*stim_test[N_test]+1e-4, test_duration, t_sim = sim_ticks_test)
ext_evts_data_test = exportAER(SL_test)
###########################################################################

print("Generating parameters files!")
cfg_train = ConfigurationNSAT(
                           sim_ticks = sim_ticks_train,
                           N_NEURONS=N_NEURONS, 
                           N_INPUTS=N_INPUTS,
                           N_STATES=N_STATES,
                           spk_rec_mon=spk_rec_mon,
                           directory=prefix + exp_name)
                   


d=cfg_train.data
# Parameters groups mapping function
d.nmap = np.zeros(N_NEURONS, dtype='int')
d.lrnmap = np.zeros((N_NEURONS, N_STATES), dtype='int')
###########################################################################

print("################### NSAT Parameter configuration ##########################")



np.random.seed(100)
Wvh = np.random.uniform(low=-initv, high=initv,size=[Nv, Nh]).astype('int')
Whp = np.random.uniform(low=-inith, high=inith,size=[Nh, Np]).astype('int')

Wggampl = 40
Wgg1 = np.random.uniform(-Wggampl,Wggampl,size=[Ng2, Nh]).astype('int')
de = np.dot(Wgg1.T, np.ones(Ng2, dtype='int'))
Wgg1 -= np.row_stack([de/2,de/2])

Wpg1 = -64*np.eye(Np, dtype = 'int')  
Wgp1 = 35*np.eye(Ng2, dtype = 'int')    
Wlg1 = 64*np.eye(Nl, dtype = 'int')   

Wgg2 = -Wgg1

Wpg2 = 64*np.eye(Np, dtype = 'int')  
Wgp2 = -35*np.eye(Ng3, dtype = 'int')    
Wlg2 = -64*np.eye(Nl, dtype = 'int')  

d.Wgain[0][1] = 4
d.Wgain[0][2] = 4
d.Wgain[1][2] = 4
d.Wgain[2][1] = 4
d.Wgain[2][2] = 4
d.Wgain[1][0] = 4

recW = np.zeros([N_NEURONS, N_NEURONS, N_STATES], 'int')
recCW = np.zeros([N_NEURONS, N_NEURONS, N_STATES], 'bool')

recW [sg2:sg2+Ng2, sH:sH+Nh,2] = Wgg1
recCW[sg2:sg2+Ng2, sH:sH+Nh,2] = Wgg1!=0

recW [sg3:sg3+Ng3, sH:sH+Nh,2] = Wgg2
recCW[sg3:sg3+Ng3, sH:sH+Nh,2] = Wgg2!=0

recW [sH:sH+Nh, sP:sP+Np, 1] = Whp
recCW[sH:sH+Nh, sP:sP+Np, 1] = True 

recW [sP:sP+Np, sg2:sg2+Ng2, 0]      = Wpg1
recCW[sP:sP+Np, sg2:sg2+Ng2, 0]      = np.eye(Ng2, dtype='bool')

recW [sg2:sg2+Ng2, sP:sP+Np, 2]      = Wgp1
recCW[sg2:sg2+Ng2, sP:sP+Np, 2]      = np.eye(Np, dtype='bool')

recW [sP:sP+Np, sg3:sg3+Ng3, 0]      = Wpg2
recCW[sP:sP+Np, sg3:sg3+Ng3, 0]      = np.eye(Ng2, dtype='bool')

recW [sg3:sg3+Ng3, sP:sP+Np, 2]      = Wgp2
recCW[sg3:sg3+Ng3, sP:sP+Np, 2]      = np.eye(Np, dtype='bool')


extW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extCW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extW [sV:sV+Nv, sH:sH+Nh, 1] = Wvh
extCW[sV:sV+Nv, sH:sH+Nh, 1] = True
extW [sL:sL+Nl, sg2:sg2+Ng2, 0] = Wlg1
extCW[sL:sL+Nl, sg2:sg2+Ng2, 0] = np.eye(Ng2,dtype='bool')
extW [sL:sL+Nl, sg3:sg3+Ng3, 0] = Wlg2
extCW[sL:sL+Nl, sg3:sg3+Ng3, 0] = np.eye(Ng3,dtype='bool')

W = np.zeros([N_NEURONS+N_INPUTS, N_NEURONS+N_INPUTS, N_STATES], 'int')
W[:N_INPUTS,N_INPUTS:] = extW
W[N_INPUTS:,N_INPUTS:] = recW

CW = np.zeros([N_NEURONS+N_INPUTS, N_NEURONS+N_INPUTS, N_STATES], 'int')
CW[:N_INPUTS,N_INPUTS:] = extCW
CW[N_INPUTS:,N_INPUTS:] = recCW

W = np.array(W)

d.nmap[sP:sP+Np] = 2
d.nmap[sH:sH+Np] = 0
d.nmap[sg1:] = 1
d.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
d.lrnmap[d.nmap[:sP+Np],1] = 1
 
d.t_ref[0] = 40;
d.t_ref[2] = 40;
d.t_ref[1] =  0;
d.tstdp[0] = 1000;
d.tstdp[1] = 1000;

#3
d.A[0]=[[ -3,  OFF,  OFF, OFF ], 
      [  4,   -7,  OFF, OFF ],
      [OFF,  OFF,   -6, OFF ],
      [OFF,  OFF,  OFF, OFF  ]];

d.A[2] = d.A[0]

d.A[1] =   [[OFF, OFF, OFF, OFF], 
          [OFF, OFF, OFF, OFF],
          [OFF, OFF, OFF, OFF],
          [OFF, OFF, OFF, OFF]];
d.sA[0] =     [[-1, 1, 1, 1],
             [ 1,-1, 1, 1],
             [ 1, 1,-1, 1],
             [ 1, 1, 1,-1]];

d.sA[2] = d.sA[0]

d.b[0] = [-5200,  0,   0, 0];
d.b[2] = [-5200,  0,   0, 0];
d.b[1] = [ 0, 0,   0, 0];

d.Xinit = np.array([[0,0,0,0] for _ in range(N_NEURONS)], 'int')
d.Xreset[0] = [MAX-1, MAX, MAX, MAX];
d.Xreset[2] = [MAX-1, MAX, MAX, MAX];
d.Xreset[1] = [0,     MAX, MAX, MAX];
d.XresetOn[0] = [True , False, False, False];
d.XresetOn[1] = [False, False, False, False];
d.XresetOn[2] = [True , False, False, False];
d.Xth[1] = 1000;

d.hiac=[[-9, 4, 0] for _ in range(NLRN_GROUPS)]

d.plasticity_en = True

d.plastic[1] = True
d.plastic[0] = False
d.stdp_en[1] = False

d.XspikeIncrVal[1] = [-1024, 0, 0, 0];
d.sigma[0] = [0, 0, 0, 0];
d.sigma[2] = [0, 0, 0, 0];
d.modstate[:] = 2
###########################################################################

cfg_train.set_ext_events(ext_evts_data_train)

print("Generating parameters files for training")
cfg_train.set_groups(W=W, CW=CW)
fname = cfg_train.write_all_cnsat_params(prefix + exp_name, write_events = True)
cfg_train.write_all_hex_params()
#fname.ext_events = './data/fa_mnist10kc_eta-10_sig0_inputfact25_inith4_wgg40_ext_events.dat'

print("Generating parameters files!")
cfg_test = ConfigurationNSAT(sim_ticks = sim_ticks_test,
                        N_NEURONS=N_NEURONS, 
                        N_INPUTS=N_INPUTS,
                        N_STATES=N_STATES,
                        spk_rec_mon=spk_rec_mon,
                        directory=prefix + exp_name_test)


cfg_test.data = copy.copy(cfg_train.data)
cfg_test.data.sigma[:] = 0
cfg_test.data.plasticity_en = False
cfg_test.set_ext_events(ext_evts_data_test)
cfg_test.set_groups(W=W, CW=CW)

fname_test = cfg_test.write_all_cnsat_params(prefix + exp_name_test)
cfg_test.write_all_hex_params()
fname_test.syn_weights = fname.syn_weights

# C NSAT arguments
argstrain = ["exec", "-c", "-mfw", "-mspk", "-lg"]   # enable weights monitors

# C NSAT arguments
argstest = ["exec", "-c", "-mspk", "-msf"]    # enable spike monitors

print("Running C NSAT!")

nepochs = 2
pip_train = []
pip_test = []

for i in range(nepochs):
    #fWb = cfg_train.read_c_nsat_weights(fname.synw_final+'0.dat')
    print("Train")
    run_c_nsat(fname, argstrain)

    fW = cfg_train.read_c_nsat_weights(fname.synw_final+'0.dat')
    print(fW.max())
    wvh = fW[sV:sV+Nv, Nv+Nl+sH:Nv+Nl+sH+Nh, 1]
    whp = fW[Nv+Nl+sH:sH+Nh+Nv+Nl, Nv+Nl+sP:sP+Np+Nv+Nl, 1]

    utils.copy_final_weights(fname)

    print("Test")
    run_c_nsat(fname_test, argstest)

    #Read events, load spike list and compute recognition accuracy
    fname_events_train = fname.events+'0.dat'
    train_spikelist = importAER(read_from_file(fname_events_train), sim_ticks = sim_ticks_train, id_list = range(N_NEURONS))
    frate_train = train_spikelist.firing_rate(train_duration)
    pip_train.append( sum(np.argmax(frate_train[range(sP,sP+Np)].T,axis=1) == targets_train[idx]))

    #Read events, load spike list and compute recognition accuracy (Somnath use these four lines and replace fname_events_test as necessary)
    fname_events_test = fname_test.events+'0.dat'
    test_spikelist = importAER(read_from_file(fname_events_test), sim_ticks = sim_ticks_test, id_list = range(N_NEURONS))
    frate_test = test_spikelist.firing_rate(test_duration)
    pip_test .append( sum(np.argmax(frate_test[range(sP,sP+Np)].T,axis=1) == targets_classify[N_test]) )

    print "Recognition Accuracy train: ", pip_train[-1], " test: ",pip_test[-1]

test_spikelist = importAER(read_from_file(fname_test.events+'0.dat'), sim_ticks = sim_ticks_test, id_list = range(N_NEURONS))
# ax1,ax2,ax3,ax4 = pretty_state_plot(N_INPUTS,states, SL_test, test_spikelist, pre=10, post=21)

states = cfg_test.read_c_nsat_states(fname_test.states+'0.dat',

                                     time_explicit=False)
cfg_test.data.W = fW.copy()
cfg_test.write_all_hex_params()

#plt.figure(figsize=(13, 13))
#for i in range(1, 24):
#    plt.subplot(4, 6, i)
#    plt.plot(states[:, i-1, 2], 'k', lw=2)
#
test_spikelist.raster_plot()
plt.show()
