#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_mnist.py
# Purpose: eCD learning of bars abd stripes with NSAT
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Mon 14 Aug 2017 02:34:23 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#
# Update Thu Feb 3

import numpy as np
from pylab import *
import time, sys, copy, os
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
sys.path.append(DATA_DIR)

import mnist
import copy
from pyNCSre import pyST
from shutil import copyfile
import pyNSATlib as nsat
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def SimSpikingStimulus(stim, time = 1000, t_sim = None, with_labels = True):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean
    data/poisson, scaled by *poisson*.
    '''
    from pyNCSre import pyST
    n = np.shape(stim)[1]
    nc = 10
    stim[stim<=0] = 1e-5
    SL = pyST.SpikeList(id_list = range(n))
    SLd = pyST.SpikeList(id_list = range(n-nc))
    SLc = pyST.SpikeList(id_list = range(n-nc,n))
    for i in range(n-nc):
        SLd[i] = pyST.STCreate.inh_poisson_generator(stim[:,i],
                                                    range(0,len(stim)*time,time),
                                                    t_stop=t_sim, refractory = 4.)
    if with_labels:
        for t in range(0,len(stim)):
            SLt= pyST.SpikeList(id_list = range(n-nc,n))
            for i in range(n-nc,n):
                if stim[t,i]>1e-2:
                    SLt[i] = pyST.STCreate.regular_generator(stim[t,i],
                                                        jitter=True,
                                                        t_start=t*time,
                                                        t_stop=(t+1)*time)            
            if len(SLt.raw_data())>0: SLc = pyST.merge_spikelists(SLc, SLt)

    if len(SLc.raw_data())>0: 
        SL = pyST.merge_spikelists(SLd,SLc)
    else:
        SL = SLd
    return SL

def plot_features(w):
    from ml_funcs import stim_show
    stim_show(w.T, (28,28), (10,10))

inp_fact,eta,sig,wgg = 25,-7,0,40

#Assign True to generate spike trains (very long for 50000 epochs!)
gen_sl = True 

#Globals
OFF = -16
MAX = 2**15-1
Nv = 784
Nl = 10
Nh = 100
Np = 10
Ng1 = 0
Ng2 = Np
Ng3 = Np

N_CORES = 1
N_NEURONS = Nh + Ng1 + Ng2 + Ng3 + Np

n_mult = 1
N_train = 500
N_test = 100
N_INPUTS = Nv + Nl
N_STATES = 2
N_GROUPS = 8

t_sample_test = 3000
t_sample_train = 1500
nepochs = 30
test_every = 1

print("##### ERBP Parameter configuration")
sV = 0; sL = sV + Nv; sH = 0; sP = sH + Nh; sg1 = sP + Np; sg2 = sg1 + Ng1; sg3 = sg2 + Ng2

initv = 16; inith = 4

spk_rec_mon = np.arange(N_NEURONS, dtype='int')


print("#####  Loading Data ")
exp_name          = 'eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)
exp_name_test     = 'test_eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)

data_train, targets_train = mnist.load_mnist(
        'data/mnist/train-images-idx3-ubyte',
        'data/mnist/train-labels-idx1-ubyte',
        50000, with_labels = True)
data_classify, targets_classify = mnist.load_mnist(
        'data/mnist/t10k-images-idx3-ubyte',
        'data/mnist/t10k-labels-idx1-ubyte',
        10000, with_labels = False)

np.random.seed(100)

###########################################################################
print("###################### Train Stimulus Creation ##################################")
sim_ticks = N_train*t_sample_train
sim_ticks_test = N_test*t_sample_test
idx = range(len(data_train))
np.random.shuffle(idx)
idx = idx[:N_train]
data_train = np.concatenate([data_train[idx,:] for _ in range(n_mult)])

if gen_sl:
    stim = np.zeros([N_train*n_mult, N_INPUTS])
    stim[:,:data_train.shape[1]] = data_train
    SL_train = SimSpikingStimulus(inp_fact*stim, t_sample_train, t_sim = sim_ticks)
    ext_evts_data = nsat.exportAER(SL_train)

    #print("###################### Test Stimulus Creation ##################################")
    stim_test = np.zeros([N_test, N_INPUTS])
    stim_test[:N_test,:data_classify.shape[1]] = data_classify[:N_test,:]
    SL_test = SimSpikingStimulus(inp_fact*stim_test, t_sample_test, t_sim = sim_ticks_test)
    ext_evts_data_test = nsat.exportAER(SL_test)
    #print("################################################################################")
else:
    ext_evts_data_test = True
    ext_evts_data = True

print("################## Setting Weight CONF #########################################")
Wvh = np.random.uniform(low=-initv, high=initv, size=[Nv, Nh]).astype('int')
Whp = np.random.uniform(low=-inith, high=inith, size=[Nh, Np]).astype('int')

#Create matrix whose rows sum to 0
Wgg1 = np.zeros([Ng2, Nh], dtype='int')
for j in range(Nh):
    for i in range(3000):
        a = np.zeros([Ng2],dtype='int')
        a[:2] = -1
        a[2:4] = 1
        np.random.shuffle(a)
        Wgg1[:,j]+=a

Wpg1 = -96*np.eye(Np, dtype = 'int')  
Wgp1 = 37*np.eye(Ng2, dtype = 'int')    
Wlg1 = 96*np.eye(Nl, dtype = 'int') 

Wpg2 = -Wpg1
Wgp2 = -Wgp1
Wlg2 = -Wlg1

Wgg2 = -Wgg1  

HIDNRN = 0
ERRNRN = 1
OUTNRN = 2

Wgain = np.zeros([N_GROUPS, N_STATES], dtype='int')
Wgain[HIDNRN][0] = 3
Wgain[HIDNRN][1] = 4
Wgain[OUTNRN][0] = 3
Wgain[OUTNRN][1] = 4
Wgain[ERRNRN][1] = 4
Wgain[ERRNRN][0] = 4



recW = np.zeros([N_NEURONS, N_NEURONS, N_STATES], 'int')
recCW = np.zeros([N_NEURONS, N_NEURONS, N_STATES], 'bool')

recW [sg2:sg2+Ng2, sH:sH+Nh,1] = Wgg1
recCW[sg2:sg2+Ng2, sH:sH+Nh,1] = Wgg1!=0

recW [sg3:sg3+Ng3, sH:sH+Nh,1] = Wgg2
recCW[sg3:sg3+Ng3, sH:sH+Nh,1] = Wgg2!=0

recW [sH:sH+Nh, sP:sP+Np, 0] = Whp
recCW[sH:sH+Nh, sP:sP+Np, 0] = True 

recW [sP:sP+Np, sg2:sg2+Ng2, 0]      = Wpg1
recCW[sP:sP+Np, sg2:sg2+Ng2, 0]      = np.eye(Ng2, dtype='bool')

recW [sg2:sg2+Ng2, sP:sP+Np, 1]      = Wgp1
recCW[sg2:sg2+Ng2, sP:sP+Np, 1]      = np.eye(Np, dtype='bool')

recW [sP:sP+Np, sg3:sg3+Ng3, 0]      = Wpg2
recCW[sP:sP+Np, sg3:sg3+Ng3, 0]      = np.eye(Ng2, dtype='bool')

recW [sg3:sg3+Ng3, sP:sP+Np, 1]      = Wgp2
recCW[sg3:sg3+Ng3, sP:sP+Np, 1]      = np.eye(Np, dtype='bool')


extW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extCW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extW [sV:sV+Nv, sH:sH+Nh, 0] = Wvh
extCW[sV:sV+Nv, sH:sH+Nh, 0] = True
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

cfg_train = nsat.ConfigurationNSAT(
                   sim_ticks = sim_ticks,
                   N_CORES = N_CORES,
                   N_NEURONS=[N_NEURONS], 
                   N_INPUTS=[N_INPUTS],
                   N_STATES=[N_STATES],
                   bm_rng=True,
                   w_check=True,
                   monitor_spikes = False,
                   monitor_states = False,
                   gated_learning = [True],
                   plasticity_en = [True])

# Parameters groups mapping function
core0_cfg = cfg_train.core_cfgs[0]
core0_cfg.nmap = np.zeros(N_NEURONS, dtype='int')
core0_cfg.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
core0_cfg.nmap[sP:sP+Np] = OUTNRN
core0_cfg.nmap[sH:sH+Np] = HIDNRN
core0_cfg.nmap[sg1:] = ERRNRN
core0_cfg.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
core0_cfg.lrnmap[core0_cfg.nmap[:sP+Np],0] = 1
 
core0_cfg.t_ref[HIDNRN] = 39;
core0_cfg.t_ref[OUTNRN] = 39;
core0_cfg.t_ref[ERRNRN] =  0;
core0_cfg.tstdp[0] = 1000;
core0_cfg.tstdp[1] = 1000;
core0_cfg.gate_upper[:] = 2560
core0_cfg.gate_lower[:] = -core0_cfg.gate_upper
core0_cfg.learn_period[:] = int(t_sample_train)
core0_cfg.learn_burnin[:] = int(t_sample_train*.25) #Time during which no update is undertaken

core0_cfg.prob_syn[HIDNRN,0]=9
core0_cfg.prob_syn[OUTNRN,0]=9

core0_cfg.A[HIDNRN]= [[ -7,  OFF],
                      [OFF,   -6]];
 
core0_cfg.A[OUTNRN] = core0_cfg.A[HIDNRN]
 
core0_cfg.A[ERRNRN] =   [[OFF, OFF], 
                         [OFF, OFF]];
core0_cfg.sA[HIDNRN] = [[-1, 1],
                        [ 1,-1]];


 
core0_cfg.sA[ERRNRN] = core0_cfg.sA[HIDNRN]
 
core0_cfg.b[HIDNRN] = [0, 0];
core0_cfg.b[OUTNRN] = [0, 0];
core0_cfg.b[ERRNRN] = [0, 0];
core0_cfg.Xinit = np.array([[0,0] for _ in range(N_NEURONS)], 'int')
core0_cfg.Xreset[HIDNRN] = [0, 0];
core0_cfg.Xreset[OUTNRN] = [0, 0];
core0_cfg.Xreset[ERRNRN] = [0, 0];
core0_cfg.XresetOn[HIDNRN] = [False, False];
core0_cfg.XresetOn[ERRNRN] = [False, False];
core0_cfg.XresetOn[OUTNRN] = [False, False];
core0_cfg.Xth[OUTNRN] = 0;
core0_cfg.Xth[HIDNRN] = 0;
core0_cfg.Xth[ERRNRN] = 1025;
core0_cfg.Xthlo[ERRNRN] = 0; 
core0_cfg.hiac[:]=OFF
core0_cfg.hiac[core0_cfg.lrnmap[core0_cfg.nmap[:sP+Np],0]]=[eta, 0, 0]
 
core0_cfg.plasticity_en = True
 
core0_cfg.plastic[1] = True
core0_cfg.plastic[0] = False
core0_cfg.stdp_en[1] = False
core0_cfg.is_rr_on[1] = True
core0_cfg.rr_num_bits[1] = 10
 
 
core0_cfg.XspikeIncrVal[ERRNRN] = [-1025, 0];
core0_cfg.sigma[HIDNRN] = [0, 0];
core0_cfg.sigma[OUTNRN] = [0, 0];
core0_cfg.modstate[:] = 1

core0_cfg.Wgain = Wgain

wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
core0_cfg.wgt_table = wgt_table
core0_cfg.ptr_table = ptr_table

cfg_train.set_ext_events(ext_evts_data)

spk_rec_mon_test = np.arange(sP,sP+Np, dtype='int')

print("############# Generating parameters files for testing #####################")
cfg_test = nsat.ConfigurationNSAT(
                   sim_ticks = sim_ticks_test,
                   N_CORES = N_CORES,
                   N_NEURONS=[N_NEURONS], 
                   N_INPUTS=[N_INPUTS],
                   N_STATES=[N_STATES],
                   bm_rng = True,
                   w_check=False,
                   plasticity_en = [False],
                   spk_rec_mon = [spk_rec_mon_test],
                   monitor_spikes = True)

cfg_test.core_cfgs[0] = copy.copy(cfg_train.core_cfgs[0])
d = cfg_test.core_cfgs[0]
d.sigma[:] = 0

# FIXME It doesn't read the weights after training
d.wgt_table = core0_cfg.wgt_table.copy()
d.ptr_table = core0_cfg.ptr_table.copy()

cfg_test.set_ext_events(ext_evts_data_test)

#fname_test = cfg_test.write_all_cnsat_params(prefix + hex_dir_name_test)
#fname_test.syn_weights = fname.syn_weights

print("################## Writing Parameters Files ##################")
c_nsat_writer_train = nsat.C_NSATWriter(cfg_train, path='/tmp/erbp_mnist_train1/', prefix='')
c_nsat_writer_train.fname.ext_events = 'mnist_train_ext_events.dat'
c_nsat_writer_train.write()

c_nsat_writer_test = nsat.C_NSATWriter(cfg_test, path='/tmp/erbp_mnist_test1/', prefix='')
c_nsat_writer_test.fname.ext_events = 'mnist_test_ext_events.dat'
c_nsat_writer_test.write()

fname_train = c_nsat_writer_train.fname
fname_test = c_nsat_writer_test.fname

c_nsat_reader_train = nsat.C_NSATReader(cfg_train, fname_train)
c_nsat_reader_test = nsat.C_NSATReader(cfg_test, fname_test)

print("############# Running simulation #####################")
pip = []
stats_nsat = []
#stats_ext = []
#n_ext = [len(s) for s in SL_train]
for i in range(nepochs):
    nsat.run_c_nsat(fname_train)

    copyfile(fname_train.shared_mem+'_core_0.dat',
             fname_test.syn_wgt_table+'_core_0.dat',)

    # c_nsat_writer_test.write_L0connectivity()
    # c_nsat_writer_train.write_L0connectivity()
    s = c_nsat_reader_train.read_stats()[0]
    stats_nsat.append(s)
    #stats_ext.append(n_ext)
    if test_every>0:
        if i%test_every == test_every-1:
            nsat.run_c_nsat(fname_test)
            test_spikelist = nsat.importAER(nsat.read_from_file(fname_test.events+'_core_0.dat'), sim_ticks = sim_ticks_test, id_list = np.arange(sP,sP+Np))

            pip .append([i, float(sum(np.argmax(test_spikelist.id_slice(range(sP,sP+Np)).firing_rate(t_sample_test).T,axis=1) == targets_classify[:N_test]))/N_test*100])

            print exp_name
            print pip
    copyfile(fname_train.shared_mem+'_core_0.dat',
         fname_train.syn_wgt_table+'_core_0.dat',)


try:
    import experimentTools as et
    d=et.mksavedir()
    et.save(cfg_test, 'cfg_test.pkl')
    et.save(cfg_train, 'cfg_train.pkl')
    et.save(stats_nsat, 'stats_nsat.pkl')
    et.save(pip, 'pip.pkl')
    et.annotate('res',text=str(pip))
    et.save(c_nsat_reader_train.read_c_nsat_weights()[0], 'W.pkl')
except ImportError:
    print('saving disabled due to missing experiment tools')
