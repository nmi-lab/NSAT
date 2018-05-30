#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_mnist.py
# Purpose: eCD learning of bars abd stripes with NSAT
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Fri 29 Sep 2017 04:04:30 PM PDT
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

inp_fact,eta,sig,wgg = 25,-6,0,40

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
N_train = 5000
N_test = 1000
N_INPUTS = Nv + Nl
N_STATES = 4
N_GROUPS = 8
t_sample_test = 3000
t_sample_train = 1500
nepochs = 50
test_every = 10

print("##### ERBP Parameter configuration")
sV = 0; sL = sV + Nv; sH = 0; sP = sH + Nh; sg1 = sP + Np; sg2 = sg1 + Ng1; sg3 = sg2 + Ng2

initv = 16; inith = 4

spk_rec_mon = np.arange(N_NEURONS, dtype='int')


print("#####  Loading Data ")
exp_name          = 'eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)
exp_name_test     = 'test_eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)

data_train, targets_train = mnist.load_mnist(
        '/shares/data/mnist/train-images-idx3-ubyte',
        '/shares/data/mnist/train-labels-idx1-ubyte',
        50000, with_labels = True)
data_classify, targets_classify = mnist.load_mnist(
        '/shares/data/mnist/t10k-images-idx3-ubyte',
        '/shares/data/mnist/t10k-labels-idx1-ubyte',
        10000, with_labels = False)

np.random.seed(100)

###########################################################################
print("###################### Train Stimulus Creation ##################################")
sim_ticks = N_train*t_sample_train
idx = range(len(data_train))
np.random.shuffle(idx)
idx = idx[:N_train]
data_train = np.concatenate([data_train[idx,:] for _ in range(n_mult)])

stim = np.zeros([N_train*n_mult, N_INPUTS])
stim[:,:data_train.shape[1]] = data_train
SL_train = SimSpikingStimulus(inp_fact*stim, t_sample_train, t_sim = sim_ticks)
ext_evts_data = nsat.exportAER(SL_train)
#
#print("###################### Test Stimulus Creation ##################################")
stim_test = np.zeros([N_test, N_INPUTS])
stim_test[:N_test,:data_classify.shape[1]] = data_classify[:N_test,:]
sim_ticks_test = len(stim_test)*t_sample_test
SL_test = SimSpikingStimulus(inp_fact*stim_test, t_sample_test, t_sim = sim_ticks_test)
ext_evts_data_test = nsat.exportAER(SL_test)
#print("################################################################################")

print("################## Setting Weight CONF #########################################")
Wvh = np.random.uniform(low=-initv, high=initv,size=[Nv, Nh]).astype('int')
Whp = np.random.uniform(low=-inith, high=inith,size=[Nh, Np]).astype('int')

#Create matrix whose rows sum to 0
Wgg1 = np.zeros([Ng2, Nh], dtype='int')
for j in range(Nh):
    for i in range(2000):
        a = np.zeros([Ng2],dtype='int')
        a[:2] = -1
        a[2:4] = 1
        np.random.shuffle(a)
        Wgg1[:,j]+=a

Wpg1 = -64*np.eye(Np, dtype = 'int')  
Wgp1 = 37*np.eye(Ng2, dtype = 'int')    
Wlg1 = 64*np.eye(Nl, dtype = 'int') 

Wpg2 = -Wpg1
Wgp2 = -Wgp1
Wlg2 = -Wlg1

Wgg2 = -Wgg1  

Wgain = np.zeros([N_GROUPS, N_STATES], dtype='int')
Wgain[0][1] = 3
Wgain[0][2] = 4
Wgain[1][2] = 4
Wgain[2][1] = 3
Wgain[2][2] = 4
Wgain[1][0] = 4

initv= initv<<Wgain[0][1]
inith= initv<<Wgain[2][1]

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

cfg_train = nsat.ConfigurationNSAT(
                   sim_ticks = sim_ticks,
                   N_CORES = N_CORES,
                   N_NEURONS=[N_NEURONS], 
                   N_INPUTS=[N_INPUTS],
                   N_STATES=[N_STATES],
                   bm_rng=True,
                   monitor_spikes = True,
                   gated_learning = [True],
                   plasticity_en = [True])

# Parameters groups mapping function
core0_cfg = cfg_train.core_cfgs[0]
core0_cfg.nmap = np.zeros(N_NEURONS, dtype='int')
core0_cfg.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
core0_cfg.nmap[sP:sP+Np] = 2
core0_cfg.nmap[sH:sH+Np] = 0
core0_cfg.nmap[sg1:] = 1
core0_cfg.lrnmap = np.zeros((N_GROUPS, N_STATES), dtype='int')
core0_cfg.lrnmap[core0_cfg.nmap[:sP+Np],1] = 1
 
core0_cfg.t_ref[0] = 39;
core0_cfg.t_ref[2] = 39;
core0_cfg.t_ref[1] =  0;
core0_cfg.tstdp[0] = 1000;
core0_cfg.tstdp[1] = 1000;

core0_cfg.prob_syn[0,1]=9
core0_cfg.prob_syn[2,1]=9
 
core0_cfg.A[0]=[[ -3,  OFF,  OFF, OFF ], 
        [  4,   -7,  OFF, OFF ],
        [OFF,  OFF,   -6, OFF ],
        [OFF,  OFF,  OFF, OFF  ]];
 
core0_cfg.A[2] = core0_cfg.A[0]
 
core0_cfg.A[1] =   [[OFF, OFF, OFF, OFF], 
            [OFF, OFF, OFF, OFF],
            [OFF, OFF, OFF, OFF],
            [OFF, OFF, OFF, OFF]];
core0_cfg.sA[0] = [[-1, 1, 1, 1],
           [ 1,-1, 1, 1],
           [ 1, 1,-1, 1],
           [ 1, 1, 1,-1]];
 
core0_cfg.sA[2] = core0_cfg.sA[0]
 
core0_cfg.b[0] = [1000,  0,   0, 0];
core0_cfg.b[2] = [1000,  0,   0, 0];
core0_cfg.b[1] = [0, 0,   0, 0];
core0_cfg.Xinit = np.array([[0,0,0,0] for _ in range(N_NEURONS)], 'int')
core0_cfg.Xreset[0] = [MAX-1, MAX, MAX, MAX];
core0_cfg.Xreset[2] = [MAX-1, MAX, MAX, MAX];
core0_cfg.Xreset[1] = [0,     MAX, MAX, MAX];
core0_cfg.XresetOn[0] = [True , False, False, False];
core0_cfg.XresetOn[1] = [False, False, False, False];
core0_cfg.XresetOn[2] = [True , False, False, False];
core0_cfg.Xth[1] = 1025;
core0_cfg.Xthlo[1] = 0; 
core0_cfg.hiac[:]=OFF
core0_cfg.hiac[core0_cfg.lrnmap[core0_cfg.nmap[:sP+Np],1]]=[eta, 0, 0]
 
core0_cfg.plasticity_en = True
 
core0_cfg.plastic[1] = True
core0_cfg.plastic[0] = False
core0_cfg.stdp_en[1] = False
core0_cfg.is_rr_on[1] = True
core0_cfg.rr_num_bits[1] = 10
 
 
core0_cfg.XspikeIncrVal[1] = [-1025, 0, 0, 0];
core0_cfg.sigma[0] = [0, 0, sig, 0];
core0_cfg.sigma[2] = [0, 0, sig, 0];
core0_cfg.modstate[:] = 2

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
c_nsat_writer_train = nsat.C_NSATWriterMultithread(cfg_train, path='/tmp/erbp_mnist_train1/', prefix='')
c_nsat_writer_train.write()

c_nsat_writer_test = nsat.C_NSATWriterMultithread(cfg_test, path='/tmp/erbp_mnist_test1/', prefix='')
c_nsat_writer_test.write()

fname_train = c_nsat_writer_train.fname
fname_test = c_nsat_writer_test.fname

c_nsat_reader_train = nsat.C_NSATReader(cfg_train, fname_train)
c_nsat_reader_test = nsat.C_NSATReader(cfg_test, fname_test)

print("############# Running simulation #####################")
pip = []
for i in range(nepochs):
    nsat.run_c_nsat(fname_train)

    copyfile(fname_train.shared_mem+'_core_0.dat',
             fname_test.syn_wgt_table+'_core_0.dat',)
    # cfg_test.core_cfgs[0].W = c_nsat_reader_train.read_c_nsat_weights()[0]
    # cfg_train.core_cfgs[0].W = cfg_test.core_cfgs[0].W.copy()
    # c_nsat_writer_test.write_L0connectivity()
    # c_nsat_writer_train.write_L0connectivity()
    if test_every>0:
        if i % test_every == test_every-1:
            nsat.run_c_nsat(fname_test)
            test_spikelist = nsat.importAER(c_nsat_reader_test.read_c_nsat_raw_events()[0],
                                            sim_ticks=sim_ticks_test,
                                            id_list=np.arange(sP, sP+Np))

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
    et.save(pip, 'pip.pkl')
    et.annotate('res',text=str(pip))
except ImportError:
    print('saving disabled due to missing experiment tools')
