#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_eCD_NSATv2_mnist.py
# Purpose: eCD learning of bars abd stripes with NSAT
#
# Author: Emre Neftci, Sadique Sheik
#
# Creation Date : 09-08-2015
# Last Modified : Thu 22 Jun 2017 12:35:56 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#
# Update Thu Feb 3

import numpy as np
from pylab import *
import time, sys, copy, os

import copy
from pyNCS import pyST
import pyNSATlib as nsat
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW

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
Ng4 = 1 #Label on neuron

N_CORES = 1
N_NEURONS = Nh + Ng1 + Ng2 + Ng3 + Ng4 + Np

n_mult = 1
N_train = 50
N_test = 10
N_INPUTS = Nv + Nl + Ng4
N_STATES = 2
N_GROUPS = 8
t_sample_test = 3000
t_sample_train = 1500

N_UNITS = N_NEURONS + N_INPUTS

print("################### ERBP Parameter configuration ##########################")
sV = 0; sL = sV + Nv; sH = 0; sP = sH + Nh; sg1 = sP + Np; sg2 = sg1 + Ng1; sg3 = sg2 + Ng2
sLo = sL + Nl
sg4 = sg3 + Ng3

initv = 4; inith = 2

spk_rec_mon = np.arange(N_NEURONS, dtype='int')


print("#####  Loading Data ")
exp_name          = 'eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)
exp_name_test     = 'test_eta'+str(eta)+'_sig'+str(sig)+'_inputfact'+str(inp_fact)+'_inith'+str(inith)+'_wgg'+str(wgg)

np.random.seed(100)

###########################################################################
print("###################### Train Stimulus Creation ##################################")
sim_ticks = 10000000


print("################## Setting Weight CONF #########################################")
Wvh = np.random.uniform(low=-initv, high=initv, size=[Nv, Nh]).astype('int')
Whp = np.random.uniform(low=-inith, high=inith, size=[Nh, Np]).astype('int')

#Create matrix whose rows sum to 0
Wgg1 = np.zeros([Ng2, Nh], dtype='int')
for j in range(Nh):
    for i in range(2000):
        a = np.zeros([Ng2],dtype='int')
        a[:2] = -1
        a[2:4] = 1
        np.random.shuffle(a)
        Wgg1[:,j]+=a

Wpg1 = -80*np.eye(Np, dtype = 'int')  
Wgp1 = 37*np.eye(Ng2, dtype = 'int')    
Wlg1 = 8*np.eye(Nl, dtype = 'int') 

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

recW [sg4:sg4+Ng4, sg3:sg3+Ng3, 0]      = 6
recCW[sg4:sg4+Ng4, sg3:sg3+Ng3, 0]      = True

recW [sg4:sg4+Ng4, sg2:sg2+Ng2, 0]      = 6
recCW[sg4:sg4+Ng4, sg2:sg2+Ng2, 0]      = True


extW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extCW = np.zeros([N_INPUTS, N_NEURONS, N_STATES])
extW [sV:sV+Nv, sH:sH+Nh, 0] = Wvh
extCW[sV:sV+Nv, sH:sH+Nh, 0] = True
extW [sL:sL+Nl, sg2:sg2+Ng2, 0] = Wlg1
extCW[sL:sL+Nl, sg2:sg2+Ng2, 0] = np.eye(Ng2,dtype='bool')
extW [sL:sL+Nl, sg3:sg3+Ng3, 0] = Wlg2
extCW[sL:sL+Nl, sg3:sg3+Ng3, 0] = np.eye(Ng3,dtype='bool')
extW[sLo:sLo+1, sg4:sg4+Ng4, 0] = 80
extCW[sLo:sLo+1, sg4:sg4+Ng4, 0] = True

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
 
core0_cfg.t_ref[HIDNRN] = 40;
core0_cfg.t_ref[OUTNRN] = 39;
core0_cfg.t_ref[ERRNRN] =  0;
core0_cfg.tstdp[0] = 1000;
core0_cfg.tstdp[1] = 1000;

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
 
core0_cfg.b[HIDNRN] = [-1, 0];
core0_cfg.b[OUTNRN] = [-1, 0];
core0_cfg.b[ERRNRN] = [-32, 0];
core0_cfg.Xinit = np.array([[0,0] for _ in range(N_NEURONS)], 'int')
core0_cfg.Xreset[HIDNRN] = [0, 0];
core0_cfg.Xreset[OUTNRN] = [0, 0];
core0_cfg.Xreset[ERRNRN] = [0, 0];
core0_cfg.XresetOn[HIDNRN] = [False, False];
core0_cfg.XresetOn[ERRNRN] = [False, False];
core0_cfg.XresetOn[OUTNRN] = [False, False];
core0_cfg.Xth[OUTNRN] = 0;
core0_cfg.Xth[HIDNRN] = 0;
core0_cfg.Xth[ERRNRN] = 1024;
core0_cfg.Xthlo[ERRNRN] = -1024; 
core0_cfg.hiac[:]=OFF
core0_cfg.hiac[core0_cfg.lrnmap[core0_cfg.nmap[:sP+Np],0]]=[eta, 0, 0]
 
core0_cfg.plasticity_en = True
 
core0_cfg.plastic[1] = True
core0_cfg.plastic[0] = False
core0_cfg.stdp_en[1] = False
core0_cfg.is_rr_on[1] = True
core0_cfg.rr_num_bits[1] = 5
 
core0_cfg.XspikeIncrVal[ERRNRN] = [-1024, 0];
core0_cfg.sigma[HIDNRN] = [0, 0];
core0_cfg.sigma[OUTNRN] = [0, 0];
core0_cfg.modstate[:] = 1

core0_cfg.Wgain = Wgain

wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])
core0_cfg.wgt_table = wgt_table
core0_cfg.ptr_table = ptr_table
cfg_train.set_ext_events('/tmp/dvs')

print("################## Writing Parameters Files ##################")
c_nsat_writer_train = nsat.C_NSATWriter(cfg_train, path='/tmp/erbp_mnist_train1', prefix='')
c_nsat_writer_train.write()

fname_train = c_nsat_writer_train.fname
fname_train.events = '/tmp/nsat'

if __name__ == "__main__":
    print("############# Running simulation #####################")
    print('run')
    nsat.run_c_nsat(fname_train)

