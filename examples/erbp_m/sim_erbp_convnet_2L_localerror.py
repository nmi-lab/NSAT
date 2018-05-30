#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_erbp_convnet_2s.py
# Purpose: eRBP learning of MNIST NSAT Spiking Convnet
#
# Author: Emre Neftci
#
# Creation Date : 02-20-2018
# Last Modified : 02-20-2018
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#
# Update Thu Feb 3
from utils import *
import numpy as np
from pylab import *
from pyNCSre import pyST
import copy, copy,shutil
from load_mnist import *
import time
from pyNSATlib.laxesis import *

N_FEAT1 = 8
N_FEAT2 = 16
stride = 2
ksize = 5

exp_name          = '/tmp/mnist_convnet_2L_LE'
exp_name_test     = '/tmp/mnist_convnet_2L_LE_test/'

#Globals
inputsize = 28
Nchannel = 1
Nxy = inputsize*inputsize
Nv = Nxy*Nchannel
Nl = 10
Nconv1 = Nv/stride/stride*N_FEAT1/Nchannel
Nconv2 = Nconv1/stride/stride*N_FEAT2/N_FEAT1
Nh = 100
Np = 10
Ng2 = Np
Ng3 = Np

N_CORES = 1
t_sample_test = 3000
t_sample_train = 1500
nepochs = 5
N_train = 500
N_test = 100
test_every = 1
inp_fact = 25

sim_ticks = N_train*t_sample_train
sim_ticks_test = N_test*t_sample_test

print("##### ERBP Parameter configuration")

np.random.seed(100)
print("################## Constructing Network Connections #########################################")

wpg  = 96
wgp  = 37    

#Custom neuron type
erbp_ptype = erbp_ptype.copy()
erbp_ptype.rr_num_bits = 14
erbp_ptype.hiac = [-7, OFF, OFF]

erf_ntype = erf_ntype.copy()
erf_ntype.plasticity_type = [erbp_ptype,nonplastic_ptype]
erf_ntype.Wgain[0] = 2

erfle_ntype = erf_ntype.copy()
erfle_ntype.plasticity_type = [nonplastic_ptype,nonplastic_ptype]


setup       = NSATSetup(ncores = N_CORES)

pop_data    = setup.create_external_population(Nv, 0)
pop_lab     = setup.create_external_population(Nl, 0)
pop_conv1   = setup.create_population(n = Nh, core = 0, neuron_cfg = erf_ntype)
pop_conv2   = setup.create_population(n = Nh, core = 0, neuron_cfg = erf_ntype)
pop_hid     = setup.create_population(n = Nh, core = 0, neuron_cfg = erf_ntype)

pop_out1    = setup.create_population(n = Nl, core = 0, neuron_cfg = erfle_ntype)
pop_out2    = setup.create_population(n = Nl, core = 0, neuron_cfg = erfle_ntype)
pop_out3    = setup.create_population(n = Nl, core = 0, neuron_cfg = erfle_ntype)
pop_out4    = setup.create_population(n = Nl, core = 0, neuron_cfg = output_ntype)

pop_err1_pos = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)
pop_err1_neg = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)

pop_err2_pos = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)
pop_err2_neg = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)

pop_err3_pos = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)
pop_err3_neg = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)

pop_err4_pos = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)
pop_err4_neg = setup.create_population(n = Nl, core = 0, neuron_cfg = error_ntype)

Connection(setup, pop_data, pop_conv1, 0).connect_random_uniform(low=-16, high=16)
Connection(setup, pop_conv1, pop_conv2, 0).connect_random_uniform(low=-16, high=16)
Connection(setup, pop_conv2, pop_hid, 0).connect_random_uniform(low=-16, high=16)
Connection(setup, pop_hid, pop_out4, 0).connect_random_uniform(low=-16, high=16)

#eRBP related connections
#errors
Connection(setup, pop_lab, pop_err1_pos, 0).connect_one2one(wpg)
Connection(setup, pop_lab, pop_err1_neg, 0).connect_one2one(-wpg)
Connection(setup, pop_out1, pop_err1_pos, 0).connect_one2one(-wpg)
Connection(setup, pop_out1, pop_err1_neg, 0).connect_one2one(wpg)
cx_p=Connection(setup, pop_err1_pos, pop_conv1, 1)
cx_p.connect_shuffle(2000)
Connection(setup, pop_err1_neg, pop_conv1, 1).connect(cx_p.ptr_table, -cx_p.wgt_table)
Connection(setup, pop_conv1, pop_out1, 0).connect(cx_p.ptr_table.T, cx_p.wgt_table)

Connection(setup, pop_lab, pop_err2_pos, 0).connect_one2one(wpg)
Connection(setup, pop_lab, pop_err2_neg, 0).connect_one2one(-wpg)
Connection(setup, pop_out2, pop_err2_pos, 0).connect_one2one(-wpg)
Connection(setup, pop_out2, pop_err2_neg, 0).connect_one2one(wpg)
cx_p=Connection(setup, pop_err2_pos, pop_conv2, 1)
cx_p.connect_shuffle(2000)
Connection(setup, pop_err2_neg, pop_conv2, 1).connect(cx_p.ptr_table, -cx_p.wgt_table)
Connection(setup, pop_conv2, pop_out2, 0).connect(cx_p.ptr_table.T, cx_p.wgt_table)

Connection(setup, pop_lab, pop_err3_pos, 0).connect_one2one(wpg)
Connection(setup, pop_lab, pop_err3_neg, 0).connect_one2one(-wpg)
Connection(setup, pop_out3, pop_err3_pos, 0).connect_one2one(-wpg)
Connection(setup, pop_out3, pop_err3_neg, 0).connect_one2one(wpg)
cx_p=Connection(setup, pop_err3_pos, pop_hid, 1)
cx_p.connect_shuffle(2000)
Connection(setup, pop_err3_neg, pop_hid, 1).connect(cx_p.ptr_table, -cx_p.wgt_table)
Connection(setup, pop_hid, pop_out3, 0).connect(cx_p.ptr_table.T, cx_p.wgt_table)

Connection(setup, pop_err4_pos, pop_out4, 1).connect_one2one(wgp)
Connection(setup, pop_err4_neg, pop_out4, 1).connect_one2one(-wgp)
Connection(setup, pop_lab, pop_err4_pos, 0).connect_one2one(wpg)
Connection(setup, pop_lab, pop_err4_neg, 0).connect_one2one(-wpg)
Connection(setup, pop_out4, pop_err4_pos, 0).connect_one2one(-wpg)
Connection(setup, pop_out4, pop_err4_neg, 0).connect_one2one(wpg)

print("################### Constructing NSAT Configuration ##############################")
spk_rec_mon = [np.arange(setup.nneurons[0], dtype='int')]
#spk_rec_mon = [[] for i in range(setup.ncores)]
#spk_rec_mon[pop_out.core] = pop_out.addr

#TODO: fold following in NSATSetup
cfg_train = nsat.ConfigurationNSAT(
                   sim_ticks = sim_ticks,
                   N_CORES = setup.ncores,
                   N_NEURONS= setup.nneurons, 
                   N_INPUTS = setup.ninputs,
                   N_STATES = setup.nstates,
                   bm_rng = True,
                   w_check = False,
                   spk_rec_mon = spk_rec_mon,
                   monitor_spikes = False,
                   gated_learning = [True],
                   plasticity_en = [True])

# Parameters groups mapping function
for i in range(setup.ncores):
    cfg_train.core_cfgs[i] = setup.create_coreconfig(i)
#cfg_train.L1_connectivity = setup.do_L1connections()

spk_rec_mon = [[] for i in range(setup.ncores)]
spk_rec_mon[pop_out4.core] = pop_out1.addr + pop_out2.addr + pop_out3.addr + pop_out4.addr

cfg_test = nsat.ConfigurationNSAT(
                   sim_ticks = sim_ticks_test,
                   N_CORES = setup.ncores,
                   N_NEURONS= setup.nneurons, 
                   N_INPUTS = setup.ninputs,
                   N_STATES = setup.nstates,
                   bm_rng = True,
                   w_check = False,
                   plasticity_en = [False],
                   spk_rec_mon = spk_rec_mon,
                   monitor_spikes = True)

for i in range(setup.ncores):
    cfg_test.core_cfgs[i] = copy.copy(cfg_train.core_cfgs[i])
#cfg_test.L1_connectivity = cfg_train.L1_connectivity


#data_classify = data_train
#targets_classify = targets_train

SL_train = create_spike_train(data_train[:N_train], t_sample_train, scaling = inp_fact, with_labels = True)
ext_evts_data_train = nsat.exportAER(SL_train)

SL_test = create_spike_train(data_classify[:N_test], t_sample_test, scaling = inp_fact, with_labels = False)
ext_evts_data_test = nsat.exportAER(SL_test)

cfg_test.set_ext_events(ext_evts_data_test)
cfg_train.set_ext_events(ext_evts_data_train)

print("################## Writing Parameters Files ##################")
c_nsat_writer_train = nsat.C_NSATWriter(cfg_train, path=exp_name, prefix='')
c_nsat_writer_train.write()

c_nsat_writer_test = nsat.C_NSATWriter(cfg_test, path=exp_name_test,prefix='')
c_nsat_writer_test.write()

fname_train = c_nsat_writer_train.fname
fname_test = c_nsat_writer_test.fname

c_nsat_reader_train = nsat.C_NSATReader(cfg_train, fname_train)
c_nsat_reader_test = nsat.C_NSATReader(cfg_test, fname_test)

# Struct contains all the file names 

# C NSAT arguments


pip = []

if __name__ == '__main__':
    print("############# Running simulation #####################")


    for i in range(nepochs):
        t0 = time.time()
        nsat.run_c_nsat(fname_train)
        print(('Run took {0} seconds'.format(time.time()-t0)))


        for j in range(setup.ncores):
            #train->test
            shutil.copy(exp_name+'/_shared_mem_core_{0}.dat'.format(j), exp_name_test+'/_wgt_table_core_{0}.dat'.format(j))
            #train->train
            shutil.copy(exp_name+'/_shared_mem_core_{0}.dat'.format(j), exp_name+'/_wgt_table_core_{0}.dat'.format(j))
        if test_every>0:
            if i%test_every == test_every-1:
                nsat.run_c_nsat(fname_test)
                acc, slout = test_accuracy(
                        c_nsat_reader_test,
                        targets = targets_classify[:N_test],
                        pop = pop_out1,
                        sim_ticks = sim_ticks_test,
                        duration = t_sample_test)

                pip.append([i, acc])

                acc, slout = test_accuracy(
                        c_nsat_reader_test,
                        targets = targets_classify[:N_test],
                        pop = pop_out2,
                        sim_ticks = sim_ticks_test,
                        duration = t_sample_test)

                pip.append([i, acc])

                acc, slout = test_accuracy(
                        c_nsat_reader_test,
                        targets = targets_classify[:N_test],
                        pop = pop_out3,
                        sim_ticks = sim_ticks_test,
                        duration = t_sample_test)

                pip.append([i, acc])

                acc, slout = test_accuracy(
                        c_nsat_reader_test,
                        targets = targets_classify[:N_test],
                        pop = pop_out4,
                        sim_ticks = sim_ticks_test,
                        duration = t_sample_test)

                pip.append([i, acc])
                print(exp_name)
                print(pip)

    try:
        import experimentTools as et
        d=et.mksavedir(pre='Results_Scripts/')
        et.save(pip, 'pip.pkl')
        et.annotate('res',text=str(pip))
    except ImportError:
        print('saving disabled due to missing experiment tools')
     
