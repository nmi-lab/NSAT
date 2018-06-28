#!/bin/python
#-----------------------------------------------------------------------------
# File Name : sim_erbp_convnet_2s.py
# Purpose: eRBP learning of MNIST NSAT Spiking Convnet
#
# Author: Emre Neftci
#
# Creation Date : 02-20-2018
# Last Modified : Tue 27 Feb 2018 08:47:07 PM PST
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

N_FEAT1 = 16
N_FEAT2 = 32
stride = 2
ksize = 5

exp_name          = '/tmp/mnist_convnet_2L'
exp_name_test     = '/tmp/mnist_convnet_2L_test/'

#Globals
inputsize = 28
Nchannel = 1
Nxy = inputsize*inputsize
Nv = Nxy*Nchannel
Nl = 10
Nconv1 = Nv//stride//stride*N_FEAT1//Nchannel
Nconv2 = Nconv1//stride//stride*N_FEAT2//N_FEAT1
Nh = 100
Np = 10
Ng2 = Np
Ng3 = Np

t_sample_test = 3000
t_sample_train = 1500
nepochs = 100
N_train = 5000
N_test = 1000
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
erbp_ptype.rr_num_bits = 12
erbp_ptype.hiac = [-7, OFF, OFF]

erf_ntype = erf_ntype.copy()
erf_ntype.plasticity_type = [erbp_ptype,nonplastic_ptype]
erf_ntype.Wgain[0] = 2

net_graph = LogicalGraphSetup()

pop_data    = net_graph.create_population(Population(name = 'pop_data ',n = Nv, core = -1, is_external = True))
pop_lab     = net_graph.create_population(Population(name = 'pop_lab  ',n = Nl, core = -1, is_external = True))
pop_conv1   = net_graph.create_population(Population(name = 'pop_conv1',n = Nconv1, core = 0, neuron_cfg = erf_ntype))
pop_conv2   = net_graph.create_population(Population(name = 'pop_conv2',n = Nconv2, core = 1, neuron_cfg = erf_ntype))
pop_hid     = net_graph.create_population(Population(name = 'pop_hid  ',n = Nh, core = 1, neuron_cfg = erf_ntype))
pop_out     = net_graph.create_population(Population(name = 'pop_out  ',n = Nl, core = 1, neuron_cfg = output_ntype))

net_graph.create_connection(pop_data,  pop_conv1, 0, connect_conv2dbank(inputsize, Nchannel, N_FEAT1, stride, ksize))
net_graph.create_connection(pop_conv1, pop_conv2, 0, connect_conv2dbank(inputsize//stride, N_FEAT1, N_FEAT2, stride, ksize)) #Repeats N_FEAT1 times
net_graph.create_connection(pop_conv2, pop_hid,   0, connect_random_uniform(low=-16, high=16))
net_graph.create_connection(pop_hid,   pop_out,   0, connect_random_uniform(low=-4, high=4))

#eRBP related connections and populations
pop_err_pos = net_graph.create_population(Population(name = 'pop_err_pos', n = Nl, core = 0, neuron_cfg = error_ntype))
pop_err_neg = net_graph.create_population(Population(name = 'pop_err_neg', n = Nl, core = 0, neuron_cfg = error_ntype))

net_graph.create_connection(pop_out, pop_err_pos, 0, connect_one2one(-wpg))
net_graph.create_connection(pop_out, pop_err_neg, 0, connect_one2one(wpg))

net_graph.create_connection(pop_lab, pop_err_pos, 0, connect_one2one(wpg))
net_graph.create_connection(pop_lab, pop_err_neg, 0, connect_one2one(-wpg))

[p,w] = connect_shuffle(2000)(pop_err_pos, pop_conv1)
net_graph.create_connection(pop_err_pos, pop_conv1, 1, [p,+w])
net_graph.create_connection(pop_err_neg, pop_conv1, 1, [p,-w])

[p, w] = connect_shuffle(2000)(pop_err_pos, pop_conv2)
net_graph.create_connection(pop_err_pos, pop_conv2, 1, [p,+w])
net_graph.create_connection(pop_err_neg, pop_conv2, 1, [p,-w])

[p, w] = connect_shuffle(3000)(pop_err_pos, pop_hid)
net_graph.create_connection(pop_err_pos, pop_hid, 1, [p,+w])
net_graph.create_connection(pop_err_neg, pop_hid, 1, [p,-w])

net_graph.create_connection(pop_err_pos, pop_out, 1, connect_one2one(wgp))
net_graph.create_connection(pop_err_neg, pop_out, 1, connect_one2one(-wgp))

setup = net_graph.generate_multicore_setup(NSATSetup)

print("################### Constructing NSAT Configuration ##############################")
#spk_rec_mon = [np.arange(setup.nneurons[0], dtype='int')]
spk_rec_mon = [[] for i in range(setup.ncores)]
#spk_rec_mon[pop_out.core] = pop_out.addr

#TODO: fold following in NSATSetup
cfg_train = setup.create_configuration_nsat(
                   sim_ticks = sim_ticks,
                   w_check = False,
                   spk_rec_mon = spk_rec_mon,
                   monitor_spikes = False,
                   gated_learning = [True,True],
                   plasticity_en = [True,True])

spk_rec_mon = [[] for i in range(setup.ncores)]
spk_rec_mon[pop_out.core] = pop_out.addr

cfg_test = cfg_train.copy()
cfg_test.sim_ticks = sim_ticks_test
cfg_test.plasticity_en[:] = False
cfg_test.spk_rec_mon = spk_rec_mon
cfg_test.monitor_spikes = True


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
                        pop = pop_out,
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
     
