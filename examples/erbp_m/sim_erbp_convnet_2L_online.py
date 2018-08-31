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
import time
from pyNSATlib.laxesis import *

N_FEAT1 = 32
N_FEAT2 = 64
stride = 2
ksize = 5

exp_name          = '/tmp/mnist_convnet_2L_online'

#Globals
inputsize = 32
Nchannel = 2
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

np.random.seed(100)

wpg  = 64
wgp  = 37  

#Custom neuron type
erbp_ptype = erbp_ptype.copy()
erbp_ptype.rr_num_bits = 12
erbp_ptype.hiac = [-7, OFF, OFF]

erf_ntype = erf_ntype.copy()
erf_ntype.plasticity_type = [erbp_ptype,nonplastic_ptype]
erf_ntype.Wgain[0] = 2

error_ntype = error_ntype.copy()
error_ntype.Wgain[0] = 4


net_graph = LogicalGraphSetup()

pop_data    = net_graph.create_population(Population(name = 'pop_data ',n = Nv, core = -1, is_external = True))
pop_lab     = net_graph.create_population(Population(name = 'pop_lab  ',n = Nl, core = -1, is_external = True))
pop_labon   = net_graph.create_population(Population(name = 'pop_labon',n = 1,  core = -1, is_external = True))
pop_conv1   = net_graph.create_population(Population(name = 'pop_conv1',n = Nconv1, core = 0, neuron_cfg = erf_ntype))
pop_conv2   = net_graph.create_population(Population(name = 'pop_conv2',n = Nconv2, core = 0, neuron_cfg = erf_ntype))
pop_hid     = net_graph.create_population(Population(name = 'pop_hid  ',n = Nh, core = 0, neuron_cfg = erf_ntype))
pop_out     = net_graph.create_population(Population(name = 'pop_out  ',n = Nl, core = 0, neuron_cfg = output_ntype))

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
net_graph.create_connection(pop_labon, pop_err_pos, 0, connect_all2all(8))
net_graph.create_connection(pop_labon, pop_err_neg, 0, connect_all2all(8))

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
spk_rec_mon[pop_out.core] = pop_out.addr

#TODO: fold following in NSATSetup
cfg_train = setup.create_configuration_nsat(
                   sim_ticks = sim_ticks,
                   w_check = False,
                   spk_rec_mon = spk_rec_mon,
                   monitor_spikes = True,
                   gated_learning = [True,True],
                   plasticity_en = [True,True])

cfg_train.set_ext_events(True)
#cfg_train.ext_evts(True)

print("################## Writing Parameters Files ##################")
c_nsat_writer_train = nsat.C_NSATWriter(cfg_train, path=exp_name, prefix='')
c_nsat_writer_train.write()

if __name__ == '__main__':
    print("############# Running simulation #####################")

    #nsat.run_c_nsat()
