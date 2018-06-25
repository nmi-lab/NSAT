# import time
import copy
import numpy as np
from pyNCSre import pyST
import pyNSATlib as nsat
import matplotlib.pylab as plt
from ml_funcs import __tile_raster_images
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
import shutil
# from ml_funcs import data_preprocess


def SimSpikingStimulus(stim1, stim2, time=1000, t_sim=None):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0] *poisson*:
    integer, output is a poisson process with mean data/poisson, scaled by
    *poisson*.
    '''
    pyST.STCreate.seed(100)
    n = stim1.shape[1]
    SL = pyST.SpikeList(id_list=range(n+n-2))

    times = range(0, len(stim1)*time, time)
    for i in range(n):
        if np.any(stim1[:, i] > 0):
            SL[i] = pyST.STCreate.inh_poisson_generator(stim1[:, i],
                                                        times,
                                                        t_stop=t_sim)
    for i in range(n-2):
        if np.any(stim2[:, i] > 0):
            SL[n+i] = pyST.STCreate.inh_poisson_generator(stim2[:, i-n+2],
                                                          times,
                                                          t_stop=t_sim)
    plus_times1 = np.array([100]*32)+np.arange(0,2*time*32,2*time)
    plus_times2 = np.array([2*time]*32)+np.arange(0,2*time*32,2*time)
    plus_times = np.concatenate([plus_times1,plus_times2])

    minus_times1 = np.array([time]*32)+np.arange(0,2*time*32,2*time)
    minus_times2 = np.array([time+100]*32)+np.arange(0,2*time*32,2*time)
    minus_times = np.concatenate([minus_times1,minus_times2])
    
    SL[0] = pyST.SpikeTrain(np.sort(plus_times))
    SL[1] = pyST.SpikeTrain(np.sort(minus_times))
    return SL


np.random.seed(100)
pyST.STCreate.seed(100)
N_CORES = 1
Ne = 18
Ni = 18
Nv = 18
Nh = 100
Ns = 2
N_NEURONS = [Nv+Nh]
N_INPUTS = [Ns+Ne+Ni]
N_STATES = [4]
N_UNITS = N_INPUTS[0] + N_NEURONS[0]

N_test = 32
t_test = 1000

XMAX = nsat.XMAX
XMIN = nsat.XMIN
OFF = -16
MAX = nsat.MAX
MIN = nsat.XMIN

initv = 180
inith = 4
input_rate_mult = 50
n_mult = 1  # 32
train_duration = 750
test_duration = 5000

print("############## Loading Data ##############")
# data_train, targets_train, _ = bs_loader_npy(dset='train',
#                                              prefix='data/')
# data_classify, targets_classify, _ = bs_loader_npy(dset='train',
#                                                    prefix='data/')
# 

data_train = np.load("/shares/data/bs/bs_train_data.npy")
targets_train = np.load("/shares/data/bs/bs_train_targets.npy")
data_classify = np.load("/shares/data/bs/bs_classify_data.npy")
targets_classify = np.load("/shares/data/bs/bs_classify_targets.npy")


# Prepare training stimulus
idx = range(32) * n_mult
np.random.shuffle(idx)
data_train = data_train[idx, :]

m = data_train.shape[1] * 2
stim_train = np.zeros([m, Nv+Ns])
stim_train_i = np.zeros([m, Nv])
silent = np.zeros((1, Nv))
ii = 0

for i in range(m):
    if i % 2 == 0:
        stim_train[i, 0] = 0.7
        stim_train[i, 1] = 0
        stim_train[i, 2:] = data_train[ii]
        stim_train_i[i, :] = 1 - data_train[ii]
    else:
        stim_train[i, 0] = 0
        stim_train[i, 1] = 0.7
        stim_train[i, 2:] = silent
        stim_train_i[i, :] = silent
        ii += 1

sim_ticks_train = len(stim_train) * train_duration

# Stimulus Creation
# stim_train[:, 2:] = stim_train[:, 2:] * input_rate_mult + 1e-4
stim_train = stim_train * input_rate_mult + 1e-4
stim_train_i = stim_train_i * input_rate_mult + 1e-4
SL_train = SimSpikingStimulus(stim_train, stim_train_i,
                              train_duration,
                              t_sim=sim_ticks_train)

ext_evts_data_train = nsat.exportAER(SL_train)

idx = range(32)
np.random.shuffle(idx)
data_classify = data_classify[idx, :]
m = data_classify.shape[0] * 2
stim_test = np.zeros([m, Nv+Ns])
stim_test_i = np.zeros([m, Nv])
silent = np.zeros((1, Nv))
labels = np.zeros((m, ))
ii = 0

for i in range(m):
    if i % 2 == 0:
        stim_test[i, 2:18] = data_classify[ii, :16]
        stim_test_i[i, :16] = 1 - data_classify[ii, :16]
        labels[i] = data_classify[ii, 17]
    else:
        stim_test[i, 2:] = silent
        stim_test_i[i, :] = silent
        ii += 1

# stim_test[:, 2:18] = stim_test[:, 2:18] * input_rate_mult + 1e-4
stim_test = stim_test * input_rate_mult + 1e-4
stim_test_i = stim_test_i * input_rate_mult + 1e-4
sim_ticks_test = len(stim_test) * test_duration
SL_test = SimSpikingStimulus(stim_test, stim_test_i,
                             test_duration,
                             t_sim=sim_ticks_test)

# SL_test.raster_plot()
# plt.show()

ext_evts_data_test = nsat.exportAER(SL_test)

###############################################################
print("########## Setting Up NSAT ##########")
# syn_ids = [[1, 45, 110]]
# spk_rec_mon = [[10, 13, 35, 55]]
cfg_train = nsat.ConfigurationNSAT(sim_ticks=sim_ticks_train,
                                   # syn_ids_rec=syn_ids,
                                   # spk_rec_mon=spk_rec_mon,
                                   N_CORES=N_CORES,
                                   N_INPUTS=N_INPUTS,
                                   N_NEURONS=N_NEURONS,
                                   N_STATES=N_STATES,
                                   rec_deltat=50,
                                   w_check=False,
                                   w_boundary=7,
                                   tstdpmax=[39],
                                   monitor_states=False,
                                   monitor_spikes=True,
                                   monitor_weights=False,
                                   monitor_weights_final=True,
                                   plasticity_en=[True],
                                   ben_clock=True)

cfg_test = nsat.ConfigurationNSAT(sim_ticks=sim_ticks_test,
                                  N_CORES=N_CORES,
                                  N_INPUTS=N_INPUTS,
                                  N_NEURONS=N_NEURONS,
                                  N_STATES=N_STATES,
                                  w_check=False,
                                  w_boundary=7,
                                  monitor_states=False,
                                  monitor_spikes=True,
                                  monitor_weights=False,
                                  monitor_weights_final=False,
                                  plasticity_en=[False],
                                  ben_clock=True)

cfg0 = cfg_train.core_cfgs[0]
# Transition matrix group 0
cfg0.A[0] = [[-3, OFF, OFF, OFF],
             [8, -5, OFF, OFF],
             [OFF, OFF, OFF, OFF],
             [8, OFF, OFF, -5]]
cfg0.A[1] = cfg0.A[0].copy()

# Sign matrix group 0
cfg0.sA[0] = [[-1, 1, 1, 1],
              [+1, -1, 1, 1],
              [1, 1, -1, 1],
              [+1, 1, 1, -1]]
cfg0.sA[1] = cfg0.sA[0].copy()

cfg0.Xth[0] = XMAX
cfg0.Xth[1] = XMAX
cfg0.t_ref[0] = 40
cfg0.t_ref[1] = 40
cfg0.Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
cfg0.Xreset[1] = np.array([0, XMAX, XMAX, XMAX], 'int')
cfg0.Xthlo[0] = np.array([XMIN, XMIN, -1, XMIN], 'int')
cfg0.Xthup[0] = np.array([XMAX, XMAX, +1, XMAX], 'int')
cfg0.Xthlo[1] = np.array([XMIN, XMIN, -1, XMIN], 'int')
cfg0.Xthup[1] = np.array([XMAX, XMAX, +1, XMAX], 'int')
cfg0.XresetOn[0] = np.array([True, False, False, False], 'bool')
cfg0.XresetOn[1] = np.array([True, False, False, False], 'bool')
cfg0.prob_syn[0] = np.array([15, 7, 15, 15], dtype='int')
cfg0.prob_syn[1] = np.array([15, 7, 15, 15], dtype='int')
cfg0.b[0] = np.array([-6000, 0, 0, 0], 'int')
cfg0.b[1] = np.array([-9500, 0, 0, 0], 'int')

# cfg0.is_rr_on = np.array([True]+[False]*7)
# cfg0.rr_num_bits = np.array([8]+[0]*7, 'int')

cfg0.plastic[0] = True
cfg0.stdp_en[0] = True
cfg0.modstate[:] = 2

# Mapping function between neurons and NSAT parameters groups
cfg0.nmap = np.zeros((N_NEURONS[0],), dtype='int')
cfg0.nmap[:Nv] = 1
cfg0.lrnmap = np.ones((nsat.N_GROUPS, N_STATES[0]), 'i')
cfg0.lrnmap[0, 1] = 0
cfg0.lrnmap[1, 1] = 0

cfg0.Wgain[0][0] = 2      # Group, State  4
cfg0.Wgain[0][1] = 1
cfg0.Wgain[1][0] = 2      # Group, State
cfg0.Wgain[1][1] = 1
cfg0.Wgain[1][3] = 5

K = 35
cfg0.tstdp[0] = K
cfg0.tca[0] = np.array([K, K])
cfg0.hica[0] = np.array([0, 0, 0])
cfg0.sica[0] = np.array([1, 1, 1])
cfg0.tac[0] = np.array([-K, -K])
cfg0.hiac[0] = np.array([0, 0, 0])
cfg0.siac[0] = np.array([1, 1, 1])

# Synaptic strengths
W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
W[0, Ns+Ne+Ni:, 2] = 1
W[1, Ns+Ne+Ni:, 2] = -1
np.fill_diagonal(W[Ns:(Ns+Ne), (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 3], 127)
np.fill_diagonal(W[(Ns+Ne):(Ns+Ne+Ni), (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 3], -60)
ww = np.random.normal(0, 10, (Nv, Nh)).astype('int')
W[(Ns+Ne+Ni):(Ns+Ne+Ni+Nv), (Ns+Ne+Ni+Nv):, 1] = ww
W[(Ns+Ne+Ni+Nv):, (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 1] = ww.T

# Adjacency matrix
CW = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
CW[0, Ns+Ne+Ni:, 2] = 1
CW[1, Ns+Ne+Ni:, 2] = 1
np.fill_diagonal(CW[Ns:Ns+Ne, (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 3], 1)
np.fill_diagonal(CW[Ns+Ne:Ns+Ne+Ni, (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 3], 1)
CW[(Ns+Ne+Ni):(Ns+Ne+Ni+Nv), (Ns+Ne+Ni+Nv):, 1] = 1
CW[(Ns+Ne+Ni+Nv):, (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 1] = 1
cfg0.CW = CW.astype('bool')

# Writing external events
cfg_train.set_ext_events(ext_evts_data_train)
cfg_test.set_ext_events(ext_evts_data_test)

wgt_table, ptr_table = gen_ptr_wgt_table_from_W_CW(W, CW, [])

cfg0.wgt_table = wgt_table
cfg0.ptr_table = ptr_table

cfg_test.core_cfgs[0] = copy.deepcopy(cfg0)
cfg_test.core_cfgs[0].plastic[0] = False
cfg_test.core_cfgs[0].stdp_en[0] = False
cfg_test.core_cfgs[0].modstate[:] = 2


# Write C NSAT parameters binary files
c_nsat_writer_train = nsat.C_NSATWriter(cfg_train, path='/tmp',
                                        prefix='test_eCD')
c_nsat_writer_train.write()

c_nsat_writer_test = nsat.C_NSATWriter(cfg_test, path='/tmp',
                                       prefix='test_eCD_test')
c_nsat_writer_test.write()

# Reader class instance
fname_train = c_nsat_writer_train.fname
c_nsat_reader_train = nsat.C_NSATReader(cfg_train, fname_train)

fname_test = c_nsat_writer_test.fname
c_nsat_reader_test = nsat.C_NSATReader(cfg_test, fname_test)

# cfg_train.core_cfgs[0].latex_print_parameters(2)

if __name__ == '__main__':
    n_epoch = 50
    e = []
    print("############## Training ##############")
    for i in range(n_epoch):
        # Call the C NSAT for learning
        print("Epoch #:  ", i)
        c_nsat_writer_train.fname.stats_nsat = \
                '/tmp/test_eCD_stats_nsat_full'+str(i)
        nsat.run_c_nsat(c_nsat_writer_train.fname)
        if n_epoch > 1:
            shutil.copyfile(fname_train.shared_mem+'_core_0.dat',
                            fname_train.syn_wgt_table+'_core_0.dat',)
            shutil.copyfile(fname_train.shared_mem+'_core_0.dat',
                            fname_test.syn_wgt_table+'_core_0.dat',)
        nsat.run_c_nsat(c_nsat_writer_test.fname)
        test_spk = nsat.importAER(c_nsat_reader_test.read_c_nsat_raw_events()[0],
                                  sim_ticks=sim_ticks_test)
        sl = test_spk.id_slice([16, 17])
        mm = np.argmax(sl.firing_rate(time_bin=test_duration), axis=0)[::2]
        print(100*sum(labels[::2] != mm)/32.0)
        e.append(sum(labels[::2] != mm) / 32.0)

    e = np.array(e)
    np.save('/tmp/error_full', e)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(e, 'x-k', lw=2)

    W_new = c_nsat_reader_train.read_c_nsat_weights()[0]
    r, c = Ns+Ne+Ni, Ns+Ne+Ni+Nv
    W = W_new[(Ns+Ne+Ni):(Ns+Ne+Ni+Nv), (Ns+Ne+Ni+Nv):, 1]
    WT = W_new[(Ns+Ne+Ni+Nv):, (Ns+Ne+Ni):(Ns+Ne+Ni+Nv), 1]
    np.save('/tmp/weights_full', W)

    # fog = plt.figure()
    # ax = fog.add_subplot(111)
    # plt.imshow(W-WT.T, interpolation='nearest', cmap=plt.cm.gray,
    #            aspect='auto')
    # plt.colorbar()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(__tile_raster_images(W_new[r:c-2, c:, 1].T, (4, 4), (8, 8),
    #                                tile_spacing=(2, 2)), cmap=plt.cm.gray)

    # out_spikelist = nsat.importAER(nsat.read_from_file(
    #     c_nsat_writer_train.fname.events+'_core_0.dat'),
    #     sim_ticks=sim_ticks_train,
    #     id_list=[0])

    # fig = plt.figure()
    # ax0 = fig.add_subplot(111)

    # ids = [i for i in range(20)]
    # SL_train.raster_plot(display=ax0, kwargs={'color': 'b'})
    # out_spikelist.raster_plot(display=ax0, kwargs={'color': 'k'})

    # ttT, adT = SL_train.convert()
    # ttN, adN = out_spikelist.convert()
    # ttN = np.array(ttN, 'i')
    # adN = np.array(adN, 'i')

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(ttN[adN < 16], adN[adN < 16], 'x', zorder=10)
    # ax1.plot(ttT, adT-2, 'r.', zorder=0, alpha=0.5)
    # ax1.plot(ttN[adN == 16], adN[adN == 16], 'c.')
    # ax1.plot(ttN[adN == 17], adN[adN == 17], 'm.')

    # test_spk = nsat.importAER(nsat.read_from_file(c_nsat_writer_test.fname.events+'_core_0.dat'),
    #                           sim_ticks=sim_ticks_test)

    # ttT, adT = SL_test.convert()
    # ttN, adN = test_spk.convert()
    # ttN = np.array(ttN, 'i')
    # adN = np.array(adN, 'i')

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(ttN[adN < 16], adN[adN < 16], '.')
    # ax1.plot(ttT, adT-2, 'r.')
    # ax1.plot(ttN[adN == 16], adN[adN == 16], 'c.')
    # ax1.plot(ttN[adN == 17], adN[adN == 17], 'm.')

    # fig = plt.figure()
    # ax2 = fig.add_subplot(111)
    # SL_test.raster_plot(display=ax2, kwargs={'color': 'r'})
    # # ids = [i for i in range(0, 16)]
    # test_spk.raster_plot(display=ax2, kwargs={'color': 'k'})
    # plt.show()
