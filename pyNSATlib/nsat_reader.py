import numpy as np
import warnings
import pyNSATlib as nsat
#from .NSATlib import ConfigurationNSAT, exportAER, build_SpikeList
import struct as st
import time

def read_from_file(fname):
#     import struct as st
    try:
        with open(fname, "rb") as f:
            cont = f.read()
        size = int(len(cont) // 4)
        return np.array(st.unpack('i' * size, cont)).astype('i')
    except:
        print('nsat_reader:read_from_file %s file not found or unreadable' % fname)

def read_from_file_weights(fname):
#     import struct as st
    with open(fname, "rb") as f:
        cont = f.read()
    size = int(len(cont) // 4)
    return np.array(st.unpack('i' * size, cont))


def read_synaptic_weights(core_cfg, wgt_file, ptr_file, return_cw=False):
    from utils import ptr_wgt_table_to_dense
    ptr = read_from_file(ptr_file)
    wgt = read_from_file(wgt_file)
    W, CW = ptr_wgt_table_to_dense(
        ptr, wgt, core_cfg.n_inputs, core_cfg.n_neurons, core_cfg.n_states)
    if return_cw:
        return W, CW
    else:
        return W


class NSATReader(object):

    def __init__(self, config_nsat, fname):
        self.cfg = config_nsat
        self.fname = fname


class C_NSATReader(NSATReader):
    
    def read_config(self):
        self.cfg = nsat.ConfigurationNSAT.readfileb(self.fname.pickled)
        return self.cfg
#         with lzma.open(self.fname.pickled, 'wb') as fh:
#             self.cfg = nsat.ConfigurationNSAT.readb(fh)
    
#     @staticmethod
#     def unpack(data, typ='i'):
# #         data = np.array(data).flatten()
#         s = struct.unpack(typ * data.shape[0], *data.astype(typ))
#         return s
# '''
#     def read_corecfgs(self):
# '''        '''
#         Write all parameters for c_nsat simulations
#         *inputs*: fnames
#         *outputs*: None
# '''        '''
#         self.cfg = cfg = nsat.ConfigurationNSAT()
#         
#         with open(self.fname.params, 'rb') as fh:
#             # Global parameters
#             cfg.N_CORES, = struct.unpack('i', fh.readline())
# #             fh.read(unpack(cfg.N_CORES, 'i'))
#             cfg.single_core, = struct.unpack('?', fh.readline())
# #             fh.read(unpack(cfg.single_core, '?'))
# #             if len(cfg.L1_connectivity) != 0:
# #                 cfg.routing_en = True
#             cfg.routing_en, = struct.unpack('?', fh.readline())
# #             fh.read(unpack(cfg.routing_en, '?'))
#             cfg.sim_ticks, = struct.unpack('i', fh.readline())
# #             fh.read(unpack(cfg.sim_ticks, 'i'))
#             cfg.seed, = struct.unpack('i', fh.readline())
# #             fh.read(unpack(cfg.seed, 'i'))
#             cfg.s_seq, = struct.unpack('i', fh.readline())
# #             fh.read(unpack(cfg.s_seq, 'i'))
#             cfg.is_bm_rng_on, = struct.unpack('?', fh.readline())
# #             fh.read(unpack(cfg.is_bm_rng_on, '?'))
#             cfg.is_clock_on, = struct.unpack('?', fh.readline())
# #             fh.read(unpack(cfg.is_clock_on, '?'))
#             cfg.w_check, = struct.unpack('?', fh.readline())
# #             fh.read(unpack(cfg.w_check, '?'))
#             cfg.w_boundary, = struct.unpack('i', fh.readline())
# #             fh.read(unpack(cfg.w_boundary, 'i'))
# 
#             # Core parameters
#             for p, core_cfg in cfg:
#                 fh.read(unpack(cfg.ext_evts, '?'))
# #                if ( p < cfg.plasticity_en.size ):
#                 try:
#                     fh.read(unpack(cfg.plasticity_en[p], '?'))
# #                else: fh.read(bytes('cfgplasticity_en[p] OOB','utf-8'))
#                 except: fh.read(bytes('cfgplasticity_en[p] OOB','utf-8'))
#                 try:
#                     fh.read(unpack(cfg.gated_learning[p], '?'))
#                 except: fh.read(bytes('cfgggated_lerning_en[p] OOB','utf-8'))
#                 fh.read(unpack(core_cfg.n_inputs, 'i'))
#                 fh.read(unpack(core_cfg.n_neurons, 'i'))
#                 fh.read(unpack(core_cfg.n_states, 'i'))
#                 fh.read(unpack(core_cfg.n_groups, 'i'))
#                 fh.read(unpack(core_cfg.n_lrngroups, 'i'))
#                 fh.read(unpack(cfg.rec_deltat, 'i'))
#                 fh.read(unpack(cfg.num_syn_ids_rec[p], 'i'))
#                 fh.read(unpack(cfg.syn_ids_rec[p], 'i'))
# 
#             # NSAT parameters
#             for p, core_cfg in cfg:
#                 for j in range(core_cfg.n_groups):
#                     fh.read(unpack(core_cfg.gate_lower[j], 'i'))
#                     fh.read(unpack(core_cfg.gate_upper[j], 'i'))
#                     fh.read(unpack(core_cfg.learn_period[j], 'i'))
#                     fh.read(unpack(core_cfg.learn_burnin[j], 'i'))
#                     fh.read(unpack(core_cfg.t_ref[j], 'i'))
#                     fh.read(unpack(core_cfg.modstate[j], 'i'))
#                     fh.read(unpack(core_cfg.prob_syn[j], 'i'))
#                     fh.read(unpack(core_cfg.A[j].T, 'i'))
#                     fh.read(unpack(core_cfg.sA[j].T, 'i'))
#                     fh.read(unpack(core_cfg.b[j], 'i'))
#                     fh.read(unpack(core_cfg.Xreset[j], 'i'))
#                     fh.read(unpack(core_cfg.Xthlo[j], 'i'))
#                     fh.read(unpack(core_cfg.XresetOn[j], '?'))
#                     fh.read(unpack(core_cfg.Xthup[j], 'i'))
#                     fh.read(unpack(core_cfg.XspikeIncrVal[j], 'i'))
#                     fh.read(unpack(core_cfg.sigma[j], 'i'))
#                     fh.read(unpack(core_cfg.flagXth[j], '?'))
#                     fh.read(unpack(core_cfg.Xth[j], 'i'))
#                     fh.read(unpack(core_cfg.Wgain[j], 'i'))
# 
#                 fh.read(unpack(core_cfg.Xinit.flatten(), 'i'))
#                 fh.read(unpack(np.shape(cfg.spk_rec_mon[p])[0], 'i'))
#                 fh.read(unpack(cfg.spk_rec_mon[p], 'i'))
# 
#             # Learning parameters
#             for p, core_cfg in cfg:
#                 try:
#                     if cfg.plasticity_en[p]:
#                         fh.read(unpack(cfg.tstdpmax[p], 'i'))
#                         for j in range(core_cfg.n_lrngroups):
#                             fh.read(unpack(core_cfg.tstdp[j], 'i'))
#                             fh.read(unpack(core_cfg.plastic[j], '?'))
#                             fh.read(unpack(core_cfg.stdp_en[j], '?'))
#                             fh.read(unpack(core_cfg.is_stdp_exp_on[j], '?'))
#                             fh.read(pack(core_cfg.tca[j], 'i'))
#                             fh.read(pack(core_cfg.hica[j], 'i'))
#                             fh.read(pack(core_cfg.sica[j], 'i'))
#                             fh.read(pack(core_cfg.slca[j], 'i'))
#                             fh.read(pack(core_cfg.tac[j], 'i'))
#                             fh.read(pack(core_cfg.hiac[j], 'i'))
#                             fh.read(pack(core_cfg.siac[j], 'i'))
#                             fh.read(pack(core_cfg.slac[j], 'i'))
#                             fh.read(pack(core_cfg.is_rr_on[j], '?'))
#                             fh.read(pack(core_cfg.rr_num_bits[j], 'i'))
#                 except: fh.read(bytes('%d does not exist'.format(p),'utf-8'))
#             # Monitor parameters
#             # TODO: Separately for every core
#             for p, core_cfg in cfg:
#                 fh.read(pack(cfg.monitor_states, '?'))
#                 fh.read(pack(cfg.monitor_weights, '?'))
#                 fh.read(pack(cfg.monitor_weights_final, '?'))
#                 fh.read(pack(cfg.monitor_spikes, '?'))
#                 fh.read(pack(cfg.monitor_stats, '?'))
# 
#             """ The following generates the mapping function.
#                 For now this is a vector with numbers in [0, 8),
#                 and every element corresponds to a NSAT neuron
#                 unit.
#                 Example:
#                     If the user would like to have three (3)
#                     different parameters groups for 30 neurons
#                     then they have to do the following:
#                     nmap = np.zeros((num_neurons, dtype='i'))
#                     nmap[10:20] = 1
#                     nmap[20:30] = 2
#                     Of course one can mix the parameters and
#                     neurons but it's not recommended.
#             """
# 
#             # nmap = np.zeros((num_neurons, ), dtype='i')
#         with open(self.fname.nsat_params_map, 'rb') as f:
#             for p, core_cfg in cfg:
#                 f.read(pack(core_cfg.nmap, 'i'))
# 
#         # nmap = np.zeros((num_neurons, ), dtype='i')
#         with open(self.fname.lrn_params_map, 'rb') as f:
#             for p, core_cfg in cfg:
#                 lrnmap_unrolled = np.zeros(
#                     [core_cfg.n_neurons, core_cfg.n_states], dtype='int')
#                 for i in range(core_cfg.n_neurons):
#                     lrnmap_unrolled[i, :] = core_cfg.lrnmap[
#                         core_cfg.nmap[i], :]
#                 lrnmap_unrolled = lrnmap_unrolled.flatten()
#                 f.read(pack(lrnmap_unrolled, 'i'))
# '''

    def read_synaptic_weights(self, return_cw=False):
        '''
        Reads final weights into a dense W.
        Inputs:
        *return_cw*: If enabled, the CW matrix is also returned. 
        WARNING: Weight sharing information is lost and reuse of C and CW should be used only if parameters are not shared.
        '''
        Wcores = []
        CWcores = []
        for p, core_cfg in self.cfg:
            ptr_file = self.fname.syn_ptr_table + '_core_' + str(p) + '.dat'
            wgt_file = self.fname.syn_wgt_table + '_core_' + str(p) + '.dat'
            W, CW = read_synaptic_weights(
                core_cfg, wgt_file, ptr_file, return_cw=True)
            Wcores.append(W)
            CWcores.append(CW)
        if return_cw:
            return Wcores, CWcores
        else:
            return Wcores

    def read_c_nsat_weights(self, *args, **kwargs):
        '''
        Alias for read_synaptic_weights
        '''
        return self.read_synaptic_weights(*args, **kwargs)

    def read_c_nsat_weights_evo(self, *args, **kwargs):
        '''
        Alias for read_synaptic_weights_history
        '''
        return self.read_synaptic_weights_history(*args, **kwargs)

    def read_synaptic_weights_history(self, post=[]):
        '''
        Read weights monitored using monitor_weights=True.
        Inputs:
        *post*: post-synaptic neuron id whose weights are read.
        Outputs:
        A list of numpy arrays of shape (timesteps, pre-neuron id (including input neurons), state). Each item in the list corresponds to a core
        '''
        W_all, Pre_ids_all = [], []
        for p, core_cfg in self.cfg:
            n_units = core_cfg.n_inputs + core_cfg.n_neurons
            ww = read_from_file_weights(
                self.fname.synw + '_core_' + str(p) + '.dat')
            # len_ww = ww.shape[0]
            W = np.zeros((self.cfg.sim_ticks, n_units,
                          core_cfg.n_states), 'i')
            
            if (len(post) == 0):
                post = range(n_units)
            pre_ids = []
            for i in range(len(ww)):
                if len(ww[i * 5:i * 5 + 5]) != 0:
                    time, pre, post_, state, val = ww[i * 5:i * 5 + 5]
                    if post_ in post:
                        W[time, pre, state] = val
                        pre_ids.append(pre)
            W_all.append(W[1:, ...])
            Pre_ids_all.append(pre_ids)
        return W_all, Pre_ids_all

    def read_c_nsat_states(self, *args, **kwargs):
        return self.read_states(*args, **kwargs)

    def read_states(self, time_explicit=True):
        S = []
        for p, core_cfg in self.cfg:
            size = len(self.cfg.spk_rec_mon[p]) * core_cfg.n_states + 1

            filename = self.fname.states + '_core_' + str(p) + '.dat'
            tmp = read_from_file(filename)
            if ( tmp is None ):
                print('Error nsat_reader:read_states() read_from_file(\'%s\') call returned None' % filename) 
                return S

            res = np.zeros((self.cfg.sim_ticks, len(self.cfg.spk_rec_mon[p]),
                            core_cfg.n_states + 1), 'int')
            for i in range(self.cfg.sim_ticks - 1):
                tmp_ = tmp[i * size: (i + 1) * size]
                res[i, :, 0] = tmp_[0]
                res[i, :, 1:] = tmp_[1:].reshape(
                    len(self.cfg.spk_rec_mon[p]), core_cfg.n_states)
            if not time_explicit:
                S.append(res)
            else:
                S.append([res[:, :, 0], res[:, :, 1:]])
        return S

    def read_stats(self):
        '''
        Reads and returns spiking stats. TODO: ext neurons.
        '''
        nsat_stats = np.fromfile(self.fname.stats_nsat, np.uint64)[2::2]
        #ext_stats = np.fromfile(self.fname.stats_ext, np.uint64)[2::2]
        return nsat_stats, None  # ext_stats

    def read_c_nsat_states_list(self):
        T, S = [], []
        for p, core_cfg in self.cfg:
            tmp = read_from_file(self.fname.states +
                                 '_core_' + str(p) + '.dat')
            stride = len(self.cfg.spk_rec_mon[p]) * 4 + 1
            size = self.cfg.sim_ticks // self.cfg.rec_deltat
            tmp = tmp.reshape(size, stride)
            T.append(tmp[:, 0])
            S.append(np.split(tmp[:, 1:],
                              len(self.cfg.spk_rec_mon[p]),
                              axis=1))
        return T, S

    def read_spikelist(self, sim_ticks=None, id_list=None, core=0):
        from NSATlib import importAER
        if sim_ticks is None:
            sim_ticks = self.cfg.sim_ticks
        if id_list is None:
            id_list = self.cfg.spk_rec_mon[core]
        filename = self.fname.events + '_core_{0}.dat'.format(core)
        spikelist = importAER(self.read_events(
            core), sim_ticks=sim_ticks, id_list=id_list)
        return spikelist

    def read_c_nsat_synaptic_weights(self):
        '''
        Describe what this function does here.
        '''
        ptr_tables = []
        pos = 0
        for p, core_cfg in self.cfg:
            ptrs = read_from_file(self.fname.synw_final +
                                  ('_core_' + str(p) + '.dat'))
            shared_mem = read_from_file(
                self.fname.shared_mem + ('_core_' + str(p) + '.dat'))
            n_units = core_cfg.n_inputs + core_cfg.n_neurons
            n_states = core_cfg.n_states
            nentries = (ptrs[pos]) * 4
            pos += 1
            pR = ptrs[pos:nentries + pos].reshape(-1, 4)
            ptr = np.zeros([n_units, n_units, n_states], 'int')
            for p in pR:
                ptr[p[0], p[1], p[2]] = shared_mem[p[3]]
            pos += nentries
            ptr_tables.append(ptr)
        return ptr_tables

    def read_events(self, core):
        from struct import unpack
        with open(self.fname.events + ('_core_' + str(core) + '.dat'), 'rb') as f:
            c = f.read()
        size = int(len(c) // 4)
        tmp = np.array(unpack('i' * size, c), 'i')
        data = np.zeros((tmp.shape[0], ), dtype='uint64')
        size = tmp.shape[0] // 2
        evts = np.zeros((size, 2))
        data[::2] = tmp[:size]
        data[1::2] = tmp[size:]
        evts = np.flip(data.reshape(size, 2), 0)
        return evts

    def read_c_nsat_syn_evo(self, pair=None):
        '''
        Describe what this function does here.
        '''
        W, P = [], []
        for p, core_cfg in self.cfg:
            if self.cfg.syn_ids_rec[p] is not None:
                fname = self.fname.synw + ('_core_' + str(p) + '.dat')
                data = read_from_file(fname)
                size = int(data.shape[0] // 5)
                data = data.reshape(size, 5)

                w = []
                for i in self.cfg.syn_ids_rec[p]:
                    w.append(data[data[:, 2] == i])

                if pair is not None:
                    p = []
                    for i in range(len(pair) - 1):
                        p.append(data[(data[:, 2] == pair[i]) &
                                      (data[:, 1] == pair[i + 1]), 4])
                        p.append(data[(data[:, 2] == pair[i + 1]) &
                                      (data[:, 1] == pair[i]), 4])
                W.append(w)
                P.append(p)
        return W, P
