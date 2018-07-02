import numpy as np
import warnings


def read_from_file(fname):
    import struct as st
    with open(fname, "rb") as f:
        cont = f.read()
    size = int(len(cont) // 4)
    return np.array(st.unpack('i' * size, cont)).astype('i')


def read_from_file_weights(fname):
    import struct as st
    with open(fname, "rb") as f:
        cont = f.read()
    size = int(len(cont) // 4)
    return np.array(st.unpack('i' * size, cont))


def read_synaptic_weights(core_cfg, wgt_file, ptr_file, return_cw=False):
    from .utils import ptr_wgt_table_to_dense
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

    def read_synaptic_weights_history(self, post=None):
        '''
        Read weights monitored using monitor_weights=True.
        Inputs:
        *post*: post-synaptic neuron id whose weights are read.
        Outputs:
        A list of numpy arrays of shape (timesteps, pre-neuron id (including input neurons), state). Each item in the list corresponds to a core
        '''
        W_all = []
        for p, core_cfg in self.cfg:
            n_units = core_cfg.n_inputs + core_cfg.n_neurons
            ww = read_from_file_weights(self.fname.synw + '_core_' + str(p) + '.dat')
            # len_ww = ww.shape[0]
            W = np.zeros((self.cfg.sim_ticks, n_units,
                          core_cfg.n_states), 'int')
            for i in range(len(ww)):
                if len(ww[i * 5:i * 5 + 5]) != 0:
                    time, pre, post_, state, val = ww[i * 5:i * 5 + 5]
                    if post_ == post:
                        W[time, pre, state] = val
            W_all.append(W[1:, ...])
        return W_all

    def read_c_nsat_states(self, *args, **kwargs):
        return self.read_states(*args, **kwargs)

    def read_states(self, time_explicit=True):
        S = []
        for p, core_cfg in self.cfg:
            size = len(self.cfg.spk_rec_mon[p]) * core_cfg.n_states + 1
            tmp = read_from_file(self.fname.states + '_core_' + str(p) + '.dat')
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
            tmp = read_from_file(self.fname.states + '_core_' + str(p) + '.dat')
            stride = len(self.cfg.spk_rec_mon[p]) * 4 + 1
            size = self.cfg.sim_ticks // self.cfg.rec_deltat
            tmp = tmp.reshape(size, stride)
            T.append(tmp[:, 0])
            S.append(np.split(tmp[:, 1:],
                              len(self.cfg.spk_rec_mon[p]),
                              axis=1))
        return T, S

    def read_spikelist(self, sim_ticks=None, id_list=None, core=0):
        from .NSATlib import importAER
        if sim_ticks is None:
            sim_ticks = self.cfg.sim_ticks
        if id_list is None:
            id_list = self.cfg.spk_rec_mon[core]
        filename = self.fname.events + '_core_{0}.dat'.format(core)
        spikelist = importAER(self.read_events(core),sim_ticks=sim_ticks, id_list=id_list)
        return spikelist

    def read_c_nsat_synaptic_weights(self):
        '''
        Describe what this function does here.
        '''
        ptr_tables = []
        pos = 0
        for p, core_cfg in self.cfg:
            ptrs = read_from_file(self.fname.synw_final + ('_core_' + str(p) + '.dat'))
            shared_mem = read_from_file(self.fname.shared_mem + ('_core_' + str(p) + '.dat'))
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
