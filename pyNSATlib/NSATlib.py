#!/bin/python
# ---------------------------------------------------------------------------
# File Name : NSATlib_v2.py
# Purpose: Main classes for NSAT version 2
#
# Author: Emre Neftci, Sadique Sheik, Georgios Detorakis
#
# Creation Date : 09-08-2015
# Last Modified : Fri 20 Jan 2017 10:09:38 PM PST
#
# Copyright : (c)
# Licence : GPLv2
# ---------------------------------------------------------------------------
import os
import numpy as np
from pyNCSre import pyST
from .global_vars import *
import copy


def find_nsat_library():
    '''
    Find libnsat.so library in the system using LD_LIBRARY_PATH
    *outputs*: path to libnsat.so
    '''
    ldlp = os.environ.get('LD_LIBRARY_PATH')
    if ldlp is None:
        raise RuntimeError('LD_LIBRARY_PATH not set')
    ldlp = ldlp.split(':')
    for p in ldlp:
        if os.path.isfile(p + '/libnsat.so'):
            return p + '/libnsat.so'
    raise RuntimeError('libnsat.so file not found. Try adding nsat lib \
                        directory to LD_LIBRARY_PATH')


def run_c_nsat(fname):
    from ctypes import POINTER, cdll, c_int
    from .nsat_writer import c_nsat_fnames, generate_c_fnames

    _nsat = cdll.LoadLibrary(find_nsat_library())

    # handle = _nsat._handle
    _nsat.iterate_nsat.argtypes = (POINTER(c_nsat_fnames),)
    _nsat.iterate_nsat.restype = c_int

    flag = _nsat.iterate_nsat(generate_c_fnames(fname))
    return flag


def build_SpikeList(evs_time,
                    evs_addr,
                    dt=1e-3,
                    id_list=None,
                    sim_ticks=None):
    '''
    build a pyST spikelist. pyST is a submodule of pyNCSre
    (https://github.com/nmi-lab/pyNCS)

    inputs:
    *evs_time*: list of event timesteps
    *evs_addr*: list of event addresses
    *dt*: scaling of timesteps (default is ms)
    *id_list*: list of neuron ids (optional). If no id_list is provided,
    neurons that have never spikes will not be represented explicitely.

    outputs:
    pyST.SpikeList object
    '''
    from pyNCSre.pyST.spikes import SpikeList
    evs_time = np.array(evs_time) / dt * 1e-3
    SL = SpikeList(list(zip(evs_addr, evs_time)), np.unique(evs_addr))
    SL.t_start = 0
    if sim_ticks is not None:
        SL.t_stop = sim_ticks
    return SL


def _extract_submatrix(matrix,
                       start_row=0,
                       end_row=None,
                       start_col=0,
                       end_col=None):
    if end_row is None:
        end_row = matrix.shape[0]
    if end_col is None:
        end_col = matrix.shape[1]
    return matrix[start_row:end_row, start_col:end_col]


class multicoreEvents(pyST.channelEvents):

    def flatten(self):
        ev = pyST.events(atype=self.atype)
        for ch in self:
            ads = self[ch].get_ad() + (ch << CHANNEL_OFFSET)
            tms = self[ch].get_tm()
            ev.add_adtm(ads, tms)
        ev.sort_tm()
        return ev


def exportAER(spikeLists,
              filename=None,
              format='t',
              sep=' ',
              addr_format='%u',
              time_format='%u',
              dt=1,
              debug=False,
              *args,
              **kwargs):
    '''
    Modified from pyNCS.pyST.exportAER
    '''

    assert format in ['t', 'a'], 'Format must be "a" or "t"'

    ev = multicoreEvents(atype='Physical')

    # Translate logical addresses to physical using a mapping

    if isinstance(spikeLists, list):
        for i in range(len(spikeLists)):
            if not isinstance(spikeLists[i], pyST.SpikeList):
                raise TypeError(
                    "Elements of spikeLists must be SpikeList objects!")

        spikeLists = dict(list(zip(list(range(len(spikeLists))), spikeLists)))
    elif isinstance(spikeLists, pyST.SpikeList):
        spikeLists = {0: spikeLists}
    elif isinstance(spikeLists, dict):
        for i in spikeLists.keys():
            if not isinstance(spikeLists[i], pyST.SpikeList):
                raise TypeError(
                    "Values of spikeLists must be SpikeList objects!")
    else:
        raise RuntimeError(
            "spikeLists must be either a: SpikeList, list or dict object")

    for ch in spikeLists:
        if isinstance(spikeLists[ch], pyST.SpikeList):
            slrd = spikeLists[ch].raw_data()
            if len(slrd) > 0:
                tmp_mapped_SL = np.fliplr(slrd)
                mapped_SL = np.zeros_like(tmp_mapped_SL, dtype='uint32')
                mapped_SL[:, 1] = tmp_mapped_SL[:, 1] * dt  # ms
                mapped_SL[:, 0] = tmp_mapped_SL[:, 0]
                ev.add_ch(ch, mapped_SL)
            else:
                print("Warning: Empty SpikeList encountered")

    # Choose desired output: no filename given, return events
    return ev


def importAER(nsat_events, id_list=None, sim_ticks=None):
    '''
    Read NSAT events output.
    *nsat_events*: events output of NSAT. expects a single dimensional
    array such as [addr0, time0, addr1, time1] etc.
    *id_list*: list of neuron ids. if none provided, the addresses in
    the nsat_events will be used
    *sim_ticks*: duration of nsat_events. If none is provided, the largest
    timestamp will be used
    '''
    from pyNCSre.pyST import events
    tm, ad = events(nsat_events.reshape(-1, 2)).get_tmad()
    if sim_ticks is None:
        sim_ticks = tm[-1]
    SL = build_SpikeList(tm, ad, sim_ticks=sim_ticks)
    if id_list is not None:
        SL.complete(id_list)
    return SL


class coreConfig(object):
    NSAT_parameters = [
        'A',
        'b',
        'ptr_table',
        'wgt_table',
        'flagXth',
        'hiac',
        'hica',
        'lrnmap',
        'modstate',
        'nmap',
        'ntypes',
        'plastic',
        'prob_syn',
        'sA',
        'siac',
        'sica',
        'slac',
        'slca',
        'sigma',
        'stdp_en',
        'is_stdp_exp_on',
        'tac',
        'tca',
        't_ref',
        'tstdp',
        'is_rr_on',
        'rr_num_bits',
        'n_states',
        'n_neurons',
        'n_inputs',
        'n_lrngroups',
        'n_groups',
        'Wgain',
        'Xinit',
        'XresetOn',
        'XspikeIncrVal',
        'Xreset',
        'Xth',
        'Xthlo',
        'Xthup',
        'gate_lower',
        'gate_upper',
        'learn_period',
        'learn_burnin']

    def __init__(self, n_states, n_neurons, n_inputs):
        for i in self.NSAT_parameters:
            setattr(self, i, None)
        self.gen_core_cfg(n_states, n_neurons, n_inputs)
        self.default_core_cfg = copy.deepcopy(self)

    def __repr__(self):
        return '''NSAT Core Configuration:
                   Nunits: {n_units}
                   Nstates: {n_states}
                   Ninputs: {n_inputs}
                   Nneurons: {n_neurons}
                   Ngroups: {n_groups}
                   L0 connections: {nnz}'''.format(nnz=self.ptr_table.nnz, **self.__dict__)

    def gen_core_cfg(core_cfg, n_states, n_neurons, n_inputs):

        rn_states = list(range(n_states))     # number of states per neuron
        N_GROUPS = core_cfg.n_groups = 8
        N_LRNGROUPS = core_cfg.n_lrngroups = 8
        rN_GROUPS = list(range(core_cfg.n_groups))
        rN_LRNGROUPS = list(range(core_cfg.n_lrngroups))

        # NSAT Dynamics parameters
        core_cfg.A = np.array(
            [[[OFF] * n_states for _ in rn_states] for _ in rN_GROUPS])
        core_cfg.sA = np.array(
            [[[-1] * n_states for _ in rn_states] for _ in rN_GROUPS])
        core_cfg.b = np.array([[0 for _ in rn_states] for _ in rN_GROUPS])
        core_cfg.ntypes = np.zeros([n_neurons], 'int')

        # Initial conditions
        core_cfg.Xinit = np.array([[0 for _ in rn_states]
                                   for _ in range(n_neurons)])

        # Spike and reset parameters
        core_cfg.XresetOn = np.array(
            [[True] + [False for _ in rn_states[:-1]]] * N_GROUPS)
        core_cfg.Xreset = np.array([[0 for _ in rn_states] for _ in rN_GROUPS])
        core_cfg.XspikeIncrVal = np.array(
            [[0 for _ in rn_states] for _ in rN_GROUPS])
        core_cfg.Xth = np.array([MAX for _ in rN_GROUPS])
        core_cfg.Xthlo = np.array([[MIN for _ in rn_states]
                                   for _ in rN_GROUPS])
        core_cfg.Xthup = np.array([[MAX for _ in rn_states]
                                   for _ in rN_GROUPS])
        core_cfg.flagXth = np.array([False for _ in rN_GROUPS])

        # Neuron and Learning Maps
        core_cfg.nmap = np.zeros((n_neurons), dtype='int')
        core_cfg.lrnmap = np.zeros((N_GROUPS, n_states), dtype='int')

        # Refractory period
        core_cfg.t_ref = np.array([0 for _ in rN_GROUPS])

        # Gate learning parameters
        core_cfg.gate_lower = np.array([-2560 for _ in rN_GROUPS])
        core_cfg.gate_upper = np.array([2560 for _ in rN_GROUPS])
        core_cfg.learn_period = np.array([1500 for _ in rN_GROUPS])
        core_cfg.learn_burnin = np.array([400 for _ in rN_GROUPS])

        # Blankout probability
        core_cfg.prob_syn = np.array(
            [[15 for _ in rn_states] for _ in rN_GROUPS])

        # Additive noise variance
        core_cfg.sigma = np.array([[0 for _ in rn_states] for _ in rN_GROUPS])

        # Modulator state
        core_cfg.modstate = np.array([1 for _ in rN_GROUPS])

        # Plasticity Parameters
        core_cfg.plastic = np.array([False for _ in rN_LRNGROUPS])
        core_cfg.stdp_en = np.array([False for _ in rN_LRNGROUPS])
        core_cfg.is_stdp_exp_on = np.array([False for _ in rN_LRNGROUPS])
        core_cfg.Wgain = np.array([[0 for _ in rn_states] for _ in rN_GROUPS])

        # STDP Parameters
        core_cfg.tstdp = np.array([64 for _ in rN_LRNGROUPS])
        core_cfg.tca = np.array([[16, 36] for _ in rN_LRNGROUPS], 'int')
        core_cfg.hica = np.array([[1, 0, -1] for _ in rN_LRNGROUPS], 'int')
        core_cfg.sica = np.array([[1, 1, 1] for _ in rN_LRNGROUPS], 'int')
        core_cfg.slca = np.array([[16, 16, 16] for _ in rN_LRNGROUPS], 'int')
        core_cfg.tac = np.array([[-16, -36] for _ in rN_LRNGROUPS], 'int')
        core_cfg.hiac = np.array([[1, 0, -1] for _ in rN_LRNGROUPS], 'int')
        core_cfg.siac = np.array([[-1, -1, -1] for _ in rN_LRNGROUPS], 'int')
        core_cfg.slac = np.array([[16, 16, 16] for _ in rN_LRNGROUPS], 'int')

        core_cfg.is_rr_on = np.array([False for _ in rN_LRNGROUPS], 'bool')
        core_cfg.rr_num_bits = np.array([0 for _ in rN_LRNGROUPS], 'int')

        core_cfg.n_states = n_states
        core_cfg.n_neurons = n_neurons
        core_cfg.n_inputs = n_inputs
        core_cfg.n_units = n_neurons + n_inputs

        core_cfg.wgt_table, core_cfg.ptr_table = check_weight_matrix(
            None, None, core_cfg.n_states, core_cfg.n_units)

        return core_cfg

    def latex_print_parameters(self, n_groups, group_names=None, n_states=None):
        if group_names == None:
            group_names = list(range(n_groups))

        if n_states == None:
            n_states = self.n_states
        from .utils import latex_print_group
        import copy
        NSAT_parameters = copy.deepcopy(self.NSAT_parameters)
        NSAT_parameters.pop(NSAT_parameters.index('lrnmap'))
        NSAT_parameters.pop(NSAT_parameters.index('nmap'))
        NSAT_parameters.pop(NSAT_parameters.index('ptr_table'))
        NSAT_parameters.pop(NSAT_parameters.index('wgt_table'))
        for s in NSAT_parameters:
            v = getattr(self, s)
            vdef = getattr(self.default_core_cfg, s)
            if hasattr(v, '__len__') and np.any(v != vdef):
                s = s.replace('_', '')
                print(latex_print_group(
                    v[:n_groups], prefix=s, group_names=group_names, n_states=n_states))

    def is_parameter_valid(self, p):
        if p not in self.NSAT_parameters:
            return False
        else:
            return True

    def is_parameter_undefined(self):
        ps = []
        for p in self.NSAT_parameters:
            if getattr(self, p) is None:
                ps.append(p)
        return ps


def complete(vm, n=N_TOT):
    if not hasattr(vm, '__len__'):
        vm = np.array([vm], 'int')
    if len(vm) < n:
        vm_cmp = np.zeros([n], dtype='int')
        vm_cmp[:len(vm)] = vm
        return vm_cmp
    else:
        return vm


def check_weight_matrix(wgt_table, ptr_table, n_states, n_units):
    if wgt_table is None:
        wgt_table = np.zeros([0], dtype='int')

    if ptr_table is None:
        from scipy.sparse import csr_matrix
        ptr_table = csr_matrix([n_units,
                                n_units *
                                n_states],
                               dtype='uint64')
    return wgt_table, ptr_table


def check_matrix(core_cfg, A):
    Ns = core_cfg.n_states
    if np.shape(A) != (core_cfg.n_groups, Ns, Ns):
        A = [A for i in range(core_cfg.n_groups)]
    assert np.shape(A) == (core_cfg.n_groups, Ns, Ns)
    return np.array(A, dtype='int')


def check_vector(core_cfg, vector, dtype='int'):
    Ns = core_cfg.n_states
    if np.shape(vector) != (core_cfg.n_groups, Ns):
        vector = [vector for i in range(core_cfg.n_groups)]
    assert np.shape(vector) == (core_cfg.n_groups, Ns)
    return np.array(vector, dtype=dtype)


def check_scalar(core_cfg, scalar, dtype='int'):
    assert np.shape(scalar)[0] == (core_cfg.n_groups)
    return np.array(scalar, dtype=dtype)


def check_xinit(core_cfg, xinit, dtype='int'):
    assert np.shape(xinit) == (core_cfg.n_neurons, core_cfg.n_states)
    return np.array(xinit, dtype=dtype)


def process_Xresets(core_cfg):
    # NSAT uses a single placeholder for spike increment and spike reset.
    n_states = core_cfg.n_states
    Xreset = check_vector(core_cfg, core_cfg.Xreset)
    XspikeIncrVal = check_vector(core_cfg, core_cfg.XspikeIncrVal)
    core_cfg.XspikeReset = np.zeros([core_cfg.n_groups, n_states], 'int')
    core_cfg.XresetOn = check_vector(core_cfg, core_cfg.XresetOn, 'bool')
    for i in range(core_cfg.n_groups):
        for j in range(n_states):
            if core_cfg.XresetOn[i, j] == True:
                core_cfg.XspikeReset[i, j] = Xreset[i, j]
            elif core_cfg.XresetOn[i, j] == False:
                core_cfg.XspikeReset[i, j] = XspikeIncrVal[i, j]
            else:
                raise Exception('Invalid value for XresetOn')


class ConfigurationNSAT(object):
    PKL_MEMBERS = ['monitor_weights_final',
                   'single_core',
                   'core_cfgs',
                   'tstdpmax',
                   'groups_set',
                   'num_syn_ids_rec',
                   's_seq',
                   'monitor_spikes',
                   'rec_deltat',
                   'is_clock_on',
                   'N_CORES',
                   'monitor_states',
                   'L1_connectivity',
                   'ext_evts',
                   'spk_rec_mon',
                   'syn_ids_rec',
                   'sim_ticks',
                   'w_boundary',
                   'w_check',
                   'monitor_stats',
                   'gated_learning',
                   'check_flag',
                   'is_bm_rng_on',
                   'routing_en',
                   'seed',
                   'monitor_weights',
                   'plasticity_en']

    def __getstate__(self):
        return {s: self.__dict__[s] for s in self.PKL_MEMBERS}

    def __init__(self,
                 sim_ticks=1000,
                 rec_deltat=1,
                 N_CORES=1,
                 N_INPUTS=[0],
                 N_NEURONS=[512],
                 N_STATES=[4],
                 monitor_states=False,
                 monitor_weights=False,
                 monitor_weights_final=True,
                 monitor_spikes=True,
                 monitor_stats=False,
                 spk_rec_mon=None,
                 syn_ids_rec=None,
                 bm_rng=True,
                 tstdpmax=None,
                 seed=0,
                 s_seq=0,
                 w_boundary=8,
                 w_check=True,
                 ben_clock=False,
                 plasticity_en=np.array([False], 'bool'),
                 gated_learning=np.array([False], 'bool')):
        self.groups_set = False
        self.sim_ticks = sim_ticks
        self.rec_deltat = rec_deltat  # timestep for the monitors (deltat)
        self.N_CORES = N_CORES
        self.L1_connectivity = dict()
        self.ext_evts = False
        self.seed = seed

        self.single_core = True
        if N_CORES > 1:
            self.single_core = False
        self.routing_en = False

        if not hasattr(plasticity_en, '__len__'):
            print("All the cores receive the same learning flag ({0})!".format(plasticity_en))
            self.plasticity_en = np.array([plasticity_en]*N_CORES, 'bool')
        else:
            self.plasticity_en = np.array(plasticity_en, 'bool')
        # if len(plasticity_en) != N_CORES:
        #     print("All the cores receive the same learning flag ({0})!".format(
        #         plasticity_en))
        #     self.plasticity_en = np.array([plasticity_en] * N_CORES, 'bool')
        # else:
        #     self.plasticity_en = np.array(plasticity_en, 'bool')

        assert(hasattr(self.plasticity_en, '__len__'))

        if not hasattr(gated_learning, '__len__'):
            print("All the cores receive the same gated learning flag ({0})!".format(gated_learning))
            self.gated_learning = np.array([gated_learning for _ in range(N_CORES)], 'bool')
        else:
            self.gated_learning = np.array(gated_learning, dtype='bool')
        # if len(gated_learning) != N_CORES:
        #     print("All the cores receive the same gated learning flag ({0})!".format(
        #         gated_learning))
        #     self.gated_learning = np.array(
        #         [gated_learning for _ in range(N_CORES)], 'bool')
        # else:
        #     self.gated_learning = np.array(gated_learning, dtype='bool')

        assert not hasattr(
            monitor_states, '__len__'), "Monitors are System Wide"
        self.monitor_states = monitor_states    # Enabled on write_hex
        assert not hasattr(
            monitor_spikes, '__len__'), "Monitors are System Wide"
        self.monitor_spikes = monitor_spikes    # Enabled on write_hex
        assert not hasattr(
            monitor_stats, '__len__'), "Monitors are System Wide"
        self.monitor_stats = monitor_stats      # Enabled on write_hex
        assert not hasattr(
            monitor_weights, '__len__'), "Monitors are System Wide"
        self.monitor_weights = monitor_weights  # Enabled on write_hex
        assert not hasattr(monitor_weights_final,
                           '__len__'), "Monitors are System Wide"
        self.monitor_weights_final = monitor_weights_final  # Enabled on write_hex

        self.s_seq = s_seq
        self.w_boundary = w_boundary
        self.w_check = w_check

        self.check_flag = False

        # HiAER NSAT wide constants
        if tstdpmax is None:
            self.tstdpmax = np.array([TSTDPMAX for _ in range(self.N_CORES)])
        else:
            self.tstdpmax = np.array(tstdpmax)

        self.is_clock_on = ben_clock
        self.is_bm_rng_on = bm_rng

        self.init_default_corecfgs(N_STATES, N_NEURONS, N_INPUTS)
        self.set_ext_events()
        self.set_default_monitors(spk_rec_mon, syn_ids_rec)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __getitem__(self, k):
        return self.core_cfgs[k]

    def __setitem__(self, k, v):
        self.core_cfgs[k] = v

    def __iter__(self):
        for i, p in enumerate(self.core_cfgs):
            yield i, p

    def set_default_monitors(self, spk_rec_mon=None, syn_ids_rec=None):
                # Neuron ids to be monitored
        self.spk_rec_mon = spk_rec_mon
        if self.spk_rec_mon is None:
            self.spk_rec_mon = [
                np.arange(p.n_neurons, dtype='int') for i, p in self]

        # Synapse ids to be monitored
        if syn_ids_rec is None and self.ext_evts is False:
            self.syn_ids_rec = [
                np.arange(p.n_inputs + p.n_neurons, dtype='int') for i, p in self]
        else:
            self.syn_ids_rec = np.array(syn_ids_rec, 'int')

        self.num_syn_ids_rec = [np.shape(s)[0] for s in self.syn_ids_rec]

    def init_default_corecfgs(self, n_states_list, n_neurons_list, n_inputs_list):
        '''
        Initializes all the parameters for NSAT to default values (see
        documentation for default values)
        '''
        self.core_cfgs = []
        for p in range(self.N_CORES):
            self.core_cfgs.append(coreConfig(
                n_states_list[p], n_neurons_list[p], n_inputs_list[p]))

    def set_ext_events(self, ext_evts_data=None):
        '''
        Set external events
        Inputs:
        *ext_evts_data*: multicoreEvents or a dictionary of time-address
        events, one entry per core. If "True", then existing file will be used.
        '''
        # if type(ext_evts_data) != type(pyST.events()):
        self.ext_evts_data = ext_evts_data
        if ext_evts_data is True:
            self.ext_evts = True
            return None
        if ext_evts_data is not None:
            self.ext_evts = True
        if bool(self.L1_connectivity):
            self.ext_evts = True
        if isinstance(self.ext_evts_data, multicoreEvents) is False:
            self.ext_evts_data = multicoreEvents(self.ext_evts_data)

    def set_groups_core(self, core_cfg, **nsat_parameters):
        '''
        Set parameter group for core
        '''
        # Store parameters in a local data structure
        for k, v in nsat_parameters.items():
            if core_cfg.is_parameter_valid(k):
                setattr(core_cfg, k, v)
            else:
                print(("Parameter {0} is invalid".format(k)))

        process_Xresets(core_cfg)
        ps = core_cfg.is_parameter_undefined()
        if len(ps) > 0:
            print('Parameters undefined:')
            print(ps)
            raise RuntimeError()

    def set_L1_connectivity(self, l1_conn):
        assert type(l1_conn) == dict, "l1_conn must be a dictionary"
        for k, v in l1_conn.items():
            assert len(k) == 2, "keys must be (src_core, src_neuron)"
        self.L1_connectivity = l1_conn.copy()

    def set_groups(self):
        '''
        Set all parameter groups for all cores
        '''
        for p, core in self:
            self.set_groups_core(core)
        self.groups_set = True


if __name__ == '__main__':
    cfg = ConfigurationNSAT(N_CORES=2,
                            N_INPUTS=[0, 10],
                            N_NEURONS=[512, 100],
                            N_STATES=[4, 2],)
