#!/bin/python
# -----------------------------------------------------------------------------
# File Name : laxesis_neurontypes.py
# Author: Emre Neftci
#
# Creation Date : Tue 20 Feb 2018 04:27:30 PM PST
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------
import numpy as np

XMAX = MAX = 2**15 - 1
XMIN = -2**15
MIN = -2**15 + 1
OFF = -16
WMAX = 128
N_GROUPS = 8
N_GROUPS_C = 8
N_LRNGROUPS = 8
N_LRNGROUPS_C = 8
CHANNEL_OFFSET = 18
ADDR_MASK = 2**CHANNEL_OFFSET - 1
N_TOT = 2048
OFF = -16
TSTDPMAX = 1023
ISIMAX = 255
rN_GROUPS = list(range(N_GROUPS))     # num of parameters groups
rN_LRNGROUPS = list(range(N_LRNGROUPS))  # #learning params groups


class BaseCoreConfig(object):
    parameter_names = []

    def __init__(self, name=''):
        self.name = name
        for i in self.parameter_names:
            setattr(self, i, None)
        self.gen_cfg()

    def __name__(self):
        return self.name

    def __repr__(self):
        return self.name

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

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def gen_cfg(self):
        pass


class neuronConfig(BaseCoreConfig):
    parameter_names = [
        'A',
        'b',
        'flagXth',
        'modstate',
        'prob_syn',
        'sA',
        'sigma',
        't_ref',
        'Wgain',
        'XresetOn',
        'XspikeIncrVal',
        'Xreset',
        'Xth',
        'Xthlo',
        'Xthup',
        'gate_lower',
        'gate_upper',
        'learn_period',
        'learn_burnin',
        'plasticity_type']

    def __init__(self, n_states=4, synapse_cfg=None, name=''):
        self.n_states = n_states
        assert len(synapse_cfg) == n_states
        self.synapse_cfg = synapse_cfg
        super(neuronConfig, self).__init__(name=name)

    def gen_cfg(cfg):
        n_states = cfg.n_states
        rn_states = list(range(n_states))     # number of states per neuron

        # NSAT Dynamics parameters
        cfg.A = np.array([[OFF] * n_states for _ in rn_states])
        cfg.sA = np.array([[-1] * n_states for _ in rn_states])
        cfg.b = np.array([0 for _ in rn_states])

        # Spike and reset parameters
        cfg.XresetOn = np.array([True] + [False for _ in rn_states[:-1]])
        cfg.Xreset = np.array([0 for _ in rn_states])
        cfg.XspikeIncrVal = np.array([0 for _ in rn_states])
        cfg.Xth = np.array(MAX)
        cfg.Xthlo = np.array([MIN for _ in rn_states])
        cfg.Xthup = np.array([MAX for _ in rn_states])
        cfg.flagXth = np.array(False)

        # Neuron and Learning Maps
        # Refractory period
        cfg.t_ref = np.array(0)

        # Gate learning parameters
        cfg.gate_lower = np.array(MIN)
        cfg.gate_upper = np.array(MAX)
        cfg.learn_period = np.array(0)
        cfg.learn_burnin = np.array(0)

        # Blankout probability
        cfg.prob_syn = np.array([15 for _ in rn_states])

        # Additive noise variance
        cfg.sigma = np.array([0 for _ in rn_states])

        # Modulator state
        cfg.modstate = np.array(1)

        cfg.Wgain = np.array([0 for _ in rn_states])

        cfg.plasticity_type = None

    def expand_n_states(self, n_states):
        # Not implemented yet.
        import copy
        return copy.copy(self)


class plasticityConfig(BaseCoreConfig):
    parameter_names = [
        'hiac',
        'hica',
        'plastic',
        'siac',
        'sica',
        'slac',
        'slca',
        'stdp_en',
        'is_stdp_exp_on',
        'tac',
        'tca',
        'tstdp',
        'is_rr_on',
        'rr_num_bits']

    def gen_cfg(cfg):
        # Plasticity Parameters
        cfg.plastic = np.array(False)
        cfg.stdp_en = np.array(False)
        cfg.is_stdp_exp_on = np.array(False)

        # STDP Parameters
        cfg.tstdp = np.array(64, 'int')
        cfg.tca = np.array([16, 36], 'int')
        cfg.hica = np.array([1, 0, -1], 'int')
        cfg.sica = np.array([1, 1, 1], 'int')
        cfg.slca = np.array([16, 16, 16], 'int')
        cfg.tac = np.array([-16, -36], 'int')
        cfg.hiac = np.array([1, 0, -1], 'int')
        cfg.siac = np.array([-1, -1, -1], 'int')
        cfg.slac = np.array([16, 16, 16], 'int')

        cfg.is_rr_on = np.array(False, 'bool')
        cfg.rr_num_bits = np.array(0, 'int')

        return cfg


# ==========
nonplastic_ptype = plasticityConfig(name='NoPlasticity')
nonplastic_ptype.plastic = False

# ==========
erbp_ptype = plasticityConfig(name='eRBP')
erbp_ptype.tstdp = 1000
erbp_ptype.plastic = True
erbp_ptype.is_rr_on = True
erbp_ptype.rr_num_bits = 10
erbp_ptype.plastic = True
erbp_ptype.tca = np.array([16, 36], 'int')
erbp_ptype.hiac = [-7, OFF, OFF]
erbp_ptype.hica = np.array([OFF, OFF, OFF], 'int')
erbp_ptype.sica = np.array([1, 1, 1], 'int')
erbp_ptype.slca = np.array([16, 16, 16], 'int')
erbp_ptype.tac = np.array([-16, -36], 'int')
erbp_ptype.siac = np.array([-1, -1, -1], 'int')
erbp_ptype.slac = np.array([16, 16, 16], 'int')

# ==========
erbp_ptype_hiRR = erbp_ptype.copy()
erbp_ptype_hiRR.rr_num_bits = 12
erbp_ptype_hiRR.name += '_hiRR'

# ==========
erbp_ptype_loRR = erbp_ptype.copy()
erbp_ptype_loRR.rr_num_bits = 11
erbp_ptype_loRR.name += '_loRR'

# ==========
# ==========
erf_ntype = neuronConfig(2, [erbp_ptype, nonplastic_ptype], name='erfneuron')
erf_ntype.t_ref = 39
erf_ntype.prob_syn[0] = 9
erf_ntype.A = [[-7,  OFF], [OFF,   -6]]
erf_ntype.sA = [[-1, 1], [1, -1]]
erf_ntype.b = [0, 0]
erf_ntype.Xreset = [0, 0]
erf_ntype.XresetOn = [False, False]
erf_ntype.sigma = [0, 0]
erf_ntype.Xth = 500
erf_ntype.modstate = 1
erf_ntype.Wgain[0] = 3
erf_ntype.Wgain[1] = 4
erf_ntype.gate_upper = 2560
erf_ntype.gate_lower = -2560
erf_ntype.learn_burnin = 1500 * .25
erf_ntype.learn_period = 1500

erf4s_ntype = neuronConfig(
    4, [nonplastic_ptype, nonplastic_ptype, erbp_ptype, nonplastic_ptype], name='erf4sneuron')
erf4s_ntype.t_ref = 39
erf4s_ntype.prob_syn[0] = 9
erf4s_ntype.A = np.array([[0,  OFF, OFF, OFF], [
                         OFF,   -2, OFF, OFF], [0,  OFF, -6, OFF], [OFF,  OFF, OFF, OFF]])
erf4s_ntype.sA = np.array(
    [[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
erf4s_ntype.b = [0, 0, 0, 0]
erf4s_ntype.Xreset = [0, 0, 0, 0]
erf4s_ntype.XresetOn = [False, False, False, False]
erf4s_ntype.sigma = [0, 0, 0, 0]
erf4s_ntype.Xth = 500
erf4s_ntype.modstate = 1
erf4s_ntype.Wgain[0] = 3
erf4s_ntype.Wgain[1] = 4
erf4s_ntype.Wgain[2] = 4
erf4s_ntype.gate_upper = 2560
erf4s_ntype.gate_lower = -2560
erf4s_ntype.learn_burnin = 1500 * .25
erf4s_ntype.learn_period = 1500

output_ntype = erf_ntype.copy()
output_ntype.plasticity_type = [erbp_ptype, nonplastic_ptype]

error_ntype = neuronConfig(
    2, [nonplastic_ptype, nonplastic_ptype], name='errorneuron')
error_ntype.Xreset = [0, 0]
error_ntype.b = [0, 0]
error_ntype.t_ref = 0
error_ntype.sA = [[-1, 1], [1, -1]]
error_ntype.A = [[OFF, OFF], [OFF, OFF]]
error_ntype.XspikeIncrVal = [-1025, 0]
error_ntype.XresetOn = [False, False]
error_ntype.Xth = 1025
error_ntype.Xthlo = [0, MIN]
error_ntype.modstate = 1
error_ntype.Wgain[0] = 4

error4s_ntype = neuronConfig(4, [nonplastic_ptype, nonplastic_ptype,
                                 nonplastic_ptype, nonplastic_ptype], name='error4sneuron')
error4s_ntype.Xreset = [0, 0, 0, 0]
error4s_ntype.b = [0, 0, 0, 0]
error4s_ntype.t_ref = 0
error4s_ntype.sA = np.array(
    [[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
error4s_ntype.A = np.array([[OFF, OFF, OFF, OFF], [OFF, OFF, OFF, OFF], [
                           OFF, OFF, OFF, OFF], [OFF, OFF, OFF, OFF]])
error4s_ntype.XspikeIncrVal = np.array([-1025, 0, 0, 0])
error4s_ntype.XresetOn = np.array([False, False, False, False])
error4s_ntype.Xth = 1025
error4s_ntype.Xthlo = [0, MIN, MIN, MIN]
error4s_ntype.modstate = 1
error4s_ntype.Wgain[0] = 4
