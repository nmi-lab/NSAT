#!/usr/bin/env python
# ---------------------------------------------------------------------------
# File Name : default_nsat_params.py
# Author: Emre Neftci, Georgios Detorakis
#
# Creation Date : Thu 01 Sep 2016 03:20:39 PM PDT
# Last Modified : Tue Nov 29 10:00:23 PST 2016
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# ---------------------------------------------------------------------------
import numpy as np

N_STATES = 4
N_GROUPS = 8
N_LRNGROUPS = 8

rN_STATES = list(range(N_STATES))             # Number of states per neuron
rN_GROUPS = list(range(N_GROUPS))             # Number of parameters groups
rN_LRNGROUPS = list(range(N_LRNGROUPS))       # Number of learning parameters groups

# Limits
MAX = 2**15-1
MIN = -2**15+1
OFF = -16

# NSAT Dynamics parameters
A = np.array([[[OFF]*N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
sA = np.array([[[-1]*N_STATES for _ in rN_STATES] for _ in rN_GROUPS])
b = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])

# Initial conditions
Xinit = np.array([0 for _ in rN_STATES])

# Spike and reset parameters
XresetOn = np.array([[True]+[False for _ in rN_STATES[:-1]]]*N_GROUPS)
Xreset = np.array([[0]+[MAX for _ in rN_STATES[:-1]] for _ in rN_GROUPS])
XspikeResetVal = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
XspikeIncrVal = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])
Xth = np.array([MAX for _ in rN_GROUPS])
Xthlo = np.array([[MIN for _ in rN_STATES] for _ in rN_GROUPS])
Xthup = np.array([[MAX for _ in rN_STATES] for _ in rN_GROUPS])
flagXth = np.array([False for _ in rN_GROUPS])

lrnmap = np.zeros((N_GROUPS, N_STATES, ), dtype='int')

# Refractory period
t_ref = np.array([0 for _ in rN_GROUPS])

# Blankout probability
prob_syn = np.array([[15 for _ in rN_STATES] for _ in rN_GROUPS])

# Additive noise variance
sigma = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])

# Modulator state
modstate = np.array([1 for _ in rN_GROUPS])

# Plasticity Parameters
plastic = np.array([False for _ in rN_LRNGROUPS])
plasticity_en = False
stdp_en = np.array([False for _ in rN_LRNGROUPS])
Wgain = np.array([[0 for _ in rN_STATES] for _ in rN_GROUPS])

# STDP Parameters
tstdpmax = np.array([64 for _ in rN_GROUPS])
tca = [[16, 36] for _ in rN_LRNGROUPS]
hica = [[1, 0, -1] for _ in rN_LRNGROUPS]
sica = [[1, 1, 1] for _ in rN_LRNGROUPS]
tac = [[-16, -36] for _ in rN_LRNGROUPS]
hiac = [[1, 0, -1] for _ in rN_LRNGROUPS]
siac = [[-1, -1, -1] for _ in rN_LRNGROUPS]
