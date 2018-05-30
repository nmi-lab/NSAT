#!/bin/python
#-----------------------------------------------------------------------------
# File Name : laxesis.py
# Author: Emre Neftci
#
# Creation Date : Tue 20 Feb 2018 12:46:56 PM PST
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------


# TODO: BaseConfig comparison may not work as expected - need to compare by contents rather than pointer
# TODO: make external connections similar to intercore connections
# TODO: Get rid of gated learning in NSATlib
# TODO: external input populations only to same core
# TODO: external inputs as a core

import numpy as np
from pyNCSre import pyST
from .laxesis_neurontypes import *
import pyNSATlib as nsat
from pyNSATlib.NSATlib import check_weight_matrix, coreConfig
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW


def connections_dense_to_sparse_nonshared(W, CW):
    from scipy.sparse import csr_matrix

    # non-shared
    rows, cols = np.argwhere(np.array(CW) == 1).T
    data = W[rows, cols]
    ptr_table = csr_matrix(
        (np.arange(len(data)), (rows, cols)), shape=CW.shape, dtype='uint64')
    wgt_table = data

    return ptr_table, wgt_table


def connections_dense_to_sparse_shared(ptr_table, wgt_table):
    from scipy.sparse import csr_matrix
    # non-shared
    # Additional 1 to set default to -1
    rows, cols = np.argwhere(np.array(ptr_table) != -1).T
    data = ptr_table[rows, cols]
    ptr_table = csr_matrix((data, (rows, cols)), shape=ptr_table.shape)
    return ptr_table, wgt_table


class NSATSetup(object):

    def __init__(self, ncores=1):
        self.ncores = ncores
        self.nneurons = [0 for _ in range(ncores)]
        self.nstates = [0 for _ in range(ncores)]
        self.ninputs = [0 for _ in range(ncores)]
        # Stores neuron configurations for neuron type
        self.ntypes = [{} for _ in range(ncores)]
        self.ntypes_order = [[] for _ in range(ncores)]
        # Stores neuron configurations for neuron type
        self.ptypes = [{} for _ in range(ncores)]
        self.ptypes_order = [[] for _ in range(ncores)]
        self.populations_external = [[] for _ in range(ncores)]
        self.populations = [[] for _ in range(ncores)]
        self.connections = [[] for _ in range(ncores)]
        self.connections_external = [[] for _ in range(ncores)]
        self.L1Connections = {}
        self.connections_intercore = {}

    def normalize_n_states(self, core):
        '''
        Expands ntypes and ptypes if necessary
        '''
        ntypesc = self.ntypes[core]
        all_n_states = [ntype.n_states for ntype in list(ntypesc.values())]
        nmax = max(all_n_states)
        for k, v in list(ntypesc.items()):
            ntypesc[k] = v.expand_n_states(nmax)
        self.nstates[core] = nmax
        return nmax

    def assign_external_core(self, n, core):
        start = self.ninputs[core]
        self.ninputs[core] += n
        end = self.ninputs[core]
        return range(start, end)

    def assign_neurons_core(self, n, core):
        start = self.nneurons[core]
        self.nneurons[core] += n
        end = self.nneurons[core]
        return range(start, end)

    def create_external_population(self, n, core, name=''):
        n = int(n)
        addr = self.assign_external_core(n, core)
        pop = Population(setup=self, addr=addr, core=core, neuron_cfg=None,
                         is_external=True, is_contiguous=True, name=name)
        self.populations_external[core].append(pop)
        return pop

    def create_population(self, n, core, neuron_cfg, name=''):
        n = int(n)
        addr = self.assign_neurons_core(n, core)
        pop = Population(setup=self, addr=addr, core=core, neuron_cfg=neuron_cfg,
                         is_external=False, is_contiguous=True, name=name)
        synapse_cfg = neuron_cfg.synapse_cfg
        self.populations[core].append(pop)

        if neuron_cfg not in self.ntypes[core]:
            self.ntypes[core][neuron_cfg] = neuron_cfg
            self.ntypes_order[core].append(neuron_cfg)

        for i, s in enumerate(synapse_cfg):
            if s not in self.ptypes[core]:
                self.ptypes[core][s] = s
                self.ptypes_order[core].append(s)
        return pop

    def create_coreconfig(self, core):
        n_neurons = self.nneurons[core]
        n_inputs = self.ninputs[core]
        n_units = n_neurons + n_inputs
        n_states = self.normalize_n_states(core)
        core_cfg = coreConfig(n_states, n_neurons, n_inputs)

        list_ntypes = [self.ntypes[core][v] for v in self.ntypes_order[core]]

        for p in neuronConfig.parameter_names:
            stacked_parameters = [getattr(ntype, p) for ntype in list_ntypes]
            # temporary workaround
            stacked_parameters += [getattr(list_ntypes[-1], p)
                                   for _ in range(len(list_ntypes), 8)]
            setattr(core_cfg, p, np.array(stacked_parameters))

        list_ptypes = [self.ptypes[core][v] for v in self.ptypes_order[core]]

        for p in plasticityConfig.parameter_names:
            stacked_parameters = [getattr(ptype, p) for ptype in list_ptypes]
            # temporary workaround
            stacked_parameters += [getattr(nonplastic_ptype, p)
                                   for _ in range(len(list_ptypes), 8)]

            setattr(core_cfg, p, np.array(stacked_parameters))

        for p in self.populations[core]:
            ntypeid = self.ntypes_order[core].index(p.ntype)
            core_cfg.nmap[p.addr] = ntypeid
            for i, s in enumerate(p.ntype.synapse_cfg):
                ptypeid = self.ptypes_order[core].index(s)
                core_cfg.lrnmap[ntypeid, i] = ptypeid

        ptr_table, wgt_table = self.do_connections(core)
        core_cfg.wgt_table = wgt_table
        core_cfg.ptr_table = ptr_table

        # temporary workaround
        core_cfg.n_groups = 8  # len(self.ntypes[core])
        core_cfg.n_lrngroups = 8  # len(self.ptypes[core])

        return core_cfg

    def do_L1connections(self):
        L1Connections = {}
        for k, v in list(self.connections_intercore.items()):
            vv = v.copy()
            vv[0, :] += self.ninputs[k[0]]
            # print(self.ninputs[k[0]])
            for i in range(len(v[0, :])):
                key = (k[0], vv[0, i])
                if key not in L1Connections:
                    L1Connections[key] = ()
                L1Connections[key] += ((k[1], v[1, i]),)
        return L1Connections

    def do_connections(self, core):
        n_neurons = self.nneurons[core]
        n_states = self.normalize_n_states(core)
        n_inputs = self.ninputs[core]

        wgt_table = []

        offset = len(wgt_table)

        from scipy.sparse import csr_matrix, isspmatrix
#        ext_ptr_table = csr_matrix((n_inputs , n_neurons* n_states), dtype='uint64')
#        rec_ptr_table = csr_matrix((n_neurons, n_neurons* n_states), dtype='uint64')

        ptr_table = csr_matrix(
            (n_neurons + n_inputs, (n_neurons + n_inputs) * n_states), dtype='uint64')
        #print (n_inputs)
        #print (n_neurons)
        for cs in self.connections_external[core]:
            if cs.src_pop.is_contiguous and cs.dst_pop.is_contiguous:
                pt = cs.ptr_table.copy()
                assert isspmatrix(
                    cs.ptr_table), "Connection ptr_table is not sparse"
                pt.data = pt.data + offset
                src_offset = 0  # state offset
                dst_offset = n_inputs + \
                    (n_inputs + n_neurons) * cs.dst_state  # state offset
                ptr_table[(src_offset + cs.src_bgn): (src_offset + cs.src_end),
                          (dst_offset + cs.dst_bgn): (dst_offset + cs.dst_end)] = pt
                #print(dst_offset,cs.dst_bgn, dst_offset , cs.dst_end)
                wgt_table += list(cs.wgt_table)
                offset += len(cs.wgt_table)
            else:
                raise NotImplementedError()

        for cs in self.connections[core]:
            if cs.src_pop.is_contiguous and cs.dst_pop.is_contiguous:
                pt = cs.ptr_table.copy()
                pt.data = pt.data + offset
                src_offset = n_inputs  # state offset
                dst_offset = n_inputs + \
                    (n_inputs + n_neurons) * cs.dst_state  # state offset
                ptr_table[(src_offset + cs.src_bgn): (src_offset + cs.src_end),
                          (dst_offset + cs.dst_bgn): (dst_offset + cs.dst_end)] = pt
                #print(dst_offset,cs.dst_bgn, dst_offset , cs.dst_end)
                wgt_table += list(cs.wgt_table)
                offset += len(cs.wgt_table)

            else:
                raise NotImplementedError()

        return ptr_table, wgt_table


class Population(object):
    """
    Population is a set of neurons and corresponding plasticitys. 
    This is on top of plasticitys and is intended to be used by the user to create neural networks.
    """

    def __len__(self):
        return len(self.addr)

    def __init__(
            self,
            name='',
            setup=None,
            addr=None,
            core=None,
            neuron_cfg=None,
            synapse_cfg=None,
            is_external=False,
            is_contiguous=False):
        """
        Init a population by name and description. Population is empty.
        Name and description are used in the graph representation for
        connectivity.
        - name: string
        - description: string
        - setup: NeuroSetup to init the population
        - neuron_cfg: neuron_cfg string to init the population, e.g. 'pixel'
        """
        if name is None or name == '':
            self.name = "Core:{0}_Size:{1}_Ntype:{2}".format(
                core, len(addr), neuron_cfg)
        else:
            self.name = name
        self.addr = addr
        self.core = core
        self.is_external = is_external
        self.is_contiguous = is_contiguous
        if not is_external:
            self.ntype = neuron_cfg
            self.ptype = neuron_cfg.synapse_cfg
        self.setup = setup

    def __repr__(self):
        return self.name


def loccon2d(imsize=28, ksize=5, stride=2, init=5):
    '''
    Locally connected layer (like conv2d but without sharing)
    '''
    paddedW = np.zeros([imsize + ksize, imsize + ksize,
                        imsize, imsize], dtype='int') - 1
    paddedCW = np.zeros([imsize + ksize, imsize + ksize,
                         imsize, imsize], dtype='int') - 1
    for i in range(imsize):
        for j in range(imsize):
            paddedW[i:i + ksize, j:j + ksize, i, j] = np.random.uniform(
                low=-init, high=init, size=[ksize, ksize]).astype('int')
            paddedCW[i:i + ksize, j:j + ksize, i, j] = 1
    k2 = ksize // 2
    k2o = ksize - ksize // 2
    padded_stridedW = paddedW[k2:-k2o, k2:-k2o, ::stride, ::stride]
    padded_stridedCW = paddedCW[k2:-k2o, k2:-k2o, ::stride, ::stride]
    W = padded_stridedW.reshape(imsize * imsize, imsize * imsize // stride**2)
    CW = padded_stridedCW.reshape(
        imsize * imsize, imsize * imsize // stride**2)
    return W, CW, []


def conv2d(imsize=28, ksize=5, stride=2):
    Y = np.arange(ksize**2).reshape(ksize, ksize)
    K = np.zeros_like(Y)
    padded = np.zeros([imsize + ksize, imsize + ksize,
                       imsize, imsize], dtype='int') - 1
    for i in range(imsize):
        for j in range(imsize):
            padded[i:i + ksize, j:j + ksize, i, j] = Y
    k2 = ksize // 2
    k2o = ksize - ksize // 2
    padded_strided = padded[k2:-k2o, k2:-k2o, ::stride, ::stride]
    Wd = padded_strided.reshape(imsize * imsize, imsize * imsize // stride**2)
    return Wd, 2 * (Wd != -1), K.flatten()


def gen_filterbank(func, nfilterin, nfilterout, low=0, high=0, **kwargs):
    W = []
    CW = []
    shared = []
    offset = 0
    for j in range(nfilterin):
        W_slice = []
        CW_slice = []
        # fix!
        for i in range(nfilterout):
            w, cw, s = func(**kwargs)
            s = np.random.uniform(low=low, high=high,
                                  size=s.shape[0]).astype('int')
            #s = (i<<4) + np.arange(3**2)
            w[w != -1] += offset
            offset += len(s)
            W_slice += [w]
            CW_slice += [cw]
            shared.extend(s)
        W += [np.column_stack(W_slice)]
        CW += [np.column_stack(CW_slice)]
    ptr_table = np.row_stack(W)
    wgt_table = np.array(shared)
    return ptr_table, wgt_table


class Connection(object):

    def __init__(
            self,
            setup,
            src_pop,
            dst_pop,
            dst_state):
        self.setup = setup
        self.src_pop = src_pop
        self.dst_pop = dst_pop
        self.dst_state = dst_state

    @property
    def src_bgn(self):
        return self.src_pop.addr[0]

    @property
    def src_end(self):
        return self.src_bgn + len(self.src_pop)

    @property
    def dst_bgn(self):
        return self.dst_pop.addr[0]

    @property
    def dst_end(self):
        return self.dst_bgn + len(self.dst_pop)

    def __repr__(self):
        return "src_pop: {0} -> dst_pop: {1}".format(self.src_pop.name, self.dst_pop)

    def connect(self, ptr_table, wgt_table):
        self.ptr_table = ptr_table
        self.wgt_table = wgt_table
        if self.src_pop.is_external and (self.src_pop.core != self.dst_pop.core):
            raise NotImplementedError(
                'Currently, external populations ans destination populations must be on the same core.')

        elif not self.src_pop.is_external and (self.src_pop.core != self.dst_pop.core):
            # merge following two
            self.is_external = True
            # TODO: make only L1 connections that exist
            popin = self.setup.create_external_population(
                len(self.src_pop), self.dst_pop.core)
            key = (self.src_pop.core, self.dst_pop.core)
            L1 = self.setup.connections_intercore
            if key not in L1:
                L1[key] = np.array([[], []], 'int')
            value = L1[key]
            L1[key] = np.column_stack(
                [value, np.array([self.src_pop.addr, popin.addr], 'int')])
            self.src_pop = popin
            self.setup.connections_external[self.dst_pop.core].append(self)
        elif self.src_pop.is_external:
            self.setup.connections_external[self.dst_pop.core].append(self)
        else:
            self.setup.connections[self.dst_pop.core].append(self)
        return ptr_table, wgt_table

    def connect_one2one(self, weight):
        nsrc = len(self.src_pop)
        ndst = len(self.dst_pop)
        assert nsrc == ndst
        CW = np.eye(nsrc, dtype='bool')
        W = np.eye(nsrc, dtype='int') * weight
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return self.connect(p, w)

    def connect_shuffle(self, iterations=2000):
        nsrc = len(self.src_pop)
        ndst = len(self.dst_pop)
        a = np.zeros([nsrc * iterations, ndst], dtype='int')
        a[:2 * iterations, :] = -1
        a[2 * iterations:4 * iterations, :] = 1
        list(map(np.random.shuffle, a.T))
        W = a.reshape(nsrc, iterations, ndst).sum(axis=1)
        CW = W != 0
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return self.connect(p, w)

    def connect_random_uniform(self, low, high, p=1.):
        nsrc = len(self.src_pop)
        ndst = len(self.dst_pop)
        if p < 1:
            CW = np.random.randn(nsrc, ndst) < p
        else:
            CW = np.ones([nsrc, ndst], 'bool')
        W = np.zeros([nsrc, ndst], 'int')
        W[CW] = np.random.uniform(low=low, high=high, size=[
                                  CW.sum()]).astype('int')
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return self.connect(p, w)

    def connect_conv2dbank(self, imsize, nchannels, nfeatures, stride, ksize, **kwargs):
        '''
        Consider separating
        '''
        nsrc = len(self.src_pop)
        ndst = len(self.dst_pop)
        ptr_table_dense, wgt_table_dense = gen_filterbank(
            conv2d,
            nchannels,
            nfeatures,
            imsize=imsize,
            ksize=ksize,
            stride=stride,
            **kwargs)
        assert nsrc == ptr_table_dense.shape[
            0], "number of neurons at source must be {0}, is {1}".format(ptr_table.shape[0], nsrc)
        assert ndst == ptr_table_dense.shape[
            1], "number of neurons at target must be {0}, is {1}".format(ptr_table.shape[1], ndst)
        ptr_table, wgt_table = connections_dense_to_sparse_shared(
            ptr_table_dense, wgt_table_dense)
        return self.connect(ptr_table, wgt_table)

    def connect_loccon2dbank(self, imsize, nchannels, nfeatures, stride, ksize):
        '''
        Consider separating
        '''

        nsrc = len(self.src_pop)
        ndst = len(self.dst_pop)
        W, CW = gen_filterbank(loccon2d, nchannels,
                               nfeatures, imsize=imsize, ksize=ksize)
        assert nsrc == ptr_table.shape[0]
        assert ndst == ptr_table.shape[1]
        ptr_table, wgt_table = connections_dense_to_sparse_nonshared(W, CW)
        return self.connect(ptr_table, wgt_table)


if __name__ == "__main__":
    modSTDP_ptype = plasticityConfig('modSTDP')
    mod2STDP_ptype = plasticityConfig('mod2STDP')
    neurone_ntype = neuronConfig(2, name='neurone', synapse_cfg=[
                                 nonplastic_ptype, modSTDP_ptype])
    neuroni_ntype = neuronConfig(2, name='neuroni', synapse_cfg=[
                                 mod2STDP_ptype, nonplastic_ptype])
    neuronj_ntype = neuronConfig(2, name='neurone', synapse_cfg=[
                                 nonplastic_ptype, modSTDP_ptype])

    # test population creatin and assignment
    setup = NSATSetup(ncores=1)
    inp1 = setup.create_external_population(20, 0)
    pop1 = setup.create_population(
        n=20,
        core=0,
        neuron_cfg=neurone_ntype)
    pop2 = setup.create_population(
        n=20,
        core=0,
        neuron_cfg=neuroni_ntype)
    pop3 = setup.create_population(
        n=40,
        core=0,
        neuron_cfg=neuronj_ntype)
    c = Connection(setup, inp1, pop2, 0)
    c.connect_one2one(5)

    core_cef = setup.create_coreconfig(0)
