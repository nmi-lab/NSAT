#!/bin/python
# -----------------------------------------------------------------------------
# File Name : laxesis.py
# Author: Emre Neftci
#
# Creation Date : Tue 20 Feb 2018 12:46:56 PM PST
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------


# TODO: BaseConfig comparison may not work as expected - need to compare by contents rather than pointer
# TODO: make external connections similar to intercore connections
# TODO: Get rid of gated learning in NSATlib
# TODO: external input populations only to same core
# TODO: external inputs as a core

import numpy as np
from pyNCSre import pyST
from .laxesis_neurontypes import *
import pyNSATlib as nsat
from pyNSATlib.NSATlib import check_weight_matrix, coreConfig, ConfigurationNSAT
from pyNSATlib.utils import gen_ptr_wgt_table_from_W_CW
#from .neurostar import *
import numpy as np
import igraph

POPCOUNTER = 0
CTYPE_LOC = 0
CTYPE_EXT = 1
CTYPE_GLO = 2

# Helper Functions


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

# Base Class


class MulticoreResourceManager(object):
    '''
    A setup class for multicore hardware
    '''

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
        self.connections_intercore = {}

    def create_coreconfig(self, core):
        NotImplementedError('Abstract method, implement')

    def do_connections(self, core):
        res = []
        for core in range(self.ncores):
            res.append(self.do_L0connections(core))
        return res

    def do_L0connections(self, core):
        NotImplementedError('Abstract method, implement')

    def do_L1connections(self):
        NotImplementedError('Abstract method, implement')

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

    def add_external_population(self, n, core, name=''):
        n = int(n)
        addr = self.assign_external_core(n, core)
        pop = Population(setup=self, addr=addr, core=core, neuron_cfg=None,
                         is_external=True, is_contiguous=True, name=name)
        self.populations_external[core].append(pop)
        return pop

    def add_population(self, pop):
        n = len(pop)
        core = pop.core
        neuron_cfg = pop.ntype
        synapse_cfg = pop.ptype
        name = pop.name
        pop.is_contiguous = True
        if pop.is_external:
            pop.addr = self.assign_external_core(n, core)
            self.populations_external[core].append(pop)
        else:
            pop.addr = self.assign_neurons_core(n, core)
            self.populations[core].append(pop)

            # TODO, add to nsat_setup
            if neuron_cfg not in self.ntypes[core]:
                self.ntypes[core][neuron_cfg] = neuron_cfg
                self.ntypes_order[core].append(neuron_cfg)

            for i, s in enumerate(synapse_cfg):
                if s not in self.ptypes[core]:
                    self.ptypes[core][s] = s
                    self.ptypes_order[core].append(s)

        return pop

    def add_connection(self, src_pop, tgt_pop, dst_state, connection_function):
        Connection(
            setup=self,
            src_pop=src_pop,
            tgt_pop=tgt_pop,
            dst_state=dst_state,
            connection_function=connection_function)


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
            n=0,
            addr=None,
            core=None,
            neuron_cfg=None,
            synapse_cfg=[],
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
        global POPCOUNTER
        self.id = POPCOUNTER
        POPCOUNTER += 1

        if name is None or name == '':
            self.name = "Pop{0}".format(self.id)
        else:
            self.name = str(name)
        if addr is None:
            self.addr = range(n)
        else:
            self.addr = addr
        self.core = core
        self.is_external = is_external
        self.is_contiguous = is_contiguous
        if neuron_cfg is None:
            self.ntype = None
            self.ptype = []
        elif not is_external:
            self.ntype = neuron_cfg
            self.ptype = neuron_cfg.synapse_cfg

    def copy(self):
        return Population(
            name=self.name + '_copy',
            n=len(self),
            addr=self.addr,
            core=self.core,
            neuron_cfg=self.ntype,
            synapse_cfg=self.ptype,
            is_external=self.is_external,
            is_contiguous=self.is_contiguous)

    def gen_ext_copy(self, core_ext):
        return Population(
            name=self.name + '_ext',
            n=len(self),
            addr=None,
            core=core_ext,
            neuron_cfg=None,
            synapse_cfg=[],
            is_external=True,
            is_contiguous=False)

    def partition(self, nparts=1):
        pop_list = []
        assert self.n // nparts == int(self.n / parts)
        for i in range(nparts):
            pop_list.append(Population(name=self.name + '_part{0}'.format(i),
                                       n=self.n // nparts,
                                       neuron_cfg=self.ntype,
                                       is_external=self.is_external))
        return pop_list

    def __repr__(self):
        return self.name + "({0},{1})".format(len(self), self.core)


# Connection function factories
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


def connect_all2all(weight=0):
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        CW = np.ones([nsrc, ndst], dtype='bool')
        W = np.ones([nsrc, ndst], dtype='int') * weight
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return p, w
    return func


def connect_one2one(weight=1):
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        assert nsrc == ndst
        CW = np.eye(nsrc, dtype='bool')
        W = np.eye(nsrc, dtype='int') * weight
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return p, w
    return func


def connect_shuffle(iterations=2000):
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        a = np.zeros([nsrc * iterations, ndst], dtype='int')
        a[:2 * iterations, :] = -1
        a[2 * iterations:4 * iterations, :] = 1
        list(map(np.random.shuffle, a.T))
        W = a.reshape(nsrc, iterations, ndst).sum(axis=1)
        CW = W != 0
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return p, w
    return func


def connect_random_uniform(low, high, prob=1.):
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        if prob < 1:
            CW = np.random.randn(nsrc, ndst) < p
        else:
            CW = np.ones([nsrc, ndst], 'bool')
        W = np.zeros([nsrc, ndst], 'int')
        W[CW] = np.random.uniform(low=low, high=high, size=[
                                  CW.sum()]).astype('int')
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return p, w
    return func


def connect_conv2dbank(imsize, nchannels, nfeatures, stride, ksize, **kwargs):
    '''
    Consider separating
    '''
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        ptr_table_dense, wgt_table_dense = gen_filterbank(
            conv2d,
            nchannels,
            nfeatures,
            imsize=imsize,
            ksize=ksize,
            stride=stride,
            **kwargs)
        assert nsrc == ptr_table_dense.shape[
            0], "number of neurons at source must be {0}, is {1}".format(ptr_table_dense.shape[0], nsrc)
        assert ndst == ptr_table_dense.shape[
            1], "number of neurons at target must be {0}, is {1}".format(ptr_table_dense.shape[1], ndst)
        ptr_table, wgt_table = connections_dense_to_sparse_shared(
            ptr_table_dense, wgt_table_dense)
        return ptr_table, wgt_table
    return func


def connect_loccon2dbank(imsize, nchannels, nfeatures, stride, ksize):
    '''
    Consider separating
    '''
    def func(src_pop, tgt_pop):
        nsrc = len(src_pop)
        ndst = len(tgt_pop)
        W, CW = gen_filterbank(loccon2d, nchannels,
                               nfeatures, imsize=imsize, ksize=ksize)
        assert nsrc == ptr_table.shape[0]
        assert ndst == ptr_table.shape[1]
        p, w = connections_dense_to_sparse_nonshared(W, CW)
        return p, w
    return func


class Connection(object):
    '''
    A placeholder class for representing connections with a sparse pointer, weight table structure.
    '''

    def __init__(
            self,
            setup,
            src_pop,
            tgt_pop,
            dst_state,
            connection_function):
        self.setup = setup
        self.src_pop = src_pop
        self.tgt_pop = tgt_pop
        self.dst_state = dst_state
        self.connection_type = -1  # CONN_TYPE_LOC := 0 is intracore
        self.cx_func = connection_function
        # CONN_TYPE_LOC := 0 is intracore
        # CONN_TYPE_EXT := 1 is external
        # CONN_TYPE_GLO := 2 is intercore
        self.__connect(setup)

    @property
    def src_bgn(self):
        return self.src_pop.addr[0]

    @property
    def src_end(self):
        return self.src_bgn + len(self.src_pop)

    @property
    def dst_bgn(self):
        return self.tgt_pop.addr[0]

    @property
    def dst_end(self):
        return self.dst_bgn + len(self.tgt_pop)

    def __repr__(self):
        return "src_pop: {0} -> tgt_pop: {1} :: state {2}".format(self.src_pop, self.tgt_pop, self.dst_state)

    def __connect(self, setup):
        if hasattr(self.cx_func, '__call__'):
            ptr_table, wgt_table = self.cx_func(self.src_pop, self.tgt_pop)
        else:
            ptr_table, wgt_table = self.cx_func
        self.ptr_table = ptr_table
        self.wgt_table = wgt_table
        if self.src_pop.is_external and (self.src_pop.core != self.tgt_pop.core):
            raise NotImplementedError(
                'Currently, external populations and destination populations must be on the same core.')

        elif not self.src_pop.is_external and (self.src_pop.core != self.tgt_pop.core):
            # merge following two
            self.connection_type = CTYPE_GLO
            key = (self.src_pop.core, self.tgt_pop.core)
            # TODO: make only L1 connections that exist
            # popin = self.setup.create_external_population(len(self.src_pop),
            #                                              self.tgt_pop.core,
            #                                              self.src_pop.name+"_EXT")
            L1 = self.setup.connections_intercore
            if key not in L1:
                L1[key] = np.array([[], []], 'int')
            value = L1[key]
            L1[key] = np.column_stack(
                [value, np.array([self.src_pop.addr, self.tgt_pop.addr], 'int')])
        elif self.src_pop.is_external:
            self.connection_type = CTYPE_EXT
            setup.connections_external[self.tgt_pop.core].append(self)
        else:
            self.connection_type = CTYPE_LOC
            setup.connections[self.tgt_pop.core].append(self)
        return ptr_table, wgt_table

# Network Base Class Move to network.py


class Graph(igraph.Graph):

    def add_vertex_and_return(self, name=None, **kwds):
        """add_vertex_return(name=None, **kwds)

        Like add_vertex but returns the created vertex instead of result
        """
        if not kwds and name is None:
            return self.add_vertices(1)

        vid = self.vcount()
        result = self.add_vertices(1)
        vertex = self.vs[vid]
        for key, value in kwds.items():
            vertex[key] = value
        if name is not None:
            vertex["name"] = name
        return vertex

    def add_edge_and_return(self, source, target, **kwds):
        """add_edge(source, target, **kwds)

        Same as add_Edge but return the created edge insteast of the result
        """
        if not kwds:
            return self.add_edges([(source, target)])

        eid = self.ecount()
        result = self.add_edges([(source, target)])
        edge = self.es[eid]
        for key, value in kwds.items():
            edge[key] = value
        return edge

    def draw(self):
        pg = self
        layout = pg.layout("kk")
        pg.vs["label"] = pg.vs["name"]
        igraph.plot(pg, layout=layout, bbox=(1024, 1024), margin=20).show()


class LogicalGraphSetup(object):
    '''
    A class for creating the network graph. Currently the main gateway into
    creating neural networks with NSAT with automatic resource allocation and
    core-level distribution
    '''

    def __init__(self):
        # logical graph
        self.g = Graph(directed=True,
                       vertex_attrs={'core': -1,
                                     'name': '',
                                     'id': -1,
                                     'data': {}},
                       edge_attrs={'core_tgt': -1,
                                   'name': '',
                                   'core_src': -1,
                                   'cx': None,
                                   'ctype': -1})

        # placement graph

    def create_population(self, pop):
        if isinstance(pop, Population):
            self.__create_node(
                name=pop.name,
                core=pop.core,
                is_external=pop.is_external,
                id=pop.id,
                data=pop)
        else:
            for p in pop:
                self.__create_node(
                    name=p.name,
                    core=p.core,
                    is_external=p.is_external,
                    id=p.id,
                    data=p)

        return pop

    def create_connection(self, src, tgt, dst_state, cx=connect_all2all()):
        if isinstance(src, Population):
            src = [src]
        if isinstance(tgt, Population):
            tgt = [tgt]

        v = []
        for src_ in src:
            for tgt_ in tgt:
                src_v = self.g.vs.find(id=src_.id)
                tgt_v = self.g.vs.find(id=tgt_.id)
                v.append(self.__create_edge(src_v, tgt_v, core_src=src_.core,
                                            core_tgt=tgt_.core, cx=cx,
                                            dst_state=dst_state))

        return v

    def __create_node(self, name='', **data):
        core = data['core']
        pop_vertex = self.g.add_vertex_and_return(name, **data)
        #core_vertices = self.pg.vs.select(core=core)
        # if len(core_vertices)==0:
        #    print('Adding core {0}'.format(core))
        #    self.pg.add_vertex(name=pop_vertex['core'], core=core, node_list = {pop_vertex:pop_vertex.index})
        # elif len(core_vertices)==1:
        #    pop_list = core_vertices[0]['node_list']
        #    pop_list[pop_vertex] = pop_vertex.index
        #    core_vertices[0].update_attributes({'node_list':pop_list})
        # else:
        #    raise ValueError('Placement graph has more than one node for core {0}'.format(core))

    def __create_edge(self, src, tgt, dst_state, **data):
        ctype = -1
        e = self.g.add_edge_and_return(
            src, tgt, dst_state=dst_state, ctype=ctype, **data)
        #src_core = src['core']
        #tgt_core = tgt['core']
        # self.pg.add_edge(src_core,
        #                 tgt_core,
        #                 src_pop = src,
        #                 tgt_pop = tgt,
        #                 edge = e)

    def generate_core_assigned_graph(self):
        physical_graph = Graph(directed=True, vertex_attrs={
                               'core': -1, 'name': '', 'id': -1, 'data': {}})
        for src_v in self.g.vs:
            physical_graph.add_vertex_and_return(**src_v.attributes())

        for src_v in self.g.vs:
            src_v_out = self.g.es.select(_source=src_v.index)
            for core in np.unique(src_v_out['core_tgt']).astype('int'):
                es = src_v_out.select(core_tgt=core)
                if core == src_v['core']:
                    for ee in es:
                        tgt_v = physical_graph.vs.find(ee.target)
                        new_local_edge = physical_graph.add_edge_and_return(
                            src_v, tgt_v, **ee.attributes())
                        new_local_edge.update_attributes({'ctype': 0})
                else:
                    if len(es) > 0:
                        pop_ext = src_v['data'].gen_ext_copy(core)
                        ext_v = physical_graph.add_vertex_and_return(name=src_v['name'] + '_ext_core{0}'.format(
                            core), id=pop_ext.id, is_external=True, data=pop_ext, core=core)
                        print(ext_v['name'])
                        new_intercore_edge = physical_graph.add_edge_and_return(
                            src_v, ext_v, core_src=src_v['core'], core_tgt=core, cx=connect_one2one(), ctype=2)
                        new_intercore_edge.update_attributes({'ctype': 2})
                        for ee in es:
                            tgt_v = physical_graph.vs.find(ee.target)
                            new_ext_edge = physical_graph.add_edge_and_return(
                                ext_v, tgt_v, **ee.attributes())
                            new_ext_edge.update_attributes({'ctype': 1})

        return physical_graph

    def generate_multicore_setup(self, setup_type=MulticoreResourceManager):
        graph = self.generate_core_assigned_graph()
        cores = np.unique(graph.vs['core'])
        ncores = np.max(cores) + 1
        mc_setup = setup_type(ncores=ncores)
        for v in graph.vs:
            # TODO: ext pop workaround
            if v['core'] != -1:
                mc_setup.add_population(v['data'])

        for ee in graph.es:
            src_v = graph.vs.find(ee.source)
            # TODO: ext pop workaround
            if src_v['core'] != -1:
                tgt_v = graph.vs.find(ee.target)
                mc_setup.add_connection(
                    src_pop=src_v['data'],
                    tgt_pop=tgt_v['data'],
                    dst_state=ee['dst_state'],
                    connection_function=ee['cx'])
        return mc_setup


if __name__ == "__main__":
    # test population creatin and assignment
    setup = LogicalGraphSetup()
    inp1 = Population(n=20,
                      core=-1,
                      is_external=True,
                      name='inp0')
    pop1 = Population(n=20,
                      core=0,
                      neuron_cfg=None)
    pop2 = Population(n=20,
                      core=1,
                      neuron_cfg=None)
    pop3 = Population(n=20,
                      core=1,
                      neuron_cfg=None)

    c1 = connect_one2one(5)

    setup.create_population(inp1)
    setup.create_population(pop1)
    setup.create_population(pop2)
    setup.create_population(pop3)
    setup.create_connection(inp1, pop1, dst_state=1, cx=c1)
    setup.create_connection(pop1, pop2, dst_state=1, cx=c1)
    setup.create_connection(pop1, pop1, dst_state=1, cx=c1)
    setup.create_connection(pop1, pop3, dst_state=1, cx=c1)

    nsat_setup = setup.generate_multicore_setup()


class NSATSetup(MulticoreResourceManager):
    '''
    A class for NSAT Resource Management
    '''

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

        ptr_table, wgt_table = self.do_L0connections(core)
        core_cfg.wgt_table = wgt_table
        core_cfg.ptr_table = ptr_table

        # temporary workaround
        core_cfg.n_groups = 8  # len(self.ntypes[core])
        core_cfg.n_lrngroups = 8  # len(self.ptypes[core])

        return core_cfg

    def create_configuration_nsat(setup, sim_ticks, **kwargs):
        # TODO: fold following in NSATSetup
        cfg = ConfigurationNSAT(
            sim_ticks=sim_ticks,
            N_CORES=setup.ncores,
            N_NEURONS=setup.nneurons,
            N_INPUTS=setup.ninputs,
            N_STATES=setup.nstates,
            bm_rng=True,
            **kwargs)
        for i in range(setup.ncores):
            cfg.core_cfgs[i] = setup.create_coreconfig(i)
        cfg.L1_connectivity = setup.do_L1connections()
        return cfg

    def do_L1connections(self):
        L1Connections = {}
        for k, v in list(self.connections_intercore.items()):
            src_core_id = k[0]
            dst_core_id = k[1]
            vv = v.copy()
            # src_neuron_id
            vv[0, :] += self.ninputs[src_core_id]
            # print(self.ninputs[k[0]])
            for i in range(len(v[0, :])):
                key = (src_core_id, vv[0, i])
                if key not in L1Connections:
                    L1Connections[key] = ()
                L1Connections[key] += ((dst_core_id, v[1, i]),)
        return L1Connections

    def do_L0connections(self, core):
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
            if cs.src_pop.is_contiguous and cs.tgt_pop.is_contiguous:
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
            if cs.src_pop.is_contiguous and cs.tgt_pop.is_contiguous:
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
    net_graph = LogicalGraphSetup()
    inp1 = Population(n=20,
                      core=-1,
                      is_external=True,
                      name='inp0')
    pop1 = Population(n=20,
                      core=0,
                      neuron_cfg=neurone_ntype)
    pop2 = Population(n=20,
                      core=1,
                      neuron_cfg=neurone_ntype)
    pop3 = Population(n=20,
                      core=1,
                      neuron_cfg=neurone_ntype)

    c1 = connect_one2one(5)

    net_graph.create_population(inp1)
    net_graph.create_population(pop1)
    net_graph.create_population(pop2)
    net_graph.create_population(pop3)
    net_graph.create_connection(inp1, pop1, dst_state=1, cx=c1)
    net_graph.create_connection(pop1, pop2, dst_state=1, cx=c1)
    net_graph.create_connection(pop1, pop1, dst_state=1, cx=c1)
    net_graph.create_connection(pop1, pop3, dst_state=1, cx=c1)

    pg = net_graph.generate_core_assigned_graph()

    layout = pg.layout("kk")
    pg.vs["label"] = pg.vs["name"]
    igraph.plot(pg, layout=layout, bbox=(300, 300), margin=20).show()

    setup = net_graph.generate_multicore_setup(NSATSetup)
    core_cef = setup.create_coreconfig(0)
