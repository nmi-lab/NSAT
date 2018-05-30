#!/bin/python
# ---------------------------------------------------------------------------
# File Name : utils.py
# Purpose: Python Utilities for NSAT version 2
#
# Author: Emre Neftci, Sadique Sheik, Georgios Detorakis
#
# Creation Date : 09-08-2015
# Last Modified : Tue 29 Nov 2016 09:28:27 AM PST
#
# Copyright : (c) UC Regents, Emre Neftci, Sadique Sheik, Georgios Detorakis
# Licence : GPLv2
# ---------------------------------------------------------------------------
import pylab
import numpy as np
from . import NSATlib
from .global_vars import *


def latex_vector_print(var, prefix='W'):
    s = '\\mathbf{{{prefix}}} = \\begin{{bmatrix}} \n'.format(prefix=prefix)
    s += ''.join(['{1} \\\\ \n'.format(i, v) for i, v in enumerate(var)])
    s += '\\end{bmatrix} \n'
    return s


def latex_matrix_print(var, prefix='W'):
    s = '{prefix} = \\begin{{bmatrix}} '.format(prefix=prefix)
    for j in range(var.shape[0]):
        #s += "W_{{{0}}} = ".format(j)
        s += ''.join(['{0} & '.format(v) for i, v in enumerate(var[j])])
        s = s[:-2]
        s += ' '
        s += "\\\\ \n"
    s += '\\end{bmatrix} \n'
    return s


def latex_print_group(var_group, prefix, group_names, n_states):
    if not hasattr(var_group, '__len__'):
        return
    ngroups = len(var_group)
    ndims = len(np.array(var_group[0]).shape)
    s = '\\begin{equation}'
    if ndims == 1:
        for j, var in enumerate(var_group):
            var = np.array(var)
            s += latex_vector_print(var[:n_states], prefix=prefix +
                                    '^{grp}'.format(grp=group_names[j]))
            s += '\quad '
    elif ndims == 2:
        for j, var in enumerate(var_group):
            var = np.array(var)
            s += latex_matrix_print(var[:n_states], prefix=prefix +
                                    '^{grp}'.format(grp=group_names[j]))
            s += '\quad '

    s += '\\end{equation} \n'
    return s


def ptr_wgt_table_to_dense(ptr, wgt, n_inputs, n_neurons, n_states):
    ptr = np.array(ptr)
    wgt = np.array(wgt)
    pos = 0
    n_units = n_inputs + n_neurons
    nentries = (ptr[pos]) * 4
    pos += 1
    ptrR = ptr[pos:nentries + pos].reshape(-1, 4)
    W = np.zeros([n_units, n_units, n_states], 'int')
    CW = np.zeros([n_units, n_units, n_states], 'bool')
    for w in ptrR:
        W[w[0], w[1], w[2]] = w[3]
        CW[w[0], w[1], w[2]] = True
    pos += nentries
    W = wgt[W]

    # TODO
    return W, CW


def gen_ptr_wgt_table_from_W_CW(W, CW, sharedW):
    from scipy.sparse import csr_matrix
    wgt_table = np.concatenate([[0], sharedW]).astype('int')
    ptr_table = np.zeros_like(CW, 'int')
    offset = len(wgt_table)
    for i in range(W.shape[2]):
        ww = W[:, :, i]
        cw = CW[:, :, i]
        w_shared = sharedW

        # non-shared
        c0 = np.argwhere(np.array(cw) == 1)
        c1 = ww[np.array(cw) == 1]
        ptr_table_non_shared = csr_matrix(
            (np.arange(offset, offset + len(c1)), (c0.T[0], c0.T[1])), shape=cw.shape)
        offset += len(c1)

        w_notshared = c1
        # shard
        c1 = np.argwhere(np.array(cw) == 2)
        shareds = ww[np.array(cw) == 2].flatten() + 1
        ptr_table_shared = csr_matrix(
            (shareds, (c1.T[0], c1.T[1])), shape=cw.shape)

        wgt_table = np.concatenate([wgt_table, w_notshared])
        ptr_table[:, :, i] = (ptr_table_non_shared +
                              ptr_table_shared).toarray()

    return wgt_table, ptr_table


def read_nsat_events_iterator(filename):
    data = np.fromfile(filename, 'int32')
    pos = 0
    while pos < len(data):
        t, nevs = data[pos:pos + 2]
        evs = data[pos + 2:(pos + 2 + nevs * 2)]
        pos += 2 + nevs * 2
        yield t, nevs, evs


def dummy_spike_expand(isi):
    '''
     Insert dummy spikes in between spikes which are wide apart in time.
     We store spikes in a delta tstep format where the spike time is stored
     as prev_tstep+delta_tstep.
     The delta_tstep is encoded as 8 bits. Hence if two spikes are apart by
     more than 255 tsteps, one or more dummy spike are inserted.
     This function expands an isi into multiple dummy spikes.
     *inputs*: interspike interval ISI (integer)
     *outputs*: number of events including dummy events, list of expanded ISIs.
    '''
    quotnt = isi // ISIMAX
    remain = isi % ISIMAX
    return quotnt, [ISIMAX] * quotnt


def read_states_hex(fname):
    states = []
    with open(fname, 'r') as f:
        for line in f:
            for word in line.split():
                tmp = int(word, 16)
                if tmp > 32767:
                    tmp -= 65536
                states.append(tmp)
    return np.array(states, dtype='int')


def plot_stdp_kernel(fname, tca=[16, 36], tac=[16, 36]):
    data = np.genfromtxt(fname)

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:, 0], data[:, 1], 'k', lw=2)
    ax.axhline(0, c='k', ls='--')
    ax.axvline(0, c='k', ls='--')
    ax.axvline(tca[0], c='r', ls='--')
    ax.axvline(tca[1], c='r', ls='--')
    ax.axvline(-tac[0], c='b', ls='--')
    ax.axvline(-tac[1], c='b', ls='--')


def generate_wgt_stats(cfg, fname, W, CW):
    self = cfg
    extW = NSATlib._extract_submatrix(W,
                                      start_row=0,
                                      end_row=self.N_INPUTS,
                                      start_col=self.N_INPUTS,
                                      end_col=self.N_UNITS)

    extCW = NSATlib._extract_submatrix(CW,
                                       start_row=0,
                                       end_row=self.N_INPUTS,
                                       start_col=self.N_INPUTS,
                                       end_col=self.N_UNITS)

    recW = NSATlib._extract_submatrix(W,
                                      start_row=self.N_INPUTS,
                                      end_row=self.N_UNITS,
                                      start_col=self.N_INPUTS,
                                      end_col=self.N_UNITS)

    recCW = NSATlib._extract_submatrix(CW,
                                       start_row=self.N_INPUTS,
                                       end_row=self.N_UNITS,
                                       start_col=self.N_INPUTS,
                                       end_col=self.N_UNITS)

    ext_arr = np.loadtxt(fname.stats_ext + '0.dat', 'int')

    if len(ext_arr) > 0:
        ext_stats_read = np.zeros([len(ext_arr[ext_arr[:, 0] == 0]), 4],
                                  'int')
        ext_stats_write = np.zeros([len(ext_arr[ext_arr[:, 0] == 1]), 4],
                                   'int')

        for n, ttriplet in enumerate(ext_arr[ext_arr[:, 0] == 0][:, 1:]):
            ext_stats_read[n] = [ttriplet[0],
                                 cfg.data.ext_memory_map[
                                     tuple(ttriplet[[1, 3, 2]])],
                                 ttriplet[2],
                                 ttriplet[4]]

        for n, ttriplet in enumerate(ext_arr[ext_arr[:, 0] == 1][:, 1:]):
            ext_stats_write[n] = [ttriplet[0],
                                  cfg.data.ext_memory_map[
                                      tuple(ttriplet[[1, 3, 2]])],
                                  ttriplet[2],
                                  ttriplet[4]]
    else:
        ext_stats_read = np.zeros([0, 4], 'int')
        ext_stats_write = np.zeros([0, 4], 'int')

    loc_arr = np.loadtxt(fname.stats_nsat + '0.dat', 'int')
    if len(loc_arr) > 0:
        loc_stats_read = np.zeros([len(loc_arr[loc_arr[:, 0] == 0]), 4],
                                  'int')
        loc_stats_write = np.zeros([len(loc_arr[loc_arr[:, 0] == 1]), 4],
                                   'int')

        for n, ttriplet in enumerate(loc_arr[loc_arr[:, 0] == 0][:, 1:]):
            loc_stats_read[n] = [ttriplet[0],
                                 cfg.data.loc_memory_map[
                                     tuple(ttriplet[[1, 3, 2]])],
                                 ttriplet[2],
                                 ttriplet[4]]

        for n, ttriplet in enumerate(loc_arr[loc_arr[:, 0] == 1][:, 1:]):
            loc_stats_write[n] = [ttriplet[0],
                                  cfg.data.loc_memory_map[
                                      tuple(ttriplet[[1, 3, 2]])],
                                  ttriplet[2],
                                  ttriplet[4]]
    else:
        loc_stats_read = np.zeros([0, 4], 'int')
        loc_stats_write = np.zeros([0, 4], 'int')

    zero_ext = np.zeros([np.sum(cfg.data.extCW), 4], dtype='int')
    kk = 0
    for i in range(extW.shape[0]):
        for j in range(extW.shape[2]):
            for k in range(extW.shape[1]):
                if extCW[i, k, j]:
                    zero_ext[kk] = [0,
                                    cfg.data.ext_memory_map[(i, j, k)],
                                    k,
                                    extW[i, k, j]]
                    kk += 1

    zero_rec = np.zeros([np.sum(recCW), 4], dtype='int')
    kk = 0
    for i in range(recW.shape[0]):
        for j in range(recW.shape[2]):
            for k in range(recW.shape[1]):
                if recCW[i, k, j]:
                    zero_rec[kk] = [0,
                                    cfg.data.loc_memory_map[(i, j, k)],
                                    k,
                                    recW[i, k, j]]
                    kk += 1

    stats_read_ext = np.concatenate([zero_ext, ext_stats_read])
    stats_read_rec = np.concatenate([zero_rec, loc_stats_read])
    stats_write_ext = np.concatenate([zero_ext, ext_stats_write])
    stats_write_rec = np.concatenate([zero_rec, loc_stats_write])
    return stats_read_ext, stats_read_rec, stats_write_ext, stats_write_rec


def write_spike_lists_hex(data, fname):
    if type(data) != pyST.spikes.SpikeList:
        raise TypeError("Data not a pyNCS object!")
    tmp = data.convert("[times,ids]")
    tmp = np.vstack([tmp[0], tmp[1]]).astype('int')
    spks = np.recarray(tmp.shape[1], dtype=[('t', int), ('id', int)])
    spks['t'] = tmp[0, :]
    spks['id'] = tmp[1, :]
    spks.sort()
    n = spks.size
    with open(fname, "w") as f:
        for i in range(n):
            f.write("{:08x}  {:08x}\n".format(spks[i][0], spks[i][1]))


def copy_final_weights(fname):
    '''
    Copies final weights (eg. from a previous run) to current weights,
    without explicitely reading the final weights.
    '''
    raise NotImplementedError()
    import shutil
    shutil.copy(fname.synw_final + '0.dat', fname.syn_weights)


def import_c_nsat_events(fname_train):
    f = nsat.read_from_file(fname_train.events + '_core_0.dat')
    i = 0
    tmad = []
    while i < len(f):
        t = f[i]
        i += 1
        n = f[i]
        i += 1
        if n > 0:
            ad = f[i:i + n]
            i += n
            tmad.append([np.ones(n) * t, ad])
    sl = nsat.importAER(np.fliplr(np.column_stack(tmad).T))
    return sl
