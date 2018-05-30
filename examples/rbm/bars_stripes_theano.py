#!/bin/python
#-----------------------------------------------------------------------------
# File Name : bars_stripes_theano.py
# Author: Emre Neftci
#
# Creation Date : Wed 08 Jun 2016 12:34:31 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pylearn2.datasets import dense_design_matrix
import scipy.misc
import numpy as np
import numpy as N
from itertools import combinations

class Bars_Stripes(dense_design_matrix.DenseDesignMatrix):
    """
    The Bars_Stripes dataset
    """

    def __init__(self, bsdim = 4, center = False, start=None, stop=None, axes=['b', 0, 1, 'c']):
        self.args = locals()

        bsdim_tot = int(np.sum([scipy.misc.comb(bsdim, i) for i in range(bsdim+1)]))

        indices = sum([list(combinations(range(bsdim), i)) for i in range(bsdim+1)], [])

        Ahorz = [np.zeros([bsdim,bsdim]) for i in range(bsdim_tot)]
        Avert = [np.zeros([bsdim,bsdim]) for i in range(bsdim_tot)]

        #Horizontal
        for i, ind in enumerate(indices):
            if len(ind)>0:
                Ahorz[i][ind,:]=1

        #Vertical
        for i, ind in enumerate(indices):
            if len(ind)>0:
                Avert[i][:,ind]=1

        self.bars = Avert
        self.stripes = Ahorz
        self.bars_stripes = np.concatenate([self.bars,self.stripes])
        self.bars_stripes_labels = np.array([0]*len(Avert)+[1]*len(Ahorz))

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        topo_view = self.bars_stripes
        y = np.atleast_2d(self.bars_stripes_labels).T

        y_labels = len(np.unique(y))

        m, r, c = topo_view.shape
        assert r == bsdim
        assert c == bsdim
        topo_view = topo_view.reshape(m, r, c, 1)

        if center:
            topo_view -= topo_view.mean(axis=0)

        super(Bars_Stripes, self).__init__(topo_view = dimshuffle(topo_view), y=y,
                                    axes=axes, y_labels=y_labels)

        assert not N.any(N.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start


    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        return N.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        .. todo::

            WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['start'] = None
        args['stop'] = None
        return Bars_Stripes(**args)
