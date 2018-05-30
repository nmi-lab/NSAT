"""
The Bars_Stripes dataset.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import warnings
import numpy as np
from ml_funcs import data_loader

try:
    from pylearn2.datasets import dense_design_matrix
    from bars_stripes_theano import Bars_Stripes
    notheano = False
except ImportError:
    warnings.warn('Could not import pylearn2, constructing bars and stripes data will not be possible (but loading preexisting data is still OK')
    notheano = True


def bs_pickle_data(filename, data, targets, **metadata):
    '''
    Pickle data, targets and metadata.
    *data*: np.array
    *targets*: np.array
    *metadata*: keywords arguments, will be pickled as a dictionary
    '''
    import pickle
    np.save(filename+'_data.npy', data)
    np.save(filename+'_targets.npy', targets)
    fd = open(filename+'_metadata.pkl', 'wb')
    pickle.dump(metadata, fd)
    fd.close()

def bs_load_and_save(prefix = 'data/'):
    '''Load the bars and stripes data, could be used in another script to save npy files'''
    if not notheano:
        data = Bars_Stripes()
        data_train, targets_train, metadata = data_loader(data)
        data = Bars_Stripes()
        data_classify, targets_classify,  metadata  = data_loader(data)
        bs_pickle_data(prefix+'bs_train', data_train, targets_train,  **metadata)
        bs_pickle_data(prefix+'bs_classify', data_classify, targets_classify,  **metadata)
    else:
        warnings.warn('Theano (pylearn2) is not loaded')

    #Save row mnist data output here (no preprocessing at this stage)



def bs_loader_npy(dset='train', prefix = 'data/'):
    '''
    Load bars and stripes data from a numpy file, as opposed to pylearn2
    dset: 'train'/'test'
    '''
    data = np.load(prefix + 'bs_{0}_data.npy'.format(dset))
    targets = np.load(prefix + 'bs_{0}_targets.npy'.format(dset))
    metadata = np.load(prefix + 'bs_{0}_metadata.pkl'.format(dset))
    return data, targets, metadata
