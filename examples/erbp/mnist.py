#!/bin/python
#-----------------------------------------------------------------------------
# File Name : 
# Author: Emre Neftci
#
# Creation Date : Wed 07 Sep 2016 12:06:25 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np

def select_equal_n_labels(n, data, labels, classes = None, seed=None):
    if classes is None:
        classes = list(range(10))    
    n_classes = len(classes)
    n_s = np.ceil(float(n)/n_classes)
    max_i = [np.nonzero(labels==i)[0] for i in classes]
    if seed is not None:
        np.random.seed(seed)
    f = lambda x, n: np.random.random_integers(0, x-1, n)
    a = np.concatenate([max_i[i][f(len(max_i[i]), n_s)] for i in classes])
    np.random.shuffle(a)
    iv_seq = data[a]
    iv_l_seq = labels[a]
    return iv_seq, iv_l_seq



def load_mnist(data_url, labels_url, n_samples=None, nc_perlabel=1, with_labels=True, randomize= False, binary = False, seed=None, **kwargs):
    '''
    Loads MNIST data. Returns randomized samples as pairs [data vectors, data labels]
    test: use test data set. If true, the first n_sample samples are used (no randomness)
    Outputs input vector, label vector and sequence of labels.
    kwargs unsed
    '''
    print(('Loading ' + data_url))
    f_image = open(data_url  ,'rb')
    print(('Loading ' + labels_url))
    f_label = open(labels_url,'rb')

    #Extracting images
    m, Nimages, dimx, dimy =  np.fromstring(f_image.read(16),dtype='>i')
    nbyte_per_image = dimx*dimy
    iv = np.fromstring(f_image.read(Nimages*nbyte_per_image),dtype='uint8').reshape(Nimages, nbyte_per_image).astype('float')/256

    if n_samples is None:
        n_samples = Nimages

    #Extracting labels
    np.fromstring(f_label.read(8),dtype='>i') #header unused
    iv_l = np.fromstring(f_label.read(Nimages),dtype='uint8')
    
    iv_clamped = iv
    
    if randomize is False:
        iv_seq, iv_l_seq  = iv_clamped[:n_samples], iv_l[:n_samples]
    elif randomize == 'within':
        idx = list(range(n_samples))
        iv_seq, iv_l_seq  = iv_clamped[:n_samples], iv_l[:n_samples]
        np.random.shuffle(idx)
        iv_seq = iv_clamped[idx]
        iv_l_seq = iv_l[idx]
    else:
        iv_seq, iv_l_seq = select_equal_n_labels(n_samples, iv_clamped, iv_l, seed = seed)



    #expand labels
    if nc_perlabel>0:
        iv_label_seq = np.zeros([n_samples, nc_perlabel*10])
        for i in range(len(iv_l_seq)):
            s = iv_l_seq[i]*nc_perlabel
            iv_label_seq[i,s:(s+nc_perlabel)] = 1
    else:
        iv_label_seq = np.zeros([n_samples,0])

    iv_label_seq = iv_label_seq

    if not with_labels:
        iv_label_seq *= 0

    data_vectors = np.concatenate([iv_seq, iv_label_seq], axis = 1)
    return data_vectors, iv_l_seq


