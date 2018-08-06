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



def load_mnist(data_url, labels_url, n_samples=None, nc_perlabel=1, with_labels=True, randomize= False, binary = False, seed=None, labels_offset = 0, **kwargs):
    '''
    Loads MNIST data. Returns randomized samples as pairs [data vectors, data labels]
    test: use test data set. If true, the first n_sample samples are used (no randomness)
    Outputs input vector, label vector and sequence of labels.
    kwargs unsed
    '''
    import gzip, pickle

    print('Loading ' + data_url)
    f_image = open(data_url  ,'rb')
    print('Loading ' + labels_url)
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

    data_vectors = np.concatenate([iv_seq, np.zeros([iv_seq.shape[0],labels_offset]), iv_label_seq], axis = 1)
    return data_vectors, iv_l_seq

def SimSpikingStimulus(stim, time = 1000, t_sim = None, with_labels = True):
    '''
    Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
    *poisson*: integer, output is a poisson process with mean
    data/poisson, scaled by *poisson*.
    '''
    from pyNCSre import pyST
    n = np.shape(stim)[1]
    nc = 10
    stim[stim<=0] = 1e-5
    SL = pyST.SpikeList(id_list = list(range(n)))
    SLd = pyST.SpikeList(id_list = list(range(n-nc)))
    SLc = pyST.SpikeList(id_list = list(range(n-nc,n)))
    for i in range(n-nc):
        SLd[i] = pyST.STCreate.inh_poisson_generator(stim[:,i],
                                                    list(range(0,len(stim)*time,time)),
                                                    t_stop=t_sim, refractory = 4.)
    if with_labels:
        for t in range(0,len(stim)):
            SLt= pyST.SpikeList(id_list = list(range(n-nc,n)))
            for i in range(n-nc,n):
                if stim[t,i]>1e-2:
                    SLt[i] = pyST.STCreate.regular_generator(stim[t,i],
                                                        jitter=True,
                                                        t_start=t*time,
                                                        t_stop=(t+1)*time)            
            if len(SLt.raw_data())>0: SLc = pyST.merge_spikelists(SLc, SLt)

    if len(SLc.raw_data())>0: 
        SL = pyST.merge_spikelists(SLd,SLc)
    else:
        SL = SLd
    return SL

def create_spike_train(data_train, t_sample, scaling=1, n_mult=1, with_labels = True): 
    data_train = np.array(data_train)
    N_train = len(data_train)
    data = np.concatenate([data_train for _ in range(n_mult)])
    t_sim = len(data)*t_sample*n_mult
    return SimSpikingStimulus(scaling*data, t_sample, t_sim = t_sim, with_labels = with_labels)



data_train, targets_train = load_mnist(
        '/shares/data/mnist/train-images-idx3-ubyte',
        '/shares/data/mnist/train-labels-idx1-ubyte',
        50000, with_labels = True)
data_classify, targets_classify = load_mnist(
        '/shares/data/mnist/t10k-images-idx3-ubyte',
        '/shares/data/mnist/t10k-labels-idx1-ubyte',
        10000, with_labels = False)


