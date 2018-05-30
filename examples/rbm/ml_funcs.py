#!/usr/bin/env/ python
# ----------------------------------------------------------------------------
# File Name : helpers.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 04-02-2016
# Last Modified : Thu Dec  8 16:34:13 PST 2016
#
# Copyright : (c)
# Licence : GPLv2
# ----------------------------------------------------------------------------
import numpy as np


class ECDStimulus(object):
    def __init__(self,
                 stim,
                 times=[0, 50, 100],
                 scale_exc=[1, 0],
                 scale_inh=[1, 0],
                 poisson=False):
        '''
        Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
        *poisson*: integer, output is a poisson process with mean
                   data/poisson, scaled by *poisson*.
        '''
        self.stim = stim / 10
        self.times = times
        self.scale_exc = scale_exc
        self.scale_inh = scale_inh
        self.poisson = poisson

    def __getitem__(self, t):
        '''
        Basically returns either the stimulus or not according to the
        time/phase
        '''
        import bisect
        mod_ = t % self.times[-1]
        res_ = t // self.times[-1]
        # fibnd time index
        idx = bisect.bisect(self.times, mod_) - 1
        if self.poisson:
            mu = self.stim[res_]
            mu_plus = mu[mu >= 0]
            mu_minus = -mu[mu < 0]
            nmu = np.zeros_like(mu)
            nmu[mu >= 0] = mu_plus * self.scale_exc[idx]
            # nmu[mu>=0] = np.random.poisson(mu_plus)*self.scale_exc[idx]
            nmu[mu < 0] = -mu_minus * self.scale_inh[idx]
            return nmu
        else:
            return self.stim[res_] * self.scale[idx]


def plot_weight_matrix(l, W, pdict):
    '''
    plot the effective weight matrix of readout neuron l
    '''
    W_l = W[pdict['Nv']:, pdict['Nv'] - pdict['N_labels'] + l]
    W_v = W[:pdict['Nv'] - pdict['N_labels'], pdict['Nv']:]
    W_vl = np.dot(W_v, W_l)
    import pylab
    pylab.figure()
    pylab.colorbar(pylab.pcolor(W_vl.reshape((28, 28))))
    return W_vl


def preprocess_bounded_logit(input_vector,
                             beta=0.0015,
                             gamma=16000,
                             t_ref=4e-4,
                             min_p=1e-5,
                             max_p=0.98):
    '''
    Bound and take elementwise scaled logit of a vector
    '''
    s = np.array(input_vector)
    s[s < min_p] = min_p
    s[s > max_p] = max_p
    # return -np.log(gamma*(1./s*t_ref-t_ref))/beta
    return -np.log(-1 + 1. / (s))


def preprocess_sigmoid(input_vector):
    rate = 1. / (4e-3 + 1. / (1e-32 + 5000 * input_vector))
    return rate


def data_preprocess(stim,
                    targets,
                    metadata,
                    preprocessor=preprocess_sigmoid,
                    clamp={'features': 1, 'targets': 1},
                    kwargs_preprocessor={},
                    ):
    '''
    'preprocessor': which preprocessor to use. preprocess_bounded_logit is
                    the default as used in Neftci et al 2014.
                    Can be any function that takes a vector and returns a
                    vector.
    '''

    Nv = metadata['Nv']
    N_labels = metadata['N_labels']
    data_specs1 = metadata['data_specs1']
    idxs = metadata['idxs']

    # build mask
    mask = np.zeros(stim.shape[1])
    for i in range(len(data_specs1)):
        if clamp in data_specs1[i]:
            mask[idxs[i]:idxs[i + 1]] = float(clamp[data_specs1[i]])

    if preprocessor is not None:
        stim = preprocessor(stim, **kwargs_preprocessor)

    stim_masked = np.array([s * mask for s in stim])

    return stim_masked, targets, Nv, N_labels


def computeInference(sl, pdict):
    '''
    Given the spiking data of the network, return the inferred label
    '''
    label_resp = sl[pdict['Nv'] - pdict['N_labels']:pdict['Nv']]
    return label_resp.mean_rates().argmax()


def confusion_matrix(SL, pdict, targets_classify):
    SL.complete(range(pdict['N']))
    # For testing
    cm = np.zeros((pdict['N_labels'], pdict['N_labels']))
    for i in range(pdict['N_test']):
        t_start = i * pdict['t_sample']
        t_stop = t_start + pdict['t_sample']
        infer = computeInference(SL.time_slice(t_start, t_stop), pdict)
        cm[targets_classify[i], infer] += 1
    return np.trace(cm) / pdict['N_test']


def recognition_rate(SL, Nv, N_test, N_labels, t_sample, targets_classify):
    t_sim = N_test * t_sample
    Sc = SL.id_slice(range(Nv - N_labels, Nv))
    Sc.t_start = 0
    Sc.t_stop = t_sim
    fr = Sc.firing_rate(t_sample).reshape(N_labels, 1, -1).mean(axis=1)
    maxfr = np.argmax(fr, axis=0)
    return (float(np.sum(maxfr == targets_classify[:N_test])) /
            len(targets_classify[:N_test]))


def pickle_data(filename, data, targets, **metadata):
    '''
    Pickle data, targets and metadata.
    *data*: np.array
    *targets*: np.array
    *metadata*: keywords arguments, will be pickled as a dictionary
    '''
    import cPickle
    np.save(filename + '_data.npy', data)
    np.save(filename + '_targets.npy', targets)
    fd = file(filename + '_metadata.pkl', 'w')
    cPickle.dump(metadata, fd)
    fd.close()


def _one_hot(x, xmax):
    x1h = np.zeros(xmax)
    x1h[x] = 1
    return x1h


def data_loader(data):
    '''
    Load data using one of the two methods:
    'data': uses pylearn2 loader, *data* is then the objet return from
            the pylearn2 data loader. Uses the sequential mode for the
            iterator, so don't forget to shuffle  before using the data
            for training.
    '''
    N_samples = len(data.X)
    N_labels = data.y_labels
    Nv = data.X.shape[1] + N_labels

    # get_data_specs is required otheriwse there are no labels
    it = data.iterator(mode='sequential', num_batches=N_samples,
                       batch_size=1, data_specs=data.get_data_specs())

    data_specs0 = data.get_data_specs()[0]
    data_specs1 = data.get_data_specs()[1]

    dcr = range(len(data_specs1))

    # get dims and indexes for dims
    dims = []
    components = data_specs0
    components_ids = data_specs1
    for i, s in enumerate(components_ids):
        if s == 'features':
            dims.append(components.components[i].dim)
        elif s == 'targets' and 'int' in components.components[i].dtype:
            N_labels = components.components[i].max_labels
            dims.append(N_labels)
        else:
            NotImplementedError('component type not supported')

    idxs = np.concatenate([[0], np.cumsum(dims)])

    # Build vector

    data = []
    data_targets = []

    for j, dc in enumerate(it):
        tmp_data = np.zeros(np.sum(dims))
        if j >= N_samples:
            break

        for i in dcr:
            if data_specs1[i] == 'targets':
                data_targets.append(dc[i][0][0])
                d = _one_hot(dc[i], N_labels)
                tmp_data[idxs[i]:idxs[i + 1]] = d
            elif data_specs1[i] == 'features':
                d = dc[i].flatten()
                tmp_data[idxs[i]:idxs[i + 1]] = d
            else:
                raise NotImplementedError

        data.append(tmp_data)

    stim = np.array(data)
    targets = np.array(data_targets)

    metadata = dict()
    metadata['Nv'] = Nv
    metadata['N_labels'] = N_labels
    metadata['N_samples'] = N_samples
    metadata['data_specs1'] = data_specs1
    metadata['idxs'] = idxs

    return stim, targets, metadata


def __tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                         scale_rows_to_unit_interval=False,
                         output_pixel_vals=False):
    """
    From deeplearning.net
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = __tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = __scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def __scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def stim_show(images, img_shape, tile_shape):
    '''
    Plots every image in images (using imshow)
    '''
    import matplotlib.pyplot as plt
    til = __tile_raster_images(images + .5,
                               img_shape,
                               tile_shape,
                               tile_spacing=(1, 1))

    f = plt.imshow(til, interpolation='nearest')
    plt.bone()
    plt.xticks([]), plt.yticks([])
    plt.show()
    return f
