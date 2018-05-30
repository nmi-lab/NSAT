import numpy as np


def preprocess_bounded_logit(input_vector, beta=0.0015, gamma=16000,
                             t_ref=4e-4, min_p=1e-5, max_p=0.98):
    '''
    Bound and take elementwise scaled logit of a vector
    '''
    s = np.array(input_vector)
    s[s < min_p] = min_p
    s[s > max_p] = max_p
    return -np.log(-1 + 1./(s))


def data_preprocess(stim,
                    targets,
                    metadata,
                    preprocessor=preprocess_bounded_logit,
                    clamp={'features': 1, 'targets': 1},
                    kwargs_preprocessor={},
                    ):
    '''
    'preprocessor': which preprocessor to use. preprocess_bounded_logit is the
     default as used in Neftci et al 2014.
     Can be any function that takes a vector and returns a vector.
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


class ECDStimulus_v2(object):

    def __init__(self, stim, times=[0, 50, 100], scale_exc=[1, 0],
                 scale_inh=[1, 0], t_sim=None):
        '''
        Times must be sorted. ex: times = [0, 1, 2] ; scale = [1,0]
        *poisson*: integer, output is a poisson process with mean data/poisson,
        scaled by *poisson*.
        '''
        self.stim = stim
        self.times = times
        self.scale_exc = scale_exc
        self.scale_inh = scale_inh
        self.t_sim = t_sim

    def __getitem__(self, t):
        '''
        Basically returns either the stimulus or npilot board masterot
        according to the time/phase
        '''
        import bisect
        if self.t_sim is not None:
            if t >= self.t_sim:
                out = np.zeros([self.stim[0].shape[0], 4], dtype='int')
                out[:, 0] -= 32000
                return out
        mod_ = t % self.times[-1]
        res_ = t // self.times[-1]
        # fibnd time index
        idx = bisect.bisect(self.times, mod_) - 1
        mu = self.stim[res_]
        mu_plus = mu[mu >= 0] / 5
        mu_minus = -mu[mu < 0]
        nmu = np.zeros_like(mu)
        nmu[mu >= 0] = mu_plus * self.scale_exc[idx]
        nmu[mu < 0] = -mu_minus * self.scale_inh[idx]
        out = np.zeros([mu.shape[0], 4], dtype='int')
        out[:, 0] = nmu
        out[:, 2] = 0
        if (t % 1000) == 0:
            out[:, 2] = 1
        elif (t % 1000) == 100:
            out[:, 2] = 1
        elif (t % 1000) == 500:
            out[:, 2] = -1
        elif (t % 1000) == 600:
            out[:, 2] = -1
        return out


if __name__ == '__main__':
    from bars_stripes import bs_loader_npy, bs_load_and_save
    N_NEURONS = 64

    bs_load_and_save()
    # data, targets, _ = bs_loader_npy(dset='train', prefix='data/')

    # stim = np.zeros((data.shape[0], N_NEURONS))
    # stim[:, :data.shape[1]] = data

    # scales_exc = np.array([2**17, 0])
    # scales_inh = np.array([2**17, 0])
    # t_sim_data = len(data) * 1000
    # t_sim = len(data) * 1000 + 1000
    # times = [0, 500, 1000]
    # ecd_stim = ECDStimulus_v2(stim, times, scales_exc, scales_inh, t_sim_data)
    # print(t_sim, t_sim_data)

    # for i in range(10000):
    #     print(ecd_stim[i])
