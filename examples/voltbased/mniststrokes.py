import numpy as np

def getDataOf(idx=0, dataset='points', training=True, path='../../data/mnist-digits-stroke-sequence-data/sequences/'):
    '''
    idx:  id of the data point
    dataset: inputdata/points/targetdata
    training: True for trainimg False for testimg
    '''
    if training:
        setname = 'trainimg'
    else:
        setname = 'testimg'
    
    assert dataset in ['inputdata', 'points', 'targetdata']
    
    fname = "{3}/{0}-{1}-{2}.txt".format(setname, idx, dataset, path)
    
    if dataset == 'inputdata':
        data = np.loadtxt(fname)
    elif dataset == 'points':
        data = np.loadtxt(fname, delimiter=',', skiprows=1)[:-1]
        # Clear any [-1,-1] points which correspond to pen lifts
        idx = data[:,0] != -1
        data = data[idx]
    elif dataset == 'targetdata':
        data = np.loadtxt(fname)[:-1]
        
    data = data.astype(int)
        
    return data


def getDatabyIdx(idx=0, training=True, path='../../data/mnist-digits-stroke-sequence-data/sequences/'):
    '''
    idx:  id of the data point
    training: True for trainimg False for testimg
    Returns: (data, label)
    '''
    targetdata = getDataOf(idx=idx, training=training, path=path, dataset='targetdata')
    points = getDataOf(idx=idx, training=training, path=path, dataset='points')-1
    label = np.argmax(targetdata[0][:10])
    return points, label
    
    
def getSpikesbyIdx(idx=0, training=True, path='../../data/mnist-digits-stroke-sequence-data/sequences/'):
    '''
    idx:  id of the data point
    training: True for trainimg False for testimg
    Returns: (data, label)
    '''
    points, label = getDatabyIdx(idx=idx, training=training, path=path)
    indx = points[:,0]*28+points[:,1] # Generate unique id for each pixel
    if np.any(indx>=28*28):
        raise Exception
    t = np.arange(len(indx))
    return indx, t, label

def filterBlankPixels(path='../../data/mnist-digits-stroke-sequence-data/sequences/'):
    '''
    Filter out pixels that contain no data
    Not very useful.. 654 pixels are still active!
    '''
    allpixels = []
    for idx in range(50000):
        indx, t, label = getSpikesbyIdx(idx, training=True, path=path)
        allpixels.append(indx)
    for idx in range(10000):
        indx, t, label = getSpikesbyIdx(idx, training=False, path=path)
        allpixels.append(indx)
    return allpixels

def genEmbeddedSpikePattern(N=28*28, rate=1000./(28*28), t=2000, pf=5.0, pd=50.0, jitter=0.0,
                            target_patterns = [0]):
    '''
    N: No. of input neurons
    rate: Input firing rate
    t: total input time
    pf: Pattern presentation frequency
    pd: Pattern duration
    target_patterns: [0,1,...9]
    Generate an input spike pattern embedded in noise.
    '''
    t = t+100 # To compensate for a bug
    spks = []
    target_arr = []
    if target_patterns is None:
        #target_patterns = ['left','right','up','down']
        target_patterns = [0]
    last_pattern = 0
    ct=0 # Current time
    tp = [] # Start time of pattern presentation
    p = []
    while ct < t:
        nd = int(np.random.exponential((1000.0/pf)-pd)) # Duration of noise
        if nd < pd:
            nd = int(pd)
        #nd = int(1000/pf)
        pspk = np.random.rand(N,nd)<=rate*0.001
        for i in np.arange(pspk.shape[0]):
            spk_ti = np.arange(nd)[pspk[i]]+ct
            if len(spk_ti) > 0:
                spks.append(list(zip([i]*len(spk_ti), spk_ti)))
        #Insert target spike pattern
        tp.append(ct+nd)
        idx = np.random.choice(target_patterns)
        label = None
        while label == None:
            idp, txp, label = getSpikesbyIdx(np.random.randint(50000))
            if label not in target_patterns:
                label = None
            else:
                target_arr = np.array(list(zip(idp, txp)))
        pd = (target_arr[:,1]).max() + 1
        last_pattern = idx
        p.append(idx)
        tg = target_arr.copy()
        tg[:,1] += ct+nd + np.random.randint(-jitter,jitter+1,len(tg)) #noise
        spks.append(list(zip(tg[:,0], tg[:,1])))
        ct += nd + pd

    # Sort by time
    spks = np.concatenate(spks)
    indx = np.argsort(spks[:,1])
    spks = spks[indx]
    ids = spks[:,0]
    spkt = spks[:,1]
    return ids,spkt,tp,p
