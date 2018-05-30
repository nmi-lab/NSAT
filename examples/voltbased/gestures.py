import numpy as np
#from dvsdata import gatherAllMovements, gatherDAVISMovements

def generateTarget(N=225.0, pd=100.0, rate=50.0, motion='left'):
    '''
    Generate a pattern that lasts for pd ms
    motion: left|right|up|down|random
    '''
    Ns = int(np.sqrt(N))
    width = 12

    N=int(N)
    pd = int(pd)
    if motion=='left':
        x = np.arange(Ns-1,-1,-1)
        y = np.arange(int(Ns/2)-int(width*0.5),
                      int(Ns/2)+int(width*0.5)+1)
        x,y = np.meshgrid(x,y)
        ids = (y*Ns+x).flatten(order='F')
        spkt = np.linspace(1, pd, len(ids))
    elif motion=='right':
        x = np.arange(0,Ns,1)
        y = np.arange(int(Ns/2)-int(width*0.5),
                      int(Ns/2)+int(width*0.5)+1)
        x,y = np.meshgrid(x,y)
        ids = (y*Ns+x).flatten(order='F')
        spkt = np.linspace(1, pd, len(ids))
    elif motion=='up':
        y = np.arange(Ns-1,-1,-1)
        x = np.arange(int(Ns/2)-int(width*0.5),
                      int(Ns/2)+int(width*0.5)+1)
        x,y = np.meshgrid(x,y)
        ids = (y*Ns+x).flatten(order='C')
        spkt = np.linspace(1, pd, len(ids))
    elif motion=='down':
        y = np.arange(0,Ns,1)
        x = np.arange(int(Ns/2)-int(width*0.5),
                      int(Ns/2)+int(width*0.5)+1)
        x,y = np.meshgrid(x,y)
        ids = (y*Ns+x).flatten(order='C')
        spkt = np.linspace(1, pd, len(ids))
    elif motion=='random':
        # Generate the target pattern
        spkt = []
        ids = []
        pspk = np.random.rand(N,pd)<=rate*0.001
        for i in np.arange(pspk.shape[0]):
            spk_ti = np.arange(pd)[pspk[i]]
            if len(spk_ti) > 0:
                spkt.append(spk_ti)
                ids.append([i]*len(spk_ti))
        spkt = np.concatenate(spkt)
        ids = np.concatenate(ids)
    target = []
    target.append(list(zip(ids, spkt)))
    target = np.concatenate(target).astype(int)
    return target


def genEmbeddedSpikePattern(N=100, rate=50, t=2000, pf=5.0, pd=50.0, jitter=0.0,
                            target=[0], dtype='random', res=16, scale=0.1, step=12, tref=10.0 ):
    '''
    N: No. of input neurons
    rate: Input firing rate
    t: total input time
    pf: Pattern presentation frequency
    pd: Pattern duration
    target: list of pattern indices
    dtype: random|motion|dvs
    Generate an input spike pattern embedded in noise.
    '''
    t = t+100 # To compensate for a bug
    spks = []
    target_arr = []
    p = []
    pds = [] # Pattern durations for each pattern
    if dtype is 'random':
        targets = [generateTarget(N=N,pd=pd,rate=rate,motion='random') for tgt in target]
    elif dtype is 'motion':
        targets = [generateTarget(N=N,pd=pd,rate=rate,motion=tgt) for tgt in np.array(['up', 'down', 'left', 'right'])[target]]
    elif dtype is 'dvs':
        pat, l = gatherAllMovements(res=res, scale=scale, tref=tref)
        targets = {str(i): pat[l==i] for i in target}
        #print([len(v[0])*1000/60/(16*16) for k, v in targets.items()])
    elif dtype is 'davis':
        pat, l = gatherDAVISMovements(res=None, scale=scale, step=step, tref=tref)
        targets = {str(i): pat[l==i] for i in target}
    elif dtype is 'raw':
        pat, l = gatherDAVISMovements(res=None, scale=scale, step=step, tref=tref, raw=True)
        targets = {str(i): pat[l==i] for i in target}
    ct=0 # Current time
    tp = [] # Start time of pattern presentation
    pid = 0
    p_shuffled = np.random.permutation(target)
    while ct < t:
        nd = int(np.random.exponential((1000.0/pf)-pd)) # Duration of noise
        mint = max(1000.0/pf*0.25, pd)
        if nd < mint:
            nd = int(mint)
        #nd = int(1000/pf)
        pspk = np.random.rand(N,nd)<=rate*0.001
        for i in np.arange(pspk.shape[0]):
            spk_ti = np.arange(nd)[pspk[i]]+ct
            if len(spk_ti) > 0:
                spks.append(list(zip([i]*len(spk_ti), spk_ti)))
        #Insert target spike pattern
        tp.append(ct+nd)
        #idx = np.random.randint(len(target))
        try:
            target_arr = targets[p_shuffled[pid]]
        except:
            tgs = targets[str(p_shuffled[pid])]
            z = np.random.choice(np.arange(len(tgs)))
            target_arr = tgs[z]
        p.append(p_shuffled[pid])
        pds.append(pd)
        # Change the counter
        pid += 1
        if pid == len(target):
            pid = 0
            p_shuffled = np.random.permutation(target)
        # Update other variables and append to spikes
        pd = (target_arr[:,1]).max() + 1
        tg = target_arr.copy()
        tg[:,1] += ct+nd + np.random.randint(-jitter,jitter+1,len(tg)) #noise
        spks.append(list(zip(tg[:,0], tg[:,1])))
        ct += nd + pd

    # Sort by time
    spks = np.concatenate(spks)
    indx = np.argsort(spks[:,1])
    spks = spks[indx]
    ids = spks[:,0].astype(int)
    spkt = spks[:,1].astype(int)
    return ids,spkt,tp,p, pds


def genCoincidencePattern(N=100, rate=50, t=2000, pf=50, Nf_co=0.2, jitter=1.0):
    '''
    N: No. of input neurons
    rate: Input firing rate
    t: total input time
    Nf_co: Fraction of neurons that spike together
    Generate an input spike pattern with Nf_co% inputs always coincident and the others disperesed randomly.
    '''
    t = int(1.1*t) # To compensate for a bug
    spks = []
    # Generate the regular inputs
    spk_t = np.arange(t)[np.random.rand(t)<=pf*0.001]
    spk_tnew = []
    t_last = -10
    for t in spk_t:
        if t-t_last > 1000./pf/2.:
            spk_ti = [t]*(int(N*Nf_co))
            spk_ti = np.random.randint(-jitter,jitter+1,len(spk_ti)) + np.array(spk_ti)
            spks.append(list(zip(range(int(N*Nf_co)),spk_ti)))
            t_last = t
            spk_tnew.append(t)

    # Generate the random inputs
    pspk = np.random.rand(int(N*(1-Nf_co)),t)<=rate*0.001
    for i in np.arange(pspk.shape[0]):
        spk_ti = np.arange(t)[pspk[i]]
        if len(spk_ti) > 0:
            spks.append(list(zip([i+int(N*Nf_co)]*len(spk_ti), spk_ti)))
    # Sort by time
    spks = np.concatenate(spks)
    indx = np.argsort(spks[:,1])
    spks = spks[indx]
    ids = spks[:,0]
    spkt = spks[:,1]
    return ids,spkt, spk_tnew

def delaySpikes(ids, spkt, delays=0):
    '''
    Delay the spike trains by given delays
    ids: neuron id
    spkt: spike times
    delays: delays
    '''
    try:
        for i, td in enumerate(delays):
            spkt[ids==i] += np.around(td)
        return ids, spkt
    except:
        return ids, spkt+delays


