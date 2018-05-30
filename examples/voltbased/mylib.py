import numpy as np


def genParamsfor(NP=4, pf=5.0, f=20.0, tp=40.0, **kwargs):
    '''
    Generate parameters for number of input patterns N
    '''
    # Parameters for 1 pattern
    ap = kwargs['Ap'] #0.0025
    am = kwargs['Am'] #0.0003
    tfr = kwargs['tfr']
    params = {
        # Vmem learning
        'Ap' : ap,
        'Am' : am,
        'V_th_stdp' : 19.0,
        # Homeostasis
        'tfr' : tfr,
        'heta' : am, #(am*1.3)/tfr,
        'tau_r' : 1000.0/(pf),
        #'R_th_lo': 0.5,
        #'R_th_hi': 2.0,
        # Input pattern dependent parameters
        'N_NEURONS' : 20*NP,
    }
    return params


def testParameters(**kwargs):
    '''
    Tests for various conditions for learning stability
    '''
    tn = int(1000.0/kwargs['pf']) - kwargs['tp']
    try:
        # Check for homeostasis
        assert kwargs['Am'] <= kwargs['tfr']*kwargs['heta'], 'Homeostasis ineffective'
        # Check for negative weight drift
        assert kwargs['tp']*kwargs['Ap'] <= 2*tn*kwargs['Am'], 'No negative weight drift'
        # Rate of learning more than rate of forgetting
        assert kwargs['Ap'] > (tn/1000.0)*kwargs['f']*kwargs['Am'], 'Potentiation not retained'
        print('All is well that ends well!')
    except (KeyError, AssertionError) as e:
        print e
        Am_min = kwargs['tp']*(kwargs['Ap'])/tn/2.0
        Am_max = kwargs['Ap']*1000.0/tn/kwargs['f']
        print('Am {2} should be in the range ({0}, {1})'.format(Am_min, Am_max, kwargs['Am']))


def tauToA(tau=20.0, dt=1.0):
    '''
    Given time constant in ms, return the corresponding A in the 1-2**A
    formulation
    '''
    A = np.log2(dt/tau)
    return int(np.round(A))

