import numpy as np
import matplotlib.pylab as plt
from struct import unpack

if __name__ == '__main__':
    error = np.load('/tmp/error.npy')
    error_rbm = np.genfromtxt('/home/gdetorak/data_paper/error_rbm.dat')

    s = []
    total = 0
    for i in range(300):
        name = '/tmp/test_eCD_stats_nsat_'+str(i)
        with open(name, 'rb') as f:
            tmp = f.read()
        size = len(tmp)
        c = np.array(unpack('Q'*int(size // 8), tmp), 'Q')
        c = c[1:].reshape(118, 2)
        c[:18, 1] *= 100
        c[18:, 1] *= 18
        total += c[:, 1].sum()
        s.append(total)
    s = np.array(s)
    e = (26432 * np.arange(1, 301)).astype('i')

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ln1 = ax.semilogx(s, error, 'ko', label='eRBM', mfc='w', alpha=0.7)
    ln2 = ax.semilogx(e, error_rbm, 'bx', label='RBM', alpha=0.7)
    ticks = ax.get_xticks()
    ax.set_xticklabels(ticks, fontsize=14, weight='bold', color='k')
    ax.set_xscale('log')
    ticks = ax.get_yticks()
    ax.set_yticklabels(ticks, fontsize=14, weight='bold')
    ax.set_ylabel('Test Set Error', fontsize=14, weight='bold')
    ax.set_xlabel('# SynOps(eRBM) / MACs(RBM)', fontsize=14, weight='bold')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.set_xlim([0, 10**10])

    plt.show()
