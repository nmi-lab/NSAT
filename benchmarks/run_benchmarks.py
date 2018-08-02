import os
import numpy as np
import matplotlib.pylab as plt
from sim_erbp_mlp_2L import erbp_mlp_2L
from sim_erbp_mlp_1L import erbp_mlp_1L
from sim_erbp_convnet_2L import erbp_convnet_2L
from sim_erbp_convnet_4L import erbp_convnet_4L
from sim_erbp_mlp_2L_multicore import erbp_mlp_2L_multicore

from load_mnist import data_train, data_classify, targets_classify

if __name__ == '__main__':
    n_epochs = 30

    case = {'mlp1-1': erbp_mlp_1L,
            'mlp2-1': erbp_mlp_2L,
            'mlp2-2': erbp_mlp_2L_multicore,
            'conv2-2': erbp_convnet_2L,
            'conv4-2': erbp_convnet_4L}
    layers = (1, 2, 2, 2, 4)
    cores = (1, 1, 2, 2, 2)
    cases = ['MLP', 'MLP', 'MLP', 'CONVNET', 'CONVNET']

    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    save = os.dup(1), os.dup(2)

    acc, t_total = [], []
    print("NSAT Benchmark suite is now running...")
    for i, key in enumerate(case):
        print("Running eRBP %s with %d hidden layer(s) on %s core(s)!"
              % (cases[i], layers[i], cores[i]))
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        accur, tt = case[key](data_train, data_classify, targets_classify,
                              nepochs=n_epochs)
        acc.append(accur)
        t_total.append(tt)
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
    print("... Done!")
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    os.close(null_fds[0])
    os.close(null_fds[1])

    print("Accuracy and total time of training and testing have been saved!")
    np.save('accuracy', acc)
    np.save('total_time', t_total)

    print("Now plotting the results!")
    labels = ['MLP-1Layer-1Core', 'MLP-2Layers-1Core', 'MLP-2Layers-2Cores',
              'CONVNET-2Layers-2Cores', 'CONVNET-4Layers-2Cores']
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('Accuracy')
    for i, a in enumerate(acc):
        ax.plot(a, label=labels[i])
    ax.legend()
    ax = fig.add_subplot(122)
    ax.set_title('Total time (train and test)')
    for i, a in enumerate(t_total):
        ax.plot(a, label=labels[i])
    ax.legend()
    plt.show()
