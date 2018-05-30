import sys
import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    im_size_x, im_size_y = 4, 4
    nn_size_x, nn_size_y = 6, 6

    fname = "/tmp/weights.npy"
    W = np.load(fname)[:16, :]
    R = np.zeros((im_size_x*nn_size_x, im_size_y*nn_size_x))

    for i in range(nn_size_x):
        for j in range(nn_size_y):
            ww = W[:, i*nn_size_x+j].reshape(im_size_x, im_size_y)
            R[i*im_size_x:(i+1)*im_size_x, j*im_size_y:(j+1)*im_size_y] = ww

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(R, interpolation='nearest', cmap=plt.cm.gray_r)
    ax.set_xticks(np.arange(-0.5, im_size_x*nn_size_x-.5, im_size_x))
    ax.set_yticks(np.arange(-0.5, im_size_y*nn_size_y-.5, im_size_y))
    ax.grid(color='r', linestyle='-', linewidth=1.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.colorbar(im)
    plt.show()
