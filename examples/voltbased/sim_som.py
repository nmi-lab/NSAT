#!/bin/python
#-----------------------------------------------------------------------------
# Author: Georgios Detorakis
#
# Creation Date : 19-08-2016
# Last Modified : 
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from generate_params import *
from default_nsat_params import *
import matplotlib.pylab as plt


def SimSpikingStimulus(rates=[5, 10], t_start=1000, t_stop=4000):
    n = np.shape(rates)[0]
    SL = pyST.SpikeList(id_list = range(n))
    for i in range(n):
        SL[i] = pyST.STCreate.regular_generator(rates[i],
                                                t_start=t_start,
                                                t_stop=t_stop,
                                                jitter=False)
        # SL[i] = pyST.STCreate.poisson_generator(rates[i], t_start, t_stop)
    return SL


def h(x, sigma=1):
    return np.exp(-(x)**2/(2*sigma**2))


def lateral_connectivity(n):
    x = np.zeros((n , n))
    for i in range(n):
        for j in range(n):
            x[i, j] = np.abs(i - j)
    return 190 * h(x, 2) - 120 * h(x, 1)


def print_dot_graph_file(w, fname, layers=[1, 1]):
    rank, size_x, size_y = 1.0, 7.5, 7.5
    num_units = w.shape[0]
    num_states = w.shape[2]

    with open(fname, 'w') as f:
        f.write("digraph G {\n")
        f.write("ranksep={:1.3f}; size = \"{:1.3f}, {:1.3f}\";\n".format(
                                                                    rank,
                                                                    size_x,
                                                                    size_y));

        for i in range(len(layers)-1):
            f.write("{ rank = same; ")
            for j in range(layers[i], layers[i+1]):
                f.write("{:d} ".format(j))
            f.write(" }\n")

        for i in range(num_units):
            for j in range(num_units):
                for k in range(num_states):
                    if w[i, j, k] != 0:
                        f.write("   {:d} -> {:d}; \n".format(i, j))
        f.write("}")


def build_synaptic_w(m, n, k):
    units = m + n
    w = np.zeros((units, units, k), dtype='i')
    cw = np.zeros((units, units, k), dtype='i')

    # input units
    w[0, 3, 0] = 10
    w[0, 4, 0] = 10
    w[0, 5, 0] = 10

    w[1, 3, 0] = 10
    w[1, 4, 0] = 10
    w[1, 5, 0] = 10

    w[2, 3, 0] = 10
    w[2, 4, 0] = 10
    w[2, 5, 0] = 10

    # excitatory units
    w[3, 6, 1] = 5
    w[4, 6, 1] = 5
    w[5, 6, 1] = 5

    # inhibitory units
    w[6, 3, 1] = -65
    w[6, 4, 1] = -65
    w[6, 5, 1] = -65

    cw[w != 0] = True
    return w.astype('i'), cw.astype('i')


if __name__ == "__main__":
    sim_ticks = 60000
    N_NEURONS = 4
    N_INPUTS = 3
    N_STATES = 4

    OFF = -16
    MAX = 2**15-1

    t_ref[0] = 40;
    t_ref[1] = 30;

    A[0] = [[-7,  OFF,  OFF, OFF ], 
            [0,  -5,  OFF, OFF ],
            [OFF,  OFF,  OFF, OFF ],
            [OFF,  OFF,  OFF, OFF]]

    A[1] = [[-7,  OFF,  OFF, OFF ], 
            [0,  -7,  OFF, OFF ],
            [OFF,  OFF,  OFF, OFF ],
            [OFF,  OFF,  OFF, OFF]]

    sA[0] = [[-1, 1, 1, 1],
             [ +1, -1, 1, 1],
             [ 1, 1, 1, 1],
             [ 1, 1, 1, 1]]

    sA[1] = sA[0].copy()

    b[0] = [0, 0, 0, 0]
    b[1] = [0, 0, 0, 0]
    # Xinit = [[0,0,0,0] for _ in range(N_NEURONS)]
    Xinit = [[0,0,0,0] for _ in range(N_NEURONS)]
    Xreset[0] = [0, MAX, MAX, MAX]
    Xreset[1] = Xreset[0].copy()
    XresetOn[0] = [True , False, False, False]
    XresetOn[1] = [True , False, False, False]

    plasticity_en = False

    plastic[0] = False
    stdp_en[0] = False

    sigma[0] = [0, 0, 0, 0]
    sigma[1] = [0, 0, 0, 0]
    modstate[0] = 3
    modstate[1] = 3

    # Synaptic weights and adjacent matrix
    W, CW = build_synaptic_w(N_INPUTS, N_NEURONS, N_STATES)
    print_dot_graph_file(W, "graph.gv", layers=[0, 3, 6])

    # Parameters groups mapping function
    nmap = np.array([0, 0, 0, 1], dtype='int')
    lrnmap = np.zeros((N_NEURONS, N_STATES), dtype='int')

    Xth[0] = 100 
    Xth[1] = 80 

    rates = [60, 30, 30]
    t_start = 1
    t_stop = 35000
    SL = SimSpikingStimulus(rates, t_start, t_stop)
    ext_evts_data = exportAER(SL)

    xH = []
    gH = []
    wH = []

    prefix = "../data/"
    hex_dir_name = exp_name = 'test_som'
    fname = generate_fnames(prefix+exp_name)

    spk_rec_mon = np.arange(N_NEURONS, dtype='int')

    print("Generating parameters files!")
    cfg = build_params(hex_dir_name,
                       fname,
                       nmap,
                       lrnmap,
                       ext_evts_data,
                       A, sA, b,
                       sim_ticks=sim_ticks,
                       N_NEURONS=N_NEURONS, 
                       N_INPUTS = N_INPUTS,
                       spk_rec_mon=spk_rec_mon,
                       Xinit=np.zeros((N_NEURONS, N_STATES), dtype='int'),
                       Xreset=Xreset,
                       XspikeResetVal=Xreset,
                       XresetOn=XresetOn,
                       Xth=Xth,
                       Xthup=Xthup,
                       Xthlo=Xthlo,
                       W=W,
                       CW=CW,
                       monitor_spikes = [0],
                       stdp_en = stdp_en,
                       plastic = plastic,
                       plasticity_en = False,
                       tstdpmax = tstdpmax,
                       hiac = hiac,
                       tca=tca,
                       hica=hica,
                       sica=sica,
                       tac=tac,
                       siac=siac,
                       ext_evts = True,
                       write_hex=False)

    # C NSAT arguments
    args = ['exec','-mw','-c','-ms','-mw', '-mspk']

    if __name__ == '__main__':
        print("Running C NSAT!")
        run_c_nsat_gen(sim_ticks, fname, args)

        # Load the results (read binary files)
        states = read_from_file(fname.states+'0.dat').reshape(sim_ticks-1,
                                                              N_NEURONS,
                                                              N_STATES)
        in_spikelist = SL
        out_spikelist = importAER(read_from_file(fname.events+'0.dat'),
                                  sim_ticks = sim_ticks,
                                  id_list = [0])

        out_spikelist.raster_plot()
        plt.axvline(x=t_start, c='r')
        plt.axvline(x=t_stop, c='r')

        fig = plt.figure()
        for i in range(1, 5):
            ax = fig.add_subplot(4, 1, i)
            ax.plot(states[:t_stop, i-1, 0])
        plt.show()

