#!/bin/python
#-----------------------------------------------------------------------------
# File Name : common.py
# Author: Emre Neftci
#
# Creation Date : Thu 22 Feb 2018 10:40:09 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

def test_accuracy(reader, targets, pop, sim_ticks, duration):
    import pyNSATlib as nsat
    import numpy as np
    N_samples = len(targets)
    SL = reader.read_spikelist(sim_ticks = sim_ticks, id_list = pop.addr, core = pop.core).id_slice(pop.addr)
    pred = np.argmax(SL.firing_rate(duration),axis=0)
    assert len(pred) == N_samples
    return float(sum( pred == targets[:N_samples]))/N_samples*100, SL


