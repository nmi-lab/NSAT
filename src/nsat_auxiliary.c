/* ************************************************************************
 * NSATlib_v2 This is a C implementation of the NSATlib_v2 python
 * script. It simulates the NSAT. 
 * Copyright (C) <2016>  UCI, Georgios Detorakis (gdetor@protonmail.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/
#include "nsat.h"


/* ************************************************************************
 * COUNT_SYNAPSES: This function counts the number of synapses per neuron.
 *
 * Args : 
 *  neuron (unit **)            : Neurons units (structure)
 *  total_num_synapses (int)    : Total number of synapses per core 
 *  num_units (int)             : Number of external and nsat neurons
 *  num_states (int)            : Number of states per NSAT neuron
 *
 * Returns :
 *  void
 **************************************************************************/
void count_synapses(unit **neuron,
                    int *total_num_synapses,
                    int num_units,
                    int num_states) {
    int j;
    int k;
    int count = 0, glob_count = 0;

    for (j = 0; j < num_units; ++j) {
        count = 0;
        for(k = 0; k < num_states; ++k) {
            count += (*neuron)[j].syn_ptr[k]->len;
        }
        (*neuron)[j].num_synapses = count;
    }

    for (j = 0; j < num_units; ++j) {
        glob_count += (*neuron)[j].num_synapses;
    }

    *total_num_synapses = glob_count;
}
