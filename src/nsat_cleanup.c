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
 * DEALLOC_NEURONS: This function deallocates the memory previously
 * allocated for the external and nsat neurons per core.
 *
 * Args : 
 *  core (nsat_core **)    : Cores struct array
 *  num_cores (int)        : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void dealloc_neurons(nsat_core **core, int num_cores) {
    int j;
    int k, p;

    for (p = 0; p < num_cores; ++p) {
        for (j = 0; j < (*core)[p].core_pms.num_inputs; ++j) {
            for (k = 0; k < (*core)[p].core_pms.num_states; ++k) {
                destroy_list_syn(&(*core)[p].ext_neuron[j].syn_ptr[k]);
                free((*core)[p].ext_neuron[j].syn_ptr[k]);
            }
            free((*core)[p].ext_neuron[j].syn_ptr);
            (*core)[p].ext_neuron[j].syn_ptr = NULL;
        }

        for(j = 0; j < (*core)[p].core_pms.num_neurons; ++j) {
            (*core)[p].nsat_neuron[j].nsat_ptr = NULL;
            dealloc((*core)[p].nsat_neuron[j].ptr_cores);
            for (k = 0; k < (*core)[p].core_pms.num_states; ++k) {
                (*core)[p].nsat_neuron[j].s[k].lrn_ptr = NULL;
            }
            dealloc((*core)[p].nsat_neuron[j].s);
            for (k = 0; k < (*core)[p].core_pms.num_states; ++k) {
                destroy_list_syn(&(*core)[p].nsat_neuron[j].syn_ptr[k]);
                free((*core)[p].nsat_neuron[j].syn_ptr[k]);
            }
            free((*core)[p].nsat_neuron[j].syn_ptr);
            (*core)[p].nsat_neuron[j].syn_ptr = NULL;
        }
    }
}


/* ************************************************************************
 * DEALLOC_CORES: This function deallocates the memory previously
 * allocated for cores.
 *
 * Args : 
 *  cores (nsat_cores **)   : Cores structures array
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void dealloc_cores(nsat_core **cores, int num_cores) {
    int p, i;

    dealloc_neurons(cores, num_cores);

    for (p = 0; p < num_cores; ++p) {

        /* Destroy core's parameters structures */
        array_list_destroy(&(*cores)[p].core_pms.ext_syn_rec_ids, 0);
        dealloc((*cores)[p].core_pms.ext_syn_rec_ids);

        array_list_destroy(&(*cores)[p].core_pms.nsat_syn_rec_ids, 0);
        dealloc((*cores)[p].core_pms.nsat_syn_rec_ids);

        /* Deallocate monitors structure */
        dealloc((*cores)[p].mon_pms);

        /* Deallocate synapses statistics structure */
        dealloc((*cores)[p].syn);

        /* Deallocate nsat parameters structure */
        for (i = 0; i < 8; ++i) {
            dealloc((*cores)[p].nsat_pms[i].prob)
            dealloc((*cores)[p].nsat_pms[i].sigma)
            dealloc((*cores)[p].nsat_pms[i].A);
            dealloc((*cores)[p].nsat_pms[i].sF);
            dealloc((*cores)[p].nsat_pms[i].b);
            dealloc((*cores)[p].nsat_pms[i].x_reset);
            dealloc((*cores)[p].nsat_pms[i].x_spike_incr);
            dealloc((*cores)[p].nsat_pms[i].x_thlow);
            dealloc((*cores)[p].nsat_pms[i].x_thup);
            dealloc((*cores)[p].nsat_pms[i].w_gain);
            dealloc((*cores)[p].nsat_pms[i].is_xreset_on);
        }
        dealloc((*cores)[p].nsat_pms);

        /* Deallocate the structure for learning parameters */
        dealloc((*cores)[p].lrn_pms);

        /* Deallocate and initialize monitoring files structure */
        (*cores)[p].files->fs = NULL;
        (*cores)[p].files->fw = NULL;
        (*cores)[p].files->fsa = NULL; 
        (*cores)[p].files->fr = NULL;
        dealloc((*cores)[p].files);

        /* Deallocate vars structure */
        dealloc((*cores)[p].vars->tX);
        dealloc((*cores)[p].vars->acm); 
        dealloc((*cores)[p].vars->g);
        dealloc((*cores)[p].vars->xinit);
        dealloc((*cores)[p].vars->rec_spk_on);
        dealloc((*cores)[p].vars);

        /* Deallocate all the lists */
        array_list_destroy(&(*cores)[p].events, 1);
        dealloc((*cores)[p].events);

        array_list_destroy(&(*cores)[p].ext_events, 1);
        dealloc((*cores)[p].ext_events);

        array_list_destroy(&(*cores)[p].nsat_events, 1);
        dealloc((*cores)[p].nsat_events);
        
        array_list_destroy(&(*cores)[p].trans_events, 1);
        dealloc((*cores)[p].trans_events);

        array_list_destroy(&(*cores)[p].mon_events, 1);
        dealloc((*cores)[p].mon_events);

        array_list_destroy(&(*cores)[p].ext_caspk, 1);
        dealloc((*cores)[p].ext_caspk);

        array_list_destroy(&(*cores)[p].nsat_caspk, 1);
        dealloc((*cores)[p].nsat_caspk);
        
        /* Free global parameters pointer */
        (*cores)[p].g_pms = NULL;
        
        dealloc((*cores)[p].shared_memory);

        /* Deallocate neurons array structures */
        dealloc((*cores)[p].ext_neuron);
        dealloc((*cores)[p].nsat_neuron);

        dealloc((*cores)[p].ext_evts_fname);
    }
}
