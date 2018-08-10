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
 * ALLOCATE_CORES: This function allocates all the memories for data
 * structures and variables per core.
 *
 * Args : 
 *  core (nsat_core *)      : NSAT cores data structure
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void allocate_cores(nsat_core **cores,
                    fnames *fname, 
                    int num_cores) {
    int i;
    int p;

    for(p = 0; p < num_cores; ++p) {
        /* Allocate and initialize monitors structure */
        (*cores)[p].mon_pms = alloc(monitors_params, 1);
        (*cores)[p].mon_pms->mon_states = false;
        (*cores)[p].mon_pms->mon_weights = false;
        (*cores)[p].mon_pms->mon_final_weights = false;
        (*cores)[p].mon_pms->mon_spikes = false;
        (*cores)[p].mon_pms->mon_stats = false;

        /* Allocate and initialize synapses statistics structure */
        (*cores)[p].syn = alloc(synapse_stat, 1);
        (*cores)[p].syn->tot_ext_syn_num = 0;
        (*cores)[p].syn->tot_nsat_syn_num = 0;

        /* Allocate and initialize nsat parameters structure */
        (*cores)[p].nsat_pms = alloc(nsat_params, 8);
        for (i = 0; i < 8; ++i) {
            (*cores)[p].nsat_pms[i].prob = NULL;
            (*cores)[p].nsat_pms[i].sigma = NULL;
            (*cores)[p].nsat_pms[i].A = NULL;
            (*cores)[p].nsat_pms[i].sF = NULL;
            (*cores)[p].nsat_pms[i].b = NULL;
            (*cores)[p].nsat_pms[i].x_reset = NULL;
            (*cores)[p].nsat_pms[i].x_spike_incr = NULL;
            (*cores)[p].nsat_pms[i].x_thlow = NULL;
            (*cores)[p].nsat_pms[i].x_thup = NULL;
            (*cores)[p].nsat_pms[i].w_gain = NULL;
            (*cores)[p].nsat_pms[i].t_ref = 0;
            (*cores)[p].nsat_pms[i].gate_low = 0;
            (*cores)[p].nsat_pms[i].gate_upper = 0;
            (*cores)[p].nsat_pms[i].period = 0;
            (*cores)[p].nsat_pms[i].burn_in = 0;
            (*cores)[p].nsat_pms[i].x_thr = 0;
            (*cores)[p].nsat_pms[i].modg_state = 2;
            (*cores)[p].nsat_pms[i].is_xreset_on = NULL;
            (*cores)[p].nsat_pms[i].is_flag_Xth = false;
        }

        /* Allocate the structure for learning parameters */
        (*cores)[p].lrn_pms = alloc(learning_params, 8);

        /* Allocate and initialize monitoring files structure */
        (*cores)[p].files = alloc(mon_files, 1);
        (*cores)[p].files->fs = NULL;
        (*cores)[p].files->fw = NULL;
        (*cores)[p].files->fsa = NULL; 
        (*cores)[p].files->fr = NULL;

        /* Build external events names for each core */
        (*cores)[p].ext_evts_fname = gen_ext_evts_fname(fname->ext_events, p);
        
        /* Allocate and initialize lists */
        (*cores)[p].events = alloc(array_list, 1);
        array_list_init(&(*cores)[p].events, 1);

        (*cores)[p].trans_events = alloc(array_list, 1);
        array_list_init(&(*cores)[p].trans_events, 1);
        
        (*cores)[p].ext_events = alloc(array_list, 1);
        array_list_init(&(*cores)[p].ext_events, 1);
        
        (*cores)[p].nsat_events = alloc(array_list, 1);
        array_list_init(&(*cores)[p].nsat_events, 1);
        
        (*cores)[p].mon_events = alloc(array_list, 1);
        array_list_init(&(*cores)[p].mon_events, 1);

        (*cores)[p].ext_caspk = alloc(array_list, 1);
        array_list_init(&(*cores)[p].ext_caspk, 1);

        (*cores)[p].nsat_caspk = alloc(array_list, 1);
        array_list_init(&(*cores)[p].nsat_caspk, 1);

        /* Allocate and initialize vars structure */
        (*cores)[p].vars = alloc(core_vars, 1);
        (*cores)[p].vars->tX = NULL;
        (*cores)[p].vars->acm = NULL;
        (*cores)[p].vars->g = NULL;
        (*cores)[p].vars->xinit = NULL;
        (*cores)[p].vars->rec_spk_on = NULL;

        (*cores)[p].shared_memory = NULL;

        (*cores)[p].curr_time = 0;
        (*cores)[p].g_pms = NULL;
        (*cores)[p].ext_neuron = NULL;
        (*cores)[p].nsat_neuron = NULL;

        (*cores)[p].core_id = p;
    }
}


/* ************************************************************************
 * INITIALIZE_CORES_VARS: This function initializes the basic parameters 
 * for each core.
 *
 * Args : 
 *  core (nsat_core *)      : NSAT core data structure
 *  num_cores (int)         : Number of cores
 *FF, OFF, OFF]]

    # Sign matrix (group 0)
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]

    # Sign matrix (group 0)
    c
 * Returns :
 *  void
 **************************************************************************/
void initialize_cores_vars(nsat_core *core, int num_cores) {
    int size, p;

    for (p = 0; p < num_cores; ++p) {
        size = core[p].core_pms.num_neurons * core[p].core_pms.num_states;

        core[p].vars->tX = alloc_zeros(STATETYPE, size);
        mem_test(core[p].vars->tX);

        core[p].vars->acm = alloc_zeros(STATETYPE, size);
        mem_test(core[p].vars->acm);

        core[p].vars->g = alloc_zeros(STATETYPE, size);
        mem_test(core[p].vars->g);
    }
}
/* ************************************************************************
 * INITIALIZE_CORES_NEURONS: This function initializes external and nsat
 * neurons units.
 * 
 * Args : 
 *  core (nsat_core *)      : NSAT core data structure
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void initialize_cores_neurons(nsat_core **cores, int num_cores) {
    int j, k, p;

    for (p = 0; p < num_cores; ++p) {
        (*cores)[p].nsat_neuron = alloc(unit, (*cores)[p].core_pms.num_neurons);
        mem_test((*cores)[p].nsat_neuron);
        (*cores)[p].ext_neuron = alloc(unit, (*cores)[p].core_pms.num_inputs);
        mem_test((*cores)[p].ext_neuron);

        for(j = 0; j < (*cores)[p].core_pms.num_inputs; ++j) {
           (*cores)[p].ext_neuron[j].is_spk_rec_on = false;
            (*cores)[p].ext_neuron[j].nsat_ptr = NULL;
            (*cores)[p].ext_neuron[j].syn_ptr = (list_syn **)malloc((*cores)[p].core_pms.num_states*sizeof(list_syn *));
            for (k = 0; k < (*cores)[p].core_pms.num_states; ++k) {
                (*cores)[p].ext_neuron[j].syn_ptr[k] = alloc_list_syn(); 
            }
            (*cores)[p].ext_neuron[j].s = NULL;
            (*cores)[p].ext_neuron[j].counter = -2000;
            (*cores)[p].ext_neuron[j].ref_period = 0;
            (*cores)[p].ext_neuron[j].ptr_cores = NULL;
            (*cores)[p].ext_neuron[j].router_size = 0;
            (*cores)[p].ext_neuron[j].is_transmitter = false;
        } 
    
        for (j = 0; j < (*cores)[p].core_pms.num_neurons; ++j) {
            (*cores)[p].nsat_neuron[j].spk_counter = 0;
            (*cores)[p].nsat_neuron[j].is_spk_rec_on = false;
            (*cores)[p].nsat_neuron[j].nsat_ptr = NULL;
            (*cores)[p].nsat_neuron[j].syn_ptr = (list_syn **)malloc((*cores)[p].core_pms.num_states*sizeof(list_syn *));
            for (k = 0; k < (*cores)[p].core_pms.num_states; ++k) {
                (*cores)[p].nsat_neuron[j].syn_ptr[k] = alloc_list_syn();
            }
            (*cores)[p].nsat_neuron[j].s = alloc(state, (*cores)[p].core_pms.num_states);
            for (k = 0; k < (*cores)[p].core_pms.num_states; ++k) {
                (*cores)[p].nsat_neuron[j].s[k].lrn_ptr = NULL;
                (*cores)[p].nsat_neuron[j].s[k].x = (*cores)[p].vars->xinit[j*(*cores)[p].core_pms.num_states+k];
            }
            (*cores)[p].nsat_neuron[j].counter = -2000;
            (*cores)[p].nsat_neuron[j].ref_period = 0;
            (*cores)[p].nsat_neuron[j].ptr_cores = NULL;
            (*cores)[p].nsat_neuron[j].router_size = 0;
            (*cores)[p].nsat_neuron[j].is_transmitter = false;
        }

        for (j = 0; j < (*cores)[p].core_pms.num_recs_on; ++j) {
            (*cores)[p].nsat_neuron[(*cores)[p].vars->rec_spk_on[j]].is_spk_rec_on = true;
        }
    }
}


int bin_file_size(FILE *fp) {
    size_t size;

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    rewind(fp);

    return size;
}


/* ************************************************************************
 * INITIALIZE_INCORES_CONNECTIONS: This function initializes the NSAT
 * synaptic connections reading the synaptic strengths from a file per core.
 *
 * Args :
 *  fname (char *)          : Input file name
 *  core (nsat_core *)      : NSAT cores data structure
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void initialize_incores_connections(fnames *fname, nsat_core **core,
                                    int num_cores) {
    int p;
    int i, j, sm_size;
    int tot_num_neurons, non_zero_elements = 0, parity_check = 0;
    int src, dst, stt, ptr;
    char *w_fname = NULL, *ptr_fname = NULL;

    FILE *fp, *fw;

    for (p = 0; p < num_cores; ++p) {
        ptr_fname = gen_fname(fname->syn_ptr_table, p, 1);
        if(!(fp = fopen(ptr_fname, "rb"))) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("File %s not found!\n", fname->syn_ptr_table);
            exit(-1);
        }
        dealloc(ptr_fname);

        w_fname = gen_fname(fname->syn_wgt_table, p, 1);
        if(!(fw = fopen(w_fname, "rb"))) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("File %s not found!\n", fname->syn_wgt_table);
            exit(-1);
        }
        dealloc(w_fname);
        sm_size = (int) bin_file_size(fw) / 4;

        /* Allocate shared memory */
        (*core)[p].sm_size = sm_size;
        (*core)[p].shared_memory = alloc_zeros(WTYPE, sm_size);
        
        /* Populate shared memory */
        fread((*core)[p].shared_memory, sizeof(WTYPE), (*core)[p].sm_size, fw);
        fclose(fw); 

        /* Check synaptic strengths range */
        for(j = 0; j < (*core)[p].sm_size; ++j) {
            if ((*core)[p].g_pms->is_check_wlim_on) {
                check_synaptic_strengths(&(*core)[p].shared_memory[j],
                                         (*core)[p].g_pms->syn_precision);
            }
        }

        if (!(*core)[p].core_pms.is_ext_evts_on) {
            non_zero_elements = 0;
            parity_check = 0;

            /* Read the pointer table */
            fread(&non_zero_elements, sizeof(int), 1, fp);
            for (i = 0; i < non_zero_elements; ++i) {
                fread(&src, sizeof(int), 1, fp);
                fread(&dst, sizeof(int), 1, fp);
                fread(&stt, sizeof(int), 1, fp);
                fread(&ptr, sizeof(int), 1, fp);

                if (src < (*core)[p].core_pms.num_neurons) {
                    push_syn(&(*core)[p].nsat_neuron[src].syn_ptr[stt],
                             dst,
                             &(*core)[p].shared_memory[ptr]);
                    parity_check++;
                } else {
                    ;
                }
            }

            /* Check if the number of input weights is valid */
            if (parity_check != non_zero_elements) {
                printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                printf("Invalid synaptic weights dimension (input, 322)!\n");
                printf("%d %d %d\n", parity_check,
                                           non_zero_elements,
                                           (*core)[p].core_pms.num_neurons);
                fclose(fp);
                exit(-1);
            }

            /* Count number of NSAT synapses */
            count_synapses(&(*core)[p].nsat_neuron,
                           &(*core)[p].syn->tot_nsat_syn_num,
                           (*core)[p].core_pms.num_neurons,
                           (*core)[p].core_pms.num_states);
            (*core)[p].syn->tot_ext_syn_num = 0;
        
        } else {
            non_zero_elements = 0;
            parity_check = 0;
            tot_num_neurons = (*core)[p].core_pms.num_inputs +
                              (*core)[p].core_pms.num_neurons;
        
            /* Read the non zero elements from fp */
            fread(&non_zero_elements, sizeof(int), 1, fp);
            for (i = 0; i < non_zero_elements; ++i) {
                fread(&src, sizeof(int), 1, fp);
                fread(&dst, sizeof(int), 1, fp);
                fread(&stt, sizeof(int), 1, fp);
                fread(&ptr, sizeof(int), 1, fp);

                if (src < (*core)[p].core_pms.num_inputs) {
                    if(dst < (*core)[p].core_pms.num_inputs){
                        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                        printf("Invalid synaptic weights destinations!\n");
                        exit(-1);
                    }
                    dst = abs(dst - (*core)[p].core_pms.num_inputs);
                    push_syn(&(*core)[p].ext_neuron[src].syn_ptr[stt],
                             dst,
                             &(*core)[p].shared_memory[ptr]);
                    parity_check++;
                } else if ((src >= (*core)[p].core_pms.num_inputs) && 
                           (src < tot_num_neurons)) {
                    src = abs(src - (*core)[p].core_pms.num_inputs);
                    dst = abs(dst - (*core)[p].core_pms.num_inputs);
                    push_syn(&(*core)[p].nsat_neuron[src].syn_ptr[stt],
                             dst,
                             &(*core)[p].shared_memory[ptr]);
                    parity_check++;
                } else {
                    ;
                }
            }

            /* Check if the number of input weights is valid */
            if (parity_check != non_zero_elements) {
                printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                printf("Invalid synaptic weights dimension (inputs)!\n");
                fclose(fp);
                exit(-1);
            }

            /* Count number of external synapses */
            count_synapses(&(*core)[p].ext_neuron,
                           &(*core)[p].syn->tot_ext_syn_num,
                           (*core)[p].core_pms.num_inputs,
                           (*core)[p].core_pms.num_states);

            /* Count number of NSAT synapses */
            count_synapses(&(*core)[p].nsat_neuron,
                           &(*core)[p].syn->tot_nsat_syn_num,
                           (*core)[p].core_pms.num_neurons,
                           (*core)[p].core_pms.num_states);
        }

        fclose(fp);
    }
}


/* ************************************************************************
 * INITIALIZE_CORES_CONNECTIONS: This function initializes inter-core
 * connections.
 *
 * Args : 
 *  fname (char *)          : Input filename (routing tableau)
 *  core (nsat_core *)      : NSAT core data structures
 *
 * Returns :
 *  void
 **************************************************************************/
void initialize_cores_connections(char *fname, nsat_core *core)
{
    int p, i;
    int src_core_id, dst_core_id;
    int src_neuron_id, dst_neuron_id;
    int num_connections = 0, num_iterations = 0;
    FILE *fp = NULL; 

    if(!(fp = fopen(fname, "rb"))) {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("File %s not found!\n", fname);
        exit(-1);
    }

    fread(&num_iterations, sizeof(int), 1, fp);
    if (!num_iterations) {
        printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
        printf("No inter-core connections found!\n");
    }

    for (p = 0; p < num_iterations; ++p) {
        /* Read source core ID */
        fread(&src_core_id, sizeof(int), 1, fp);
        if (src_core_id > core[0].g_pms->num_cores - 1) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("Invalid source core ID %d detected in intercore initialization process!\n",
                    src_core_id);
            exit(-1);
        }
        /* Read source neuron ID */
        fread(&src_neuron_id, sizeof(int), 1, fp);
        src_neuron_id = abs(src_neuron_id - core[src_core_id].core_pms.num_inputs);
        if (src_neuron_id > core[src_core_id].core_pms.num_neurons - 1) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("Invalid source neuron ID %d detected in intercore initialization process!\n",
                   src_neuron_id);
            exit(-1);
        }
        /* Read number of connections */
        fread(&num_connections, sizeof(int), 1, fp);
        
        /* Assign all the values and allocate memories */
        core[src_core_id].nsat_neuron[src_neuron_id].router_size = num_connections;
        core[src_core_id].nsat_neuron[src_neuron_id].ptr_cores = alloc(router, num_connections);
        core[src_core_id].nsat_neuron[src_neuron_id].is_transmitter = true;
        /* Read all the destinations */
        for (i = 0; i < num_connections; ++i) {
            fread(&dst_core_id, sizeof(int), 1, fp);
            if (dst_core_id > core[0].g_pms->num_cores - 1) {
                printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                printf("Invalid destination core ID %d detected in intercore initialization process!\n",
                        dst_core_id);
                exit(-1);
            }
            fread(&dst_neuron_id, sizeof(int), 1, fp);
            if (dst_neuron_id > core[dst_core_id].core_pms.num_inputs - 1) {
                printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                printf("Invalid destination neuron ID %d detected in intercore initialization process!\n",
                        dst_neuron_id);
                exit(-1);
            }
            core[src_core_id].nsat_neuron[src_neuron_id].ptr_cores[i].dst_core_id = dst_core_id;
            core[src_core_id].nsat_neuron[src_neuron_id].ptr_cores[i].dst_neuron_id = dst_neuron_id;
        }
    }
    
    fclose(fp);
}
