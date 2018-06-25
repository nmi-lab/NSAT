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
 * PMS_GRPOUPS_MAP_FILE: This function maps the parameters groups to the
 * NSAT neurons. The input file should contain as many as the NSAT neurons 
 * integers and each integer should represent the number of the corresponding
 * parameters group. 
 *
 * Args : 
 *  fname (char *)          : Input file name
 *  cores (nsat_cores *)    : NSAT cores data structures array
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void nsat_pms_groups_map_file(char *fname, nsat_core *cores,
                              int num_cores) {
    int p, res, j;
    int *map = NULL;
    FILE *fp = NULL;

    fp = fopen(fname, "rb");
    file_test(fp, fname);

    for (p = 0; p < num_cores; ++p) {
        map = alloc_zeros(int, cores[p].core_pms.num_neurons);

        res = fread(map, sizeof(int), cores[p].core_pms.num_neurons, fp);
        fread_test(res, cores[p].core_pms.num_neurons);

        if (max(map, cores[p].core_pms.num_neurons) > 
                cores[p].core_pms.num_nsat_params_groups) {
            dealloc(map);
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("The map exceeds the size of parameters groups!\n");
            exit(-1);
        }

        for(j = 0; j < cores[p].core_pms.num_neurons; ++j) {
            cores[p].nsat_neuron[j].nsat_ptr = &cores[p].nsat_pms[map[j]];
        }
        
        dealloc(map);
    }

    fclose(fp);
}


/* ************************************************************************
 * LEARNING_PMS_GROUPS_MAP_FILE: This function maps learning parameters
 * groups to the NSAT neurons. The input file should contain as many as
 * the NSAT neurons integers and each integer should represent the number
 * of the corresponding parameters group. 
 *
 * Args : 
 *  fname (char *)           : Input file name
 *  core (nsat_core *)       : NSAT core data structure
 *  num_neurons (int)        : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void learning_pms_groups_map_file(char *fname, nsat_core *core,
                                  int num_cores) {
    int k, p, res; 
    int j, tmp;
    int *map = NULL;
    FILE *fp = NULL;

    fp = fopen(fname, "rb");
    file_test(fp, fname);

    for(p = 0; p < num_cores; ++p) {
        if (core[p].core_pms.is_learning_on) {
            tmp = core[p].core_pms.num_neurons * (int) core[p].core_pms.num_states;
            map = alloc_zeros(int, tmp);
                                   
            res = fread(map, sizeof(int), tmp, fp);
            fread_test(res, tmp);

            if (max(map, core[p].core_pms.num_neurons) > 
                    core[p].core_pms.num_learning_pms_groups) {
                dealloc(map);
                printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                printf("The map exceeds the size of parameters groups!\n");
                exit(-1);
            }

            for (j = 0; j < core[p].core_pms.num_neurons; ++j) {
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    core[p].nsat_neuron[j].s[k].lrn_ptr = 
                        &core[p].lrn_pms[map[j*core[p].core_pms.num_states+k]];
                }
            }

            dealloc(map);
            /* check_stdp_kernel(fname->stdp_fun, core->lrn_pms,
                              core->core_pms.tstdpmax);  */
        }
    }
    fclose(fp);
}


/* ************************************************************************
 * READ_GLOBAL_PARAMS: This function loads simulations global paremeters 
 * from the corresponding binary file.
 *
 * Args : 
 *  fp (FILE *)             : Input file pointer
 *  pms (global_params *)   : Global parameters (by reference)
 *
 * Returns :
 *  void
 **************************************************************************/
void read_global_params(FILE *fp, global_params *pms) {
    int res;
    int synapse_prec = 0;

    res = fread(&pms->num_cores, sizeof(int), 1, fp);
    fread_test(res, 1);
    if (pms->num_cores <= 0) {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("No NSAT cores found!\n");
        printf("Please give the number of cores (>0)\n");
        exit(-1);
    }
    res = fread(&pms->is_single_core, sizeof(bool), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->is_routing_on, sizeof(bool), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->ticks, sizeof(int), 1, fp);
    fread_test(res, 1);
    if(pms->ticks > INT_MAX) {
        printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
        printf("Simulation ticks overflow! Spike list will re-iterate!\n");
    }
    res = fread(&pms->rng_init_state, sizeof(int), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->rng_init_seq, sizeof(int), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->is_bm_rng_on, sizeof(bool), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->is_clock_on, sizeof(bool), 1, fp);
    fread_test(res, 1);
    res = fread(&pms->is_check_wlim_on, sizeof(bool), 1, fp);
    fread_test(res, 1);
    res = fread(&synapse_prec, sizeof(int), 1, fp);
    fread_test(res, 1);
    pms->syn_precision = pow(2, synapse_prec);
} 


/* ************************************************************************
 * READ_CORE_PARAMS: This function loads simulations basic cores 
 * paremeters from a binary file.
 *
 * Args : 
 *  fp (FILE *)         : Input file pointer
 *  core (nsat_core *)  : NSAT core data structure
 *  num_cores (int)     : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void read_core_params(FILE *fp, nsat_core *core, int num_cores) {
    int size_v, tmp, i;
    int p, res;

    for (p = 0; p < num_cores; ++p) {
        core[p].core_pms.ext_syn_rec_ids = alloc(array_list, 1);
        array_list_init(&core[p].core_pms.ext_syn_rec_ids, 0);
        core[p].core_pms.nsat_syn_rec_ids = alloc(array_list, 1);
        array_list_init(&core[p].core_pms.nsat_syn_rec_ids, 0);
        
        res = fread(&core[p].core_pms.is_ext_evts_on, sizeof(bool), 1, fp);
        fread_test(res, 1);
        res = fread(&core[p].core_pms.is_learning_on, sizeof(bool), 1, fp);
        fread_test(res, 1);
        res = fread(&core[p].core_pms.is_learning_gated, sizeof(bool), 1, fp);
        fread_test(res, 1);
        res = fread(&core[p].core_pms.num_inputs, sizeof(int), 1, fp);
        fread_test(res, 1);

        if (check_compliance(core[p].core_pms.is_ext_evts_on,
                             core[p].core_pms.num_inputs) == -1) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("Core #%d: Number of input units is 0 and external events are ENABLED!\n", p);
            exit(-1);
        }

        if (check_compliance(core[p].core_pms.is_ext_evts_on,
                             core[p].core_pms.num_inputs) == -2) {
            printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Number of input units has been set to 0!\n", p);
            // core[p].core_pms.num_inputs = 0;
        }

        res = fread(&core[p].core_pms.num_neurons, sizeof(int), 1, fp);
        fread_test(res, 1);
        res = fread(&core[p].core_pms.num_states, sizeof(int), 1, fp);
        fread_test(res, 1);

        res = fread(&core[p].core_pms.num_nsat_params_groups, sizeof(int), 1, fp);
        fread_test(res, 1);
        if (core[p].core_pms.num_nsat_params_groups > N_NSATGROUPS) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Invalid number of NSAT parameters groups!\n", p);
            exit(-1);
        }

        res = fread(&core[p].core_pms.num_learning_pms_groups, sizeof(int), 1, fp);
        fread_test(res, 1);
        if (core[p].core_pms.num_learning_pms_groups > N_LRNGROUPS) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Invalid number of NSAT parameters groups!\n", p);
            exit(-1);
        }

        res = fread(&core[p].core_pms.timestamp, sizeof(int), 1, fp);
        fread_test(res, 1);
        if (check_compliance(core[p].core_pms.is_learning_on,
                             core[p].core_pms.num_learning_pms_groups) == -1) {
            printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Number of learning groups is 0 and learning is ENABLED!\n", p);
            exit(-1);
        }

        if (check_compliance(core[p].core_pms.is_learning_on,
                             core[p].core_pms.num_learning_pms_groups) == -2) {
            printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Number of learning groups has been set to 0!\n", p);
            core[p].core_pms.num_learning_pms_groups = 0;
        }

        res = fread(&size_v, sizeof(int), 1, fp);
        fread_test(res, 1);
        if (size_v > core[p].core_pms.num_neurons +
                     core[p].core_pms.num_inputs) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("CORE #%d: Invalid neurons range for monitoring synaptic weights!\n", p);
            exit(-1);
        }

        for (i = 0; i < size_v; ++i) {
            res = fread(&tmp, sizeof(int), 1, fp);
            fread_test(res, 1);
            if (tmp < core[p].core_pms.num_inputs) { 
                array_list_push(&core[p].core_pms.ext_syn_rec_ids, tmp, 0, 0);
            } else {
                tmp = tmp - core[p].core_pms.num_inputs;
                array_list_push(&core[p].core_pms.nsat_syn_rec_ids, tmp, 0, 0);
            }
        }
    }
}


/* ************************************************************************
 * READ_NSAT_PARAMS: This function loads NSAT paremeters per core from a
 * corresponding binary file.
 *
 * Args : 
 *  fp (FILE *)                 : Input file pointer
 *  core (nsat_core *)          : NSAT core data structure 
 *  num_cores (int)             : Number of cores
 * Returns :
 *  void
 **************************************************************************/
void read_nsat_params(FILE *fp, nsat_core *core, int num_cores) {
    int i, p; 
    int size_v, size_m;
    int size_i, size_n = 0;

    for (p = 0; p < num_cores; ++p) {
        /* Compute proper sizes for NSAT parameters vectors */
        size_m = core[p].core_pms.num_states * core[p].core_pms.num_states;
        size_v = core[p].core_pms.num_states;
        size_i = (int) core[p].core_pms.num_states * core[p].core_pms.num_neurons;

        for(i = 0; i < core[p].core_pms.num_nsat_params_groups; ++i) {
            fread(&core[p].nsat_pms[i].gate_low, sizeof(int), 1, fp);

            fread(&core[p].nsat_pms[i].gate_upper, sizeof(int), 1, fp);           
            
            fread(&core[p].nsat_pms[i].period, sizeof(int), 1, fp);           
            
            fread(&core[p].nsat_pms[i].burn_in, sizeof(int), 1, fp);           

            fread(&core[p].nsat_pms[i].t_ref, sizeof(int), 1, fp);

            fread(&core[p].nsat_pms[i].modg_state, sizeof(int), 1, fp);

            core[p].nsat_pms[i].prob = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].prob, sizeof(int), size_v, fp);
            
            core[p].nsat_pms[i].A = alloc_zeros(int, size_m);
            fread(core[p].nsat_pms[i].A, sizeof(int), size_m, fp);

            core[p].nsat_pms[i].sF = alloc_zeros(int, size_m);
            fread(core[p].nsat_pms[i].sF, sizeof(int), size_m, fp);

            core[p].nsat_pms[i].b = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].b, sizeof(int), size_v, fp);

            core[p].nsat_pms[i].x_reset = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].x_reset, sizeof(int), size_v, fp);

            core[p].nsat_pms[i].x_thlow = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].x_thlow, sizeof(int), size_v, fp);
            /* Check if there is an underflow and correct it */
            if (!check_intervals(core[p].nsat_pms[i].x_thlow, XTHLOW, XTHUP, size_v)) {
                printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
                printf("Values of x_thlow are not in [-XMAX, XMAX]\n");
            }
            check_underflow(&core[p].nsat_pms[i].x_reset,
                            core[p].nsat_pms[i].x_thlow, size_v);

            core[p].nsat_pms[i].is_xreset_on = alloc(bool, size_v);
            fread(core[p].nsat_pms[i].is_xreset_on, sizeof(bool), size_v, fp);

            core[p].nsat_pms[i].x_thup = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].x_thup, sizeof(int), size_v, fp);
            if (!check_intervals(core[p].nsat_pms[i].x_thup, XTHLOW, XTHUP, size_v)) {
                printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
                printf("Values of x_thup are not in [-XMAX, XMAX]\n");
            }

            core[p].nsat_pms[i].x_spike_incr = alloc_zeros(int, size_v);
            fread(core[p].nsat_pms[i].x_spike_incr, sizeof(int), size_v, fp);
            if(!check_upper_boundary(core[p].nsat_pms[i].x_spike_incr, size_v)) {
                printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
                printf("Spike increment is positive! Potential overflow!\n");
            }

            core[p].nsat_pms[i].sigma = alloc_zeros(int, size_v); 
            fread(core[p].nsat_pms[i].sigma, sizeof(int), size_v, fp);

            fread(&core[p].nsat_pms[i].is_flag_Xth, sizeof(bool), 1, fp);

            fread(&core[p].nsat_pms[i].x_thr, sizeof(int), 1, fp);

            core[p].nsat_pms[i].w_gain = alloc_zeros(int, size_v); 
            fread(core[p].nsat_pms[i].w_gain, sizeof(int), size_v, fp);

        }

        /* Read the initial conditions for every state of each neuron at once */
        core[p].vars->xinit = alloc_zeros(STATETYPE, size_i);
        mem_test(core[p].vars->xinit);
        fread(core[p].vars->xinit, sizeof(STATETYPE), size_i, fp);

        /* Read the recording flags (ON/OFF) for every neuron (NSAT) */
        fread(&size_n, sizeof(int), 1, fp);      /* number of ON recs */
        core[p].core_pms.num_recs_on = size_n;   /* Storage for other uses */
        core[p].vars->rec_spk_on = alloc(int, size_n);
        mem_test(core[p].vars->rec_spk_on);
        fread(core[p].vars->rec_spk_on, sizeof(int), size_n, fp);
        if (!check_recs_flags(core[p].vars->rec_spk_on, size_n,
                              core[p].core_pms.num_neurons)) {
            printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
            printf("Mismatched recording neurons indices!\n");
            exit(-1);
        }
    }
}


/* ************************************************************************
 * READ_LRN_PARAMS: This function loads learning parameters per core from 
 * a binary file. 
 *
 * Args : 
 *  fp (FILE *)             : Input file pointer
 *  core (nsat_core *)      : NSAT core data structure
 *  num_cores (int)         : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void read_lrn_params(FILE *fp, nsat_core *core, int num_cores) 
{
    int p;
    int i;

    for (p = 0; p < num_cores; ++p) {
        if (core[p].core_pms.is_learning_on) {
            fread(&core[p].core_pms.tstdpmax, sizeof(int), 1, fp);
            for (i = 0; i < core[p].core_pms.num_learning_pms_groups; ++i) {
                fread(&core[p].lrn_pms[i].tstdp, sizeof(int), 1, fp);
                fread(&core[p].lrn_pms[i].is_plastic_state, sizeof(bool), 1, fp);
                fread(&core[p].lrn_pms[i].is_stdp_on, sizeof(bool), 1, fp);
                fread(&core[p].lrn_pms[i].is_stdp_exp_on, sizeof(bool), 1, fp);
                if (!core[p].lrn_pms[i].is_stdp_on && 
                    core[p].lrn_pms[i].is_stdp_exp_on) {
                    printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
                    printf("STDP is disabled but expSTDP is enabled!\n");
                    exit(-1);
                }
                fread(core[p].lrn_pms[i].tca, sizeof(int), 2, fp);
                fread(core[p].lrn_pms[i].hica, sizeof(int), 3, fp);
                fread(core[p].lrn_pms[i].sica, sizeof(int), 3, fp);
                fread(core[p].lrn_pms[i].slca, sizeof(int), 3, fp);
                fread(core[p].lrn_pms[i].tac, sizeof(int), 2, fp);
                fread(core[p].lrn_pms[i].hiac, sizeof(int), 3, fp);
                fread(core[p].lrn_pms[i].siac, sizeof(int), 3, fp);
                fread(core[p].lrn_pms[i].slac, sizeof(int), 3, fp);
                fread(&core[p].lrn_pms[i].is_rr_on, sizeof(bool), 1, fp);
                fread(&core[p].lrn_pms[i].rr_num_bits, sizeof(int), 1, fp);
            }
        }
    }
}


/* ************************************************************************
 * READ_MONITOR_PARAMS: This function sets loads all the monitor parameters
 * per core (file pointers).
 *
 * Args :
 *  fp (FILE *)           : Input data file pointer  
 *  core (nsat_core *)    : NSAT core data structure
 *  num_cores (int)       : Number of cores
 *
 * Returns :
 *  void
 **************************************************************************/
void read_monitor_params(FILE *fp, nsat_core *cores, int num_cores) {
    int p;

    for (p = 0; p < num_cores; ++p) {
        fread(&cores[p].mon_pms->mon_states, sizeof(bool), 1, fp);
        fread(&cores[p].mon_pms->mon_weights, sizeof(bool), 1, fp);
        fread(&cores[p].mon_pms->mon_final_weights, sizeof(bool), 1, fp);
        fread(&cores[p].mon_pms->mon_spikes, sizeof(bool), 1, fp);
        fread(&cores[p].mon_pms->mon_stats, sizeof(bool), 1, fp);
    }
}

/* ************************************************************************
 * PRINT_PARAMS2FILE: This function prints all the parameters in a file.
 *
 * Args : 
 *  fn (fnames *)               : Data structure containing filenames
 *  core (nsat_core *)          : NSAT core data structure
 *  g_pms (global_params *)     : Global parameters struct (by reference)
 *
 * Returns :
 *  void
 **************************************************************************/
void print_params2file(fnames *fn, nsat_core *core, global_params *g_pms) {
    int j;
    int k, l, p;
    FILE *fp = NULL;
    
    if(!(fp = fopen(fn->check_pms, "w"))) {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("File %s not found!\n", fn->check_pms);
        exit(-1);
    }

    fprintf(fp, "Simulation files names\n");
    fprintf(fp, "NSAT parameters map: %s\n", fn->nsat_params_map);
    fprintf(fp, "Learning parameters map: %s\n", fn->lrn_params_map);
    fprintf(fp, "NSAT parameters file: %s\n", fn->params);
    fprintf(fp, "Synaptic weights table (IN): %s\n", fn->syn_wgt_table);
    fprintf(fp, "Synaptic weights pointer table (IN): %s\n", fn->syn_ptr_table);
    fprintf(fp, "External events: %s\n", fn->ext_events);
    fprintf(fp, "Synaptic weights matrix (OUT): %s\n", fn->synw);
    fprintf(fp, "Synaptic weights matrix final (OUT): %s\n", fn->synw_final);
    fprintf(fp, "Spikes list (time, id): %s\n", fn->events);
    fprintf(fp, "NSAT neurons states: %s\n", fn->states);
    fprintf(fp, "Check parameters: %s\n", fn->check_pms);
    fprintf(fp, "\n \n");

    fprintf(fp, "Total number of NSAT cores: %d\n", g_pms->num_cores);
    fprintf(fp, "Single core is ");
    print_enabled_or_disabled(fp, g_pms->is_single_core);
    fprintf(fp, "\n");
    fprintf(fp, "Routing scheme is ");
    print_enabled_or_disabled(fp, g_pms->is_routing_on);
    fprintf(fp, "\n");
    fprintf(fp, "Total number of simulation ticks: %d\n", g_pms->ticks);
    fprintf(fp, "Check synaptic strengths boundary: ");
    print_enabled_or_disabled(fp, g_pms->is_check_wlim_on);
    fprintf(fp, "\n");
    fprintf(fp, "Boundary of synaptic strengths: %d\n", g_pms->syn_precision);
    fprintf(fp, "RNG: ");
    print_rng_choice(fp, g_pms->is_bm_rng_on);
    fprintf(fp, "\n");
    fprintf(fp, "RNG seed: %d\n", g_pms->rng_init_state);
    fprintf(fp, "RNG initial sequence: %d\n", g_pms->rng_init_seq);
    fprintf(fp, "\n");
    fprintf(fp, "Benchmark clock is ");
    print_enabled_or_disabled(fp, g_pms->is_clock_on);
    fprintf(fp, "\n");

    for (p = 0; p < g_pms->num_cores; ++p) {
        fprintf(fp, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        fprintf(fp, "Core ID: %d\n", p);
        fprintf(fp, "Total number of inputs: %d\n", core[p].core_pms.num_inputs);
        fprintf(fp, "Total number of neurons: %d\n", core[p].core_pms.num_neurons);
        fprintf(fp, "Total number of states/neuron: %d\n", core[p].core_pms.num_states);
        fprintf(fp, "Dimension of NSAT parameters space: %d\n", core[p].core_pms.num_nsat_params_groups);
        fprintf(fp, "Dimension of NSAT learning parameters space: %d\n", core[p].core_pms.num_learning_pms_groups);
        fprintf(fp, "External events are ");
        print_enabled_or_disabled(fp, core[p].core_pms.is_ext_evts_on);
        fprintf(fp, "\n");
        fprintf(fp, "Learning is ");
        print_enabled_or_disabled(fp, core[p].core_pms.is_learning_on);
        fprintf(fp, "\n");


        fprintf(fp, "\n \n");

        fprintf(fp, "==================================\n");
        for (j = 0; j < core[p].core_pms.num_neurons; ++j) {
            fprintf(fp, "--------- Neuron ID ---------\n");
            fprintf(fp, "%d \n", j);

            fprintf(fp, "--------- Threshold value ---------\n");
            fprintf(fp, "%d \n", core[p].nsat_neuron[j].nsat_ptr->x_thr);

            fprintf(fp, "--------- Global modulator state ---------\n");
            fprintf(fp, "%d \n", core[p].nsat_neuron[j].nsat_ptr->modg_state);


            fprintf(fp, "--------- Matrix A ---------\n");
            for (k = 0; k < core[p].core_pms.num_states; ++k) {
                for(l = 0; l < core[p].core_pms.num_states; ++l) {
                    fprintf(fp, "%3d ", core[p].nsat_neuron[j].nsat_ptr->A[k*core[p].core_pms.num_states+l]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");

            fprintf(fp, "--------- Matrix sF ---------\n");
            for (k = 0; k < core[p].core_pms.num_states; ++k) {
                for (l = 0; l < core[p].core_pms.num_states; ++l) {
                    fprintf(fp, "%3d ", core[p].nsat_neuron[j].nsat_ptr->sF[k*core[p].core_pms.num_states+l]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");

            fprintf(fp, "--------- Refractory period ---------\n");
            fprintf(fp, "%d \n", core[p].nsat_neuron[j].nsat_ptr->t_ref);

            fprintf(fp, "--------- Threshold State Flag ---------\n");
            fprintf(fp, "%d \n", core[p].nsat_neuron[j].nsat_ptr->is_flag_Xth);


            fprintf(fp, "--------- NSAT Neuron Dynamics Parameters: ---------\n");
            fprintf(fp, "            b        x_init       x_reset   is_reset_on         x_thup       x_thlow          prob  x_spike_incr         sigma          W gain\n");
            for (k = 0; k < core[p].core_pms.num_states; ++k) {
                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->b[k]);

                fprintf(fp, "%13d ", core[p].nsat_neuron[j].s[k].x);

                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->x_reset[k]);

                print_enabled_or_disabled(fp, core[p].nsat_neuron[j].nsat_ptr->is_xreset_on[k]);

                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->x_thup[k]);

                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->x_thlow[k]);

                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->prob[k]);
                
                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->x_spike_incr[k]);
                
                fprintf(fp, "%13d ", core[p].nsat_neuron[j].nsat_ptr->sigma[k]);

                fprintf(fp, "%13d \n", core[p].nsat_neuron[j].nsat_ptr->w_gain[k]);
            }
            fprintf(fp, "\n");

            if (core[p].core_pms.is_learning_on) {
                fprintf(fp, "--------- Enabled/Disabled Randomized Rounding ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  ", k);
                    print_true_or_false(fp, core[p].nsat_neuron[j].s[k].lrn_ptr->is_rr_on);
                    fprintf(fp, "\n");
                }

                fprintf(fp, "--------- Randomized Rounding bits ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->rr_num_bits);
                }

                fprintf(fp, "STDP maximal time interval: %d\n", core[p].core_pms.tstdpmax);
                fprintf(fp, "--------- Enabled/Disabled State Plasticity ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  ", k);
                    print_true_or_false(fp, core[p].nsat_neuron[j].s[k].lrn_ptr->is_plastic_state);
                    fprintf(fp, "\n");
                }

                fprintf(fp, "--------- STDP Maximum Time ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d\n", k, core[p].nsat_neuron[j].s[k].lrn_ptr->tstdp);
                }

                fprintf(fp, "--------- Enabled/Disabled STDP ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  ", k);
                    print_true_or_false(fp, core[p].nsat_neuron[j].s[k].lrn_ptr->is_stdp_on);
                    fprintf(fp, "\n");
                }

                fprintf(fp, "--------- Enabled/Disabled expSTDP ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  ", k);
                    print_true_or_false(fp, core[p].nsat_neuron[j].s[k].lrn_ptr->is_stdp_exp_on);
                    fprintf(fp, "\n");
                }

                fprintf(fp, "--------- STDP Causal window widths ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->tca[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->tca[1]);
                }

                fprintf(fp, "------ STDP height of the causal box ------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hica[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hica[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hica[2]);
                }

                fprintf(fp, "--------- STDP sign of the causal update ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->sica[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->sica[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->sica[2]);
                }

                fprintf(fp, "--------- STDP causal slopes ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slca[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slca[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slca[2]);
                }

                fprintf(fp, "--------- STDP Acausal window widths ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->tac[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->tac[1]);
                }

                fprintf(fp, "------ STDP height of the acausal box ------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hiac[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hiac[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->hiac[2]);
                }

                fprintf(fp, "--------- STDP sign of the acausal update ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->siac[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->siac[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->siac[2]);
                }

                fprintf(fp, "--------- STDP acausal slopes ---------\n");
                for (k = 0; k < core[p].core_pms.num_states; ++k) {
                    fprintf(fp, "STATE: %u  %d  %d  %d\n", k,
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slac[0],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slac[1],
                            core[p].nsat_neuron[j].s[k].lrn_ptr->slac[2]);
                }

            }
            fprintf(fp, "\n");
            fprintf(fp, "==================================\n");
            fprintf(fp, "\n");
        }
        fprintf(fp, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    }
    fclose(fp);
}


/* ************************************************************************
 * CHECK_STDP_KERNEL: This function prints out in a file an instance of
 * the provided STDP kernel function. 
 *
 * Args : 
 *  fname (char *)             : Output filename
 *  pms (learning_params *)    : Learning parameters (STDP kernel description)
 *  tstdpmax (int)             : Maximum STDP time 
 *
 * Returns :
 *  void
 **************************************************************************/
void print_stdp_kernel(char *fname, learning_params *pms, int tstdpmax) {
    int i, res;
    int sign, dt = -(tstdpmax-1), tmp, tau;
    FILE *fp = NULL;
    
    if (!(fp = fopen(fname, "w"))) {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("File could not be opened!\n");
        exit(-1);
    }

    res = (tstdpmax - 1) + (tstdpmax - 1);
    for (i = 0; i <= res; ++i) {
        tmp = K_W(pms, dt, &sign, &tau);
        fprintf(fp, "%d  %f\n", dt, sign * pow(2, tmp));
        dt++;
    }
    fclose(fp);
}
