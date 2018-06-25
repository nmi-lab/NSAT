/* ************************************************************************
 * NSATlib_v2 This is a C implementation of the NSATlib_v2 python
 * script. It simulates the NSAT. 
 * Copyright (C) <2016>  UCI, Georgios Detorakis (gdetor@protonmail.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
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

extern inline void progress_bar(int x, int n);

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_barrier_t barrier;


void *nsat_thread(void *args)
{
    int t, q, i;

    nsat_core *core = (nsat_core *)args;
    clock_t t_s, t_f;
    int id, err_flag = false;
    FILE *fext;

    if (core->core_pms.is_ext_evts_on) {
        fext = fopen(core->ext_evts_fname, "rb");
        if (!fext) {
            printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
            printf("No external events file for Core %u !\n", core->core_id);
            err_flag = true;
        }
    }

    t_s = clock();
    for (t = 1; t < core->g_pms->ticks; ++t) {
        if (core->core_pms.is_ext_evts_on && err_flag == false) {
            get_external_events_per_core(fext, &core, t);
        }

        if ((t%1000)==0 && core->core_id == 0) printf("Time %d\n",t);
        core->curr_time = t;
        nsat_dynamics((void *)&core[0]);

        pthread_barrier_wait(&barrier);

        pthread_mutex_lock(&lock);
        if (core->g_pms->is_routing_on) {
            for(i = 0; i < core->trans_events->length; ++i) {
                id = core->trans_events->array[i];
                for (q = 0; q < core->nsat_neuron[id].router_size; ++q) {
                    array_list_push(&core[core->nsat_neuron[id].ptr_cores[q].dst_core_id].ext_events,
                                    core->nsat_neuron[id].ptr_cores[q].dst_neuron_id,
                                    t, 1);
                }
            }
            array_list_clean(&core->trans_events, 1);
        }
        pthread_mutex_unlock(&lock);

        nsat_events_and_learning((void *)&core[0]);

        pthread_barrier_wait(&barrier);
    }
    t_f = clock();
    printf("Thread %u execution time: %lf seconds\n",
           core->core_id, (double) (t_f - t_s) / CLOCKS_PER_SEC);

    if (core->core_pms.is_ext_evts_on && fext!=NULL) {
        fclose(fext);
    }

    return NULL;
}


int iterate_nsat(fnames *fname) {
    int p;
    clock_t t0, tf;
    FILE *fp=NULL;

    global_params g_pms;
    nsat_core *cores = NULL;

    pthread_t *cores_t=NULL;

    /* Open parameters file */
    fp = fopen(fname->params, "rb");
    file_test(fp, fname->params);

    /* Read global parameters */
    read_global_params(fp, &g_pms);

    /* Allocate memory for all the cores */
    cores = alloc(nsat_core, g_pms.num_cores);
    allocate_cores(&cores, fname, g_pms.num_cores);
    for(p = 0; p < g_pms.num_cores; ++p) { cores[p].g_pms = &g_pms; }

    /* Initialize RNG with seed */
    if (g_pms.is_bm_rng_on) {
        pcg32_srandom(g_pms.rng_init_state, g_pms.rng_init_seq);
    }

    /* Read/Load cores basic parameters */
    read_core_params(fp, cores, g_pms.num_cores); 

    /* Read/Load cores neurons NSAT parameters */
    read_nsat_params(fp, cores, g_pms.num_cores);

    /* Read/Load cores learning parameters */
    read_lrn_params(fp, cores, g_pms.num_cores);

    /* Read/Load monitors params */
    read_monitor_params(fp, cores, g_pms.num_cores);
    fclose(fp);

    /* Initialize cores' temporary arrays */
    initialize_cores_vars(cores, g_pms.num_cores);

    /* Initialize units */
    initialize_cores_neurons(&cores, g_pms.num_cores);

    /* Neurons point to their NSAT parameters group */
    nsat_pms_groups_map_file(fname->nsat_params_map, cores, g_pms.num_cores);

    /* Neurons states point to their learning parameters group */
    learning_pms_groups_map_file(fname->lrn_params_map, cores, g_pms.num_cores);

    /* Print parameters in a file */
    print_params2file(fname, cores, &g_pms);

    /* Load all the synaptic weights to units */
    initialize_incores_connections(fname, &cores, g_pms.num_cores);

    /* Load all the inter-core connections */
    initialize_cores_connections(fname->l1_conn, cores);

    /* Open all necessary monitor files */
    open_cores_monitor_files(cores, fname, g_pms.num_cores);
    
    /* Initialize all threads variables */
    cores_t = alloc(pthread_t, g_pms.num_cores);
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_barrier_init(&barrier, NULL, g_pms.num_cores);

    /* Check if clock is on */
    t0 = clock();

    /* Create and run threads (NSAT Cores) */
    for (p = 0; p < g_pms.num_cores; ++p) {
        pthread_create(&cores_t[p], NULL, nsat_thread, (void *)&cores[p]);
    }

    /* Join threads */
    for (p = 0; p < g_pms.num_cores; ++p) {
        pthread_join(cores_t[p], NULL);
    }

    /* If clock is turned on then print out the execution time */
    tf = clock();
    if (g_pms.is_clock_on) {
        printf("Simulation execution time: %lf seconds\n",
               (double) (tf - t0) / CLOCKS_PER_SEC);
    }

    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&cond);
    pthread_barrier_destroy(&barrier);

    /* Write spikes events */
    write_spikes_events(fname, cores, g_pms.num_cores);

    /* Write final synaptic strengths */
    write_final_weights(fname, cores, g_pms.num_cores);

    /* Write the shared memories per core */
    write_shared_memories(fname, cores, g_pms.num_cores);

    /* Close all the monitor files */
    close_cores_monitor_files(cores, g_pms.num_cores);

    /* Write spike statistics */
    write_spike_statistics(fname, cores, g_pms.num_cores);

    /* Destroy neurons parameters groups and clean up memories */
    dealloc_cores(&cores, g_pms.num_cores);
    dealloc(cores);
    if (!g_pms.is_single_core)
        dealloc(cores_t);

    return 0;
}
