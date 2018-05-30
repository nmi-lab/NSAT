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
    unsigned long long t, q, i;

    nsat_core *core = (nsat_core *)args;
    clock_t t_s, t_f;
    unsigned long long id;
    FILE *fext;

    if (core->core_pms.is_ext_evts_on) {
        fext = fopen(core->ext_evts_fname, "rb");
        if (!fext) {
            printf(ANSI_COLOR_YELLOW "WARNING:  " ANSI_COLOR_RESET);
            printf("No external events file for Core %u !\n", core->core_id);
        }
    }

    t_s = clock();
    for (t = 1; t < core->g_pms->ticks; ++t) {
        if (core->core_pms.is_ext_evts_on) {
            get_external_events_per_core(fext, &core, t);
        }

        core->curr_time = t;
        nsat_dynamics((void *)&core[0]);

        pthread_barrier_wait(&barrier);

        if (core->g_pms->is_routing_on) {
            pthread_mutex_lock(&lock);
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

        pthread_barrier_wait(&barrier);

        nsat_events_and_learning((void *)&core[0]);
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
#if OLD == 1
    iterate_nsat_old(fname);
#else
    iterate_nsat_new(fname);
#endif
    return 0;
}

#if OLD == 0
int iterate_nsat_new(fnames *fname) {
    unsigned int p;
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
#endif

/* ************************************************************************
 * ITERATE_NSAT: This function serves as interface for the NSAT function.
 * It iterates over a number of epochs and is used by Python wrappers in 
 * order to execute NSAT simulations. 
 *
 * Args : 
 *  fname (fnames *)            : File names structs
 *
 * Returns :
 *  void
 **************************************************************************/
#if OLD == 1
int iterate_nsat_old(fnames *fname) {
    unsigned int p;
    unsigned long long t;
    clock_t t0, tf;
    FILE *fp=NULL;
#if DAVIS==0
    FILE *fext=NULL;
    char *ext_events_fname=NULL;
#else
    int fd;
#endif
        
    global_params g_pms;
    nsat_core *cores = NULL;

    void *exit_status=NULL;
    pthread_t *cores_t=NULL;

    int q;
    spk_list_node *ptr = NULL;

    /* Open parameters file */
    fp = fopen(fname->params, "rb");
    file_test(fp, fname->params);

    /* Read global parameters */
    read_global_params(fp, &g_pms);

    /* Allocate memory for all the cores */
    cores_t = alloc(pthread_t, g_pms.num_cores);
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
    initialize_cores_neurons(cores, g_pms.num_cores);

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

    /* Check if clock is on */
    t0 = clock();

#if DAVIS==0
    if (cores[0].core_pms.is_ext_evts_on) {
        /* ext_events_fname = add_extension(fname->ext_events); */
        /* fext = fopen(ext_events_fname, "rb"); */
        fext = fopen(fname->ext_events, "rb");
        file_test(fext, ext_events_fname);
    }
#else
    if ((fd = open(fname->ext_events, O_RDONLY)) < 0) {
        printf("File %s not found!\n", fname->ext_events);
        exit(-1);
    }
#endif

    /* Simulation time loop */
#if DAVIS == 0
    // progress_bar(0, g_pms.ticks);
#endif

    for(t = 1; t < g_pms.ticks; ++t) {
        /* Get external events */
#if DAVIS==0
        get_external_events(fext, &cores, t, g_pms.num_cores);
#else
        get_davis_events(fd, &cores);
        if ((t%1000) == 0)  printf("Timestep %d\n", t);
#endif
        if (!g_pms.is_single_core) {
            /* Run the threads */
            for (p = 0; p < g_pms.num_cores; ++p) {
                cores[p].curr_time = t;
                pthread_create(&cores_t[p],
                               NULL,
                               nsat_dynamics,
                               (void *)&cores[p]);
            }

            /* Threads barrier */
            for (p = 0; p < g_pms.num_cores; ++p) {
                pthread_join(cores_t[p], &exit_status);
            }
        } else {
            cores[0].curr_time = t;
            nsat_dynamics((void *)&cores[0]);
        }

        /* Routing spikes between NSAT cores */
        if (g_pms.is_routing_on) {
            for (p = 0; p < g_pms.num_cores; ++p) {
                /* Route NSAT spikes between cores */
                ptr = cores[p].trans_events->head;
                while(ptr != NULL) {
                    for (q = 0; q < cores[p].nsat_neuron[ptr->id].router_size; ++q) {
                        push_spk(&cores[cores[p].nsat_neuron[ptr->id].ptr_cores[q].dst_core_id].ext_events,
                                 cores[p].nsat_neuron[ptr->id].ptr_cores[q].dst_neuron_id, t);   
                    }
                    ptr = ptr->next;
                }
                ptr = NULL;

                /* Destroy temporary list */
                destroy_list_spk(&cores[p].trans_events);
            }
        }

        if (!g_pms.is_single_core) {
            /* Run the intercore spikes */
            for (p = 0; p < g_pms.num_cores; ++p) {
                pthread_create(&cores_t[p],
                               NULL,
                               nsat_events_and_learning,
                               (void *)&cores[p]);
            }

            /* Threads barrier */
            for (p = 0; p < g_pms.num_cores; ++p) {
                pthread_join(cores_t[p], &exit_status);
            }
        } else {
            nsat_events_and_learning((void *)&cores[0]);
        }

#if DAVIS == 0
        // progress_bar(t, g_pms.ticks);
#endif

#if DAVIS == 1
        if ((t % 1000) == 0)
            write_shared_memories(fname, cores, g_pms.num_cores);
#endif
    }

    /* If clock is turned on then print out the execution time */
    tf = clock();
    //if (g_pms.is_clock_on) {
        printf("Simulation execution time: %lf seconds\n",
               (double) (tf - t0) / CLOCKS_PER_SEC);
    //}

#if DAVIS==0
    if (cores[0].core_pms.is_ext_evts_on) {
        fclose(fext);
        dealloc(ext_events_fname);
    }
#else
   /* Close external events file */
    if (cores[0].core_pms.is_ext_evts_on) {
        (void) close(fd);
    }

    for(p = 0; p < g_pms.num_cores; ++p) {
        fclose(cores[p].files->event_file);
    }
#endif

#if DAVIS == 0
    /* Write spikes events */
    write_spikes_events(fname, cores, g_pms.num_cores);
#endif

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
#endif
