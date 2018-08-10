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
 * GET_EXTERNAL_EVENTS: This function reads the external events (spikes)
 * from a file.
 *
 * Args : 
 *  fp (FILE *)             : Input file name pointer
 *  core (nsat_core *)      : Cores structs array
 *  curr_time (int)         : Current time step
 *  num_cores (int)         : Number of neurons
 *
 * Returns :
 *  void
 **************************************************************************/
void get_external_events(FILE *fp, nsat_core **core,
                         int curr_time,
                         int num_cores)
{
    int res;
    int i, num_nonzeros, time, neuron_id;
    int core_id;

    if ((*core)[0].core_pms.is_ext_evts_on) {
        res = fread(&time, sizeof(int), 1, fp);
        fread_test(res, 1);
        res = fread(&num_nonzeros, sizeof(int), 1, fp);
        if (num_nonzeros != 0) {
            for (i = 0; i < num_nonzeros; ++i) {
                res = fread(&core_id, sizeof(int), 1, fp);
                fread_test(res, 1);
                if (core_id >= num_cores) {
                    printf("False core ID (%d) detected in external events file!\n",
                           core_id);
                    break;
                }
                res = fread(&neuron_id, sizeof(int), 1, fp);
                fread_test(res, 1);
                if (neuron_id >= (*core)[core_id].core_pms.num_inputs) {
                    printf("Core #%d: False destination neuron ID (%d) detected in exterrnal events file %s!\n",
                           core_id, neuron_id, (*core)[core_id].ext_evts_fname);
                    break;
                }
                if (time == curr_time) {
                    array_list_push(&(*core)[core_id].ext_events, neuron_id, curr_time, 1);
                }
            }
       }
    }
}


void get_external_events_per_core(FILE *fp, nsat_core **core,
                                  int curr_time)
{
    int res;
    int i, num_nonzeros, time, neuron_id;

    if (fp != NULL) {
        res = fread(&time, sizeof(int), 1, fp);
        fread_test(res, 1);
        res = fread(&num_nonzeros, sizeof(int), 1, fp);
        fread_test(res, 1);
        if (num_nonzeros != 0) {
            for (i = 0; i < num_nonzeros; ++i) {
                fread(&neuron_id, sizeof(int), 1, fp);
                fread_test(res, 1);
                if (neuron_id >= (*core)->core_pms.num_inputs) {
                    printf("False destination neuron ID (%d) detected in exterrnal events file %s!\n",
                           neuron_id,
                           (*core)->ext_evts_fname);
                    break;
                }
                if (time == curr_time) {
                    array_list_push(&(*core)->ext_events, neuron_id, curr_time, 1);
                }
            }
        }
    }
}


#if DAVIS == 1
/* extern pthread_mutex_t lock; */
/* extern route_list_ *davis_; */

void get_davis_events(int fd) {
    /* int i, time, num_events, n; */
    /* int buffer, core_id, neuron_id; */
    /* static int times = 0; */

    /* struct timespec tm; */

    /* tm.tv_sec = 0; */
    /* tm.tv_nsec = 10000000; */

    /* for(;;) { */
    /*     n = read(fd, &buffer, sizeof(int)); */
    /*     if (n > 0) { */
    /*         while (n != 4) { */
    /*             lseek(fd, -n, SEEK_CUR); */
    /*             n = read(fd, &buffer, sizeof(int)); */
    /*             nanosleep(&tm, &tm); */
    /*         } */
    /*         time = buffer; */

    /*         n = read(fd, &buffer, sizeof(int)); */
    /*         while (n != 4) { */
    /*             lseek(fd, -n, SEEK_CUR); */
    /*             n = read(fd, &buffer, sizeof(int)); */
    /*             nanosleep(&tm, &tm); */
    /*         } */
    /*         num_events = buffer; */

    /*         printf(" %d  %d ", time, num_events); */
    /*         for (i = 0; i < num_events; ++i) { */
    /*             n = read(fd, &buffer, sizeof(int)); */
    /*             while (n != 4) { */
    /*                 lseek(fd, -n, SEEK_CUR); */
    /*                 n = read(fd, &buffer, sizeof(int)); */
    /*                 nanosleep(&tm, &tm); */
    /*             } */
    /*             core_id = buffer; */

    /*             n = read(fd, &buffer, sizeof(int)); */
    /*             while (n != 4) { */
    /*                 lseek(fd, -n, SEEK_CUR); */
    /*                 n = read(fd, &buffer, sizeof(int)); */
    /*                 nanosleep(&tm, &tm); */
    /*             } */
    /*             neuron_id = buffer; */
    
    /*             printf(" %d  %d ", core_id, neuron_id); */
    /*             pthread_mutex_lock(&lock); */
    /*             array_list_push(&davis_[core_id].units, neuron_id, time, 1); */
    /*             /1* array_list_push(&davis_[core_id].units, neuron_id, *1/ */
    /*             /1*                 (*core)->curr_time, 1); *1/ */
    /*             pthread_mutex_unlock(&lock); */
    /*         } */
    /*         printf("\n"); */
    /*         break; */
    /*     } else if (n < 0) { */
    /*         break; */
    /*     } else { */
    /*         nanosleep(&tm, &tm); */
    /*         continue; */
    /*     } */
    /* } */
}
#endif


/* ************************************************************************
 * WRITE_FINAL_WEIGHTS: This function stores to a file the final 
 * synaptic weights. 
 *
 * Args : 
 *  fname (char *)         : Output file name
 *  core (nsat_core *)     : NSAT core pointer (to struct)
 *  num_cores (int)        : Number of states per NSAT neuron
 *
 * Returns :
 *  void
 **************************************************************************/
void write_final_weights(fnames *fname, nsat_core *core,
                         int num_cores) {
    int j;
    int k, p;
    int tmp = 0, tot = 0;
    size_t add;
    FILE *fp = NULL;
    char *filename = NULL;
    syn_list_node *ptr = NULL;

    for (p = 0; p < num_cores; ++p) {
        filename = gen_fname(fname->synw_final, p, 1);
        if (!(fp = fopen(filename, "wb"))){
            printf("File %s cannot be opened!\n", filename);
            exit(-1);
        }
        
        tot = core->syn->tot_ext_syn_num + core->syn->tot_nsat_syn_num;

        /* Write external neurons synaptic weights */
        fwrite(&tot, sizeof(int), 1, fp);
        for(j = 0; j < core->core_pms.num_inputs; ++j) {
            for(k = 0; k < core->core_pms.num_states; ++k) {
                ptr = core->ext_neuron[j].syn_ptr[k]->head;
                while(ptr != NULL) {
                    fwrite(&j, sizeof(int), 1, fp);
                    tmp = ptr->id + core->core_pms.num_inputs;
                    fwrite(&tmp, sizeof(int), 1, fp);
                    fwrite(&k, sizeof(int), 1, fp);
                    add = &core->shared_memory[0] - ptr->w_ptr;
                    fwrite(&add, sizeof(int), 1, fp);
                    ptr = ptr->next;
                }
                ptr = NULL;
            }
        }

        /* Write NSAT neurons synaptic weights */
        for(j = 0; j < core->core_pms.num_neurons; ++j) {
            for(k = 0; k < core->core_pms.num_states; ++k) {
                ptr = core->nsat_neuron[j].syn_ptr[k]->head;
                while(ptr != NULL) {
                    tmp = j + core->core_pms.num_inputs;
                    fwrite(&tmp, sizeof(int), 1, fp);
                    tmp = ptr->id + core->core_pms.num_inputs;
                    fwrite(&tmp, sizeof(int), 1, fp);
                    fwrite(&k, sizeof(int), 1, fp);
                    add = &core->shared_memory[0] - ptr->w_ptr;
                    fwrite(ptr->w_ptr, sizeof(int), 1, fp);
                    ptr = ptr->next;
                }
                ptr = NULL;
            }
        }
        fclose(fp);
        dealloc(filename);
    }
}


/* ************************************************************************
 * WRITE_SPIKES_EVENTS: This function stores to a file the spike events. 
 *
 * Args : 
 *  fname (char *)          : Output file name
 *  core (nsat_core *)      : Cores structs array
 *  num_cores (int)         : Cores numbers
 *
 * Returns :
 *  void
 **************************************************************************/
void write_spikes_events(fnames *fname, nsat_core *core, int num_cores) {
    int p;
    char *filename=NULL;                     /* Tmp filename */  
    FILE *fp = NULL;

    for(p = 0; p < num_cores; ++p) {
        filename = gen_fname(fname->events, p, 1);

        if(!(fp = fopen(filename, "wb"))) {
            printf("File %s cannot be opened!\n", filename);
            exit(-1);
        }

        fwrite(core[p].events->array,
               sizeof(int),
               core[p].events->length,
               fp);

        fwrite(core[p].events->times,
               sizeof(int),
               core[p].events->length,
               fp);

        fclose(fp);
        dealloc(filename);
    }
}


void write_spikes_events_online(nsat_core *core) {
    fwrite(&core->curr_time,
           sizeof(int),
           1,
           core->files->event_file);

    fwrite(&core->mon_events->length-1,
           sizeof(int),
           1,
           core->files->event_file);
    
    fwrite(&core->mon_events->array,
           sizeof(int),
           core->mon_events->length-1,
           core->files->event_file);
}


/* ************************************************************************
 * WRITE_SHARED_MEMORIES: This function stores to a file the shared memory
 * per core.
 *
 * Args : 
 *  fname (char *)          : Output file name
 *  core (nsat_core *)      : Cores structs array
 *  num_cores (int)         : Cores numbers
 *
 * Returns :
 *  void
 **************************************************************************/
void write_shared_memories(fnames *fname, nsat_core *core, int num_cores) {
    int p;
    char *filename = NULL;
    FILE *fp = NULL;

    for(p = 0; p < num_cores; ++p) {
        if (core[p].mon_pms->mon_final_weights) {
            filename = gen_fname(fname->shared_mem, p, 1);
            if(!(fp = fopen(filename, "wb"))) {
                printf("File %s cannot be opened!\n", filename);
                exit(-1);
            } 
            dealloc(filename);
            fwrite(core[p].shared_memory, sizeof(int), core[p].sm_size, fp);
            fclose(fp);
        }
    }
}


void write_spike_statistics(fnames *fname, nsat_core *core, int num_cores) {
    int p, j;
    FILE *fp;

    if(!(fp = fopen(fname->stats_nsat, "wb"))) {
        printf("File %s cannot open!\n", fname->stats_nsat);   
        printf("Nothing written to the file!\n");   
    } else {
        for (p = 0; p < (int) num_cores; ++p) {
            fwrite(&p, sizeof(int), 1, fp);
            for (j = 0; j < (int) core[p].core_pms.num_neurons; ++j) {
                fwrite(&j, sizeof(int), 1, fp);
                fwrite(&core[p].nsat_neuron[j].spk_counter, sizeof(int), 1, fp);
            }
        }
        fclose(fp);
    }
}
