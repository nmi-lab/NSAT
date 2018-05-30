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
#ifndef NSAT_H
#define NSAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h> 
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <inttypes.h>

#include <pthread.h>

#include "pcg_basic.h"      /* Random Number Generator */ 
#include "list.h"
#include "array_list.h"
#include "nsat_math.h"
#include "nsat_checks.h"
#include "dtypes.h"

/* Macros definitions for memory allocation and deallocation */
#define alloc(type_t, size) (type_t *) malloc((size) * sizeof(type_t))
#define alloc_zeros(type_t, size) (type_t *) calloc(size, sizeof(type_t))
#define mem_test(mem) {if(!mem) { \
                         perror("Cannot allocate memory!\n");   \
                         exit(-2);} }
#define dealloc(ptr) {free(ptr); \
                      ptr = NULL; }
#define file_test(file, fname) {if(!file) { \
                                 printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET); \
                                 printf("File %s not found!\n", fname); \
                                 exit(-1);} }

#define handle_error_en(en, msg) \
       do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)


#define N_NSATGROUPS 8
#define N_LRNGROUPS 8

#define OLD 0

#define XTHUP 32767
#define XTHLOW -32768

#define DAVIS 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"


/* Neuron parameters struct */
struct nsat_parameters_s {
    int *prob;
    int *sigma;
    int *A;
    int *sF;
    int *b;
    int *x_reset;
    int *x_spike_incr;
    int *x_thlow;
    int *x_thup;
    int *w_gain;
    int t_ref;
    int x_thr;
    int modg_state;
    int gate_low;
    int gate_upper;
    unsigned int period;
    unsigned int burn_in;
    bool *is_xreset_on;
    bool is_flag_Xth;
} __attribute__ ((aligned));
typedef struct nsat_parameters_s nsat_params;


/* Learning parameters per state */
struct learning_params_s {
    STATETYPE tca[2];
    STATETYPE hica[3];
    STATETYPE sica[3];
    STATETYPE slca[3];
    STATETYPE tac[2];
    STATETYPE hiac[3];
    STATETYPE siac[3];
    STATETYPE slac[3];
    STATETYPE tstdp;
    int rr_num_bits;
    bool is_plastic_state;
    bool is_stdp_on;
    bool is_stdp_exp_on;
    bool is_rr_on;
} __attribute__ ((aligned));
typedef struct learning_params_s learning_params;


/* Simulation paremeters struct */
struct global_pms_s {
    unsigned long long ticks;
    unsigned long long rng_init_state;
    unsigned long long rng_init_seq;
    unsigned int num_cores;
    int syn_precision;
    bool is_single_core;
    bool is_routing_on;
    bool is_bm_rng_on;
    bool is_clock_on;
    bool is_check_wlim_on;
} __attribute__ ((aligned));
typedef struct global_pms_s global_params;


/* Local parameters per core */
struct cores_params_s {
    list_id *ext_syn_rec_ids;
    list_id *nsat_syn_rec_ids;
    unsigned long long num_inputs;
    unsigned long long num_neurons;
    unsigned int num_states;
    unsigned long long num_recs_on;
    unsigned long long timestamp;
    unsigned int num_nsat_params_groups;
    unsigned int num_learning_pms_groups;
    int tstdpmax;
    bool is_learning_on;
    bool is_ext_evts_on;
    bool is_learning_gated;
} __attribute__ ((aligned));
typedef struct cores_params_s cores_params;


/* Monitors parameters struct */
struct monitors_params_s {
    bool mon_states;
    bool mon_states_fpga;
    bool mon_weights;
    bool mon_final_weights;
    bool mon_spikes; 
    bool mon_stats; 
} __attribute__ ((aligned));
typedef struct monitors_params_s monitors_params;


/* NSAT neuron's state struct */
struct state_s {
    learning_params *lrn_ptr;
    STATETYPE x;
} __attribute__ ((aligned));
typedef struct state_s state;


/* Events router struct */
struct router_s {
    unsigned int dst_core_id;
    unsigned long long dst_neuron_id;
} __attribute__ ((aligned));
typedef struct router_s router;


/* Unit struct (input or neuron)*/
struct unit_s {
    nsat_params *nsat_ptr;
    list_syn **syn_ptr;
    state *s;
    router *ptr_cores;
    unsigned long long spk_counter; 
    unsigned int router_size;
    unsigned long long counter;
    unsigned int ref_period;     /* That serves as delay for innput units */
    int num_synapses;
    bool is_spk_rec_on;
    bool is_transmitter;
} __attribute__ ((aligned));
typedef struct unit_s unit;


/* Synapse statistics struct */
struct synapse_stat_s {
    unsigned long long tot_ext_syn_num;
    unsigned long long tot_nsat_syn_num;
} __attribute__ ((aligned));
typedef struct synapse_stat_s synapse_stat;


/* File names struct */
typedef struct fname_s {
    char *nsat_params_map;
    char *lrn_params_map;
    char *params;
    char *syn_wgt_table;
    char *syn_ptr_table;
    char *ext_events;
    char *synw;
    char *synw_final;
    char *events;
    char *states;
    char *check_pms;
    char *stdp_fun;
    char *stats_nsat;
    char *stats_ext;
    char *l1_conn;
    char *shared_mem;
} fnames;


/* Monitor files pointers struct */
typedef struct mon_files_s {
    FILE *fs;
    FILE *fw;
    FILE *fsa;
    FILE *fr;
    FILE *event_file;
} mon_files;


/* Temporary cores' variables */
struct core_vars_s {
    STATETYPE *tX;
    STATETYPE *acm;
    STATETYPE *g;
    STATETYPE *xinit;
    unsigned long long *rec_spk_on;
} __attribute__ ((aligned));
typedef struct core_vars_s core_vars;


/* NSAT core struct */
struct nsat_core_s {
    cores_params core_pms;
    monitors_params *mon_pms;
    synapse_stat *syn;
    nsat_params *nsat_pms;
    learning_params *lrn_pms;
    mon_files *files;
    unit *ext_neuron;
    unit *nsat_neuron;
    array_list *events;
    array_list *ext_events;
    array_list *nsat_events;
    array_list *trans_events;
    array_list *mon_events;
    array_list *nsat_caspk;
    array_list *ext_caspk;
    global_params *g_pms;
    core_vars *vars;
    WTYPE *shared_memory;
    unsigned int core_id;
    unsigned long long curr_time;
    size_t sm_size;
    char *ext_evts_fname;
} __attribute__ ((aligned));
typedef struct nsat_core_s nsat_core;



/********************************************************************/
/*  Functions declarations 
*********************************************************************/

/* Printing/Visualization functions declarations */
void print_stdp_kernel(char *, learning_params *, int);
void print_params2file(fnames *, nsat_core *, global_params *);


/* Monitors functions declarations */
FILE *open_monitor_file(char *);
void open_cores_monitor_files(nsat_core *, fnames *, size_t);
void close_cores_monitor_files(nsat_core *, size_t);
void store_fpga_states(nsat_core *);
void update_state_monitor_file(nsat_core *);
void update_monitor_stats(int, int, int, int, FILE *, int, bool);
void update_state_monitor_online(nsat_core *);
void update_synaptic_strength_monitor_file(nsat_core *);
void update_monitor_next_state(int *, FILE *, int, int);
void open_online_spike_monitor(nsat_core **, fnames *);


/* STDP window functions declarations */
int K_W(learning_params *, int, int *, int *);


int bin_file_size(FILE *);

/* String manipulation functions declarations */
char *gen_fname(char *, int, int);
char *gen_ext_evts_fname(char *, unsigned int);
char *add_extension(char *);


/* Files I/O functions declarations */
void get_external_events(FILE *, nsat_core **, unsigned long long,
                         unsigned int);
void get_external_events_per_core(FILE *, nsat_core **, unsigned long long);
void get_davis_events(int fd, nsat_core **cores);
void write_spikes_events(fnames *, nsat_core *, int);
void write_final_weights(fnames *, nsat_core *, unsigned int);
void write_shared_memories(fnames *, nsat_core *, int);
void write_spikes_events_online(nsat_core *);
void write_spike_statistics(fnames *, nsat_core *, int);


/* Read/Load parameters functions declarations */
void read_core_params(FILE *, nsat_core *, unsigned int);
void read_nsat_params(FILE *, nsat_core *, unsigned int);
void read_lrn_params(FILE *, nsat_core *, unsigned int);
void read_monitor_params(FILE *, nsat_core *, unsigned int);
void read_global_params(FILE *, global_params *pms);


/* Initialization functions declarations */
void allocate_cores(nsat_core **, fnames *, unsigned int);
void initialize_cores_vars(nsat_core *, unsigned int);
void initialize_cores_neurons(nsat_core **, unsigned int);
void initialize_monitor_spk(char *, unit **);
void initialize_cores_connections(char *, nsat_core *);
void initialize_incores_connections(fnames *, nsat_core **, unsigned int);


/* Cleanup functions declarations */
void dealloc_neurons(nsat_core **, unsigned int);
void dealloc_cores(nsat_core **, unsigned int);


/* Mapping functions declarations */
void nsat_pms_groups_map_file(char *, nsat_core *, unsigned int);
void learning_pms_groups_map_file(char *, nsat_core *, unsigned int);


/* NSAT auxiliary functions */
void count_synapses(unit **, unsigned long long *, unsigned long long,
                    unsigned int);
void max_stdp_time(learning_params *, cores_params *);

/* NSAT auxiliary functions declarations */
int one_bit_shift(int, int);
int zero_bit_shift(int, int);
int *blank_out_prob(int, size_t);
WTYPE randomized_rounding(WTYPE, int);


/* Threads ready functions */
#if OLD == 1
void *nsat_dynamics(void *);
#else
void nsat_dynamics(nsat_core *);
#endif
#if OLD == 1
void *nsat_events_and_learning(void *);
#else
void nsat_events_and_learning(nsat_core *);
#endif
void *nsat_thread(void *);

int iterate_nsat(fnames *);
int iterate_nsat_new(fnames *);
int iterate_nsat_old(fnames *);

/* Core NSAT functions declarations */
void refractory_period(STATETYPE **, unit *, unsigned long long, unsigned int);
void state_reset(STATETYPE **, unit *, array_list *, unsigned int);
void shift_synaptic_events(STATETYPE **, unit *, unsigned long long, unsigned int);
void set_counters(unit *, unit *, array_list *, array_list *, int);
void set_global_modulator(STATETYPE **, STATETYPE *, unit *, array_list *,
                          unsigned int);
void integrate_nsat(STATETYPE **, STATETYPE *, unit *, unsigned long long,
                    unsigned int);
void expand_spike_list(unit *, array_list *, array_list **, unsigned long long,
                       unsigned long long, int);        
void accumulate_synaptic_events(STATETYPE **, unit *, unit *, array_list *,
                                unsigned int, unsigned int, unsigned long long);
void spike_events(STATETYPE *, unit *, array_list *, array_list *, array_list *,
                  array_list *, unsigned long long, unsigned long long,
                  unsigned int, unsigned int);
void causal_stdp(unit *, unit *, STATETYPE *, STATETYPE *, array_list *,
                 unsigned long long, unsigned int, int, bool, bool);
void acausal_stdp(unit *, unit *, STATETYPE *, array_list *, unsigned long long,
                  unsigned int, int, bool, bool);


/********************************************************************/
/* Inline functions definitions */
/********************************************************************/

/* ************************************************************************
 * OVER_UNDER_FLOW: This function checks if a neuron's state exceeds the 
 * lower boundaries of an 32-bit integer.
 *
 * Args : 
 *  core (nsat_core *)  : NSAT core data structure pointer
 *
 * Returns :
 *  void
 **************************************************************************/
inline void over_under_flow(nsat_core *core) {
    size_t j = 0;
	unsigned int k = 0;
    unsigned int num_states = core->core_pms.num_states;

	for(j = 0; j < core->core_pms.num_neurons; ++j) {
		for(k = 0; k < num_states; ++k) {
			if (core->vars->tX[j*num_states+k] < core->nsat_neuron[j].nsat_ptr->x_thlow[k]) {
				core->vars->tX[j*num_states+k] = core->nsat_neuron[j].nsat_ptr->x_thlow[k];
			}
			if (core->vars->tX[j*num_states+k] > core->nsat_neuron[j].nsat_ptr->x_thup[k]) {
                core->vars->tX[j*num_states+k] = core->nsat_neuron[j].nsat_ptr->x_thup[k];
			}
		}
	}
}


/* ************************************************************************
  COPY_STATES: This function copies neurons states from a temporary array 
 * (src) to each neuron state vector (dest).
 *
 * Args : 
 *  dest (int **)       : Destination array
 *  src (int *)         : Source array
 *  num_neurons (int)   : Number of neuron units (neurons)
 *  num_states (int)    : Number of states per neuron
 *
 * Returns :
 *  void
 **************************************************************************/
inline void copy_states(unit **dest,
                        STATETYPE *src,
                        unsigned long long size_x,
                        unsigned int size_y) {
    size_t j, k;

    for(j = 0; j < size_x; ++j) {
        for(k = 0; k < size_y; ++k) {
            (*dest)[j].s[k].x = src[j*size_y+k];
        }
    }
}


inline void progress_bar(int x, int n) {
    int i;
    float ratio;
    int width = 30, etp=0;

    if (x % ((n / 100)) != 0) return;
    ratio = x / (float) (n-1);
    etp = (int) (ratio * width);

    printf("Progress: [%d %%]  [", (int) (ratio * 100));
    for (i = 0; i < etp; ++i) {
        printf("#");
    }

    for (i = etp; i < width; ++i) {
        printf(" ");
    }
    printf("]");
    fflush(stdout);
    printf("\n\033[F\033[J");
}

#endif  /* NSAT_H */
