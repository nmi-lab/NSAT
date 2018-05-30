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


extern inline void over_under_flow(nsat_core *core);

extern inline void copy_states(unit **dest,
                               STATETYPE *src,
                               unsigned long long num_neurons,
                               unsigned int num_states); 


/* ************************************************************************
 * ZERO_BIT_SHIFT: This function performs a power of two multiplication
 * using bit shift and rounds towards zero (in NSAT is used for learning).
 *
 * Args : 
 *  x (int)     : Base
 *  a (int)     : Exponent
 *
 * Returns :
 *  x * 2^a
 **************************************************************************/
int zero_bit_shift(int x, int a)
{
    if (a == 0)
        return x;
    else if (a < 0)
        return sign(x) * (abs(x) >> -a);
    else
        return (x << a);
}


/* ************************************************************************
 * ZERO_BIT_SHIFT_DIV: This function performs a power of two division
 * using bit shift and rounds towards zero (in NSAT is used for learning).
 *
 * Args : 
 *  x (int)     : Base
 *  a (int)     : Exponent
 *
 * Returns :
 *  x / 2^a
 **************************************************************************/
int zero_bit_shift_div(int x, int a)
{
    if (a == 0)
        return x;
    else if (a < 0)
        return sign(x) * (abs(x) << -a);
    else
        return (x >> a);
}


/* ************************************************************************
 * ONE_BIT_SHIFT: This function performs a power of two multiplication
 * using bit shift and rounds towards one (in NSAT is used dynamics).
 *
 * Args : 
 *  x (int)     : Base
 *  a (int)     : Exponent
 *
 * Returns :
 *  x * 2^a
 **************************************************************************/
int one_bit_shift(int x, int a)
{
    int res = 0;
    if ((a > -16) && (a < 0)) {
        if (x < 0)
            res = (x + (1 << (-a)) -1 ) >> (-a);
        else
            res = x >> -a;
        if (!res)
            res = sign(x);
        return res;
    } else if (a >= 0) {
        res = x << a;
        return res;
    } else {
        return res;
    }
}


/* ************************************************************************
 * BLANK_OUT_PROBABILITY: This function returns 1 or 0 depending on a 
 * probability (flip a coin).
 *
 * Args : 
 *  prob (float)            : Probability
 *  dim (size_t)            : Size of output array
 *
 * Returns :
 *  A vectror of zeros and ones depending on the probability (prob). 
 *  [This models the stochasticity of the synapses]
 **************************************************************************/
int *blank_out_prob(int prob, size_t dim)
{
    size_t i;
    int *res = NULL;
    int rnd;

    res = alloc_zeros(int, dim);

    if (prob == 15) {
        for (i = 0; i < dim; ++i)
            res[i] = 1;
    } else if (prob == 0) {
        ;
    } else {
        for (i = 0; i < dim; ++i) { 
            rnd = pcg32_boundedrand(15);
            if (rnd < prob)
                res[i] = 1;
            else
                res[i] = 0;
        }
    }
    return res;
}


/* ************************************************************************
 * K_W: This function reflects the STDP learning window. It takes as 
 * as arguments the temporal difference (delta), and a pointer to an 
 * integer, sgn.
 *
 * Args : 
 *  pms (learning_params *)     : Learning parameters pointer (this points to
 *                                  all the necessary parameters for building
 *                                  the STDP Kernel)
 *  dt (int)                    : Temporal difference (Dt)
 *  sgn (int *)                 : A pointer to the sign of the temporal
 *                                difference. Pass sgn by reference when the
 *                                function is called.
 *
 * Returns :
 *  The STDP value and the signum of the Dt. 
 **************************************************************************/
int K_W(learning_params *pms, int dt, int *sign, int *tau) {
    int a = -16;

    if ((dt > 0) && (dt < pms->tca[0])) {
        a = pms->hica[0];
        (*sign) = pms->sica[0];
        (*tau) = pms->slca[0];
    } else if ((dt >= pms->tca[0]) && (dt < pms->tca[1])) {
        a = pms->hica[1];
        (*sign) = pms->sica[1];
        (*tau) = pms->slca[1];
    } else if ((dt >= pms->tca[1]) && (dt < pms->tstdp)) {
        a = pms->hica[2];
        (*sign) = pms->sica[2];
        (*tau) = pms->slca[2];
    } else if ((dt < 0) && (dt > pms->tac[0])) {
        a = pms->hiac[0];
        (*sign) = pms->siac[0];
        (*tau) = pms->slac[0];
    } else if ((dt <= pms->tac[0]) && (dt > pms->tac[1])) {
        a = pms->hiac[1];
        (*sign) = pms->siac[1];
        (*tau) = pms->slac[1];
    } else if ((dt <= pms->tac[1]) && (dt > -pms->tstdp)) {
        a = pms->hiac[2];
        (*sign) = pms->siac[2];
        (*tau) = pms->slac[2];
    } else {
        a = -16;
        (*sign) = 0;
    }
    return a;
}


/* ************************************************************************
 * RANDOMIZED_ROUNDING: This function implements the randomized rounding 
 * at the synaptic level. It gets the least significant bits of the synaptic
 * weights increament (dw), and bit shifts them. 
 *
 * Args : 
 *  dw (int)            : Synaptic weights increament
 *  shift_size  (int)   : Number of bits to shift the dw
 *
 * Returns :
 *  The randomized rounding synaptic increament (dw) -- int
 **************************************************************************/
WTYPE randomized_rounding(WTYPE dw, int shift_size)
{
    WTYPE a = dw >> shift_size, dw_round;
    WTYPE exp = pow(2, shift_size);
    WTYPE p = dw & (exp-1);
    WTYPE rnd = pcg32_boundedrand(exp);

    dw_round = a + (rnd < p);
    return dw_round;
}


/* ************************************************************************
 * EXPAND_SPIKE_LIST: This function takes as argument a spike list and
 * fills in it with new spikes. In addition, it sets units counters to 
 * their new values. 
 *
 * Args : 
 *  neuron  (unit *)            : Neuron unit structure
 *  spikes  (list_spk *)        : Spike list (previous time steps)
 *  spikes4lrn (list_spk **)    : List to be populated for learning
 *  curr_time (int)             : Current time step
 *  tstdpmax  (int)             : STDP maximum time
 *  size  (int)                 : Number of units
 *
 * Returns :
 *  void
 **************************************************************************/
void expand_spike_list(unit *neuron,
                       array_list *spikes,
                       array_list **spikes4lrn,
                       unsigned long long curr_time,
                       unsigned long long num_units,
                       int tstdpmax)
{
    unsigned long long j, id;
    int dlpt;

    for (j = 0; j < spikes->length; ++j) {
        id = spikes->array[j];
        dlpt = curr_time - neuron[id].counter;
        if ((dlpt > 0) && (dlpt < tstdpmax)) {
            array_list_push(spikes4lrn, id, curr_time, 1);
        }
    }

    for (j = 0; j < num_units; ++j) {
        dlpt = curr_time - neuron[j].counter;
        if (dlpt == tstdpmax) {
            array_list_push(spikes4lrn, j, curr_time, 1);
        }
    }
}


/* ************************************************************************
 * SET_COUNTERS: This function sets the counters of external or NSAT
 * neuron units to their new values. 
 *
 * Args : 
 *  nsat_neuron (unit *)        : NSAT neuron units structure
 *  ext_neuron (unit *)         : Input units structure
 *  ext_events (list_spk *)     : External events spike list
 *  nsat_events (list_spk *)    : NSAT events spike list
 *  curr_time  (int)            : Current time step
 *
 * Returns :
 *  void
 **************************************************************************/
void set_counters(unit *nsat_neuron,
                  unit *ext_neuron,
                  array_list *nsat_events,
                  array_list *ext_events,
                  int curr_time)
{
    unsigned long long j, id;

    for (j = 0; j < ext_events->length; ++j) {
        id = ext_events->array[j];
        ext_neuron[id].counter = curr_time;
    }

    for (j = 0; j < nsat_events->length; ++j) {
        id = nsat_events->array[j];
        nsat_neuron[id].counter = curr_time;
    }
}


/* ************************************************************************
 * SET_GLOBAL_MODULATOR: This function sets the new values for the 
 * global modulator (m in NSAT framework).
 *
 * Args : 
 *  x (int **)                  : Modulatory state array (to be populated)
 *  y (int *)                   : State array (tX)
 *  nsat_neuron (unit *)        : NSAT neuron unit structure
 *  nsat_events (list_spk *)    : NSAT events spike list
 *  num_states (int)            : Number of state components
 *
 * Returns :
 *  void
 **************************************************************************/
void set_global_modulator(STATETYPE **x,
                          STATETYPE *y,
                          unit *nsat_neuron,
                          array_list *nsat_events,
                          unsigned int num_states)
{
    unsigned long long j, id;

    for (j = 0; j < nsat_events->length; ++j) {
        id = nsat_events->array[j];
        (*x)[id] = y[id * num_states + nsat_neuron[id].nsat_ptr->modg_state];
    }
}


/* ************************************************************************
 * REFRACTORY_PERIOD: This function applies the reftacory period on a
 * spiked neuron.
 *
 * Args : 
 *  x   (int **)            : 0 Component of neuron's state (temp variable)
 *  nsat_neuron (unit *)    : NSAT neuron unit struct
 *  num_neurons (int)       : Number of neurons
 *  num_states (int)        : Number of states
 *
 * Returns :
 *  void
 **************************************************************************/
void refractory_period(STATETYPE **x,
                       unit *nsat_neuron,
                       unsigned long long num_neurons,
                       unsigned int num_states)
{
    unsigned long long j;

    for (j = 0; j < num_neurons; ++j) {
        if (nsat_neuron[j].ref_period > 0){
            if (nsat_neuron[j].nsat_ptr->is_xreset_on[0])
                (*x)[j*num_states] = nsat_neuron[j].nsat_ptr->x_reset[0];
            nsat_neuron[j].ref_period -= 1;
        }
    }
}


/* ************************************************************************
 * INTEGRATE_NSAT: This function integrates the dynamics of NSAT neurons.
 *
 * Args : 
 *  x   (int **)                : State array to populate (temp variable)
 *  acm (int *)                 : Accumulator (synaptic inputs)
 *  nsat_neuron (unit *)        : NSAT neuron structure
 *  num_neurons (int)           : Number of neurons
 *  num_states (int)            : Number of states
 *
 * Returns :
 *  void
 **************************************************************************/
void integrate_nsat(STATETYPE **x,
                    STATETYPE *acm,
                    unit *nsat_neuron,
                    unsigned long long num_neurons,
                    unsigned int num_states)
{
    unsigned long long j;
    unsigned int k, l;
    int mysum = 0;

    for (j = 0; j < num_neurons; ++j) {
        for (k = 0; k < num_states; ++k) {
            mysum = 0;
            for (l = 0; l < num_states; ++l) {
                if (nsat_neuron[j].nsat_ptr->A[k*num_states+l] != -16) {
                    mysum += nsat_neuron[j].nsat_ptr->sF[k*num_states+l] * 
                             one_bit_shift(nsat_neuron[j].s[l].x,
                                           nsat_neuron[j].nsat_ptr->A[k*num_states+l]);
                }
            }
            (*x)[j*num_states+k] = nsat_neuron[j].s[k].x + mysum
                                   + acm[j*num_states+k]
                                   + nsat_neuron[j].nsat_ptr->b[k];
            if (nsat_neuron[j].nsat_ptr->sigma[k] != 0) {
                (*x)[j*num_states+k] += normal(0.0, 1.0) * nsat_neuron[j].nsat_ptr->sigma[k];
            }
        }
    }
}


/* ************************************************************************
 * ACCUMULATE_SYNAPTIC_EVENTS: This function accumulates the synaptic
 * events into NSAT neurons dynamics. 
 *
 * Args : 
 *  acm  (int **)               : Accumulator (synaptic inputs)
 *  pre_unit (unit *)           : Pre-synaptic units
 *  post_unit (unit *)          : Post-synaptic units
 *  spikes_list (list_spk *)    : Spike list
 *  g_pms (global_params *)     : Global parameters
 *  curr_time (int)             : Current simulation step    
 *  num_states (int)            : Number of state's components
 *
 * Returns :
 *  void
 **************************************************************************/
void accumulate_synaptic_events(STATETYPE **acm,
                                unit *pre_unit,
                                unit *post_unit,
                                array_list *spikes_list,
                                unsigned int num_states,
                                unsigned int core_id,
                                unsigned long long time)
{
    unsigned long long j, pre;
    unsigned int k;
    int l, num_pre_spks;
    int *prb=NULL;

    syn_list_node *ptr_post = NULL;

    for (j = 0; j < spikes_list->length; ++j) {
        pre = spikes_list->array[j];
        for (k = 0; k < num_states; ++k) {
            ptr_post = pre_unit[pre].syn_ptr[k]->head;
            num_pre_spks = pre_unit[pre].syn_ptr[k]->len;
            if (num_pre_spks > 0) {
                prb = blank_out_prob(post_unit[ptr_post->id].nsat_ptr->prob[k], num_pre_spks);
                l = 0; 
                while (ptr_post != NULL) {
                    (*acm)[ptr_post->id * num_states + k] += *(ptr_post->w_ptr) * prb[l];
                    ptr_post = ptr_post->next;
                    l++;
                }
                ptr_post = NULL;
                dealloc(prb);
            }
        }
    }
} 


/* ************************************************************************
 * SHIFT_SYNAPTIC_EVENTS: This function multiplies the synaptic weights
 * with a constant gain (the multiplication is performed by zero_bit_shift
 * function).
 *
 * Args : 
 *  acm  (int **)           : Accumulator (synaptic inputs)
 *  nsat_neuron (unit *)    : NSAT units vector    
 *  num_neurons (int)       : Number of neurons
 *  num_states (int)        : Number of state's components
 *
 * Returns :
 *  void
 **************************************************************************/
void shift_synaptic_events(STATETYPE **acm,
                           unit *nsat_neuron,
                           unsigned long long num_neurons,
                           unsigned int num_states)
{
    unsigned long long j;
    unsigned int k;
    int tmp;

    for (j = 0; j < num_neurons; ++j) {
        for (k = 0; k < num_states; ++k) {
            tmp = zero_bit_shift((*acm)[j*num_states+k],
                                 nsat_neuron[j].nsat_ptr->w_gain[k]);
            if (tmp > XTHUP)
                (*acm)[j*num_states+k] = XTHUP;
            else if (tmp < XTHLOW)
                (*acm)[j*num_states+k] = XTHLOW;
            else
                (*acm)[j*num_states+k] = tmp;
        }
    }
}


/* ************************************************************************
 * SPIKE_EVENTS: This function checks if a neuron's state has crossed its
 * threshold value and fills out the corresponding spike list with the
 * proper events times and addresses. 
 *
 * Args : 
 *  x   (int *)                 : Neuron state components (tX)
 *  nsat_neuron  (unit *)       : NSAT neurons array
 *  events (list_spk *)         : Total spike events list
 *  mon_events (list_spk *)     : Monitors spikes events list
 *  trans_events (list_spk *)   : Transmitted to other cores spikes list
 *  curr_time  (int)            : Current simulation time step
 *  num_neurons (int)           : Number of neurons
 *  num_states (int)            : Number of states
 *
 * Returns :
 *  void
 **************************************************************************/
void spike_events(STATETYPE *x,
                  unit *nsat_neuron,
                  array_list *nsat_events,
                  array_list *events,
                  array_list *mon_events,
                  array_list *trans_events,
                  unsigned long long curr_time,
                  unsigned long long num_neurons,
                  unsigned int core_id,
                  unsigned int num_states)
{
    unsigned long long j;

    for (j = 0; j < num_neurons; ++j) {
        /* Check for spike - fixed threshold value*/
        if (nsat_neuron[j].ref_period == 0) {
            if (nsat_neuron[j].nsat_ptr->is_flag_Xth == false) {
                if (x[j*num_states] >= nsat_neuron[j].nsat_ptr->x_thr) {
                    array_list_push(&nsat_events, j, curr_time, 1);
                    nsat_neuron[j].spk_counter++;
                    if (nsat_neuron[j].is_transmitter) {
                        array_list_push(&trans_events, j, curr_time, 1);
                    }

                    if (nsat_neuron[j].is_spk_rec_on) {
                        array_list_push(&events, j, curr_time, 1);
                        array_list_push(&mon_events, j, curr_time, 1);
                    }
                    nsat_neuron[j].ref_period = nsat_neuron[j].nsat_ptr->t_ref;
                }
            } else {
                if (x[j*num_states] >= x[j*num_states+1]) {
                    array_list_push(&nsat_events, j, curr_time, 1);
                    nsat_neuron[j].spk_counter++;
                    if (nsat_neuron[j].is_transmitter)
                        array_list_push(&trans_events, j, curr_time, 1);

                    if (nsat_neuron[j].is_spk_rec_on) {
                        array_list_push(&events, j, curr_time, 1);
                        array_list_push(&mon_events, j, curr_time, 1);
                    }
                    nsat_neuron[j].ref_period = nsat_neuron[j].nsat_ptr->t_ref;
                }
            }
        }
    }
}


/* ************************************************************************
 * STATE_RESET: This function resets all the NSAT states according to 
 * a threshold value. 
 *
 * Args : 
 *  x   (int *)                 : Neuron states components (temp variable, tX)
 *  nsat_neuron  (unit *)       : NSAT neuron structure array    
 *  nsat_events  (list_spk *)   : NSAT events spike list
 *  num_states  (int)           : Number of states
 *
 * Returns :
 *  void
 **************************************************************************/
void state_reset(STATETYPE **x,
                 unit *nsat_neuron,
                 array_list *nsat_events,
                 unsigned int num_states)
{
    unsigned long long j, id, t;
    unsigned int k;

    for (j = 0; j < nsat_events->length; ++j) {
        id = nsat_events->array[j];
        for (k = 0; k < num_states; ++k) {
            if (nsat_neuron[id].nsat_ptr->is_xreset_on[k] == true)
                (*x)[id*num_states+k] = nsat_neuron[id].nsat_ptr->x_reset[k];
            else
                (*x)[id*num_states+k] += nsat_neuron[id].nsat_ptr->x_spike_incr[k];
        }
    
    }
}


/* ************************************************************************
 * CAUSAL_STDP: This function implements the causal direction of the STDP
 * learning rule.
 *
 * Args : 
 *  core (nsat_core *)      : NSAT core data structure
 *  spikes (list_spk *)     : Spikes list
 *  flag (int)              : Indicates internal or external events (0 or 1)
 *
 * Returns :
 *  void
 **************************************************************************/
void causal_stdp(unit *pre_unit,
                 unit *post_unit,
                 STATETYPE *x,
                 STATETYPE *g,
                 array_list *spikes,
                 unsigned long long curr_time,
                 unsigned int num_states,
                 int syn_precision,
                 bool is_learning_gated,
                 bool is_check_wlim_on)
{
    unsigned long long j, pre, ddt = 0;
    unsigned int k;
    int tmp;
    int dw = 0, kdtca = 0, sca = 0, tau = 0;

    syn_list_node *ptr_post = NULL; 

    for (j = 0; j < spikes->length; ++j) {
        pre = spikes->array[j];
        for(k = 0; k < num_states; ++k) {
            ptr_post = pre_unit[pre].syn_ptr[k]->head;
            while(ptr_post != NULL) {
                dw = 0;
                if (post_unit[ptr_post->id].s[k].lrn_ptr->is_plastic_state) {
                    if (!is_learning_gated ||
                        (x[ptr_post->id*num_states+k] > post_unit[ptr_post->id].nsat_ptr->gate_low &&
                        x[ptr_post->id*num_states+k] < post_unit[ptr_post->id].nsat_ptr->gate_upper)  
                        && ((curr_time % post_unit[ptr_post->id].nsat_ptr->period) >= post_unit[ptr_post->id].nsat_ptr->burn_in)
                        ){
                        if (post_unit[ptr_post->id].s[k].lrn_ptr->is_stdp_on) {
                            ddt = post_unit[ptr_post->id].counter - pre_unit[pre].counter;
                            if (ddt > 0) {
                                kdtca = K_W(post_unit[ptr_post->id].s[k].lrn_ptr, ddt, &sca, &tau);
                                if (post_unit[ptr_post->id].s[k].lrn_ptr->is_stdp_exp_on) {
                                    tmp = -zero_bit_shift_div(ddt, tau) + kdtca;
                                    dw = sca * zero_bit_shift(g[ptr_post->id], tmp);
                                } else {
                                    dw = sca * zero_bit_shift(g[ptr_post->id], kdtca);
                                }
                                if (post_unit[ptr_post->id].s[k].lrn_ptr->is_rr_on) {
                                    int ww = randomized_rounding(dw, post_unit[ptr_post->id].s[k].lrn_ptr->rr_num_bits);
                                    *(ptr_post->w_ptr) += ww;
                                } else {
                                    *(ptr_post->w_ptr) += dw;
                                }
                                if (is_check_wlim_on) {
                                    check_synaptic_strengths(ptr_post->w_ptr,
                                                             syn_precision);
                                }
                            }
                        }
                    }
                }
                ptr_post = ptr_post->next;
            }
            ptr_post = NULL;
        }
    }
}




/* ************************************************************************
 * ACAUSAL_STDP: This function implements the acausal direction of STDP
 * learning.
 *
 * Args : 
 *  core (nsat_core *)      : NSAT core data structure
 *  flag (int)              : Indicates internal or external events (0 or 1)
 *
 * Returns :
 *  void
 **************************************************************************/
void acausal_stdp(unit *pre_unit,
                  unit *post_unit,
                  STATETYPE *x,
                  array_list *spikes,
                  unsigned long long curr_time,
                  unsigned int num_states,
                  int syn_precision,
                  bool is_learning_gated,
                  bool is_check_wlim_on)
{
    unsigned long long j, pre;
    unsigned int k;
    int tmp;
    int detac = 0, kdtac = 0, sac = 0, dw = 0, tau = 0;

    syn_list_node *ptr_post = NULL;

    for (j = 0; j < spikes->length; ++j) {
        pre = spikes->array[j];
        for(k = 0; k < num_states; ++k) {
            ptr_post = pre_unit[pre].syn_ptr[k]->head;
            while(ptr_post != NULL) {
                dw = 0;
                if (post_unit[ptr_post->id].s[k].lrn_ptr->is_plastic_state) {
                    if (!is_learning_gated ||
                            (x[ptr_post->id*num_states+k] > post_unit[ptr_post->id].nsat_ptr->gate_low &&
                             x[ptr_post->id*num_states+k] < post_unit[ptr_post->id].nsat_ptr->gate_upper)  
                            && ((curr_time % post_unit[ptr_post->id].nsat_ptr->period) >= post_unit[ptr_post->id].nsat_ptr->burn_in)
                            ){
                        if (post_unit[ptr_post->id].s[k].lrn_ptr->is_stdp_on) {
                            detac = post_unit[ptr_post->id].counter - curr_time;
                            kdtac = K_W(post_unit[ptr_post->id].s[k].lrn_ptr, detac, &sac, &tau);
                            if (post_unit[ptr_post->id].s[k].lrn_ptr->is_stdp_exp_on) {
                                tmp = -zero_bit_shift_div(detac, tau) + kdtac;
                                /* CHECK tmp = -zero_bit_shift_div(detac, tau) + kdtac; */
                                dw = sac * zero_bit_shift(x[ptr_post->id * num_states 
                                        + post_unit[ptr_post->id].nsat_ptr->modg_state],
                                        tmp);
                            } else {
                                dw = sac * zero_bit_shift(x[ptr_post->id * num_states 
                                        + post_unit[ptr_post->id].nsat_ptr->modg_state], 
                                        kdtac);
                            }
                        } else {
                            dw = zero_bit_shift(x[ptr_post->id*num_states+post_unit[ptr_post->id].nsat_ptr->modg_state], 
                                    post_unit[ptr_post->id].s[k].lrn_ptr->hiac[0]);
                        }
                        if (post_unit[ptr_post->id].s[k].lrn_ptr->is_rr_on) {
                            WTYPE ww = randomized_rounding(dw, post_unit[ptr_post->id].s[k].lrn_ptr->rr_num_bits);
                            *(ptr_post->w_ptr) += ww;
                        } else {
                            *(ptr_post->w_ptr) += dw; 
                        }
                        if (is_check_wlim_on) {
                            check_synaptic_strengths(ptr_post->w_ptr,
                                                     syn_precision);
                        }
                    }
                }
                ptr_post = ptr_post->next;
            }
            ptr_post = NULL;
        }
    }
}



/* ************************************************************************
 * NSAT_DYNAMICS: This function implementes the nsat dynamics. Is either
 * called per thread (core) or as is when the user operates in single-core
 * mode.
 *
 * Args :
 *  arg (void *)        : Void pointer (implicit struct of NSAT core)
 *
 * Returns :
 *  NULL *
 **************************************************************************/
#if OLD == 1
void *nsat_dynamics(void *arg) {
#else
void nsat_dynamics(nsat_core *core) {
#endif
    int stamps = 0;                          /* Time stamps for monitors */  

    /* Cast arg struct to nsat_core struct */
#if OLD == 1 
    nsat_core *core = (nsat_core *)arg;
#endif
    stamps = (int) core->curr_time % core->core_pms.timestamp;

    /* Update state monitors (binary file) */
    if ((core->mon_pms->mon_states) && (core->curr_time == 1)) {
#if DAVIS == 0
        /* update_state_monitor_file(core); */
        update_state_monitor_online(core);
#else
        update_state_monitor_online(core);
#endif
    }

    /* Update state monitors (write hex format - ascii file) */
    if ((core->mon_pms->mon_states_fpga) && (core->curr_time == 0)) {
        store_fpga_states(core);
    }

    /* Update weight monitors */
    if ((core->mon_pms->mon_weights) && (stamps == 0)) {
        update_synaptic_strength_monitor_file(core);
    }

    /* Integrate NSAT equations */
    integrate_nsat(&core->vars->tX, core->vars->acm, core->nsat_neuron,
                   core->core_pms.num_neurons, core->core_pms.num_states);

    /* Refractory period */
    refractory_period(&core->vars->tX, core->nsat_neuron,
                      core->core_pms.num_neurons, core->core_pms.num_states);

#if DAVIS == 1
    write_spikes_events_online(core);
#endif

    /* Check for spikes and resets, and fill out events list */
    array_list_clean(&core->nsat_events, 1);
    array_list_clean(&core->mon_events, 1);
    spike_events(core->vars->tX, core->nsat_neuron, core->nsat_events,
                 core->events, core->mon_events, core->trans_events,
                 core->curr_time, core->core_pms.num_neurons,
                 core->core_id,
                 core->core_pms.num_states);

    /* Check for underflows */
    over_under_flow(core);

#if OLD == 1
     return NULL;
#endif
}


/* ************************************************************************
 * NSAT_EVENTS_AND_LEARNING: This function implementes the nsat events 
 * accumulation and the learning process. It can be called per thread (core)
 * or as is when the user operates in single-core mode.
 *
 * Args :
 *  arg (void *)        : Void pointer (implicit struct of NSAT core)
 *
 * Returns :
 *  NULL *
 **************************************************************************/
#if OLD == 1
void *nsat_events_and_learning(void *arg) {
#else
void nsat_events_and_learning(nsat_core *core) {
#endif
    int stamps = 0;                          /* Time stamps for monitors */  

#if OLD == 1
    nsat_core *core = (nsat_core *) arg;
#endif

    stamps = (int) core->curr_time % core->core_pms.timestamp;

    /* Clean up the accumulator */
    memset(core->vars->acm, 0, core->core_pms.num_neurons * 
           core->core_pms.num_states*sizeof(STATETYPE));

    /* Add NSAT synaptic events if it's necessary */
    if (core->nsat_events->length > 0) {
        accumulate_synaptic_events(&core->vars->acm, core->nsat_neuron,
                                   core->nsat_neuron, core->nsat_events,
                                   core->core_pms.num_states,
                                   core->core_id, core->curr_time);
    }

    /* Add external synaptic events if it's necessary */
    if (core->ext_events->length > 0) {
        accumulate_synaptic_events(&core->vars->acm, core->ext_neuron,
                                   core->nsat_neuron, core->ext_events,
                                   core->core_pms.num_states,
                                   core->core_id, core->curr_time);
    }
    
    /* Shift the synaptic weights according to a constant gain */
    shift_synaptic_events(&core->vars->acm, core->nsat_neuron,
                          core->core_pms.num_neurons, core->core_pms.num_states);

    /* Causal STDP update - IsLearning suppresses learning in validation */
    if (core->core_pms.is_learning_on) {

        /* Fill out external events spike list for STDP */
        expand_spike_list(core->ext_neuron, core->ext_events, &core->ext_caspk,
                          core->curr_time, core->core_pms.num_inputs,
                          core->core_pms.tstdpmax);

        /* Compute causal STDP on external events */
        if (core->ext_caspk->capacity > 1 && core->syn->tot_ext_syn_num != 0) {
            causal_stdp(core->ext_neuron,
                        core->nsat_neuron,
                        core->vars->tX,
                        core->vars->g,
                        core->ext_caspk,
                        core->curr_time,
                        core->core_pms.num_states,
                        core->g_pms->syn_precision,
                        core->core_pms.is_learning_gated,
                        core->g_pms->is_check_wlim_on);
        }
        array_list_clean(&core->ext_caspk, 1);

        /* Fill out NSAT events spike list for STDP */
        expand_spike_list(core->nsat_neuron, core->nsat_events, &core->nsat_caspk,
                          core->curr_time, core->core_pms.num_neurons,
                          core->core_pms.tstdpmax);

        /* Compute causal STDP on NSAT events */
        if (core->nsat_caspk->length > 0 && core->syn->tot_nsat_syn_num != 0) {
            causal_stdp(core->nsat_neuron,
                        core->nsat_neuron,
                        core->vars->tX,
                        core->vars->g,
                        core->nsat_caspk,
                        core->curr_time,
                        core->core_pms.num_states,
                        core->g_pms->syn_precision,
                        core->core_pms.is_learning_gated,
                        core->g_pms->is_check_wlim_on);
        }
        array_list_clean(&core->nsat_caspk, 1);
      
        /* Acausal STDP update */
        /* External events */
        if (core->ext_events->length > 0 && core->syn->tot_ext_syn_num != 0) {
            acausal_stdp(core->ext_neuron,
                         core->nsat_neuron,
                         core->vars->tX,
                         core->ext_events,
                         core->curr_time,
                         core->core_pms.num_states,
                         core->g_pms->syn_precision,
                         core->core_pms.is_learning_gated,
                         core->g_pms->is_check_wlim_on);
        }

        /* NSAT events */
        if (core->nsat_events->length > 0 && core->syn->tot_nsat_syn_num != 0) {
            acausal_stdp(core->nsat_neuron,
                         core->nsat_neuron,
                         core->vars->tX,
                         core->nsat_events,
                         core->curr_time,
                         core->core_pms.num_states,
                         core->g_pms->syn_precision,
                         core->core_pms.is_learning_gated,
                         core->g_pms->is_check_wlim_on);
        }
    }

    /* Reset spiked neurons states */
    state_reset(&core->vars->tX, core->nsat_neuron, core->nsat_events,
                core->core_pms.num_states);

    /* Check for underflows */
    over_under_flow(core);

    /* Update last spike time - set counters */
    set_counters(core->nsat_neuron, core->ext_neuron, core->nsat_events,
                 core->ext_events, core->curr_time);

    /* Update global modulator's state */
                         
    set_global_modulator(&core->vars->g, core->vars->tX, core->nsat_neuron,
                         core->nsat_events, core->core_pms.num_states);

    /* Swap old with new values */
    copy_states(&core->nsat_neuron,
                core->vars->tX,
                core->core_pms.num_neurons,
                core->core_pms.num_states);

    /* Update state monitors (binary file) */
    if ((core->mon_pms->mon_states) && (stamps == 0)) {
#if DAVIS == 0
        /* update_state_monitor_file(core);  */
        update_state_monitor_online(core);
#else
        update_state_monitor_online(core);
#endif
    }

    /* Update state monitors (write hex format - ascii file) */
    /* if ((core->mon_pms->mon_states_fpga) && (stamps == 0)) { */
    /*     store_fpga_states(core); */
    /* } */

    /* Hard reset tX */ 
    memset(core->vars->tX, 0, core->core_pms.num_neurons * 
           core->core_pms.num_states*sizeof(int));

    /* Delete previous time step external spikes */
    array_list_clean(&core->ext_events, 1);
    array_list_clean(&core->nsat_caspk, 1);
    array_list_clean(&core->ext_caspk, 1);
    
#if OLD == 1
    return NULL;
#endif
}
