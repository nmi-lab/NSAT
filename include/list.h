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
#ifndef LIST_H
#define LIST_H

#include "dtypes.h"

/* Integers list node struct */
typedef struct identity_list_s {
    unsigned long long id;
    struct identity_list_s *next;
} id_list_node;


/* Integers list struct */
typedef struct list_ids_s {
    struct identity_list_s *head;
    int len;
} list_id;


/* Spike list node struct */
typedef struct spike_list_s {
    unsigned long long id;
    unsigned long long t;
    struct spike_list_s *next;
} spk_list_node;


/* Spike list struct */
typedef struct list_spk_s {
    struct spike_list_s *head;
    struct spike_list_s *tail;
    int len;
} list_spk;


/* Synaptic strength list node struct */
typedef struct syn_list_s {
    unsigned long long id;
    WTYPE *w_ptr;
    struct syn_list_s *next;
} syn_list_node;


/* Synapses list struct */
typedef struct list_syn_s {
    struct syn_list_s *head;
    struct syn_list_s *tail;
    int len;
} list_syn;


/* Spike list functions declarations */
list_spk *alloc_list_spk(void);
list_syn *alloc_list_syn(void);
list_id *alloc_list_id(void);
void push_spk(list_spk **, unsigned long long, unsigned long long);
void push_back_spk(list_spk **, unsigned long long);
void push_syn(list_syn **, unsigned long long, WTYPE *);
void push_id(list_id **, unsigned long long);
void destroy_list_spk(list_spk **);
void destroy_list_syn(list_syn **);
void destroy_list_id(list_id **);
void print_list_spk(list_spk *);
void print_list_syn(list_syn *);

#endif  /* LIST_H */
