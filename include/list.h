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

/* Synaptic strength list node struct */
typedef struct syn_list_s {
    int id;
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
list_syn *alloc_list_syn(void);
void push_syn(list_syn **, int, WTYPE *);
void destroy_list_syn(list_syn **);
void print_list_syn(list_syn *);

#endif  /* LIST_H */
