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


list_syn *alloc_list_syn(void) {
    list_syn *tmp_lst = NULL; 
    tmp_lst = alloc(list_syn, 1);
    tmp_lst->head = NULL;
    tmp_lst->len = 0;
    return tmp_lst;
}


/* ************************************************************************
 * PUSH_SYN: This function pushes (append) a node into a list (synaptic
 * strengths).
 *
 * Args : 
 *  List (node **)   : Double pointer to spikes list
 *  id (int)         : The id value for the new inserted node
 *                      (i.e. neuron ID)
 *  W (int)          : The value for the current synapse
 *
 * Returns :
 *  void
 **************************************************************************/
void push_syn(list_syn **List, int id, WTYPE *W) {
    syn_list_node *tmp = NULL;
    tmp = alloc(syn_list_node, 1);
    tmp->id = id;
    tmp->w_ptr = W;
    tmp->next = (*List)->head;
    (*List)->head = tmp;
    (*List)->len++;
}


/* ************************************************************************
 * PRINT_LIST_SYN: This function prints all the id values in a list.
 *
 * Args : 
 *  List (spk list *)   : A pointer to synapses list
 *
 * Returns :
 *  void
 **************************************************************************/
void print_list_syn(list_syn *List) {
    syn_list_node *ptr = NULL;

    printf("[");
    ptr = List->head;
    while(ptr != NULL) {
        printf(" %d", ptr->id);
        ptr = ptr->next;
    }
    printf("]\n");
    ptr = NULL;
}


/* ************************************************************************
 * DESTROY_LIST_SYN: This function destroys a list and frees the memory.
 *
 * Args : 
 *  head (node **)   : Double pointer to the head of the list
 *
 * Returns :
 *  void
 **************************************************************************/
void destroy_list_syn(list_syn **List) {
    syn_list_node* current=NULL;
    syn_list_node* next=NULL;

    current = (*List)->head;
    while (current != NULL) {
        next = current->next;
        free(current);
        current = next;
    }
    dealloc(current);
    dealloc(next);
    (*List)->head = NULL;
    (*List)->len = 0;
}
