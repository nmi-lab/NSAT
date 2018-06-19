#ifndef ARRAY_LIST_H
#define ARRAY_LIST_H

#include <stdio.h>
#include <stdlib.h>

typedef struct array_list_s {
    int *array;
    int *times;
    int length;
    int capacity;
} array_list;

void array_list_init(array_list **, int);
void array_list_clean(array_list **, int);
void array_list_destroy(array_list **, int);
void array_list_push(array_list **, int, int, int);


#endif  /* ARRAY_LIST_H */
