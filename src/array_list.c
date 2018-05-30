#include "array_list.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"


void array_list_init(array_list **vector, int flag)
{
    (*vector)->array = NULL;
    (*vector)->times = NULL;

    (*vector)->array = (unsigned long long *) calloc(1, sizeof(unsigned long long));
    if (flag == 1) {
        (*vector)->times = (unsigned long long *) calloc(1, sizeof(unsigned long long));
    }
    (*vector)->capacity = 1;
    (*vector)->length = 0;
}


void array_list_push(array_list **vector, unsigned long long value,
                     unsigned long long time, int flag)
{
    if ((*vector)->capacity < 1) {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("PUSH_ARRAY_LIST: The array list has not been initialized!\n");
        exit(-1);
    } 

    (*vector)->array[(*vector)->length] = value;
    (*vector)->array = realloc((*vector)->array,
                               ((*vector)->capacity+1)*sizeof(unsigned long long));
    if (flag == 1) {
        (*vector)->times[(*vector)->length] = time;
        (*vector)->times = realloc((*vector)->times,
                                   ((*vector)->capacity+1)*sizeof(unsigned long long));
    }
    (*vector)->capacity++;
    (*vector)->length++;
}


void array_list_destroy(array_list **vector, int flag)
{
    if ((*vector)->capacity < 1) {
        (*vector)->array = NULL;
        (*vector)->times = NULL;
    } 
    
    free((*vector)->array);
    if (flag == 1) {
        free((*vector)->times);
    }
    (*vector)->capacity = 0;
    (*vector)->length = 0;
}


void array_list_clean(array_list **vector, int flag)
{
    free((*vector)->array);
    (*vector)->array = (unsigned long long *) calloc(1, sizeof(unsigned long long));
    if (flag == 1) {
        free((*vector)->times);
        (*vector)->times = (unsigned long long *) calloc(1, sizeof(unsigned long long));
    }
    (*vector)->capacity = 1;
    (*vector)->length = 0;
}
