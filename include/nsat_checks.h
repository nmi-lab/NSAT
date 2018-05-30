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
#ifndef NSAT_CHECK_H
#define NSAT_CHECK_H

#include "dtypes.h"

/* For brief description of these functions see nsat_checks.c file */

inline bool is_power_of_two(int x) {
    return x && !(x & (x - 1));

}


inline bool check_recs_flags(unsigned long long *x, unsigned long long size,
                             unsigned long long num_neurons) {
    unsigned long long j;

    for(j = 0; j < size; ++j) {
        if (x[j] > num_neurons) {
            return false;
        }
    }
    return true;
}


inline bool comp(int x, int y) {
    return (x > y) ? true : false;  
}


inline bool check_upper_boundary(int *x, int num_states) {
    int k;

    for (k = 0; k < num_states; ++k) {
        if (x[k] > 0) {
            return false;
        }
    }
    return true;
}


inline void check_synaptic_strengths(WTYPE *x, WTYPE boundary) {
    
    if (*x >= -boundary && *x <= boundary-1) {
        ;
    } else if (*x < -boundary) {
        *x = -boundary;
    } else {
        *x = boundary-1; 
    }
}


inline int check_compliance(bool flag, int y) {
    if (flag) {
        if (!y) {
            return -1;
        }
    } else {
        if (y) {
            return -2;
        }
    }
    return 0;
}


inline bool check_intervals(int *x, int inf, int sup, unsigned int size) {
    unsigned int k;

    if (!size) {
        if (((*x-1 >= inf) && (*x-1 <= sup)) == false) {
            return false;
        }
    } else {
        for(k = 0; k < size; ++k) {
            if (((x[k] >= inf) && (x[k] <= sup)) == false) {
                return false;
            }
        }
    }
    return true;
}


inline void check_underflow(int **x, int *y, int num_states) {
    int k;

    for(k = 0; k < num_states; ++k) {
        if (abs((*x)[k]) > abs(y[k])) {
            (*x)[k] = y[k];
        }
    }
}


inline void print_true_or_false(FILE *fp, int x) {
    if (x == 1)
        fprintf(fp, "True");
    else
        fprintf(fp, "False");
}



inline void print_enabled_or_disabled(FILE *fp, int x) {
    if (x == 1) 
        fprintf(fp, "Enabled");
    else
        fprintf(fp, "Disabled");
}

inline void print_rng_choice(FILE *fp, bool x) {
    if (x) 
        fprintf(fp, "Box-Muller");
    else
        fprintf(fp, "Hardware");
}

#endif  /* NSAT_CHECK_H */
