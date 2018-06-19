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
#ifndef NSAT_MATH_H
#define NSAT_MATH_H

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <float.h>


/* max function - For more details see nsat_math.c */
inline int max(int *x, int size) {
    int i, tmp = x[0];

    for(i = 1; i < size; ++i) {
        if (x[i] > tmp) {
            tmp = x[i];
        }
    }
    return tmp;
}


/* sign function - For more details see nsat_math.c */
inline int sign(int x) {
    return (x > 0) - (x < 0);
}


/* normal distribution function - For more details see nsat_math.c */
double normal(double, double);

#endif /* NSAT_MATH_H */
