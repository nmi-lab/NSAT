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
#include "nsat_math.h"
#include "pcg_basic.h"


/* ************************************************************************
 * MAX: This function returns the maximum element of an array of integers.
 *
 * Args : 
 *  *x (pointer to int)   : Input integers array
 *  size (int)            : The length of the array
 *
 * Returns :
 *  The maximum element of the input integers array.
 **************************************************************************/
extern inline unsigned long long max(int *x, int size);


/* ************************************************************************
 * SIGN: This function takes as argument an integer number and returns 
 * its signum. 
 *
 * Args : 
 *  x (int)   : Input integer number
 *
 * Returns :
 *  1  if x > 0
 *  0  if x == 0
 *  -1 if x < 0
 **************************************************************************/
extern inline int sign(int x);


/* ************************************************************************
 * NORMAL: This function returns a random number drawn from a normal 
 * distribution. The normal distribution is implemented using the 
 * algorithm of Box-Muller. 
 *
 * Args : 
 *  mu (double)      : Mean of distribution
 *  sigma (double)   : Sigma (variance) of distribution
 *
 * Returns :
 *  A random number drawn from a normal distribution.
 **************************************************************************/
double normal(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   /* u1 = rand() * (1.0 / RAND_MAX); */
       u1 = (double) pcg32_boundedrand(100) / (double) 100;
	   /* u2 = rand() * (1.0 / RAND_MAX); */
       u2 = (double) pcg32_boundedrand(100) / (double) 100;
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}
