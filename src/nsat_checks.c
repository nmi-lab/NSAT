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
#include "nsat_checks.h"


/* ************************************************************************
 * IS_POWER_OF_TWO: This function checks if an integer is a power of two.
 *
 * Args : 
 *  x (int)           : Integer input
 *
 * Returns :
 *  True if integer x is a power of 2, False otherwise
 **************************************************************************/
extern inline bool is_power_of_two(int x);


/* ************************************************************************
 * CHECK_RECS_FLAGS: This function checks if the IDs of NSAT neurons are 
 * properly included in the recording list.
 *
 * Args : 
 *  x (int *)           : List with recording neurons IDs
 *  size (int)          : Size of recording list
 *  num_neurons (int)   : Number of neurons
 *
 * Returns :
 *  True if all the recording IDs match neurons IDs, false otherwise
 **************************************************************************/
extern inline bool check_recs_flags(int *x, 
                                    int size,
                                    int num_neurons);


/* ************************************************************************
 * COMP: This function compares two values, and returs true if the first 
 * argument is greater than the second one. 
 *
 * Args : 
 *  x (int)     :  
 *  y (int)     : 
 *
 * Returns :
 *  True if x > y, false otherwise
 **************************************************************************/
extern inline bool comp(int x, int y);


/* ************************************************************************
 * CHECK_UPPER_BOUNDARY: This function checks if the states fulfill the 
 * upper boundary requirement.
 *
 * Args : 
 *  x (int)            : Integer array contains the states
 *  num_states (int)   : Number of states per neuron
 *
 * Returns :
 *  True if all the states fulfill the requirements for the upper boundary,
 *  false otherwise
 **************************************************************************/
extern inline bool check_upper_boundary(int *x, int num_states);


/* ************************************************************************
 * CHECK_SYNAPTIC_STRENGTHS: This function checks if the synaptic strengths
 * lie in between the upper and the lower limits and corrects respectively.
 *
 * Args : 
 *  x (int)         : Synaptic strength
 *
 * Returns :
 *  -
 **************************************************************************/
extern inline void check_synaptic_strengths(WTYPE *x, WTYPE boundary);


/* ************************************************************************
 * CHECK_COMPLIANCE: This function checks if a NSAT flag complies with 
 * the corresponding numerical values (e.g. number of input neurons and 
 * is_ext_evts_on).
 *
 * Args : 
 *  flag (bool)     : A boolean flag (NSAT flags)
 *  y (int)         : Corresponding to the flag value
 *
 * Returns :
 *  -1 if the flag is true and y is zero
 *  -2 if the flag is false and y is not zero
 *  0 otherwise
 **************************************************************************/
extern inline int check_compliance(bool flag, int y);


/* ************************************************************************
 * CHECK_INTERVALS: This function checks if a number belongs to a 
 * predefined interval.
 *
 * Args : 
 *  x (int *)   : A pointer to an integer represents the number to test
 *  int (int)   : Lower boundary of the interval
 *  sup (int)   : Upper boundary of the interval
 *  size (int)  : The size of the input array (if any)
 *
 * Returns :
 *  false if the number (or any number in the input array) does/do not
 *  belong to the interval, true otherwise.
 **************************************************************************/
extern inline bool check_intervals(int *x, int inf, int sup, int size);


/* ************************************************************************
 * CHECK_UNDERFLOW: This function checks for underflows and replaces the
 * underflowed number with the minimum integer. 
 *
 * Args : 
 *  x (int **)        : Double pointer to integer (input array)
 *  y (int *)         : Pointer to integer (lower integer value - boundary)
 *  num_states (int)  : Number of states
 *
 * Returns :
 *  void
 **************************************************************************/
extern inline void check_underflow(int **x, int *y, int num_states);


/* ************************************************************************
 * PRINT_TRUE_OR_FALSE: This function returns a True or False depending on
 * its input argument.
 *
 * Args : 
 *  x (int)   : Input integer (to be inspected)
 *
 * Returns :
 *  char * ("true" if x is other than zero and "false" otherwise).
 **************************************************************************/
extern inline void print_true_or_false(FILE *fp, int x);


/* ************************************************************************
 * PRINT_ENABLED_OR_DISABLED: This function returns ENABLED or DISABLED 
 * depeding on its input argument.
 *
 * Args : 
 *  x (int **)       : Double pointer to an integer
 *
 * Returns :
 *  char * ("ENABLED" if x is other than zero and "DISABLED" otherwise).
 **************************************************************************/
extern inline void print_enabled_or_disabled(FILE *fp, int x);
 

/* ************************************************************************
 * PRINT_RNG_CHOICE: This function prints the RNG used in a simulation.
 *
 * Args : 
 *  fp (FILE *)      : A pointer to the output file
 *  x (int **)       : A boolean defining Box-Muller or LFSR-CASR RNG method
 *
 * Returns :
 *
 **************************************************************************/
extern inline void print_rng_choice(FILE *fp, bool x);
