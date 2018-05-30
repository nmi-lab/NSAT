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


/* ************************************************************************
 * GEN_FNAME: This function generates a finelame by concatenating the 
 * filename and the number of core.
 *
 * Args : 
 *  ifname (char *)   : Filename
 *  x (int)           : Number of core ID
 *
 * Returns :
 *  A string containing the full filename
 **************************************************************************/
char *gen_fname(char *ifname, int x, int flag)
{
    char base[] = "_core_", core_id[5];
    char dat_ext[5]=".dat", hex_ext[5]=".hex";
    char *tmp = NULL;
    size_t size = strlen(ifname)+7+5+5+1;

    tmp = alloc(char, size);
    sprintf(core_id, "%d", x);
    strcpy(tmp, ifname);
    strcat(tmp, base);
    strcat(tmp, core_id);
    if (flag) {
        strcat(tmp, dat_ext);
    } else {
        strcat(tmp, hex_ext);
    }

    return tmp;
}


char *gen_ext_evts_fname(char *fname, unsigned int id)
{
    char ext[10] = ".dat", postfix[20] = "_core_";
    char *tmp = NULL;
    char core_num[100];

    tmp = alloc(char, 200);

    strcpy(tmp, fname);
    strcat(tmp, postfix);
    sprintf(core_num, "%u", id);
    strcat(tmp, core_num);
    strcat(tmp, ext);

    return tmp;
}


char *add_extension(char *fname)
{
    char *tmp = NULL;
    char ext[10] = ".dat";
    tmp = alloc(char, 200);

    strcpy(tmp, fname);
    strcat(tmp, ext);
    
    return tmp;
}
