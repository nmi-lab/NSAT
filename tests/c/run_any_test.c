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
#include "../../include/nsat.h"


char *set_name(char *base, char *in_name) {
    char *res = NULL;

    res = (char *) malloc(sizeof(char) * 500);

    strcpy(res, base);
    strcat(res, in_name);
    return res;
}


int main(int argc, char **argv) {
    if (argc == 2) {
        fnames fname;
        char *names[16] = {"_nsat_params_map.dat", "_lrn_params_map.dat",
                           "_params.dat", "_wgt_table", "_ext_events",
                           "_events", "_states", "_weights", "_weights_final",
                           "_cpms.dat", "_stdp_fun.dat", "_nsat_stats",
                           "_ext_stats", "_l1_conn.dat", "_shared_mem",
                           "_ptr_table"};

        fname.nsat_params_map = set_name(argv[1], names[0]);
        fname.lrn_params_map = set_name(argv[1], names[1]);
        fname.params = set_name(argv[1], names[2]);
        fname.syn_wgt_table = set_name(argv[1], names[3]);
        fname.syn_ptr_table = set_name(argv[1], names[15]);
        fname.ext_events = set_name(argv[1], names[4]);
        fname.events = set_name(argv[1], names[5]);
        fname.states = set_name(argv[1], names[6]);
        fname.synw = set_name(argv[1], names[7]);
        fname.synw_final = set_name(argv[1], names[8]);
        fname.check_pms = set_name(argv[1], names[9]);
        fname.stdp_fun = set_name(argv[1], names[10]);
        fname.stats_nsat = set_name(argv[1], names[11]);
        fname.stats_ext = set_name(argv[1], names[12]);
        fname.l1_conn = set_name(argv[1], names[13]);
        fname.shared_mem = set_name(argv[1], names[14]);

        iterate_nsat(&fname);

        dealloc(fname.nsat_params_map);
        dealloc(fname.lrn_params_map);
        dealloc(fname.params);
        dealloc(fname.syn_wgt_table);
        dealloc(fname.syn_ptr_table);
        dealloc(fname.ext_events);
        dealloc(fname.events);
        dealloc(fname.states);
        dealloc(fname.synw);
        dealloc(fname.synw_final);
        dealloc(fname.check_pms);
        dealloc(fname.stdp_fun);
        dealloc(fname.stats_nsat);
        dealloc(fname.stats_ext);
        dealloc(fname.l1_conn);
        dealloc(fname.shared_mem);
    } else {
        printf(ANSI_COLOR_RED "ERROR:  " ANSI_COLOR_RESET);
        printf("Missing input argument (path where files are)!\n");
        exit(-1);
    }

    return 0;
}

