from ctypes import Structure, c_char_p

# Limits
XMAX = MAX = 2**15-1
XMIN = -2**15
MIN = -2**15+1
OFF = -16
WMAX = 128
CHANNEL_OFFSET = 18
ADDR_MASK = 2**CHANNEL_OFFSET-1
N_TOT = 2048
OFF = -16
TSTDPMAX = 1023
ISIMAX = 255
N_GROUPS = 8

"""
File Compression strategies, use instead of 'with open(filename):', use like
'with compression_strategy(filename):' for writing and reading to files gzip 
is the fastest, lzma is the smallest, bz2 is in-between.  Gzip is within epsilon
of _pickle run time but with 10x smaller file size, probably good enough.  Lzma 
file size is within epsilon of _pickle, but run time is 10x slower (still very
small in absolute terms).
""" 
# import lzma
# compression_strategy = lzma.open
# import bz2
# compression_strategy = bz2.open
import gzip
compression_strategy = gzip.open

# Fnames are needed outside NSAT and thereby moved here
FNAME_FIELDS = ['nsat_params_map',
                'lrn_params_map',
                'params',
                'syn_wgt_table',
                'syn_ptr_table',
                'ext_events',
                'synw',
                'synw_final',
                'events',
                'states',
                'check_pms',
                'stdp_fun',
                'stats_nsat',
                'stats_ext',
                'l1_conn',
                'shared_mem',
                'pickled']

    
class c_nsat_fnames(Structure):
    """ fnames class implements the C struct: fnames. Contains the
        filenames of all the necessary input files.
    """
    _fields_ = [(s, c_char_p) for s in FNAME_FIELDS]
    
    def __init__(self, fname=None):
#         self._fields = [(s, c_char_p) for s in FNAME_FIELDS]
        if fname is not None:
            for f in fname._fields:
                setattr(self, f, getattr(fname, f).encode('utf-8'))

#     def generate(self, fname=None):
#         if fname is not None:
#             for f in fname.fields:
#                 setattr(self, f, getattr(fname, f).encode('utf-8'))
#         return self

'''    
def generate_c_fnames(fname=None):
    c_fnames = c_nsat_fnames()
    if fname is not None:
        for f in fname.fields:
            setattr(c_fnames, f, getattr(fname, f).encode('utf-8'))
    return c_fnames
'''

class nsat_fnames(object):
    """ fnames class implements the C struct: fnames. Contains the
        filenames of all the necessary input files.
    """
    _fields = FNAME_FIELDS

    def __init__(self):
        for f in self._fields:
            setattr(self, f, '')
            
    def generate(self, path):
        fname = self
        fname.nsat_params_map = path + '_nsat_params_map.dat'
        fname.lrn_params_map = path + '_lrn_params_map.dat'
        fname.params = path + '_params.dat'
        fname.syn_wgt_table = path + '_wgt_table'
        fname.syn_ptr_table = path + '_ptr_table'
        fname.ext_events = path + '_ext_events'
        fname.synw = path + '_weights'
        fname.synw_final = path + '_weights_final'
        fname.events = path + '_events'
        fname.states = path + '_states'
        fname.check_pms = path + '_cpms.dat'
        fname.stdp_fun = path + '_stdp_fun.dat'
        fname.stats_nsat = path + '_stats_nsat'
        fname.stats_ext = path + '_stats_ext'
        fname.l1_conn = path + '_l1_conn.dat'
        fname.shared_mem = path + '_shared_mem'
        fname.pickled = path + '_pickled_config'
        return fname

'''
def generate_default_fnames(path):
    fname = nsat_fnames()
    fname.nsat_params_map = path + "_nsat_params_map.dat"
    fname.lrn_params_map = path + "_lrn_params_map.dat"
    fname.params = path + "_params.dat"
    fname.syn_wgt_table = path + "_wgt_table"
    fname.syn_ptr_table = path + "_ptr_table"
    fname.ext_events = path + "_ext_events"
    fname.synw = path + "_weights"
    fname.synw_final = path + "_weights_final"
    fname.events = path + "_events"
    fname.states = path + "_states"
    fname.check_pms = path + "_cpms.dat"
    fname.stdp_fun = path + "_stdp_fun.dat"
    fname.stats_nsat = path + "_stats_nsat"
    fname.stats_ext = path + "_stats_ext"
    fname.l1_conn = path + "_l1_conn.dat"
    fname.shared_mem = path + "_shared_mem"
    fname.pickled = path + '_pickled_config'
    return fname
'''
fnames = nsat_fnames()