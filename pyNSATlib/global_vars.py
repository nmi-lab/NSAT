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