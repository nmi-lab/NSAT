#-----------------------------------------------------------------------------
# Purpose:
#
# Author: Emre Neftci
#
# Copyright : University of Zurich, Giacomo Indiveri, Emre Neftci, Sadique Sheik, Fabio Stefanini
# Licence : GPLv2
#-----------------------------------------------------------------------------
# -*- coding: utf-8 -*-

#For jaer monitoring
import tarfile
import glob
import time
import os
import fnmatch
import warnings
import pickle
#try:
#    import dill as pickle
#except ImportError:
#    warnings.warn('Cannot import dill, falling back to pickle. Several pyNCS objects cannot be pickled')
#    import pickle 

import numpy as np
from shutil import rmtree

### The globals class
class datacontainer:
    def __init__(self):
        self.directory = './'

global globaldata
globaldata = datacontainer()

REC_FN_SEQ = 'seq_address_specification'
REC_FN_MON = 'mon_address_specification'

def get_figsize(fig_width_pt, ratio='g'):
    """
    Method to generate figure size.
    """
    inches_per_pt = 1.0 / 72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    if ratio is 'g':
        fig_height = fig_width * golden_mean      # height in inches
    elif ratio is 's':
        fig_height = fig_width                  # square figure
    else:
        fig_height = 1. * fig_width / ratio
    fig_size = [fig_width, fig_height]      # exact figsize
    return fig_size


def loadPlotParameters(size=0.5, fontsize=18.0):
    import pylab
    """
    Load default matplotlib parameters for pretty plotting
    size: 0.5 -- two column page.
          0.33 -- three column page.
          0.25 -- two column double figure.
    fontsize: universal font size
    """
    if size <= 0.25:
        border = 0.22
    elif size <= 0.33:
        border = 0.20
    else:
        border = 0.15
    params0 = {'backend': 'pdf',
          'savefig.dpi': 300.,
          'axes.labelsize': fontsize,
          'figure.subplot.bottom': border,
          'figure.subplot.left': border,
          'text.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'legend.pad': 0.1,    # empty space around the legend box
          'legend.fontsize': fontsize,
          'lines.markersize': 5,
          'lines.linewidth': 1,
          'font.size': fontsize,
          'text.usetex': True,
          'figure.figsize': get_figsize(1000 * size, ratio=1.3)}  # size in inches
    pylab.rcParams.update(params0)
    
def load_compatibility(filename):
    """
    Same as experimentTools.load(), but works around recent module renaming problems
    Code from http://wiki.python.org/moin/UsingPickle/RenamingModules
    """
    import pickle as pickle #Does not work with dill
    renametable = {
        'pyST': 'pyNCS.pyST',
        'pyST.spikes': 'pyNCS.pyST.spikes',
        'pyST.STas': 'pyNCS.pyST.STas',
        'pyST.STsl': 'pyNCS.pyST.STsl',
        'pyST.stgen': 'pyNCS.pyST.stgen',
        }
    
    def mapname(name):
        if name in renametable:
            return renametable[name]
        return name
    
    def mapped_load_global(self):
        module = mapname(self.readline()[:-1])
        name = mapname(self.readline()[:-1])
        klass = self.find_class(module, name)
        self.append(klass)
        
    def loads(filename_):
        with open(filename_,'rb') as fh:
            unpickler = pickle.Unpickler(fh)
        unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
        return unpickler.load()
    
    return loads(filename)

def load(filename=None, compatibility=False):
    """
    Unpickles file named 'filename' from the results directory. If no 'filename' is given, then 'globaldata.pickle' is loaded
    """
    if filename == None: 
        filename = globaldata.directory + 'globaldata.pickle'
    else:
        filename = globaldata.directory + filename        
    if compatibility:
        return load_compatibility(filename)
    else:
        return pickle.load(open(filename, 'rb'))    


def save_source():
    """
    Save all the python scripts from the current directory into the results directory
    """
    h = tarfile.open(globaldata.directory + 'exp_scripts.tar.bz2', 'w:bz2')
    all_src = []
    all_src += glob.glob('*.py')
    all_src += glob.glob('*.cpp')
    all_src += glob.glob('*.h')
    all_src += glob.glob('*.java')
    all_src += glob.glob('*.m')
    for i in all_src:
        h.add(i)
    h.close()
    
def save_file(filename):
    """
    Save all the python scripts from the current directory into the results directory
    """
    import shutil
    shutil.copy(filename, globaldata.directory+filename)


def save(obj=None, filename=None):
    if obj == None and filename == None:
        with open(globaldata.directory + 'globaldata.pickle', 'w') as f:
            pickle.dump(globaldata, f)
        save_source()
    elif obj == None and filename != None:
        with open(globaldata.directory + filename, 'w') as f:
            pickle.dump(globaldata, f)
    else:
        with open(globaldata.directory + filename, 'w') as f:
            pickle.dump(obj, f)
    return None


def savetxt(obj, filename):
    np.savetxt(globaldata.directory + filename, obj)


def mksavedir(pre='Results/', exp_dir=None):
    """
    Creates a results directory in the subdirectory 'pre'. The directory name is given by ###__dd_mm_yy, where ### is the next unused 3 digit number
    """

    if pre[-1] != '/':
        pre + '/'

    if not os.path.exists(pre):
        os.makedirs(pre)
    prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))

    if exp_dir == None:
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (
                int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        direct = time.strftime(
            pre + expDirN + "__" + "%d-%m-%Y", time.localtime())
        assert not os.path.exists(direct)

    elif isinstance(exp_dir, str):
        direct = pre + exp_dir
        if os.path.exists(direct):
            print(("Warning: overwriting directory {0}".format(direct)))
            rmtree(direct)

    else:
        raise TypeError('exp_dir should be a string')

    os.mkdir(direct)

    globaldata.directory = direct + str('/')

    print(("Created experiment directory {0}".format(globaldata.directory)))
    return globaldata.directory


def savefig(filename, *args, **kwargs):
    """
    Like pylab.savefig but appends the Results directory
    """
    import pylab
    pylab.savefig(globaldata.directory + filename, *args, **kwargs)


def annotate(filename='', text=''):
    "Create a file in the Results directory, with contents text"
    with open(globaldata.directory + filename, 'w') as f:
        f.write(text)

def savefigs(filename='fig', extension = 'png', close=True, *args, **kwargs):
    """
    Saves all figures with filename *filename#* where # is the figure number.
    The order is: last opened last saved.
    Inputs:
    *filename*: figure name prefix
    *extension*: figure extension. savefig should resolve the format from the extension
    *close*: whether to close the figure after it is saved
    *args, **kwargs  are passed to savefig  
    """
    import matplotlib,pylab
    figures = [manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    print(('Saving {0} figures'.format(len(figures))))
    for i, f in enumerate(figures):
        savefig('fig'+str(i)+'.'+extension, *args, **kwargs)
        if close: pylab.close()
