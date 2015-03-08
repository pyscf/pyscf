#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
import h5py
import pyscf.lib
from pyscf.lib import logger
import pyscf.gto
from pyscf import ao2mo
from pyscf.scf import _vhf
from pyscf.df import incore

class load(ao2mo.load):
    '''load 3c2e integrals from hdf5 file
    Usage:
        with load(cderifile) as eri:
            print eri.shape
    '''
    pass

