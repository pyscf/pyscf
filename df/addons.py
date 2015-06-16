#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import ao2mo

class load(ao2mo.load):
    '''load 3c2e integrals from hdf5 file
    Usage:
        with load(cderifile) as eri:
            print eri.shape
    '''
    pass

