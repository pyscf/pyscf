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

def load_buf(cderi_or_h5file, start_id, count=160, dataname='eri_mo'):
    if isinstance(cderi_or_h5file, str):
        return _load_file(cderi_or_h5file, start_id, count, dataname)
    elif isinstance(cderi_or_h5file, numpy.ndarray):
        return _load_array(cderi_or_h5file, start_id, count)
    else:
        raise ValueError('Unknown eri date type %s', type(cderi_or_h5file))

def _load_file(cderi_file, start_id, count=160, dataname='eri_mo'):
    feri = h5py.File(cderi_file, 'r')
    if ('%s/0/0'%dataname) in feri:
        comp = len(feri[dataname])
        nset = len(feri['%s/0'%dataname])
        ncol = sum([feri['%s/0/%d'%(dataname,i)].shape[1] for i in range(nset)])
        nrow = min(feri['%s/0/0'%dataname].shape[0]-start_id, count)
        end = start_id + nrow
        buf = numpy.empty((comp,nrow,ncol))
        for icomp in range(comp):
            p0 = 0
            for i in range(nset):
                dat = feri['%s/%d/%d'%(dataname,icomp,i)]
                p1 = p0 + dat.shape[2]
                buf[icomp,:,p0:p1] = dat[start_id:end]
                p0 = p1
    else:
        nset = len(feri[dataname])
        ncol = sum([feri['%s/%d'%(dataname,i)].shape[1] for i in range(nset)])
        nrow = min(feri['%s/0'%dataname].shape[0]-start_id, count)
        end = start_id + nrow
        buf = numpy.empty((nrow,ncol))
        p0 = 0
        for i in range(nset):
            dat = feri['%s/%d'%(dataname,i)]
            p1 = p0 + dat.shape[1]
            buf[:,p0:p1] = dat[start_id:end]
            p0 = p1
    feri.close()
    return buf

def _load_array(cderi, start_id, count=160):
    if cderi.ndim == 2:
        nrow, ncol = cderi.shape
        end = min(nrow, start_id+count)
        return cderi[start_id:end]
    else:
        comp, nrow, ncol = cderi.shape
        end = min(nrow, start_id+count)
        return cderi[:,start_id:end]

