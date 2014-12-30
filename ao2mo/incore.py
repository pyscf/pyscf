#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import numpy
import ctypes
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

BLOCK = 56

def full(eri_ao, mo_coeff, verbose=0, compact=True):
    return general(eri_ao, (mo_coeff,)*4, verbose, compact)

# It consumes two times of the memory needed by MO integrals
def general(eri_ao, mo_coeffs, verbose=0, compact=True):
    if isinstance(verbose, int):
        log = logger.Logger(sys.stdout, verbose)
    elif isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, 0)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1)//2

    if compact and ijsame:
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        nij_pair = nmoi*nmoj

    if compact and klsame:
        klmosym = 's2'
        nkl_pair = nmok*(nmok+1) // 2
        mokl = numpy.array(mo_coeffs[2], order='F', copy=False)
        klshape = (0, nmok, 0, nmok)
    else:
        klmosym = 's1'
        nkl_pair = nmok*nmol
        mokl = numpy.array(numpy.hstack((mo_coeffs[2],mo_coeffs[3])), \
                           order='F', copy=False)
        klshape = (0, nmok, nmok, nmol)

    if nij_pair == 0 or nkl_pair == 0:
        # 0 dimension sometimes causes blas problem
        return numpy.zeros((nij_pair,nkl_pair))

#    if nij_pair > nkl_pair:
#        log.warn('low efficiency for AO to MO trans!')

# transform e1
    eri1 = half_e1(eri_ao, mo_coeffs, compact)

# transform e2
    eri1 = _ao2mo.nr_e2_(eri1, mokl, klshape, aosym='s4', mosym=klmosym)
    return eri1

def half_e1(eri_ao, mo_coeffs, compact=True):
    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1)//2

    if compact and ijsame:
        ijmosym = 's2'
        nij_pair = nmoi*(nmoi+1) // 2
        moij = numpy.array(mo_coeffs[0], order='F', copy=False)
        ijshape = (0, nmoi, 0, nmoi)
    else:
        ijmosym = 's1'
        nij_pair = nmoi*nmoj
        moij = numpy.array(numpy.hstack((mo_coeffs[0],mo_coeffs[1])), \
                           order='F', copy=False)
        ijshape = (0, nmoi, nmoi, nmoj)

    if eri_ao.size == nao_pair**2: # 4-fold symmetry
        ftrans = _ao2mo._fpointer('AO2MOtranse1_incore_s4')
    else:
        ftrans = _ao2mo._fpointer('AO2MOtranse1_incore_s8')
    if ijmosym == 's2':
        fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_s2')
    elif nmoi <= nmoj:
        fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_iltj')
    else:
        fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_igtj')
    fdrv = getattr(_ao2mo.libao2mo, 'AO2MOnr_e1incore_drv')
    eri1 = numpy.empty((nij_pair,nao_pair))

    for blk0 in range(0, nao_pair, BLOCK):
        blk1 = min(blk0+BLOCK, nao_pair)
        buf = numpy.empty((blk1-blk0,nij_pair))
        fdrv(ftrans, fmmm,
             buf.ctypes.data_as(ctypes.c_void_p),
             eri_ao.ctypes.data_as(ctypes.c_void_p),
             moij.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(blk0), ctypes.c_int(blk1-blk0),
             ctypes.c_int(nao),
             ctypes.c_int(ijshape[0]), ctypes.c_int(ijshape[1]),
             ctypes.c_int(ijshape[2]), ctypes.c_int(ijshape[3]))
        eri1[:,blk0:blk1] = buf.T
    return eri1

def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) or \
            (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()
    import time
    print(time.clock())
    eri0 = full(rhf._eri, rhf.mo_coeff)
    print(abs(eri0).sum()-5384.460843787659) # should = 0
    eri0 = general(rhf._eri, (rhf.mo_coeff,)*4)
    print(abs(eri0).sum()-5384.460843787659)
    print(time.clock())

