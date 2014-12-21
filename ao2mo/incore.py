#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import _ao2mo

def full(eri_ao, mo_coeff, verbose=None, compact=True):
    return general(eri_ao, (mo_coeff,)*4, verbose, compact)

# It consumes two times of the memory needed by MO integrals
def general(eri_ao, mo_coeffs, verbose=None, compact=True):
    def iden_coeffs(mo1, mo2):
        return (id(mo1) == id(mo2)) \
                or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    if ijsame:
        nij_pair = nmoi*(nmoi+1) / 2
    else:
        nij_pair = nmoi*nmoj

    if klsame:
        nkl_pair = nmok*(nmok+1) / 2
    else:
        nkl_pair = nmok*nmol
    if nij_pair == 0 or nkl_pair == 0:
        # 0 dimension sometimes causes blas' complaint
        return numpy.zeros((nij_pair,nkl_pair))

    if nij_pair > nkl_pair:
        print('low efficiency for AO to MO trans!')

    if ijsame:
        moji = numpy.array(mo_coeffs[0], order='F', copy=False)
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moji = numpy.array(numpy.hstack((mo_coeffs[1],mo_coeffs[0])), \
                           order='F', copy=False)
        ijshape = (nmoj, nmoi, 0, nmoj)

    if klsame:
        molk = numpy.array(mo_coeffs[2], order='F', copy=False)
        klshape = (0, nmok, 0, nmok)
    else:
        molk = numpy.array(numpy.hstack((mo_coeffs[3],mo_coeffs[2])), \
                           order='F', copy=False)
        klshape = (nmol, nmok, 0, nmol)

    buf = _ao2mo.nr_e1_incore_(eri_ao, moji, ijshape)
    buf = _ao2mo.nr_e2_(buf, molk, klshape)
    return buf

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

