#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import numpy
import lib._ao2mo as _ao2mo

# full_eri in Mulliken notation
def gen_int2e_from_full_eri(full_eri, nao=None):
    if nao is None:
        nao = full_eri.shape[0]
    else:
        full_eri = full_eri.reshape(nao,nao,nao,nao)
    int2e = numpy.empty((nao*(nao+1)/2,nao*(nao+1)/2))
    for i in range(nao):
        for j in range(i+1):
            ij = i*(i+1)/2 + j
            for k in range(nao):
                for l in range(k+1):
                    kl = k*(k+1)/2 + l
                    int2e[ij,kl] = full_eri[i,j,k,l]
    return int2e

def get_int2e_from_partial_eri(eri_ao, mo_coeff):
    return full(eri_ao, mo_coeff)

def full(eri_ao, mo_coeff):
    if mo_coeff.flags.c_contiguous:
        mo_coeff = mo_coeff.copy('F')
    if eri_ao.ndim == 1:
        return _ao2mo.partial_eri_ao2mo_o3(eri_ao, mo_coeff)
    elif eri_ao.ndim == 2:
        return _ao2mo.partial_eri_ao2mo_o2(eri_ao, mo_coeff)


# It consumes two times of the memory needed by MO integrals
def general(eri_ao, mo_coeffs, compact=True):
    def iden_coeffs(mo1, mo2):
        return (id(mo1) == id(mo2)) \
                or (mo1.shape==mo2.shape and abs(mo1-mo2).sum()<1e-12)

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

    if nij_pair > nkl_pair:
        print 'low efficiency for AO to MO trans!'

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1) / 2

    if ijsame:
        moji = numpy.array(mo_coeffs[0], order='F')
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moji = numpy.array(numpy.hstack((mo_coeffs[1],mo_coeffs[0])), order='F')
        ijshape = (nmoj, nmoi, 0, nmoj)

    if klsame:
        molk = numpy.array(mo_coeffs[2], order='F')
        klshape = (0, nmok, 0, nmok)
    else:
        molk = numpy.array(numpy.hstack((mo_coeffs[3],mo_coeffs[2])), order='F')
        klshape = (nmol, nmok, 0, nmol)

    buf = _ao2mo.nr_e1_ao2mo_incore(eri_ao, moji, ijshape)
    buf = _ao2mo.nr_e2_ao2mo(buf, molk, klshape)
    return buf

if __name__ == '__main__':
    import scf
    import gto
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
    print time.clock()
    eri0 = full(rhf._eri, rhf.mo_coeff)
    print abs(eri0).sum()-5384.460843787659 # should = 0
    print time.clock()

