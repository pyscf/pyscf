#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf.gto import mole
from pyscf.gto import moleintor
from pyscf.lib import logger
import pyscf.symm
from pyscf.scf import hf


def frac_occ(mf, tol=1e-3):
    assert(isinstance(mf, hf.RHF))
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron // 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            nsocc = int(lst.sum())
            ndocc = nocc - int(lst[:nocc].sum())
            frac = 2.*(nocc-ndocc)/nsocc
            mo_occ[nsocc:ndocc] = frac
            logger.warn(mf, 'fraction occ = %6g  [%d:%d]',
                        frac, ndocc, ndocc+nsocc)
        if nocc < mo_occ.size:
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])
        logger.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return get_occ
def frac_occ_(mf, tol=1e-3):
    mf.get_occ = frac_occ(mf, tol)
    return mf.get_occ

def dynamic_occ(mf, tol=1e-3):
    assert(isinstance(mf, hf.RHF))
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron // 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            ndocc = nocc - int(lst[:nocc].sum())
            mo_occ[ndocc:nocc] = 0
            logger.warn(mf, 'set charge = %d', mol.charge+(nocc-ndocc)*2)
        if nocc < mo_occ.size:
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])
        logger.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return get_occ
def dynamic_occ_(mf, tol=1e-3):
    mf.get_occ = dynamic_occ(mf, tol)
    return mf.get_occ

def float_occ(mf):
    '''for UHF, do not fix the nelec_alpha. determine occupation based on energy spectrum'''
    from pyscf.scf import uhf
    assert(isinstance(mf, uhf.UHF))
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        ee = numpy.sort(numpy.hstack(mo_energy))
        n_a = int((mo_energy[0]<(ee[mol.nelectron-1]+1e-3)).sum())
        n_b = mol.nelectron - n_a
        if n_a != mf.nelec[0]:
            logger.info(mf, 'change num. alpha/beta electrons '
                        ' %d / %d -> %d / %d',
                        mf.nelec[0], mf.nelec[1], n_a, n_b)
            mf.nelec = (n_a, n_b)
        return uhf.UHF.get_occ(mf, mo_energy, mo_coeff)
    return get_occ
def float_occ_(mf):
    mf.get_occ = float_occ(mf)
    return mf.get_occ

def symm_allow_occ(mf, tol=1e-3):
    '''search the unoccupied orbitals, choose the lowest sets which do not
break symmetry as the occupied orbitals'''
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron // 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            nocc_left = int(lst[:nocc].sum())
            ndocc = nocc - nocc_left
            mo_occ[ndocc:nocc] = 0
            i = ndocc
            nmo = len(mo_energy)
            logger.info(mf, 'symm_allow_occ [:%d] = 2', ndocc)
            while i < nmo and nocc_left > 0:
                deg = (abs(mo_energy[i:i+5]-mo_energy[i]) < tol).sum()
                if deg <= nocc_left:
                    mo_occ[i:i+deg] = 2
                    nocc_left -= deg
                    logger.info(mf, 'symm_allow_occ [%d:%d] = 2, energy = %.12g',
                                i, i+nocc_left, mo_energy[i])
                    break
                else:
                    i += deg
        logger.info(mf, 'HOMO = %.12g, LUMO = %.12g,',
                    mo_energy[nocc-1], mo_energy[nocc])
        logger.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return get_occ
def symm_allow_occ_(mf, tol=1e-3):
    mf.get_occ = symm_allow_occ(mf, tol)
    return mf.get_occ

def follow_state(mf, occorb=None):
    occstat = [occorb]
    old_get_occ = mf.get_occ
    def get_occ(mo_energy, mo_coeff=None):
        if occstat[0] is None:
            mo_occ = old_get_occ(mo_energy, mo_coeff)
        else:
            mo_occ = numpy.zeros_like(mo_energy)
            s = reduce(numpy.dot, (occstat[0].T, mf.get_ovlp(), mo_coeff))
            nocc = mf.mol.nelectron // 2
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx = numpy.argsort(numpy.einsum('ij,ij->j', s, s))
            mo_occ[idx[-nocc:]] = 2
            logger.debug(mf, '  mo_occ = %s', mo_occ)
            logger.debug(mf, '  mo_energy = %s', mo_energy)
        occstat[0] = mo_coeff[:,mo_occ>0]
        return mo_occ
    return get_occ
def follow_state_(mf, occorb=None):
    mf.get_occ = follow_state_(mf, occorb)
    return mf.get_occ



def project_mo_nr2nr(mol1, mo1, mol2):
    ''' Project orbital coefficients

    |psi1> = |AO1> C1
    |psi2> = P |psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2
    C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = mol2.intor_symmetric('cint1e_ovlp_sph')
    s21 = mole.intor_cross('cint1e_ovlp_sph', mol2, mol1)
    return numpy.linalg.solve(s22, numpy.dot(s21, mo1))

def project_mo_nr2r(mol1, mo1, mol2):
    s22 = mol2.intor_symmetric('cint1e_ovlp')
    s21 = mole.intor_cross('cint1e_ovlp_sph', mol2, mol1)

    ua, ub = pyscf.symm.cg.real2spinor_whole(mol2)
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    mo2 = numpy.dot(s21, mo1)
    return numpy.linalg.solve(s22, mo2)

def project_mo_r2r(mol1, mo1, mol2):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atm, bas, env = mole.conc_env(mol2._atm, mol2._bas, mol2._env,
                                  mol1._atm, mol1._bas, mol1._env)
    bras = kets = range(nbas2)
    s22 = moleintor.getints('cint1e_ovlp', atm, bas, env,
                            bras, kets, comp=1, hermi=1)
    t22 = moleintor.getints('cint1e_spsp', atm, bas, env,
                            bras, kets, comp=1, hermi=1)
    bras = range(nbas2)
    kets = range(nbas2, nbas1+nbas2)
    s21 = moleintor.getints('cint1e_ovlp', atm, bas, env,
                            bras, kets, comp=1, hermi=0)
    t21 = moleintor.getints('cint1e_spsp', atm, bas, env,
                            bras, kets, comp=1, hermi=0)
    n2c = s21.shape[1]
    pl = numpy.linalg.solve(s22, s21)
    ps = numpy.linalg.solve(t22, t21)
    return numpy.vstack((numpy.dot(pl, mo1[:n2c]),
                         numpy.dot(ps, mo1[n2c:])))


