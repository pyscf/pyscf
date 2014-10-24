#!/usr/bin/env python

import numpy
from pyscf import gto
from pyscf import symm
import pyscf.lib.logger as log
import pyscf.gto.moleintor
import chkfile

def frac_occ(scf, tol=1e-3):
    def set_occ(mo_energy, mo_coeff=None):
        mol = scf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron / 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            nsocc = int(lst.sum())
            ndocc = nocc - int(lst[:nocc].sum())
            frac = 2.*(nocc-ndocc)/nsocc
            mo_occ[nsocc:ndocc] = frac
            log.warn(scf, 'fraction occ = %6g, [%d:%d]', frac, ndocc, ndocc+nsocc)
        if nocc < mo_occ.size:
            log.info(scf, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(scf, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(scf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return set_occ

def dynamic_occ(scf, tol=1e-3):
    def set_occ(mo_energy, mo_coeff=None):
        mol = scf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron / 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            ndocc = nocc - int(lst[:nocc].sum())
            mo_occ[ndocc:nocc] = 0
            log.warn(scf, 'set charge = %d', (nocc-ndocc)*2)
        if nocc < mo_occ.size:
            log.info(scf, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(scf, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(scf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return set_occ

def follow_state():
    pass



def project_mo_nr2nr(mol1, mo1, mol2):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atm, bas, env = gto.mole.conc_env(mol2._atm, mol2._bas, mol2._env, \
                                      mol1._atm, mol1._bas, mol1._env)
    bras = kets = range(nbas2)
    s22 = gto.moleintor.getints('cint1e_ovlp_sph', atm, bas, env, \
                                bras, kets, dim3=1, hermi=1)
    bras = range(nbas2)
    kets = range(nbas2, nbas1+nbas2)
    s21 = gto.moleintor.getints('cint1e_ovlp_sph', atm, bas, env, \
                                bras, kets, dim3=1, hermi=0)
    return numpy.linalg.solve(s22, numpy.dot(s21, mo1))

def project_mo_nr2r(mol1, mo1, mol2):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atm, bas, env = gto.mole.conc_env(mol2._atm, mol2._bas, mol2._env, \
                                      mol1._atm, mol1._bas, mol1._env)
    bras = kets = range(nbas2)
    s22 = gto.moleintor.getints('cint1e_ovlp', atm, bas, env, \
                                bras, kets, dim3=1, hermi=1)
    bras = range(nbas2)
    kets = range(nbas2, nbas1+nbas2)
    s21 = gto.moleintor.getints('cint1e_ovlp_sph', atm, bas, env, \
                                bras, kets, dim3=1, hermi=0)

    ua, ub = symm.cg.real2spinor_whole(mol2)
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    mo2 = numpy.dot(s21, mo1)
    return numpy.linalg.solve(s22, mo2)

def project_mo_r2r(mol1, mo1, mol2):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atm, bas, env = gto.mole.conc_env(mol2._atm, mol2._bas, mol2._env, \
                                      mol1._atm, mol1._bas, mol1._env)
    bras = kets = range(nbas2)
    s22 = gto.moleintor.getints('cint1e_ovlp', atm, bas, env, \
                                bras, kets, dim3=1, hermi=1)
    t22 = gto.moleintor.getints('cint1e_spsp', atm, bas, env, \
                                bras, kets, dim3=1, hermi=1)
    bras = range(nbas2)
    kets = range(nbas2, nbas1+nbas2)
    s21 = gto.moleintor.getints('cint1e_ovlp', atm, bas, env, \
                                bras, kets, dim3=1, hermi=0)
    t21 = gto.moleintor.getints('cint1e_spsp', atm, bas, env, \
                                bras, kets, dim3=1, hermi=0)
    n2c = s21.shape[1]
    pl = numpy.linalg.solve(s22, s21)
    ps = numpy.linalg.solve(t22, t21)
    return numpy.vstack((numpy.dot(pl, mo1[:n2c]),
                         numpy.dot(ps, mo1[n2c:])))

def init_guess_by_chkfile(mf, chkfile_name, projection=True):
    import hf
    import dhf
    mol = mf.mol
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)

    fproj = lambda mo: mo
    fdm = lambda: numpy.dot(mo*mo_occ, mo.T.conj())
    if isinstance(mf, dhf.UHF) or 'dhf' in str(mf.__class__):
        if numpy.iscomplexobj(scf_rec['mo_coeff']):
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            fproj = lambda mo: project_mo_r2r(chk_mol, mo, mol)
        else:
            if scf_rec['mo_coeff'].ndim == 2: # nr-RHF
                mo = project_mo_nr2r(chk_mol, scf_rec['mo_coeff'], mol)
                mo_occ = scf_rec['mo_occ'] * .5
            else: # nr-UHF
                mo = project_mo_nr2r(chk_mol, scf_rec['mo_coeff'][0], mol)
                mo_occ = scf_rec['mo_occ'][0]
            n2c = mo.shape[0]
            dm = numpy.zeros((n2c*2,n2c*2), dtype=complex)
            dm_ll = numpy.dot(mo*mo_occ, mo.T.conj())
            dm[:n2c,:n2c] = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
            fdm = lambda: dm
    elif isinstance(mf, hf.RHF) or 'RHF' in str(mf.__class__): # nr-RHF
        if scf_rec['mo_coeff'].ndim == 2:
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
        else:
            mo = scf_rec['mo_coeff'][0]
            mo_occ = scf_rec['mo_occ'][0]
        fproj = lambda mo: project_mo_nr2nr(chk_mol, mo, mol)
    else: # nr-UHF
        if scf_rec['mo_coeff'].ndim == 2:
            mo = (scf_rec['mo_coeff'],)*2
            mo_occ = (scf_rec['mo_occ'],)*2
        else:
            mo = (scf_rec['mo_coeff'][0], scf_rec['mo_coeff'][1])
            mo_occ = (scf_rec['mo_occ'][0], scf_rec['mo_occ'][1])
        fproj = lambda mo: (project_mo_nr2nr(chk_mol, mo[0], mol),
                            project_mo_nr2nr(chk_mol, mo[1], mol))
        fdm = lambda: numpy.array(numpy.dot(mo[0]*mo_occ[0], mo[0].T), \
                                  numpy.dot(mo[1]*mo_occ[1], mo[1].T))

    if projection:
        mo = fproj(mo)

    dm = fdm()

    hf_energy = scf_rec['hf_energy'] + 0
    mo = mo_occ = chk_mol = scf_rec = None

    def finit(*args):
        return hf_energy, dm
    return finit

