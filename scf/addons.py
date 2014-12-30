#!/usr/bin/env python

from functools import reduce
import weakref
import numpy
import pyscf.gto.mole as mole
import pyscf.gto.moleintor as moleintor
import pyscf.lib.logger as log
import pyscf.symm
from pyscf.scf import hf
from pyscf.scf import chkfile

def frac_occ(mf, tol=1e-3):
    mf = weakref.ref(mf)
    def set_occ(mo_energy, mo_coeff=None):
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
            log.warn(mf, 'fraction occ = %6g, [%d:%d]', frac, ndocc, ndocc+nsocc)
        if nocc < mo_occ.size:
            log.info(mf, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(mf, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return set_occ

def dynamic_occ(mf, tol=1e-3):
    mf = weakref.ref(mf)
    def set_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron // 2
        mo_occ[:nocc] = 2
        if abs(mo_energy[nocc-1] - mo_energy[nocc]) < tol:
            lst = abs(mo_energy - mo_energy[nocc-1]) < tol
            ndocc = nocc - int(lst[:nocc].sum())
            mo_occ[ndocc:nocc] = 0
            log.warn(mf, 'set charge = %d', (nocc-ndocc)*2)
        if nocc < mo_occ.size:
            log.info(mf, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(mf, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return set_occ

def float_occ(uhf):
    '''for UHF, do not fix the nelec_alpha. determine occupation based on energy spectrum'''
    assert(isinstance(uhf, hf.UHF))
    uhf = weakref.ref(uhf)
    def set_occ(mo_energy, mo_coeff=None):
        mol = uhf.mol
        ee = sorted([(e,0) for e in mo_energy[0]] \
                    + [(e,1) for e in mo_energy[1]])
        n_a = len([x for x in ee[:mol.nelectron] if x[1]==0])
        n_b = mol.nelectron - n_a
        if n_a != uhf.nelectron_alpha:
            log.info(uhf, 'change num. alpha/beta electrons ' \
                     ' %d / %d -> %d / %d', \
                     uhf.nelectron_alpha,
                     mol.nelectron-uhf.nelectron_alpha, n_a, n_b)
            uhf.nelectron_alpha = n_a
        return hf.UHF.set_occ(uhf, mo_energy, mo_coeff)
    return set_occ

def symm_allow_occ(mf, tol=1e-3):
    '''search the unoccupied orbitals, choose the lowest sets which do not
break symmetry as the occupied orbitals'''
    mf = weakref.ref(mf)
    def set_occ(mo_energy, mo_coeff=None):
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
            log.info(mf, 'occ [:%d] = 2', ndocc)
            while i < nmo and nocc_left > 0:
                deg = (abs(mo_energy[i:i+5]-mo_energy[i]) < tol).sum()
                if deg <= nocc_left:
                    mo_occ[i:i+deg] = 2
                    nocc_left -= deg
                    log.info(mf, 'occ [%d:%d] = 2, energy = %.12g',
                             i, i+nocc_left, mo_energy[i])
                    break
                else:
                    i += deg
        log.info(mf, 'HOMO = %.12g, LUMO = %.12g,', \
                  mo_energy[ndocc-1], mo_energy[ndocc])
        log.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ
    return set_occ

def follow_state(mf):
    pass

def break_spin_sym(mol, mo_coeff, level=1):
    '''level = 1, mix 5 HOMO_beta plus 5 LUMO_beta
level = 2, HOMO_beta = LUMO_alpha
level = 3, HOMO_beta = 0'''
    # break symmetry between alpha and beta
    mo_coeff = mo_coeff.copy()
    nocc = mol.nelectron // 2
    if opt == 1: # break spatial symmetry
        nmo = mo_coeff[0].shape[1]
        nvir = nmo - nocc
        if nvir < 5:
            for i in range(nocc-1,nmo):
                mo_coeff[1][:,nocc-1] += mo_coeff[0][:,i]
            mo_coeff[1][:,nocc-1] *= 1./(nvir+1)
        else:
            for i in range(nocc-1,nocc+5):
                mo_coeff[1][:,nocc-1] += mo_coeff[0][:,i]
            mo_coeff[1][:,nocc-1] *= 1./2
    elif opt == 2:
        mo_coeff[1][:,nocc-1] = mo_coeff[0][:,nocc]
    else:
        if nocc == 1:
            mo_coeff[1][:,:nocc] = 0
        else:
            mo_coeff[1][:,nocc-1] = 0
    return mo_coeff



def project_mo_nr2nr(mol1, mo1, mol2):
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

def init_guess_by_chkfile(mf, chkfile_name, projection=True):
    import dhf
    mol = mf.mol
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)

    fproj = lambda mo: mo
    fdm = lambda mo: numpy.dot(mo*mo_occ, mo.T.conj())
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
            fdm = lambda mo: dm
    elif isinstance(mf, hf.RHF) or 'RHF' in str(mf.__class__): # nr-RHF
        if scf_rec['mo_coeff'].ndim == 2:
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
        else:
            mo = scf_rec['mo_coeff'][0]
            mo_occ = scf_rec['mo_occ'][0]
        fproj = lambda mo: project_mo_nr2nr(chk_mol, mo, mol)
    elif isinstance(mf, hf.ROHF) or 'ROHF' in str(mf.__class__): # nr-ROHF
        if scf_rec['mo_coeff'].ndim == 2:
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            fproj = lambda mo: project_mo_nr2nr(chk_mol, mo, mol)
            fdm = lambda mo: numpy.array((numpy.dot(mo[:,mo_occ>0],
                                                    mo[:,mo_occ>0].T), \
                                          numpy.dot(mo[:,mo_occ==2],
                                                    mo[:,mo_occ==2].T)))
        else:
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            fproj = lambda mo: (project_mo_nr2nr(chk_mol, mo[0], mol),
                                project_mo_nr2nr(chk_mol, mo[1], mol))
            fdm = lambda mo: numpy.array((numpy.dot(mo[0][:,mo_occ[0]>0],
                                                    mo[0][:,mo_occ[0]>0].T), \
                                          numpy.dot(mo[1][:,mo_occ[1]>0],
                                                    mo[1][:,mo_occ[1]>0].T)))
    else: # nr-UHF
        if scf_rec['mo_coeff'].ndim == 2:
            mo = (scf_rec['mo_coeff'],)*2
            mo_occ = (scf_rec['mo_occ'],)*2
        else:
            mo = (scf_rec['mo_coeff'][0], scf_rec['mo_coeff'][1])
            mo_occ = (scf_rec['mo_occ'][0], scf_rec['mo_occ'][1])
        fproj = lambda mo: (project_mo_nr2nr(chk_mol, mo[0], mol),
                            project_mo_nr2nr(chk_mol, mo[1], mol))
        fdm = lambda mo: numpy.array((numpy.dot(mo[0]*mo_occ[0], mo[0].T), \
                                      numpy.dot(mo[1]*mo_occ[1], mo[1].T)))

    if projection:
        mo = fproj(mo)

    dm = fdm(mo)

    hf_energy = scf_rec['hf_energy'] + 0
    mo = mo_occ = chk_mol = scf_rec = None

    def finit(*args):
        return hf_energy, dm
    return finit


def init_guess_by_1e(mf):
    '''Initial guess from one electron system.'''
    mol = mf.mol
    log.info(mf, '\n')
    log.info(mf, 'Initial guess from one electron system.')
    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    mo_energy, mo_coeff = mf.eig(h1e, s1e)
    mo_occ = mf.set_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return lambda *args: (0, dm)


def init_guess_by_atom(mf):
    '''Initial guess from atom calculation.'''
    import atom_hf
    mol = mf.mol
    atm_scf = atom_hf.get_atm_nrhf_result(mol)
    nbf = mol.num_NR_function()
    dm = numpy.zeros((nbf, nbf))
    hf_energy = 0
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        if symb in atm_scf:
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        else:
            symb = mol.pure_symbol_of_atm(ia)
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        p1 = p0 + mo_e.__len__()
        dm[p0:p1,p0:p1] = numpy.dot(mo_c*mo_occ, mo_c.T.conj())
        hf_energy += e_hf
        p0 = p1

    log.info(mf, '\n')
    log.info(mf, 'Initial guess from superpostion of atomic densties.')
    for k,v in atm_scf.items():
        log.debug(mf, 'Atom %s, E = %.12g', k, v[0])
    log.debug(mf, 'total atomic SCF energy = %.12g', hf_energy)

    hf_energy -= mol.nuclear_repulsion()

    if isinstance(mf, dhf.UHF) or 'dhf' in str(mf.__class__):
        s0 = mol.intor_symmetric('cint1e_ovlp_sph')
        ua, ub = symm.cg.real2spinor_whole(mol)
        s = numpy.dot(ua.T.conj(), s0) + numpy.dot(ub.T.conj(), s0) # (*)
        proj = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp'), s)

        n2c = ua.shape[1]
        n4c = n2c * 2
        dm = numpy.zeros((n4c,n4c), dtype=complex)
        # *.5 because alpha and beta are summed in Eq. (*)
        dm_ll = reduce(numpy.dot, (proj, dm0*.5, proj.T.conj()))
        dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
        return lambda *args: (hf_energy, dm)
    elif isinstance(mf, hf.RHF) or 'RHF' in str(mf.__class__): # nr-RHF
        return lambda *args: (hf_energy, dm)
    elif isinstance(mf, hf.ROHF) or 'ROHF' in str(mf.__class__): # nr-ROHF
        return lambda *args: (hf_energy, (dm*.5,dm*.5))
    else: # UHF
        return lambda *args: (hf_energy, (dm*.5,dm*.5))
