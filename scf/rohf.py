#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restricted Open-shell Hartree-Fock
'''

from functools import reduce
import numpy
import pyscf.gto
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import hf
from pyscf.scf import uhf


def init_guess_by_minao(mol):
    dm = hf.init_guess_by_minao(mol)
    return numpy.array((dm*.5, dm*.5))

def init_guess_by_atom(mol):
    dm = hf.init_guess_by_atom(mol)
    return numpy.array((dm*.5, dm*.5))

def init_guess_by_chkfile(mol, chk, project=True):
    from pyscf.scf import addons
    if isinstance(chk, pyscf.gto.Mole):
        raise RuntimeError('''
You see this error message because of the API updates.
The first argument is chk file name.''')
    chk_mol, scf_rec = chkfile.load_scf(chk)

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_mol, mo, mol)
        else:
            return mo
    if scf_rec['mo_coeff'].ndim == 2:
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        if numpy.iscomplexobj(mo):
            raise RuntimeError('TODO: project DHF orbital to ROHF orbital')
        dm = make_rdm1(fproj(mo), mo_occ)
    else:  # UHF
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = uhf.make_rdm1((fproj(mo[0]),fproj(mo[1])), mo_occ)
    return dm

def get_fock_(mf, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
              diis_start_cycle=None, level_shift_factor=None,
              damp_factor=None):
    '''Roothaan's effective fock.
    Ref. http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html

    ======== ======== ====== =========
    space     closed   open   virtual
    ======== ======== ====== =========
    closed      Fc      Fb     Fc
    open        Fb      Fc     Fa
    virtual     Fc      Fa     Fc
    ======== ======== ====== =========

    where Fc stands for the Fa + Fb

    Returns:
        (fock_eff, fock_alpha, fock_beta)
        Roothaan effective Fock matrix, with UHF alpha and beta Fock matrices.
        Attach alpha and beta fock, because Roothaan effective Fock cannot
        provide correct orbital energies.
    '''
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift_factor
    if damp_factor is None:
        damp_factor = mf.damp_factor
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
# Fc = (Fa+Fb)/2
    mf._focka_ao = h1e + vhf[0]
    fockb_ao = h1e + vhf[1]
    fc = (mf._focka_ao + fockb_ao) * .5
# Projector for core, open-shell, and virtual
    nao = s1e.shape[0]
    pc = numpy.dot(dm[1], s1e)
    po = numpy.dot(dm[0]-dm[1], s1e)
    pv = numpy.eye(nao) - numpy.dot(dm[0], s1e)
    f  = reduce(numpy.dot, (pc.T, fc, pc)) * .5
    f += reduce(numpy.dot, (po.T, fc, po)) * .5
    f += reduce(numpy.dot, (pv.T, fc, pv)) * .5
    f += reduce(numpy.dot, (po.T, fockb_ao, pc))
    f += reduce(numpy.dot, (po.T, mf._focka_ao, pv))
    f += reduce(numpy.dot, (pv.T, fc, pc))
    f = f + f.T

    if 0 <= cycle < diis_start_cycle-1:
        f = hf.damping(s1e, dm[0], f, damp_factor)
    if adiis and cycle >= diis_start_cycle:
        #f = adiis.update(s1e, dmsf*.5, f)
        f = adiis.update(s1e, dm[0], f)
    #f = level_shift(s1e, dmsf*.5, f, level_shift_factor)
    f = hf.level_shift(s1e, dm[0], f, level_shift_factor)
# attach alpha and beta fock, because Roothaan effective Fock cannot provide
# correct orbital energy.  To define orbital energy in mf.eig, we use alpha
# fock and beta fock.
# TODO, check other treatment  J. Chem. Phys. 133, 141102
    return f

def get_grad(mo_coeff, mo_occ, fock=None):
    occidxa = numpy.where(mo_occ>0)[0]
    occidxb = numpy.where(mo_occ==2)[0]
    viridxa = numpy.where(mo_occ==0)[0]
    viridxb = numpy.where(mo_occ<2)[0]
    mask = hf.uniq_var_indices(mo_occ)

    focka = reduce(numpy.dot, (mo_coeff.T, fock[0], mo_coeff))
    fockb = reduce(numpy.dot, (mo_coeff.T, fock[1], mo_coeff))

    g = numpy.zeros_like(focka)
    g[viridxa[:,None],occidxa]  = focka[viridxa[:,None],occidxa]
    g[viridxb[:,None],occidxb] += fockb[viridxb[:,None],occidxb]
    return g[mask]

def make_rdm1(mo_coeff, mo_occ):
    mo_a = mo_coeff[:,mo_occ>0]
    mo_b = mo_coeff[:,mo_occ==2]
    dm_a = numpy.dot(mo_a, mo_a.T)
    dm_b = numpy.dot(mo_b, mo_b.T)
    return numpy.array((dm_a, dm_b))

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    elif isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    ee, ecoul = uhf.energy_elec(mf, dm, h1e, vhf)
    logger.debug(mf, 'Ecoul = %.15g', ecoul)
    return ee, ecoul

# pass in a set of density matrix in dm as (alpha,alpha,...,beta,beta,...)
def get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None):
    return uhf.get_veff(mol, dm, dm_last, vhf_last, hermi, vhfopt)


# use UHF init_guess, get_veff, diis, and intermediates such as fock, vhf, dm
# keep mo_energy, mo_coeff, mo_occ as RHF structure

class ROHF(hf.RHF):
    __doc__ = hf.SCF.__doc__

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        n_a = (mol.nelectron + mol.spin) // 2
        self.nelec = (n_a, mol.nelectron - n_a)
        self._focka_ao = None
        self._keys = self._keys.union(['nelec'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        if hasattr(self, 'nelectron_alpha'):
            logger.warn(self, 'Note the API updates: attribute nelectron_alpha was replaced by attribute nelec')
            #raise RuntimeError('API updates')
            self.nelec = (self.nelectron_alpha,
                          self.mol.nelectron-self.nelectron_alpha)
            delattr(self, 'nelectron_alpha')
        logger.info(self, 'num. doubly occ = %d  num. singly occ = %d',
                    self.nelec[1], self.nelec[0]-self.nelec[1])

    def init_guess_by_minao(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from superpostion of atomic densties.')
        return init_guess_by_atom(mol)

    def init_guess_by_1e(self, mol=None):
        if mol is None:
            mol = self.mol
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = hf.RHF.eig(self, h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def init_guess_by_chkfile(self, chk=None, project=True):
        if chk is None: chk = self.chkfile
        return init_guess_by_chkfile(self.mol, chk, project=project)

    def eig(self, h, s):
        '''Solve the generalized eigenvalue problem for Roothan effective Fock
        matrix.  Note Roothaan effective Fock do not provide correct orbital
        energies.  We use spectrum of alpha fock and beta fock to sort the
        orbitals.  The energies of doubly occupied orbitals are the eigenvalue
        of Roothaan Fock matrix.  But the energies of singly occupied orbitals
        and virtual orbitals are the expection value of alpha-Fock matrix.

        Args:
            h : a list of 3 ndarrays
                (fock_eff, fock_alpha, fock_beta), which is provided by :func:`ROHF.get_fock_`
        '''
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        ncore = self.nelec[1]
        mo_energy, mo_coeff = hf.SCF.eig(self, h, s)
        mopen = mo_coeff[:,ncore:]
        ea = numpy.einsum('ik,ik->k', mopen, self._focka_ao.dot(mopen))
        idx = ea.argsort()
        mo_energy[ncore:] = ea[idx]
        mo_coeff[:,ncore:] = mopen[:,idx]
        return mo_energy, mo_coeff

    def get_fock_(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                  diis_start_cycle=None, level_shift_factor=None,
                  damp_factor=None):
        return get_fock_(self, h1e, s1e, vhf, dm, cycle, adiis,
                         diis_start_cycle, level_shift_factor, damp_factor)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mo_occ = numpy.zeros_like(mo_energy)
        ncore = self.nelec[1]
        nopen = self.nelec[0] - ncore
        nocc = ncore + nopen
        mo_occ[:ncore] = 2
        mo_occ[ncore:nocc] = 1
        if nocc < len(mo_energy):
            logger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self.mol, '!! HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'HOMO = %.12g  no LUMO', mo_energy[nocc-1])
        if nopen > 0:
            for i in range(ncore, nocc):
                logger.debug(self, 'singly occupied orbital energy = %.12g',
                             mo_energy[i])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        return energy_elec(self, dm, h1e, vhf)

    # pass in a set of density matrix in dm as (alpha,alpha,...,beta,beta,...)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Unrestricted Hartree-Fock potential matrix of alpha and beta spins,
        for the given density matrix.  See :func:`scf.uhf.get_veff`.
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5, dm*.5))
        nset = len(dm) // 2
        if (self._eri is not None or not self.direct_scf or
            mol.incore_anyway or self._is_mem_enough()):
            dm = numpy.array(dm, copy=False)
            dm = numpy.vstack((dm[nset:], dm[:nset]-dm[nset:]))
            vj, vk = self.get_jk(mol, dm, hermi)
            vj = numpy.vstack((vj[:nset]+vj[nset:], vj[:nset]))
            vk = numpy.vstack((vk[:nset]+vk[nset:], vk[:nset]))
            vhf = uhf._makevhf(vj, vk, nset)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            ddm = numpy.vstack((ddm[nset:],             # closed shell
                                ddm[:nset]-ddm[nset:])) # open shell
            vj, vk = self.get_jk(mol, ddm, hermi)
            vj = numpy.vstack((vj[:nset]+vj[nset:], vj[:nset]))
            vk = numpy.vstack((vk[:nset]+vk[nset:], vk[:nset]))
            vhf = uhf._makevhf(vj, vk, nset) \
                + numpy.array(vhf_last, copy=False)
        return vhf


