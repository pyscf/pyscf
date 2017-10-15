#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic unrestricted Hartree-Fock analytical nuclear gradients
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import rhf_grad


def grad_elec(grad_mf, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = grad_mf._scf
    mol = grad_mf.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(grad_mf.stdout, grad_mf.verbose)

    h1 = grad_mf.get_hcore(mol)
    s1 = grad_mf.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = grad_mf.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    f1 = h1 + vhf
    dme0 = grad_mf.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    atom_slices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = atom_slices[ia]
# h1, s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        vrinv = grad_mf._grad_rinv(mol, ia)
        de[k] += numpy.einsum('sxij,sij->x', f1[:,:,p0:p1], dm0[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vrinv, dm0_sf) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de

def get_veff(mf_grad, mol, dm):
    vj, vk = mf_grad.get_jk(mol, dm)
    return vj[0]+vj[1] - vk

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    return numpy.asarray((rhf_grad.make_rdm1e(mo_energy[0], mo_coeff[0], mo_occ[0]),
                          rhf_grad.make_rdm1e(mo_energy[1], mo_coeff[1], mo_occ[1])))


class Gradients(rhf_grad.Gradients):
    '''Non-relativistic unrestricted Hartree-Fock gradients
    '''
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self._scf.make_rdm1()
        return get_veff(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None: mo_occ = self._scf.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    mf = scf.UHF(mol)
    mf.scf()
    g = Gradients(mf)
    print(g.grad())

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    rhf = scf.UHF(mol)
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    rhf = scf.UHF(mol)
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0                3.27774948e-03]
# [ 0   4.31591309e-02  -1.63887474e-03]
# [ 0  -4.31591309e-02  -1.63887474e-03]]
