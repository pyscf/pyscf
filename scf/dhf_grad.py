#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

"""
Relativistic Dirac-Hartree-Fock
"""

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
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    n2c = dm0.shape[0] // 2

    t0 = (time.clock(), time.time())
    log.debug('Compute Gradients of NR Hartree-Fock Coulomb repulsion')
    vhf = grad_mf.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    f1 = h1 + vhf
    dme0 = grad_mf.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = grad_mf.aorange_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        vrinv = grad_mf._grad_rinv(mol, ia)
        de[k] +=(numpy.einsum('xij,ji->x', f1[:,p0:p1], dm0[:,p0:p1])
               + numpy.einsum('xji,ji->x', f1[:,p0:p1].conj(), dm0[p0:p1])).real
        de[k] +=(numpy.einsum('xij,ji->x', vrinv, dm0)
               + numpy.einsum('xji,ji->x', vrinv.conj(), dm0)).real
        de[k] -=(numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[:,p0:p1])
               + numpy.einsum('xji,ji->x', s1[:,p0:p1].conj(), dme0[p0:p1])).real
# small components
        p0 += n2c
        p1 += n2c
        de[k] +=(numpy.einsum('xij,ji->x', f1[:,p0:p1], dm0[:,p0:p1])
               + numpy.einsum('xji,ji->x', f1[:,p0:p1].conj(), dm0[p0:p1])).real
        de[k] -=(numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[:,p0:p1])
               + numpy.einsum('xji,ji->x', s1[:,p0:p1].conj(), dme0[p0:p1])).real
    log.debug('gradients of electronic part')
    log.debug(str(de))
    return de

def grad_nuc(mol, atmlst=None):
    return rhf_grad.grad_nuc(mol, atmlst)

def get_hcore(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    t  = mol.intor('int1e_ipkin_spinor', comp=3)
    vn = mol.intor('int1e_ipnuc_spinor', comp=3)
    wn = mol.intor('int1e_ipspnucsp_spinor', comp=3)
    h1e = numpy.zeros((3,n4c,n4c), numpy.complex)
    h1e[:,:n2c,:n2c] = vn
    h1e[:,n2c:,:n2c] = t
    h1e[:,:n2c,n2c:] = t
    h1e[:,n2c:,n2c:] = wn * (.25/c**2) - t
    return -h1e

def get_ovlp(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    s  = mol.intor('int1e_ipovlp_spinor', comp=3)
    t  = mol.intor('int1e_ipkin_spinor', comp=3)
    s1e = numpy.zeros((3,n4c,n4c), numpy.complex)
    s1e[:,:n2c,:n2c] = s
    s1e[:,n2c:,n2c:] = t * (.5/c**2)
    return -s1e

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    return rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

# 2C AO spinor range
def aorange_by_atom(mol):
    aorange = []
    p0 = p1 = 0
    b0 = b1 = 0
    ia0 = 0
    for ib in range(mol.nbas):
        if ia0 != mol.bas_atom(ib):
            aorange.append((b0, ib, p0, p1))
            ia0 = mol.bas_atom(ib)
            p0 = p1
            b0 = ib
        p1 += mol.bas_len_spinor(ib) * mol.bas_nctr(ib)
    aorange.append((b0, mol.nbas, p0, p1))
    return aorange

def get_veff(mol, dm, level='SSSS'):
    return get_coulomb_hf(mol, dm, level)
def get_coulomb_hf(mol, dm, level='SSSS'):
    '''Dirac-Hartree-Fock Coulomb repulsion'''
    if level.upper() == 'LLLL':
        logger.info(mol, 'Compute Gradients: (LL|LL)')
        vj, vk = _call_vhf1_llll(mol, dm)
#L2SL the response of the large and small components on the large component density
#LS2L the response of the large component on the L+S density
#NOSS just exclude SSSS
#TODO    elif level.upper() == 'LS2L':
#TODO        logger.info(mol, 'Compute Gradients: (LL|LL) + (SS|dLL)')
#TODO        vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_ls2l_o1, mol, dm)
#TODO    elif level.upper() == 'L2SL':
#TODO        logger.info(mol, 'Compute Gradients: (LL|LL) + (dSS|LL)')
#TODO        vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_l2sl_o1, mol, dm)
#TODO    elif level.upper() == 'NOSS':
#TODO        logger.info(mol, 'Compute Gradients: (LL|LL) + (dSS|LL) + (SS|dLL)')
#TODO        vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_xss_o1, mol, dm)
    else:
        logger.info(mol, 'Compute Gradients: (LL|LL) + (SS|LL) + (SS|SS)')
        vj, vk = _call_vhf1(mol, dm)
    return -(vj - vk)


class Gradients(rhf_grad.Gradients):
    '''Unrestricted Dirac-Hartree-Fock gradients'''
    def __init__(self, scf_method):
        rhf_grad.Gradients.__init__(self, scf_method)
        if scf_method.with_ssss:
            self.level = 'SSSS'
        else:
            #self.level = 'NOSS'
            self.level = 'LLLL'

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)

    def _grad_rinv(self, mol, ia):
        n2c = mol.nao_2c()
        n4c = n2c * 2
        c = lib.param.LIGHT_SPEED
        v = numpy.zeros((3,n4c,n4c), numpy.complex)
        mol.set_rinv_origin(mol.atom_coord(ia))
        vn = mol.atom_charge(ia) * mol.intor('int1e_iprinv_spinor', comp=3)
        wn = mol.atom_charge(ia) * mol.intor('int1e_ipsprinvsp_spinor', comp=3)
        v[:,:n2c,:n2c] = vn
        v[:,n2c:,n2c:] = wn * (.25/c**2)
        return -v

    def get_veff(self, mol, dm):
        return get_coulomb_hf(mol, dm, level=self.level)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None,
                  atmlst=None):
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None: mo_occ = self._scf.mo_occ
        return grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst)

    def aorange_by_atom(self):
        return aorange_by_atom(self.mol)



def _call_vhf1_llll(mol, dm):
    n2c = dm.shape[0] // 2
    dmll = dm[:n2c,:n2c].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,:n2c,:n2c], vk[:,:n2c,:n2c] = \
            _vhf.rdirect_mapdm('int2e_ip1_spinor', 's2kl',
                               ('lk->s1ij', 'jk->s1il'), dmll, 3,
                               mol._atm, mol._bas, mol._env)
    return vj, vk

def _call_vhf1(mol, dm):
    c1 = .5 / lib.param.LIGHT_SPEED
    n2c = dm.shape[0] // 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,:n2c,:n2c], vk[:,:n2c,:n2c] = \
            _vhf.rdirect_mapdm('int2e_ip1_spinor', 's2kl',
                               ('lk->s1ij', 'jk->s1il'), dmll, 3,
                               mol._atm, mol._bas, mol._env)
    vj[:,n2c:,n2c:], vk[:,n2c:,n2c:] = \
            _vhf.rdirect_mapdm('int2e_ipspsp1spsp2_spinor', 's2kl',
                               ('lk->s1ij', 'jk->s1il'), dmss, 3,
                               mol._atm, mol._bas, mol._env) * c1**4
    vx = _vhf.rdirect_bindm('int2e_ipspsp1_spinor', 's2kl',
                            ('lk->s1ij', 'jk->s1il'), (dmll, dmsl), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,n2c:,n2c:] += vx[0]
    vk[:,n2c:,:n2c] += vx[1]
    vx = _vhf.rdirect_bindm('int2e_ip1spsp2_spinor', 's2kl',
                            ('lk->s1ij', 'jk->s1il'), (dmss, dmls), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,:n2c,:n2c] += vx[0]
    vk[:,:n2c,n2c:] += vx[1]
    return vj, vk


if __name__ == "__main__":
    from pyscf import gto
    from pyscf import scf

    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#"out_h2o"
    h2o.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = {"H": '6-31g',
                 "O": '6-31g',}
    h2o.build()
    method = scf.dhf.UHF(h2o)
    print(method.scf())
    g = Gradients(method)
    print(g.grad())
#[[ 0   0               -2.40120097e-02]
# [ 0   4.27565134e-03   1.20060029e-02]
# [ 0  -4.27565134e-03   1.20060029e-02]]

