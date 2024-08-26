#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Relativistic Dirac-Hartree-Fock
"""


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.grad import rhf as rhf_grad


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    n2c = dm0.shape[0] // 2

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Compute Gradients of NR Hartree-Fock Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_2c_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm0).real
# large components
        de[k] +=(numpy.einsum('xij,ji->x', vhf[:,p0:p1], dm0[:,p0:p1]) +
                 numpy.einsum('xji,ji->x', vhf[:,p0:p1].conj(), dm0[p0:p1])).real
        de[k] -=(numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[:,p0:p1]) +
                 numpy.einsum('xji,ji->x', s1[:,p0:p1].conj(), dme0[p0:p1])).real
# small components
        p0 += n2c
        p1 += n2c
        de[k] +=(numpy.einsum('xij,ji->x', vhf[:,p0:p1], dm0[:,p0:p1]) +
                 numpy.einsum('xji,ji->x', vhf[:,p0:p1].conj(), dm0[p0:p1])).real
        de[k] -=(numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[:,p0:p1]) +
                 numpy.einsum('xji,ji->x', s1[:,p0:p1].conj(), dme0[p0:p1])).real
    log.debug('gradients of electronic part')
    log.debug(str(de))
    return de

grad_nuc = rhf_grad.grad_nuc

def get_hcore(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    t  = mol.intor('int1e_ipkin_spinor', comp=3)
    vn = mol.intor('int1e_ipnuc_spinor', comp=3)
    wn = mol.intor('int1e_ipspnucsp_spinor', comp=3)
    h1e = numpy.zeros((3,n4c,n4c), numpy.complex128)
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
    s1e = numpy.zeros((3,n4c,n4c), numpy.complex128)
    s1e[:,:n2c,:n2c] = s
    s1e[:,n2c:,n2c:] = t * (.5/c**2)
    return -s1e

make_rdm1e = rhf_grad.make_rdm1e

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
get_veff = get_coulomb_hf


class GradientsBase(rhf_grad.GradientsBase):
    '''
    Basic nuclear gradient functions for 4C relativistic methods
    '''
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def hcore_generator(self, mol):
        aoslices = mol.aoslice_2c_by_atom()
        h1 = self.get_hcore(mol)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        c = lib.param.LIGHT_SPEED
        def hcore_deriv(atm_id):
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_at_nucleus(atm_id):
                z = -mol.atom_charge(atm_id)
                vn = z * mol.intor('int1e_iprinv_spinor', comp=3)
                wn = z * mol.intor('int1e_ipsprinvsp_spinor', comp=3)

            v = numpy.zeros((3,n4c,n4c), numpy.complex128)
            v[:,:n2c,:n2c] = vn
            v[:,n2c:,n2c:] = wn * (.25/c**2)
            v[:,p0:p1]         += h1[:,p0:p1]
            v[:,n2c+p0:n2c+p1] += h1[:,n2c+p0:n2c+p1]
            return v + v.conj().transpose(0,2,1)
        return hcore_deriv

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)


class Gradients(GradientsBase):
    '''Unrestricted Dirac-Hartree-Fock gradients'''

    _keys = {'level'}

    def __init__(self, scf_method):
        GradientsBase.__init__(self, scf_method)
        if scf_method.with_ssss:
            self.level = 'SSSS'
        else:
            #self.level = 'NOSS'
            #self.level = 'LLLL'
            raise NotImplementedError

    def get_veff(self, mol, dm):
        return get_coulomb_hf(mol, dm, level=self.level)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        return 0

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

    as_scanner = rhf_grad.as_scanner

    to_gpu = lib.to_gpu

Grad = Gradients

from pyscf import scf
scf.dhf.UHF.Gradients = lib.class_as_method(Gradients)


def _call_vhf1_llll(mol, dm):
    n2c = dm.shape[0] // 2
    dmll = dm[:n2c,:n2c].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex128)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex128)
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
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex128)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex128)
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
    from pyscf import lib

    with lib.light_speed(30):
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
        method = scf.dhf.UHF(h2o).run()
        g = method.Gradients().kernel()
        print(g)

        ms = method.as_scanner()
        h2o.set_geom_([["O" , (0. , 0.     ,-0.001)],
                       [1   , (0. , -0.757 , 0.587)],
                       [1   , (0. , 0.757  , 0.587)]], unit='Ang')
        e1 = ms(h2o)
        h2o.set_geom_([["O" , (0. , 0.     , 0.001)],
                       [1   , (0. , -0.757 , 0.587)],
                       [1   , (0. , 0.757  , 0.587)]], unit='Ang')
        e2 = ms(h2o)
        print(g[0,2], (e2-e1)/0.002*lib.param.BOHR)
