#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

"""
Relativistic Dirac-Hartree-Fock
"""


import numpy
from pyscf import lib
from pyscf import scf
import pyscf.scf._vhf
import pyscf.lib.logger as log
import hf

WITH_LLLL = 1
WITH_L2SL = 2 # the response of the large and small components on the large component density
WITH_LS2L = 3 # the response of the large component on the L+S density
WITH_NOSS = 4 # just exclude SSSS
WITH_SSSS = 5

class UHF(hf.RHF):
    '''Unrestricted Dirac-Hartree-Fock gradients'''
    def __init__(self, scf_method, restart=False):
        hf.RHF.__init__(self, scf_method, restart)
        if scf_method.with_ssss:
            self.vhf_level = WITH_SSSS
        else:
            self.vhf_level = WITH_NOSS

    @lib.omnimethod
    def get_hcore(self, mol):
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        s  = mol.intor('cint1e_ipovlp', dim3=3)
        t  = mol.intor('cint1e_ipkin', dim3=3)
        vn = mol.intor('cint1e_ipnuc', dim3=3)
        wn = mol.intor('cint1e_ipspnucsp', dim3=3)
        h1e = numpy.zeros((3,n4c,n4c), numpy.complex)
        h1e[:,:n2c,:n2c] = vn
        h1e[:,n2c:,:n2c] = t
        h1e[:,:n2c,n2c:] = t
        h1e[:,n2c:,n2c:] = wn * (.25/c**2) - t
        return h1e

    @lib.omnimethod
    def get_ovlp(self, mol):
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        s  = mol.intor('cint1e_ipovlp', dim3=3)
        t  = mol.intor('cint1e_ipkin', dim3=3)
        s1e = numpy.zeros((3,n4c,n4c), numpy.complex)
        s1e[:,:n2c,:n2c] = s
        s1e[:,n2c:,n2c:] = t * (.5/c**2)
        return s1e

    def _grad_rinv(self, mol, ia):
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed
        v = numpy.zeros((3,n4c,n4c), numpy.complex)
        mol.set_rinv_orig(mol.coord_of_atm(ia))
        vn = mol.charge_of_atm(ia) * mol.intor('cint1e_iprinv', dim3=3)
        wn = mol.charge_of_atm(ia) * mol.intor('cint1e_ipsprinvsp', dim3=3)
        v[:,:n2c,:n2c] = vn
        v[:,n2c:,n2c:] = wn * (.25/c**2)
        return v

    def get_coulomb_hf(self, mol, dm):
        '''Dirac-Hartree-Fock Coulomb repulsion'''
        if self.vhf_level == WITH_LLLL:
            log.info(mol, 'Compute Gradients: (LL|LL)')
            #vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_ll_o1, mol, dm)
            vj, vk = _call_vhf1_llll(mol, dm)
        elif self.vhf_level == WITH_LS2L:
            log.info(mol, 'Compute Gradients: (LL|LL) + (SS|dLL)')
            vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_ls2l_o1, mol, dm)
        elif self.vhf_level == WITH_L2SL:
            log.info(mol, 'Compute Gradients: (LL|LL) + (dSS|LL)')
            vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_l2sl_o1, mol, dm)
        elif self.vhf_level == WITH_NOSS:
            log.info(mol, 'Compute Gradients: (LL|LL) + (dSS|LL) + (SS|dLL)')
            vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_xss_o1, mol, dm)
        else:
            log.info(mol, 'Compute Gradients: (LL|LL) + (SS|LL) + (SS|SS)')
            #vj, vk = scf.hf.get_vj_vk(pycint.rkb_vhf_coul_grad_o1, mol, dm)
            vj, vk = _call_vhf1(mol, dm)
        return vj - vk

    def atom_of_aos(self, mol):
        #TODO: labels = mol.labels_of_spinor_GTO()
        ao_lab = []
        for ib in range(mol.nbas):
            n = mol.len_spinor_of_bas(ib) * mol.nctr_of_bas(ib)
            ao_lab.extend([mol.atom_of_bas(ib)] * n)
        return ao_lab

    def frac_atoms(self, mol, atm_id, mat):
        '''extract row band for each atom'''
        v = numpy.zeros_like(mat)
        # *2 for small components
        blk = numpy.array(self.atom_of_aos(mol)*2) == atm_id
        v[:,blk,:] = mat[:,blk,:]
        return v


def _call_vhf1_llll(mol, dm):
    c1 = .5/mol.light_speed
    n2c = dm.shape[0] / 2
    dmll = dm[:n2c,:n2c].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,:n2c,:n2c], vk[:,:n2c,:n2c] = \
            scf._vhf.rdirect_mapdm('cint2e_ip1', 'CVHFdot_rs2kl',
                                   ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_jk_s1il'),
                                   dmll, 3, mol._atm, mol._bas, mol._env)
    return vj, vk

def _call_vhf1(mol, dm):
    c1 = .5/mol.light_speed
    n2c = dm.shape[0] / 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,:n2c,:n2c], vk[:,:n2c,:n2c] = \
            scf._vhf.rdirect_mapdm('cint2e_ip1', 'CVHFdot_rs2kl',
                                   ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_jk_s1il'),
                                   dmll, 3, mol._atm, mol._bas, mol._env)
    vj[:,n2c:,n2c:], vk[:,n2c:,n2c:] = \
            scf._vhf.rdirect_mapdm('cint2e_ipspsp1spsp2', 'CVHFdot_rs2kl',
                                   ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_jk_s1il'),
                                   dmss, 3, mol._atm, mol._bas, mol._env) * c1**4
    vx = scf._vhf.rdirect_bindm('cint2e_ipspsp1', 'CVHFdot_rs2kl',
                                ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_jk_s1il'),
                                (dmll, dmsl), 3,
                                mol._atm, mol._bas, mol._env) * c1**2
    vj[:,n2c:,n2c:] += vx[0]
    vk[:,n2c:,:n2c] += vx[1]
    vx = scf._vhf.rdirect_bindm('cint2e_ip1spsp2', 'CVHFdot_rs2kl',
                                ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_jk_s1il'),
                                (dmss, dmls), 3,
                                mol._atm, mol._bas, mol._env) * c1**2
    vj[:,:n2c,:n2c] += vx[0]
    vk[:,:n2c,n2c:] += vx[1]
    return vj, vk



if __name__ == "__main__":
    from pyscf import gto
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
    g = UHF(method)
    print(g.grad())
#[[ 0   0                0             ]
# [ 0  -4.27565134e-03  -1.20060029e-02]
# [ 0   4.27565134e-03  -1.20060029e-02]]

