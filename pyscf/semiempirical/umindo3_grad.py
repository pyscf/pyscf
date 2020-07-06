#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf import gto
from pyscf.grad import uhf as uhf_grad
from pyscf.semiempirical import mopac_param
from pyscf.semiempirical import mindo3
from pyscf.semiempirical import rmindo3_grad


class Gradients(uhf_grad.Gradients):
    get_hcore = None
    hcore_generator = rmindo3_grad.hcore_generator

    def get_ovlp(self, mol=None):
        nao = self.base._mindo_mol.nao
        return numpy.zeros((3,nao,nao))

    def get_jk(self, mol=None, dm=None, hermi=0):
        if dm is None: dm = self.base.make_rdm1()
        vj, vk = rmindo3_grad.get_jk(self.base._mindo_mol, dm)
        return vj, vk

    def grad_nuc(self, mol=None, atmlst=None):
        mol = self.base._mindo_mol
        return rmindo3_grad.grad_nuc(mol, atmlst)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        with lib.temporary_env(self, mol=self.base._mindo_mol):
            return uhf_grad.grad_elec(self, mo_energy, mo_coeff, mo_occ,
                                      atmlst) * lib.param.BOHR

Grad = Gradients


if __name__ == '__main__':
    from pyscf.data.nist import HARTREE2EV
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.spin = 2
    mol.verbose = 0
    mol.build()
    mfs = mindo3.UMINDO3(mol).set(conv_tol=1e-8).as_scanner()
    mfs(mol)
    print(mfs.e_tot - -336.25080977434175/HARTREE2EV)

    mol1 = mol.copy()
    mol1.set_geom_([['O' , (0. , 0.     , 0.0001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)]])
    mol2 = mol.copy()
    mindo_mol1 = mindo3._make_mindo_mol(mol1)
    mol2.set_geom_([['O' , (0. , 0.     ,-0.0001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)]])
    mindo_mol2 = mindo3._make_mindo_mol(mol2)

    g1 = mfs.nuc_grad_method().kernel()
    e1 = mfs(mol1)
    e2 = mfs(mol2)
    print(abs((e1-e2)/0.0002*lib.param.BOHR - g1[0,2]))
