#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import time
import numpy
import h5py
import pyscf.lib
import pyscf.symm
import pyscf.scf
import pyscf.ao2mo
import casci


class CASCI(casci.CASCI):
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        assert(mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = pyscf.symm.param.CHARACTER_TABLE[mmol.pgname][0][0]
        casci.CASCI.__init__(self, mol, mf, ncas, nelecas, ncore)

    def casci(self, mo=None, ci0=None):
        if mo is None:
            mo = self.mo_coeff
        else:
            self.mo_coeff = mo
        if ci0 is None:
            ci0 = self.ci

        self.mol.check_sanity(self)

        self.dump_flags()

        #irrep_name = self.mol.irrep_name
        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff)
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.e_tot, e_cas, self.ci = \
                casci.kernel(self, mo, ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci

    def get_hcore(self, mol=None):
        h = self.mol.intor_symmetric('cint1e_kin_sph') \
          + self.mol.intor_symmetric('cint1e_nuc_sph')
        return h

    def get_veff(self, dm):
        return pyscf.scf.hf.RHF.get_veff(self._scf, self.mol, dm)



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.symmetry = 1
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASCI(mol, m, 4, 4)
    emc = mc.casci()[0] + mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

    mc = CASCI(mol, m, 4, (3,1))
    mc.fcisolver = pyscf.fci.direct_spin1
    emc = mc.casci()[0] + mol.nuclear_repulsion()
    print(emc - -75.439016172976)
