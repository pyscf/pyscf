#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pyscf.lib
import pyscf.gto
from pyscf.lib import logger
import pyscf.symm
import pyscf.scf
import pyscf.ao2mo
from pyscf.mcscf import casci


class CASCI(casci.CASCI):
    def __init__(self, mf, ncas, nelecas, ncore=None):
        assert(mf.mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = pyscf.symm.param.CHARACTER_TABLE[mol.groupname][0][0]
        self.orbsym = []
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)

    def casci(self, mo_coeff=None, ci0=None, **cikwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci

        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.dump_flags()

        #irrep_name = self.mol.irrep_name
        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff)
        if not hasattr(self.fcisolver, 'orbsym') or \
           not self.fcisolver.orbsym:
            ncore = self.ncore
            nocc = self.ncore + self.ncas
            self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.e_tot, e_cas, self.ci = \
                casci.kernel(self, mo_coeff, ci0=ci0, verbose=self.verbose, **cikwargs)

        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci



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
    mc = CASCI(m, 4, 4)
    emc = mc.casci()[0]
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

    mc = CASCI(m, 4, (3,1))
    mc.fcisolver = pyscf.fci.direct_spin1
    emc = mc.casci()[0]
    print(emc - -75.439016172976)
