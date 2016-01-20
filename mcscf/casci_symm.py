#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
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

    def kernel(self, mo_coeff=None, ci0=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci

        label_symmetry_(self, self.mo_coeff)
        return casci.CASCI.kernel(self, mo_coeff, ci0)

    def _eig(self, mat, b0, b1):
        return eig(mat, numpy.array(self.orbsym[b0:b1]))

    def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                    casdm1=None, verbose=None):
        self.mo_coeff, self.ci, occ = cas_natorb(self, mo_coeff, ci, eris,
                                                 sort, casdm1, verbose)
        if sort:
            casci_symm.label_symmetry_(self, self.mo_coeff)
        return self.mo_coeff, self.ci, occ

    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None):
        self.mo_coeff, ci, self.mo_energy = \
                self.canonicalize(mo_coeff, ci, eris,
                                  sort, cas_natorb, casdm1, verbose)
        if sort:
            casci_symm.label_symmetry_(self, self.mo_coeff)
        if cas_natorb:  # When active space is changed, the ci solution needs to be updated
            self.ci = ci
        return self.mo_coeff, ci, self.mo_energy

def eig(mat, orbsym):
    orbsym = numpy.asarray(orbsym)
    norb = mat.shape[0]
    e = numpy.zeros(norb)
    c = numpy.zeros((norb,norb))
    for i0 in set(orbsym):
        lst = numpy.where(orbsym == i0)[0]
        if len(lst) > 0:
            w, v = pyscf.scf.hf.eig(mat[lst[:,None],lst], None)
            e[lst] = w
            c[lst[:,None],lst] = v
    return e, c

def label_symmetry_(mc, mo_coeff):
    #irrep_name = mc.mol.irrep_name
    irrep_name = mc.mol.irrep_id
    s = mc._scf.get_ovlp()
    try:
        mc.orbsym = pyscf.symm.label_orb_symm(mc.mol, irrep_name,
                                              mc.mol.symm_orb, mo_coeff, s=s)
    except ValueError:
        logger.warn(mc, 'mc1step_symm symmetrizes input orbitals')
        mo_coeff = pyscf.symm.symmetrize_orb(mc.mol, mo_coeff, s=s)
        diag = numpy.einsum('ki,ki->i', mo_coeff, numpy.dot(s, mo_coeff))
        mo_coeff = numpy.einsum('ki,i->ki', mo_coeff, 1/numpy.sqrt(diag))
        mc.orbsym = pyscf.symm.label_orb_symm(mc.mol, irrep_name,
                                              mc.mol.symm_orb, mo_coeff, s=s)

    if not hasattr(mc.fcisolver, 'orbsym') or not mc.fcisolver.orbsym:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        mc.fcisolver.orbsym = mc.orbsym[ncore:nocc]
    logger.debug(mc, 'Active space irreps %s', str(mc.fcisolver.orbsym))



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
