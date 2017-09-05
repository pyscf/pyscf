#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf.lib import logger
from pyscf import scf
from pyscf import symm
from pyscf import fci
from pyscf.mcscf import casci
from pyscf.mcscf import addons


class CASCI(casci.CASCI):
    def __init__(self, mf, ncas, nelecas, ncore=None):
        assert(mf.mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = symm.param.CHARACTER_TABLE[mol.groupname][0][0]
        self.orbsym = []
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        #self.fcisolver = fci.solver(mf.mol, self.nelecas[0]==self.nelecas[1], True)
        self.fcisolver = fci.solver(mf.mol, singlet=False, symm=True)

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

    def sort_mo_by_irrep(self, cas_irrep_nocc,
                         cas_irrep_ncore=None, mo_coeff=None, s=None):
        '''Select active space based on symmetry information.
        See also :func:`pyscf.mcscf.addons.sort_mo_by_irrep`
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo_by_irrep(self, mo_coeff, cas_irrep_nocc,
                                       cas_irrep_ncore, s)

def eig(mat, orbsym):
    orbsym = numpy.asarray(orbsym)
    norb = mat.shape[0]
    e = numpy.zeros(norb)
    c = numpy.zeros((norb,norb))
    for ir in set(orbsym):
        lst = numpy.where(orbsym == ir)[0]
        if len(lst) > 0:
            w, v = scf.hf.eig(mat[lst[:,None],lst], None)
            e[lst] = w
            c[lst[:,None],lst] = v
    return e, c

def label_symmetry_(mc, mo_coeff):
    #irrep_name = mc.mol.irrep_name
    irrep_name = mc.mol.irrep_id
    s = mc._scf.get_ovlp()
    try:
        mc.orbsym = scf.hf_symm.get_orbsym(mc._scf.mol, mo_coeff, s)
    except ValueError:
        logger.warn(mc, 'mc1step_symm symmetrizes input orbitals')
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        mo_cor = symm.symmetrize_space(mc.mol, mo_coeff[:,    :ncore], s=s, check=False)
        mo_act = symm.symmetrize_space(mc.mol, mo_coeff[:,ncore:nocc], s=s, check=False)
        mo_vir = symm.symmetrize_space(mc.mol, mo_coeff[:,nocc:     ], s=s, check=False)
        mo_coeff = numpy.hstack((mo_cor,mo_act,mo_vir))
        mc.orbsym = symm.label_orb_symm(mc.mol, irrep_name,
                                        mc.mol.symm_orb, mo_coeff, s=s)

    if (not hasattr(mc.fcisolver, 'orbsym') or mc.fcisolver.orbsym is None
        or (hasattr(mc.fcisolver.orbsym, '__len__')
            and len(mc.fcisolver.orbsym) == 0)):
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        mc.fcisolver.orbsym = mc.orbsym[ncore:nocc]
    logger.debug(mc, 'Active space irreps %s', str(mc.fcisolver.orbsym))



if __name__ == '__main__':
    from pyscf import gto
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
    mc.fcisolver = fci.direct_spin1
    emc = mc.casci()[0]
    print(emc - -75.439016172976)
