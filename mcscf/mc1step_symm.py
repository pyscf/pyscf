#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import copy
import tempfile
import numpy
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.scf
import casci
import aug_hessian
import mc1step
import mc2step


class CASSCF(mc1step.CASSCF):
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        assert(mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = pyscf.symm.param.CHARACTER_TABLE[mmol.groupname][0][0]
        self.orbsym = []
        mc1step.CASSCF.__init__(self, mol, mf, ncas, nelecas, ncore)

    def mc1step(self, mo=None, ci0=None, macro=None, micro=None):
        if mo is None:
            mo = self.mo_coeff
        else:
            self.mo_coeff = mo
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.mol.check_sanity(self)

        self.dump_flags()

        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff)

        if not hasattr(self.fcisolver, 'orbsym') or \
           not self.fcisolver.orbsym:
            ncore = self.ncore
            nocc = self.ncore + self.ncas
            self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc1step.kernel(self, mo, \
                               tol=self.conv_threshold, macro=macro, micro=micro, \
                               ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo=None, ci0=None, macro=None, micro=None, restart=False):
        if mo is None:
            mo = self.mo_coeff
        else:
            self.mo_coeff = mo
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.mol.check_sanity(self)

        self.dump_flags()

        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff)
        if not hasattr(self.fcisolver, 'orbsym') or \
           not self.fcisolver.orbsym:
            ncore = self.ncore
            nocc = self.ncore + self.ncas
            self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc2step.kernel(self, mo, \
                               tol=self.conv_threshold, macro=macro, micro=micro, \
                               ci0=ci0, verbose=self.verbose, restart=restart)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def gen_g_hop(self, mo, casdm1, casdm2, eris):
        g_orb, h_op, h_diag = mc1step.gen_g_hop(self, mo, casdm1, casdm2, eris)
        g_orb = _symmetrize(self.unpack_uniq_var(g_orb), self.orbsym,
                            self.mol.groupname)
        h_diag = _symmetrize(self.unpack_uniq_var(h_diag), self.orbsym,
                             self.mol.groupname)
        def sym_h_op(x):
            hx = h_op(x)
            hx = _symmetrize(self.unpack_uniq_var(hx), self.orbsym,
                             self.mol.groupname)
            return self.pack_uniq_var(hx)
        return self.pack_uniq_var(g_orb), sym_h_op, \
               self.pack_uniq_var(h_diag)

    def rotate_orb(self, mo, fcivec, e_ci, eris, dx=0):
        u, dx, g_orb, jkcnt = \
                mc1step.rotate_orb_ah(self, mo, fcivec, e_ci, eris, dx,
                                      self.verbose)
        u = _symmetrize(u, self.orbsym, self.mol.groupname)
        dx = _symmetrize(self.unpack_uniq_var(dx), self.orbsym,
                         self.mol.groupname)
        return u, self.pack_uniq_var(dx), g_orb, jkcnt

    def get_hcore(self, mol=None):
        h = self.mol.intor_symmetric('cint1e_kin_sph') \
          + self.mol.intor_symmetric('cint1e_nuc_sph')
        return h

    def get_veff(self, dm):
        return pyscf.scf.hf.RHF.get_veff(self._scf, self.mol, dm)

def _symmetrize(mat, orbsym, groupname, wfnsym=0):
    irreptab = pyscf.symm.param.IRREP_ID_TABLE[groupname]
    if isinstance(wfnsym, str):
        wfnsym = irreptab[wfnsym]

    mat1 = numpy.zeros_like(mat)
    for i0 in set(orbsym):
        irallow = wfnsym ^ i0
        lst = [j for j,i in enumerate(orbsym) if i == irallow]
        for j in lst:
            mat1[j,lst] = mat[j,lst]
    return mat1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
    import addons

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.symmetry = 1
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASSCF(mol, m, 6, 4)
    mc.fcisolver = pyscf.fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0] + mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(mol, m, 6, (3,1))
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    print(emc - -84.9038216713284)
