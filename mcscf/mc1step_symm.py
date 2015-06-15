#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import pyscf.lib.logger as logger
import pyscf.gto
import pyscf.symm
from pyscf.mcscf import mc1step
from pyscf.mcscf import mc2step
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.tools.mo_mapping import mo_1to1map


class CASSCF(mc1step.CASSCF):
    def __init__(self, mf, ncas, nelecas, ncore=None, frozen=[]):
        assert(mf.mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = pyscf.symm.param.CHARACTER_TABLE[mol.groupname][0][0]
        self.orbsym = []
        mc1step.CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)

    def mc1step(self, mo_coeff=None, ci0=None, macro=None, micro=None,
                callback=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None: macro = self.max_cycle_macro
        if micro is None: micro = self.max_cycle_micro
        if callback is None: callback = self.callback

        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.dump_flags()

        #irrep_name = self.mol.irrep_name
        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff,
                                                s=self._scf.get_ovlp())

        if not hasattr(self.fcisolver, 'orbsym') or \
           not self.fcisolver.orbsym:
            ncore = self.ncore
            nocc = self.ncore + self.ncas
            self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc1step.kernel(self, mo_coeff,
                               tol=self.conv_tol, macro=macro, micro=micro,
                               ci0=ci0, callback=callback, verbose=self.verbose)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo_coeff=None, ci0=None, macro=None, micro=None,
                callback=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None: macro = self.max_cycle_macro
        if micro is None: micro = 1 # self.max_cycle_micro
        if callback is None: callback = self.callback

        self.mol.check_sanity(self)

        self.dump_flags()

        #irrep_name = self.mol.irrep_name
        irrep_name = self.mol.irrep_id
        self.orbsym = pyscf.symm.label_orb_symm(self.mol, irrep_name,
                                                self.mol.symm_orb,
                                                self.mo_coeff,
                                                s=self._scf.get_ovlp())
        if not hasattr(self.fcisolver, 'orbsym') or \
           not self.fcisolver.orbsym:
            ncore = self.ncore
            nocc = self.ncore + self.ncas
            self.fcisolver.orbsym = self.orbsym[ncore:nocc]

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc2step.kernel(self, mo_coeff,
                               tol=self.conv_tol, macro=macro, micro=micro,
                               ci0=ci0, callback=callback, verbose=self.verbose)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def gen_g_hop(self, mo, casdm1, casdm2, eris):
        casdm1 = _symmetrize(casdm1, self.orbsym[self.ncore:self.ncore+self.ncas],
                             self.mol.groupname)
        g_orb, h_op1, h_opjk, h_diag = mc1step.gen_g_hop(self, mo, casdm1, casdm2, eris)
        g_orb = _symmetrize(self.unpack_uniq_var(g_orb), self.orbsym,
                            self.mol.groupname)
        h_diag = _symmetrize(self.unpack_uniq_var(h_diag), self.orbsym,
                             self.mol.groupname)
        def sym_h_op1(x):
            hx = h_op1(x)
            hx = _symmetrize(self.unpack_uniq_var(hx), self.orbsym,
                             self.mol.groupname)
            return self.pack_uniq_var(hx)
        def sym_h_opjk(x):
            hx = h_opjk(x)
            hx = _symmetrize(self.unpack_uniq_var(hx), self.orbsym,
                             self.mol.groupname)
            return self.pack_uniq_var(hx)
        return self.pack_uniq_var(g_orb), sym_h_op1, sym_h_opjk, \
               self.pack_uniq_var(h_diag)

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        dr = _symmetrize(dr, self.orbsym, self.mol.groupname)
        return numpy.dot(u0, mc1step.expmat(dr))

    def _eig(self, mat, b0, b1):
        orbsym = numpy.array(self.orbsym[b0:b1])
        norb = mat.shape[0]
        e = numpy.zeros(norb)
        c = numpy.zeros((norb,norb))
        for i0 in set(orbsym):
            lst = numpy.where(orbsym == i0)[0]
            if len(lst) > 0:
                w, v = scf.hf.eig(mat[lst[:,None],lst], None)
                e[lst] = w
                c[lst[:,None],lst] = v
        return e, c

def _symmetrize(mat, orbsym, groupname, wfnsym=0):
    if wfnsym != 0:
        raise RuntimeError('TODO: specify symmetry for %s' % groupname)
    mat1 = numpy.zeros_like(mat)
    orbsym = numpy.array(orbsym)
    for i0 in set(orbsym):
        lst = numpy.where(orbsym == i0)[0]
        mat1[lst[:,None],lst] = mat[lst[:,None],lst]
    return mat1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
    from pyscf.mcscf import addons

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
    mc = CASSCF(m, 6, 4)
    mc.fcisolver = pyscf.fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)
