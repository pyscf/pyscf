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
import pyscf.symm
from pyscf.mcscf import casci
from pyscf.mcscf import aug_hessian
from pyscf.mcscf import mc1step
from pyscf.mcscf import mc2step
from pyscf import ao2mo
from pyscf import fci
from pyscf.tools.mo_mapping import mo_1to1map


class CASSCF(mc1step.CASSCF):
    def __init__(self, mf, ncas, nelecas, ncore=None):
        assert(mf.mol.symmetry)
# Ag, A1 or A
#TODO:        self.wfnsym = pyscf.symm.param.CHARACTER_TABLE[mol.groupname][0][0]
        self.orbsym = []
        mc1step.CASSCF.__init__(self, mf, ncas, nelecas, ncore)

    def mc1step(self, mo_coeff=None, ci0=None, macro=None, micro=None, **cikwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.mol.check_sanity(self)

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

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc1step.kernel(self, mo_coeff, \
                               tol=self.conv_tol, macro=macro, micro=micro, \
                               ci0=ci0, verbose=self.verbose, **cikwargs)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo_coeff=None, ci0=None, macro=None, micro=None, **cikwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.mol.check_sanity(self)

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

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc2step.kernel(self, mo_coeff, \
                               tol=self.conv_tol, macro=macro, micro=micro, \
                               ci0=ci0, verbose=self.verbose, **cikwargs)
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

    def rotate_orb_cc(self, mo, fcasdm1, fcasdm2, eris, verbose):
        for u, g_orb, njk \
                in mc1step.rotate_orb_cc(self, mo, fcasdm1, fcasdm2, eris, verbose):
            yield _symmetrize(u, self.orbsym, self.mol.groupname), g_orb, njk

    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if eris is None: eris = self.ao2mo(mo_coeff)
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nelecas = self.nelecas
        casdm1 = self.fcisolver.make_rdm1(ci, ncas, nelecas)
        occ, ucas = _symm_eigh(-casdm1, self.orbsym[ncore:nocc])
        occ = -occ
        log.info('Natural occ %s', str(occ))
# restore phase
        for i, k in enumerate(numpy.argmax(abs(ucas), axis=0)):
            if ucas[k,i] < 0:
                ucas[:,i] *= -1
        mo_coeff1 = mo_coeff.copy()
        mo_coeff1[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)

        where_natorb = mo_1to1map(ucas)
        ci0 = fci.addons.reorder(ci, nelecas, where_natorb)

        h1eff =(reduce(numpy.dot, (mo_coeff[:,ncore:nocc].T, self.get_hcore(),
                                   mo_coeff[:,ncore:nocc]))
              + eris.vhf_c[ncore:nocc,ncore:nocc])
        h1eff = reduce(numpy.dot, (ucas.T, h1eff, ucas))
        aaaa = eris.aapp[:,:,ncore:nocc,ncore:nocc].copy()
        aaaa = ao2mo.incore.full(ao2mo.restore(8, aaaa, ncas), ucas)
        e_cas, fcivec = self.fcisolver.kernel(h1eff, aaaa, ncas, nelecas, ci0=ci0)
        log.debug('In Natural orbital, CI energy = %.12g', e_cas)
        return mo_coeff1, fcivec, occ
    def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, verbose=None):
        self.mo_coeff, self.ci, occ = self.cas_natorb(mo_coeff, ci, eris, verbose)
        return self.ci, self.mo_coeff

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if eris is None: eris = self.ao2mo(mo_coeff)
        ncore = self.ncore
        nocc = ncore + self.ncas
        nmo = mo_coeff.shape[1]
        mo_coeff1 = numpy.empty_like(mo_coeff)
        mo_coeff1[:,ncore:nocc] = mo_coeff[:,ncore:nocc]
        #mo_coeff1, ci, occ = mc.cas_natorb(mo_coeff, ci, eris, verbose)
        fock = self.get_fock(mo_coeff, ci, eris)
        if ncore > 0:
            w, c1 = _symm_eigh(fock[:ncore,:ncore], self.orbsym[:ncore])
            mo_coeff1[:,:ncore] = numpy.dot(mo_coeff[:,:ncore], c1)
        if nmo-nocc > 0:
            w, c1 = _symm_eigh(fock[nocc:,nocc:], self.orbsym[nocc:])
            mo_coeff1[:,nocc:] = numpy.dot(mo_coeff[:,nocc:], c1)
        return mo_coeff1, ci
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, verbose=None):
        self.mo_coeff, self.ci = self.canonicalize(self, mo_coeff, ci, verbose=verbose)
        return self.mo_coeff, self.ci

def _symmetrize(mat, orbsym, groupname, wfnsym=0):
    if wfnsym != 0:
        raise RuntimeError('TODO: specify symmetry for %s' % groupname)
    mat1 = numpy.zeros_like(mat)
    orbsym = numpy.array(orbsym)
    for i0 in set(orbsym):
        lst = numpy.where(orbsym == i0)[0]
        mat1[lst[:,None],lst] = mat[lst[:,None],lst]
    return mat1

def _symm_eigh(mat, orbsym):
    orbsym = numpy.array(orbsym)
    norb = mat.shape[0]
    e = numpy.zeros(norb)
    c = numpy.zeros((norb,norb))
    for i0 in set(orbsym):
        lst = numpy.where(orbsym == i0)[0]
        w, v = scipy.linalg.eigh(mat[lst[:,None],lst])
        e[lst] = w
        c[lst[:,None],lst] = v
    return e, c


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
