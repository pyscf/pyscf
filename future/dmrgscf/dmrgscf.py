#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Sandeep Sharma <sanshar@gmail.com>
#

import time
import numpy
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.mcscf.mc1step as mc1step
import pyscf.mcscf.mc2step as mc2step
import pyscf.mcscf.mc_ao2mo as mc_ao2mo
import dmrgci


class NameMeCASSCF(mc1step.CASSCF):
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        mc1step.CASSCF.__init__(self, mol, mf, ncas, nelecas, ncore)
#TODO: tune default value
#TODO:# the max orbital rotation and CI increment, prefer small step size
#TODO:        self.max_orb_stepsize = .03
#TODO:# small max_ci_stepsize is good to converge, since steepest descent is used
#TODO:        self.max_ci_stepsize = .01
#TODO:        self.max_cycle_macro = 50
#TODO:        self.max_cycle_micro = 2
#TODO:# num steps to approx orbital rotation without integral transformation.
#TODO:# Increasing steps do not help converge since the approx gradient might be
#TODO:# very diff to real gradient after few steps. If the step predicted by AH is
#TODO:# good enough, it can be set to 1 or 2 steps.
#TODO:        self.max_cycle_micro_inner = 4
#TODO:        self.conv_threshold = 1e-7
#TODO:        self.conv_threshold_grad = 1e-4
#TODO:        # for augmented hessian
#TODO:        self.ah_level_shift = 0#1e-2
#TODO:        self.ah_conv_threshold = 1e-7
#TODO:        self.ah_max_cycle = 15
#TODO:        self.ah_lindep = self.ah_conv_threshold**2

        self.fcisolver = dmrgci
        self.scratch_file = None
        self.nproc = 1

        self.current_macro_iter = 0

        self._keys = set(self.__dict__.keys() + ['_keys'])

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** NameMeCASSCF flags ********')
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        log.info('max. macro cycles = %d', self.max_cycle_macro)
        log.info('max. micro cycles = %d', self.max_cycle_micro)
        log.info('conv_threshold = %g, (%g for gradients)', \
                 self.conv_threshold, self.conv_threshold_grad)
        log.info('max_cycle_micro_inner = %d', self.max_cycle_micro_inner)
        log.info('max. orb step = %g', self.max_orb_stepsize)
        log.info('max. ci step = %g', self.max_ci_stepsize)
        log.info('augmented hessian max. cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian conv_threshold = %g', self.ah_conv_threshold)
        log.info('augmented hessian linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian level shift = %d', self.ah_level_shift)
        log.info('max_memory %d MB', self.max_memory)

        log.info('scratch_file = %s', self.scratch_file)
        log.info('nproc = %d', self.nproc)

    def mc2step(self, mo=None, ci0=None, macro=None, micro=None):
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

        self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc2step.kernel(self, mo, \
                               tol=self.conv_threshold, macro=macro, micro=micro, \
                               ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def rotate_orb(self, mo, fcivec, e_ci, eris, dx=0):
        res = mc1step.rotate_orb_ah(self, mo, fcivec, e_ci, eris, dx, self.verbose)
#TODO: follow input mo or symmetrize orbital rotation
        return res

    def update_ao2mo(self, mo):
        eris = mc_ao2mo._ERIS(self, mo)
#TODO: symmetrize eris.aaaa integrals
        return eris

    def save_mo_coeff(self, mo_coeff, imacro, imicro):
        self.current_macro_iter = imacro
        key = []
        #key.append(tmpdir)
        key.append('dmrgscfmo-%d-%d'%(imacro,imicro))
        #key.append('-'+time.strftime('%y%m%d%H%M%S'))
        key = ''.join(key)
        numpy.save(key, mo_coeff)



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = NameMeCASSCF(mol, m, 4, 4)
    mc.verbose = 4
    ecasci = mc.casci(m.mo_coeff)[0] + mol.nuclear_repulsion()
    emc = mc.mc2step()[0] + mol.nuclear_repulsion()
    print(ehf, ecasci, emc, emc-ehf)
    print(emc - -3.22013929407)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = NameMeCASSCF(mol, m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()
    mo[:,2:5] = m.mo_coeff[:,[4,2,3]]
    emc = mc.mc2step(mo)[0] + mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)


