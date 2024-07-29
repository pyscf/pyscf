#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CASSCF analytical nuclear gradients

Ref.
J. Comput. Chem., 5, 589

MRH: copied from pyscf.grad.casscf.py on 12/07/2019
Contains my modifications for SA-CASSCF gradients
1. Generalized Fock has nonzero i->a and u->a
2. Memory footprint for differentiated eris bugfix
'''

import inspect
import time
from functools import reduce
from itertools import product
import numpy
from scipy import linalg
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.grad import casci as casci_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf.df.grad.casdm2_util import (solve_df_rdm2, grad_elec_dferi,
                                       grad_elec_auxresponse_dferi)
from pyscf.mcscf.addons import StateAverageMCSCFSolver

def grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    mc = mc_grad.base
    with_df = mc.with_df
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mc.frozen is not None:
        raise NotImplementedError

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    # Necessary kludge because gfock isn't zero in occ-virt space in SA-CASSCf
    # Among many other potential applications!
    if hasattr (mc, '_tag_gfock_ov_nonzero'):
        if mc._tag_gfock_ov_nonzero:
            nocc = nmo

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(numpy.dot, (mo_cas, casdm1, mo_cas.T))
    # MRH flag: this is one of my kludges
    # It would be better to just pass the ERIS object used in orbital optimization
    # But I am too lazy at the moment
    aapa = with_df.ao2mo ((mo_cas, mo_cas, mo_occ, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nocc,ncas)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    gfock = numpy.zeros ((nocc, nocc))
    gfock[:,:ncore] = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c + vhf_a, mo_core)) * 2
    gfock[:,ncore:ncore+ncas] = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c, mo_cas, casdm1))
    gfock[:,ncore:ncore+ncas] += numpy.einsum('uviw,vuwt->it', aapa, casdm2)
    dme0 = reduce(numpy.dot, (mo_occ, (gfock+gfock.T)*.5, mo_occ.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    dm1 = dm_core + dm_cas
    vj, vk = mc_grad.get_jk(mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj - vk * .5
    hcore_deriv = mc_grad.hcore_generator(mol)
    s1 = mc_grad.get_ovlp(mol)

    dfcasdm2 = casdm2 = solve_df_rdm2 (mc_grad, mo_cas=mo_cas, casdm2=casdm2)
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = grad_elec_dferi (mc_grad, mo_cas=mo_cas, dfcasdm2=dfcasdm2, atmlst=atmlst,
        max_memory=mc_grad.max_memory)[0]
    if mc_grad.auxbasis_response:
        de_aux = vj.aux - vk.aux * .5
        de_aux = de_aux.sum ((0,1)) - de_aux[1,1]
        de_aux += grad_elec_auxresponse_dferi (mc_grad, mo_cas=mo_cas, dfcasdm2=dfcasdm2,
            atmlst=atmlst, max_memory=mc_grad.max_memory)[0]
        de += de_aux
    dfcasdm2 = casdm2 = None

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1)
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1c[:,p0:p1], dm1[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    log.timer('CASSCF nuclear gradients', *time0)
    return de

def as_scanner(mcscf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.1', verbose=0)
    >>> mc_grad_scanner = mcscf.CASSCF(scf.RHF(mol), 4, 4).nuc_grad_method().as_scanner()
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)
    name = mcscf_grad.__class__.__name__ + CASSCF_GradScanner.__name_mixin__
    return lib.set_class(CASSCF_GradScanner(mcscf_grad),
                         (CASSCF_GradScanner, mcscf_grad.__class__), name)

class CASSCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        mc_scanner = self.base
        e_tot = mc_scanner(mol)
        if isinstance(mc_scanner, StateAverageMCSCFSolver):
            e_tot = mc_scanner.e_average

        self.mol = mol
        de = self.kernel(**kwargs)
        return e_tot, de


class Gradients(casci_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mc):
        self.with_df = mc.with_df
        self.auxbasis_response = True
        casci_grad.Gradients.__init__(self, mc)

    grad_elec = grad_elec

    def get_jk (self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        vj, vk = dfrhf_grad.get_jk(self, mol, dm)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def kernel (self, mo_coeff=None, ci=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_coeff, ci, atmlst, log)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

Grad = Gradients

#from pyscf import mcscf
#mcscf.mc1step.CASSCF.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import mcscf
    from pyscf import df
    #from pyscf.grad import numeric

    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
    mol.basis = '631g'
    mol.build()
    aux = df.aug_etb (mol)
    mf = scf.RHF(mol).density_fit (auxbasis=aux).run()
    mc = mcscf.CASSCF(mf, 4, 4).run()
    mc.conv_tol = 1e-10
    de = Gradients (mc).kernel()
    #de_num = numeric.Gradients (mc).kernel ()
    #print(lib.finger(de) - 0.019602220578635747)
    #print(lib.finger(de) - lib.finger (de_num))

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'N 0 0 0; N 0 0 1.2'
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol).density_fit (auxbasis=aux).run()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-10
    mc.kernel ()
    de = Gradients (mc).kernel()

    mcs = mc.as_scanner()
    mol.set_geom_('N 0 0 0; N 0 0 1.201')
    e1 = mcs(mol)
    mol.set_geom_('N 0 0 0; N 0 0 1.199')
    e2 = mcs(mol)
    print(de[1,2], (e1-e2)/0.002*lib.param.BOHR)
