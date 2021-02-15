#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
CASCI analytical nuclear gradients

Ref.
J. Comput. Chem., 5, 589
'''

import sys
import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.scf import cphf

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range


def grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc._scf.mo_coeff
    if ci is None: ci = mc.ci

    time0 = time.clock(), time.time()
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    mo_energy = mc._scf.mo_energy

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    neleca, nelecb = mol.nelec
    assert(neleca == nelecb)
    orbo = mo_coeff[:,:neleca]
    orbv = mo_coeff[:,neleca:]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(numpy.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nmo,ncas)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    # Imat = h1_{pi} gamma1_{iq} + h2_{pijk} gamma_{iqkj}
    Imat = numpy.zeros((nmo,nmo))
    Imat[:,:nocc] = reduce(numpy.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
    Imat[:,ncore:nocc] = reduce(numpy.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    Imat[:,ncore:nocc] += lib.einsum('uviw,vuwt->it', aapa, casdm2)
    aapa = vj = vk = vhf_c = vhf_a = h1 = None

    ee = mo_energy[:,None] - mo_energy
    zvec = numpy.zeros_like(Imat)
    zvec[:ncore,ncore:neleca] = Imat[:ncore,ncore:neleca] / -ee[:ncore,ncore:neleca]
    zvec[ncore:neleca,:ncore] = Imat[ncore:neleca,:ncore] / -ee[ncore:neleca,:ncore]
    zvec[nocc:,neleca:nocc] = Imat[nocc:,neleca:nocc] / -ee[nocc:,neleca:nocc]
    zvec[neleca:nocc,nocc:] = Imat[neleca:nocc,nocc:] / -ee[neleca:nocc,nocc:]

    zvec_ao = reduce(numpy.dot, (mo_coeff, zvec+zvec.T, mo_coeff.T))
    vhf = mc._scf.get_veff(mol, zvec_ao) * 2
    xvo = reduce(numpy.dot, (orbv.T, vhf, orbo))
    xvo += Imat[neleca:,:neleca] - Imat[:neleca,neleca:].T
    def fvind(x):
        x = x.reshape(xvo.shape)
        dm = reduce(numpy.dot, (orbv, x, orbo.T))
        v = mc._scf.get_veff(mol, dm + dm.T)
        v = reduce(numpy.dot, (orbv.T, v, orbo))
        return v * 2
    dm1resp = cphf.solve(fvind, mo_energy, mc._scf.mo_occ, xvo, max_cycle=30)[0]
    zvec[neleca:,:neleca] = dm1resp

    zeta = numpy.einsum('ij,j->ij', zvec, mo_energy)
    zeta = reduce(numpy.dot, (mo_coeff, zeta, mo_coeff.T))

    zvec_ao = reduce(numpy.dot, (mo_coeff, zvec+zvec.T, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:neleca], mo_coeff[:,:neleca].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mc._scf.get_veff(mol, zvec_ao), p1))

    Imat[:ncore,ncore:neleca] = 0
    Imat[ncore:neleca,:ncore] = 0
    Imat[nocc:,neleca:nocc] = 0
    Imat[neleca:nocc,nocc:] = 0
    Imat[neleca:,:neleca] = Imat[:neleca,neleca:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))

    casci_dm1 = dm_core + dm_cas
    hf_dm1 = mc._scf.make_rdm1(mo_coeff, mc._scf.mo_occ)
    hcore_deriv = mc_grad.hcore_generator(mol)
    s1 = mc_grad.get_ovlp(mol)

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    max_memory = mc_grad.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, casci_dm1)
        de[k] += numpy.einsum('xij,ij->x', h1ao, zvec_ao)

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                de[k,i] -= numpy.einsum('ijkl,ij,kl', eri1tmp, hf_dm1[p0:p1,q0:q1], zvec_ao) * 2
                de[k,i] -= numpy.einsum('ijkl,kl,ij', eri1tmp, hf_dm1, zvec_ao[p0:p1,q0:q1]) * 2
                de[k,i] += numpy.einsum('ijkl,il,kj', eri1tmp, hf_dm1[p0:p1], zvec_ao[q0:q1])
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, hf_dm1[q0:q1], zvec_ao[p0:p1])

                #:vhf1c, vhf1a = mc_grad.get_veff(mol, (dm_core, dm_cas))
                #:de[k] += numpy.einsum('xij,ij->x', vhf1c[:,p0:p1], casci_dm1[p0:p1]) * 2
                #:de[k] += numpy.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2
                de[k,i] -= numpy.einsum('ijkl,lk,ij', eri1tmp, dm_core[q0:q1], casci_dm1[p0:p1]) * 2
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, dm_core[q0:q1], casci_dm1[p0:p1])
                de[k,i] -= numpy.einsum('ijkl,lk,ij', eri1tmp, dm_cas[q0:q1], dm_core[p0:p1]) * 2
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, dm_cas[q0:q1], dm_core[p0:p1])
            eri1 = eri1tmp = None

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], im1[:,p0:p1])

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], zeta[:,p0:p1]) * 2

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], vhf_s1occ[:,p0:p1]) * 2

    log.timer('CASCI nuclear gradients', *time0)
    return de


def as_scanner(mcscf_grad, state=None):
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
    >>> mc_grad_scanner = mcscf.CASCI(scf.RHF(mol), 4, 4).nuc_grad_method().as_scanner()
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    from pyscf import gto
    from pyscf.mcscf.addons import StateAverageMCSCFSolver
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad
    if (state is not None and
        isinstance(mcscf_grad.base, StateAverageMCSCFSolver)):
        raise RuntimeError('State-Average MCSCF Gradients does not support '
                           'state-specific nuclear gradients.')

    logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)

    class CASCI_GradScanner(mcscf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, state=state, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            if state is None:
                state = self.state

            mc_scanner = self.base
# TODO: Check root flip
            e_tot = mc_scanner(mol)
            ci = mc_scanner.ci
            if isinstance(mc_scanner, StateAverageMCSCFSolver):
                e_tot = mc_scanner.e_average
            elif not isinstance(e_tot, float):
                if state >= mc_scanner.fcisolver.nroots:
                    raise ValueError('State ID greater than the number of CASCI roots')
                e_tot = e_tot[state]
                # target at a specific state, to avoid overwriting self.state
                # in self.kernel
                ci = ci[state]

            self.mol = mol
            de = self.kernel(ci=ci, state=state, **kwargs)
            return e_tot, de
    return CASCI_GradScanner(mcscf_grad)


class Gradients(rhf_grad.GradientsBasics):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, mc):
        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        if isinstance(mc, StateAverageMCSCFSolver):
            self.state = None  # not a specific state
        else:
            self.state = 0  # of which the gradients to be computed.
        rhf_grad.GradientsBasics.__init__(self, mc)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state %s not converged', self.base.__class__)
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        if self.state is None:
            weights = self.base.weights
            log.info('State-average gradients over %d states with weights %s',
                     len(weights), weights)
        elif self.state != 0 and self.base.fcisolver.nroots > 1:
            log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    grad_elec = grad_elec

    def kernel(self, mo_coeff=None, ci=None, atmlst=None,
               state=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if ci is None: ci = self.base.ci
        if self.state is None:  # state average MCSCF calculations
            assert(state is None)
        elif isinstance(ci, (list, tuple, RANGE_TYPE)):
            if state is None:
                state = self.state
            else:
                self.state = state
            ci = ci[state]
            log.info('Multiple roots are found in CASCI solver. '
                     'Nuclear gradients of root %d are computed.', state)

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

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. x2c, QM/MM, solvent) modifies the SCF object only.
    def hcore_generator(self, mol=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.hcore_generator(mol)

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            if self.state is None:
                logger.note(self, '--------- %s gradients ----------',
                            self.base.__class__.__name__)
            else:
                logger.note(self, '--------- %s gradients for state %d ----------',
                            self.base.__class__.__name__, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

Grad = Gradients

from pyscf import mcscf
mcscf.casci.CASCI.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mc = mcscf.CASCI(mf, 4, 4).run()
    g1 = mc.Gradients().kernel()
    print(lib.finger(g1) - -0.066025991364829367)

    mcs = mc.as_scanner()
    mol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2')
    e1 = mcs(mol)
    mol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2')
    e2 = mcs(mol)
    print(g1[1,2], (e1-e2)/0.002*lib.param.BOHR)
