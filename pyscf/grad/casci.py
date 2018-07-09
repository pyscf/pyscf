#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.scf import cphf


def kernel(mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None):
    if mo_coeff is None: mo_coeff = mc._scf.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    assert(isinstance(ci, numpy.ndarray))

    mol = mc.mol
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
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

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

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, casci_dm1)
        de[k] += numpy.einsum('xij,ij->x', h1ao, zvec_ao)

        vhf1 = numpy.zeros((3,nao,nao))
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

                #:vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))
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

    de += mf_grad.grad_nuc(mol, atmlst)
    return de


def as_scanner(mcscf_grad, state=0):
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
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)

    class CASCI_GradScanner(mcscf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, state=state, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mc_scanner = self.base
            if (mc_scanner.fcisolver.nroots > 1 and
                state >= mc_scanner.fcisolver.nroots):
                raise ValueError('State ID greater than the number of CASCI roots')

# TODO: Check root flip
            e_tot = mc_scanner(mol)
            if mc_scanner.fcisolver.nroots > 1:
                e_tot = e_tot[state]
                ci = mc_scanner.ci[state]
            else:
                ci = mc_scanner.ci

            self.mol = mol
            de = self.kernel(ci=ci, state=state, **kwargs)
            return e_tot, de
    return CASCI_GradScanner(mcscf_grad)


class Gradients(lib.StreamObject):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, mc):
        self.base = mc
        self.mol = mc.mol
        self.stdout = mc.stdout
        self.verbose = mc.verbose
        self.max_memory = mc.max_memory
        self.state = 0  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state CASCI not converged')
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        if self.state != 0 and self.base.fcisolver.nroots > 1:
            log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self, mo_coeff=None, ci=None, atmlst=None, mf_grad=None,
               state=None, verbose=None):
        cput0 = (time.clock(), time.time())
        log = logger.new_logger(self, verbose)
        if ci is None: ci = self.base.ci
        if isinstance(ci, (list, tuple)):
            if state is None:
                state = self.state
            else:
                self.state = state

            ci = ci[state]
            logger.info(self, 'Multiple roots are found in CASCI solver. '
                        'Nuclear gradients of root %d are computed.', state)

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.de = kernel(self.base, mo_coeff, ci, atmlst, mf_grad, log)
        log.timer('CASCI gradients', *cput0)
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mc = mcscf.CASCI(mf, 4, 4).run()
    g1 = mc.nuc_grad_method().kernel()
    print(lib.finger(g1) - -0.066025991364829367)

    mcs = mc.as_scanner()
    mol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2')
    e1 = mcs(mol)
    mol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2')
    e2 = mcs(mol)
    print(g1[1,2], (e1-e2)/0.002*lib.param.BOHR)
