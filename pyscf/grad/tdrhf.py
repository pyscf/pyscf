#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Ref:
# J. Chem. Phys. 117, 7433
#


from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import cphf
from pyscf import __config__


def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    vj, vk = mf.get_jk(mol, (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T), hermi=0)
    veff0doo = vj[0] * 2 - vk[0]
    wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(numpy.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao+z1ao.T)

    im0 = numpy.zeros((nmo,nmo))
    im0[:nocc,:nocc] = reduce(numpy.dot, (orbo.T, veff0doo+veff, orbo))
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy)
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,nocc:] = numpy.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy)
    im0[nocc:,nocc:]+= numpy.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,:nocc] = numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy)*2
    im0[nocc:,:nocc]+= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy)*2

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo
    dm1[nocc:,nocc:] = dvv
    dm1[nocc:,:nocc] = z1
    dm1[:nocc,:nocc] += numpy.eye(nocc)*2 # for ground state
    im0 = reduce(numpy.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo
    oo0 = reduce(numpy.dot, (orbo, orbo.T))
    vj, vk = td_grad.get_jk(mol, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T,
                                  dmxmy-dmxmy.T))
    vj = vj.reshape(-1,3,nao,nao)
    vk = vk.reshape(-1,3,nao,nao)
    vhf1 = -vk
    if singlet:
        vhf1 += vj * 2
    else:
        vhf1[:2] += vj[:2]*2
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += vhf1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += vhf1[0,:,p0:p1].transpose(0,2,1)
        # oo0*2 for doubly occupied orbitals
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0) * 2
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doo)

        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', vhf1[1,:,p0:p1], oo0[p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1[2,:,p0:p1], dmxpy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1[3,:,p0:p1], dmxmy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1[2,:,p0:p1], dmxpy[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1[3,:,p0:p1], dmxmy[:,p0:p1]) * 2

        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de


def as_scanner(td_grad, state=1):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, tdscf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> td_grad_scanner = scf.RHF(mol).apply(tdscf.TDA).nuc_grad_method().as_scanner()
    >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(td_grad, lib.GradScanner):
        return td_grad

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)

    class TDSCF_GradScanner(td_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
            self._keys = self._keys.union(['e_tot'])
        def __call__(self, mol_or_geom, state=state, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            self.reset(mol)

            td_scanner = self.base
            td_scanner(mol)
# TODO: Check root flip.  Maybe avoid the initial guess in TDHF otherwise
# large error may be found in the excited states amplitudes
            de = self.kernel(state=state, **kwargs)
            e_tot = self.e_tot[state-1]
            return e_tot, de
        @property
        def converged(self):
            td_scanner = self.base
            return all((td_scanner._scf.converged,
                        td_scanner.converged[self.state]))

    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()
    else:
        return TDSCF_GradScanner(td_grad)


class Gradients(rhf_grad.GradientsMixin):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.max_memory = td.max_memory
        self.state = 1  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None
        keys = set(('cphf_max_cycle', 'cphf_conv_tol'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** LR %s gradients for %s ********',
                 self.base.__class__, self.base._scf.__class__)
        log.info('cphf_conv_tol = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        '''
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        '''
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(self, 'state=0 found in the input. '
                            'Gradients of ground state is computed.')
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

            xy = self.base.xy[state-1]

        if singlet is None: singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, singlet, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- %s gradients for state %d ----------',
                        self.base.__class__.__name__, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

Grad = Gradients

from pyscf import tdscf
tdscf.rhf.TDA.Gradients = tdscf.rhf.TDHF.Gradients = lib.class_as_method(Gradients)
