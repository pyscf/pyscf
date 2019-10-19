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

'''
CISD analytical nuclear gradients
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ci import cisd
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import ccsd as ccsd_grad


def grad_elec(cigrad, civec=None, eris=None, atmlst=None, verbose=logger.INFO):
    myci = cigrad.base
    if civec is None: civec = myci.ci
    assert(not isinstance(civec, (list, tuple)))
    nocc = myci.nocc
    nmo = myci.nmo
    d1 = cisd._gamma1_intermediates(myci, civec, nmo, nocc)
    fd2intermediate = lib.H5TmpFile()
    d2 = cisd._gamma2_outcore(myci, civec, nmo, nocc, fd2intermediate, True)
    t1 = t2 = l1 = l2 = civec
    return ccsd_grad.grad_elec(cigrad, t1, t2, l1, l2, eris, atmlst, d1, d2, verbose)


def as_scanner(grad_ci, state=0):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CISD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CISD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> ci_scanner = ci.CISD(scf.RHF(mol)).nuc_grad_method().as_scanner()
    >>> e_tot, grad = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(grad_ci, lib.GradScanner):
        return grad_ci

    logger.info(grad_ci, 'Create scanner for %s', grad_ci.__class__)

    class CISD_GradScanner(grad_ci.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)

            # cache eris. It is used multiple times when calculating gradients
            g_ao2mo = g.base.__class__.ao2mo
            def _save_eris(self, *args, **kwargs):
                self._eris = g_ao2mo(self, *args, **kwargs)
                return self._eris
            self.base.__class__.ao2mo = _save_eris

        def __call__(self, mol_or_geom, state=state, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            ci_scanner = self.base
            if ci_scanner.nroots > 1 and state >= ci_scanner.nroots:
                raise ValueError('State ID greater than the number of CISD roots')

# TODO: Check root flip
            ci_scanner(mol)
            if ci_scanner.nroots > 1:
                e_tot = ci_scanner.e_tot[state]
                civec = ci_scanner.ci[state]
            else:
                e_tot = ci_scanner.e_tot
                civec = ci_scanner.ci

            self.mol = mol
            de = self.kernel(civec, eris=ci_scanner._eris, **kwargs)
            ci_scanner._eris = None  # release the resources occupied by .eris
            return e_tot, de
        @property
        def converged(self):
            ci_scanner = self.base
            if ci_scanner.nroots > 1:
                ci_conv = ci_scanner.converged[state]
            else:
                ci_conv = ci_scanner.converged
            return all((ci_scanner._scf.converged, ci_conv))
    return CISD_GradScanner(grad_ci)

class Gradients(rhf_grad.GradientsBasics):
    def __init__(self, myci):
        self.state = 0  # of which the gradients to be computed.
        rhf_grad.GradientsBasics.__init__(self, myci)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state %s not converged',
                     self.base.__class__.__name__)
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        if self.state != 0 and self.base.nroots > 1:
            log.info('State ID = %d', self.state)
        return self

    grad_elec = grad_elec

    def kernel(self, civec=None, eris=None, atmlst=None, state=None,
               verbose=None):
        log = logger.new_logger(self, verbose)
        myci = self.base
        if civec is None: civec = myci.ci
        if civec is None: civec = myci.kernel(eris=eris)
        if (isinstance(civec, (list, tuple)) or
            (isinstance(civec, numpy.ndarray) and civec.ndim > 1)):
            if state is None:
                state = self.state
            else:
                self.state = state

            civec = civec[state]
            logger.info(self, 'Multiple roots are found in CISD solver. '
                        'Nuclear gradients of root %d are computed.', state)

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(civec, eris, atmlst, verbose=log)
        self.de = de + self.grad_nuc(atmlst=atmlst)
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

cisd.CISD.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import grad

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    myci = cisd.CISD(mf)
    myci.kernel()
    g1 = myci.Gradients().kernel()
# O     0.0000000000    -0.0000000000     0.0065498854
# H    -0.0000000000     0.0208760610    -0.0032749427
# H    -0.0000000000    -0.0208760610    -0.0032749427
    print(lib.finger(g1) - -0.032562200777204092)

    mcs = myci.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2] - (e1-e2)/0.002*lib.param.BOHR)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    myci = cisd.CISD(mf)
    myci.frozen = [0,1,10,11,12]
    myci.max_memory = 1
    myci.kernel()
    g1 = Gradients(myci).kernel()
# O    -0.0000000000     0.0000000000     0.0106763547
# H     0.0000000000    -0.0763194988    -0.0053381773
# H     0.0000000000     0.0763194988    -0.0053381773
    print(lib.finger(g1) - 0.1022427304650084)

    mcs = myci.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2] - (e1-e2)/0.002*lib.param.BOHR)

    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.76',
        basis = '631g',
        unit='Bohr')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    myci = cisd.CISD(mf)
    myci.conv_tol = 1e-10
    myci.kernel()
    g1 = Gradients(myci).kernel()
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]
