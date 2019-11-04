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
Dirac Hartree-Fock hyperfine coupling tensor (In testing)

Refs: JCP, 134, 044111
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.prop.ssc import dhf as dhf_ssc
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

# TODO: 3 SCF for sx, sy, sz

def make_h01(mol, atm_id):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    t1 = mol.intor('int1e_sa01sp_spinor', 3)
    n2c = t1.shape[2]
    n4c = n2c * 2
    h1 = numpy.zeros((3, n4c, n4c), complex)
    for i in range(3):
        h1[i,:n2c,n2c:] += .5 * t1[i]
        h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
    return h1

def kernel(hfcobj, with_gaunt=False, verbose=None):
    log = lib.logger.new_logger(hfcobj, verbose)
    mf = hfcobj._scf
    mol = mf.mol
# Add finite field to remove degeneracy
    nuc_spin = numpy.ones(3) * 1e-6
    sc = numpy.dot(mf.get_ovlp(), mf.mo_coeff)
    h0 = reduce(numpy.dot, (sc*mf.mo_energy, sc.conj().T))
    c = lib.param.LIGHT_SPEED
    n4c = h0.shape[0]
    n2c = n4c // 2
    Sigma = numpy.zeros((3,n4c,n4c), dtype=h0.dtype)
    Sigma[:,:n2c,:n2c] = mol.intor('int1e_sigma_spinor', comp=3)
    Sigma[:,n2c:,n2c:] = .25/c**2 * mol.intor('int1e_spsigmasp_spinor', comp=3)

    hfc = []
    for atm_id in range(mol.natm):
        symb = mol.atom_symbol(atm_id)
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
        nuc_gyro = get_nuc_g_factor(symb) * nuc_mag
        e_gyro = .5 * nist.G_ELECTRON
        au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
        fac = nist.ALPHA**2 * nuc_gyro * e_gyro * au2MHz
        #logger.debug('factor (MHz) %s', fac)

        h01 = make_h01(mol, 0)
        mo_occ = mf.mo_occ
        mo_coeff = mf.mo_coeff
        if 0:
            h01b = h0 + numpy.einsum('xij,x->ij', h01, nuc_spin)
            h01b = reduce(numpy.dot, (mf.mo_coeff.conj().T, h01b, mf.mo_coeff))
            mo_energy, v = numpy.linalg.eigh(h01b)
            mo_coeff = numpy.dot(mf.mo_coeff, v)
            mo_occ = mf.get_occ(mo_energy, mo_coeff)

        occidx = mo_occ > 0
        orbo = mo_coeff[:,occidx]
        dm0 = numpy.dot(orbo, orbo.T.conj())
        e01 = numpy.einsum('xij,ji->x', h01, dm0) * fac

        effspin = numpy.einsum('xij,ji->x', Sigma, dm0) * .5
        log.debug('atom %d Eff-spin %s', atm_id, effspin.real)

        e01 = (e01 / effspin).real
        hfc.append(e01)
    return numpy.asarray(hfc)

class HyperfineCoupling(lib.StreamObject):
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        self.mb = 'sternheim' # or RMB, RKB

        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s (In testing) ********',
                 self.__class__, self._scf.__class__)
        log.warn('DHF-HFC is an experimental feature. It is '
                 'still in testing.\nFeatures and APIs may be changed '
                 'in the future.')
        log.info('nuc_pair %s', self.nuc_pair)
        log.info('mb = %s', self.mb)
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    kernel = kernel

HFC = HyperfineCoupling

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.M(
        atom = [['Li', (0.,0.,0.)],
                #['He', (.4,.7,0.)],
               ],
        basis = 'ccpvdz', spin=1)
    mf = scf.DHF(mol).run()
    hfc = HFC(mf)
    print(hfc.kernel())
