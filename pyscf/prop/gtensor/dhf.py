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
Dirac Hartree-Fock g-tensor (In testing)

Refs: TCA, 129, 715
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.prop.nmr import dhf as dhf_nmr
from pyscf.data import nist


# TODO: 3 SCF for sx, sy, sz

def kernel(gobj, gauge_orig=None, mb='RKB', with_gaunt=False, verbose=None):
    log = lib.logger.new_logger(gobj, verbose)
    mf = gobj._scf
    mol = mf.mol
# Add finite field to remove degeneracy
    mag_field = numpy.ones(3) * 1e-6
    h10 = dhf_nmr.make_h10rkb(mol, None, None, False, log)
    sc = numpy.dot(mf.get_ovlp(), mf.mo_coeff)
    h0 = reduce(numpy.dot, (sc*mf.mo_energy, sc.conj().T))
    h10b = h0 + numpy.einsum('xij,x->ij', h10, mag_field)
    h10b = reduce(numpy.dot, (mf.mo_coeff.conj().T, h10b, mf.mo_coeff))
    mo_energy, v = numpy.linalg.eigh(h10b)
    mo_coeff = numpy.dot(mf.mo_coeff, v)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)

    occidx = mo_occ > 0
    orbo = mo_coeff[:,occidx]
    dm0 = numpy.dot(orbo, orbo.T.conj())
    dme = numpy.dot(orbo * mo_energy[occidx], orbo.conj().T)
    h10 = dhf_nmr.make_h10(mol, dm0, gauge_orig, mb, with_gaunt, log)
    s10 = dhf_nmr.make_s10(mol, gauge_orig, mb)

# Intrinsic muB = eh/2mc
# First order Dirac operator is 1/c * h10 => g ~ Tr(h10,DM)/c / mu_B = 2 Tr(h10,DM)
    muB = .5  # Bohr magneton
    g = (numpy.einsum('xij,ji->x', h10, dm0) -
         numpy.einsum('xij,ji->x', s10, dme)) / muB
    c = lib.param.LIGHT_SPEED
    n4c = dm0.shape[0]
    n2c = n4c // 2
    Sigma = numpy.zeros_like(s10)
    Sigma[:,:n2c,:n2c] = mol.intor('int1e_sigma_spinor', comp=3)
    Sigma[:,n2c:,n2c:] = .25/c**2 * mol.intor('int1e_spsigmasp_spinor', comp=3)
    effspin = numpy.einsum('xij,ji->x', Sigma, dm0) * .5
    log.debug('Eff-spin %s', effspin.real)
    g = (g / effspin).real

    facppt = 1e3
    gshift = (g - nist.G_ELECTRON) * facppt
    log.note('G shift (ppt) %s', gshift)
    return g

class GTensor(dhf_nmr.NMR):
    def __init__(self, mf):
        dhf_nmr.NMR.__init__(self, mf)
        lib.logger.warn(self, 'DHF-gtensor is an experimental feature. It is '
                        'still in testing.\nFeatures and APIs may be changed '
                        'in the future.')
    kernel = kernel

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.M(
        atom = [['Ne', (0.,0.,0.)],
                #['He', (.4,.7,0.)],
               ],
        basis = 'ccpvdz', spin=2, charge=2)
    mf = scf.DHF(mol).run()
    print(GTensor(mf).kernel((0,0,0)))
    print(GTensor(mf).kernel(mb='RMB'))
