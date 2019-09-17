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
Non-relativistic rotational g-tensor for RHF

Refs:
[1] J. Gauss, K. Ruud, T. Helgaker, J. Chem. Phys., 105, 2804 (1996)
[2] S. Sauer et al., Mol. Phys., 76, 445 (1991)
'''


import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import jk
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.magnetizability import rhf as rhf_mag
from pyscf.data import nist


def dia(magobj, gauge_orig=None):
    '''Part of rotational g-tensors. It is the direct second derivatives of
    the Lagrangian (corresponding to the zeroth order wavefunction).  Unit
    hbar/mu_N is not included.  This part may be different to the conventional
    dia-magnetic contributions of rotational g-tensors.
    '''
    mol = magobj.mol
    im, mass_center = inertia_tensor(mol)
    if gauge_orig is None:
        # Eq. (35) of JCP, 105, 2804
        e2 = rhf_mag.dia(magobj, gauge_orig)
        e2-= rhf_mag.dia(magobj, mass_center)
        e2 = _safe_solve(im, e2)
        return -4 * nist.PROTON_MASS_AU * e2
    else:
        dm0 = magobj._scf.make_rdm1()
        with mol.with_common_origin(gauge_orig):
            int_r = mol.intor('int1e_r', comp=3)
        cm = mass_center - gauge_orig
        e2 = numpy.einsum('xpq,qp,y->xy', int_r, dm0, cm)

        e2 = numpy.eye(3) * e2.trace() - e2
        e2 *= .5
        e2 = _safe_solve(im, e2)
        return -2 * nist.PROTON_MASS_AU * e2


# Note mo10 is the imaginary part of MO^1
def para(magobj, gauge_orig=None, h1=None, s1=None, with_cphf=None):
    '''Part of rotational g-tensors from the first order wavefunctions. Unit
    hbar/mu_N is not included.  This part may be different to the conventional
    para-magnetic contributions of rotational g-tensors.
    '''
    mol = magobj.mol
    im, mass_center = inertia_tensor(mol)

    if gauge_orig is None:
        # The first order Hamiltonian for rotation part is the same to the
        # first order Hamiltonian for magnetic field except a factor of 2. It can
        # be computed using the magnetizability code.
        mag_para = rhf_mag.para(magobj, gauge_orig, h1, s1, with_cphf) * 2

    else:
        mf = magobj._scf
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orbo = mo_coeff[:,mo_occ>0]

        # for magnetic field
        with mol.with_common_origin(mass_center):
            h10 = .5 * mol.intor('int1e_cg_irxp', 3)
            h10 = lib.einsum('xpq,pi,qj->xij', h10, mo_coeff.conj(), orbo)

        # for rotation part
        with mol.with_common_origin(gauge_orig):
            h01 = -mol.intor('int1e_cg_irxp', 3)
            h01 = lib.einsum('xpq,pi,qj->xij', h01, mo_coeff.conj(), orbo)

        s10 = numpy.zeros_like(h10)
        mo10 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h10, s10)[0]

        mag_para = numpy.einsum('xji,yji->xy', mo10.conj(), h01)
        mag_para = (mag_para + mag_para.conj()) * 2  # *2 for double occupancy

    mag_para = _safe_solve(im, mag_para)
    # unit = hbar/mu_N, mu_N is nuclear magneton
    unit = -2 * nist.PROTON_MASS_AU
    return mag_para * unit


def inertia_tensor(mol):
    '''Inertia tensor, mass center
    '''
    mass = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords()
    mass_center = numpy.einsum('i,ij->j', mass, coords) / mass.sum()
    coords = coords - mass_center
    im = numpy.einsum('i,ij,ik->jk', mass, coords, coords)
    im = numpy.eye(3) * im.trace() - im
    return im, mass_center


def nuc(mol):
    '''Nuclear contributions'''
    im, mass_center = inertia_tensor(mol)
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    coords = coords - mass_center
    gnuc = numpy.einsum('z,zx,zy->xy', charges, coords, coords)
    gnuc = numpy.eye(3) * gnuc.trace() - gnuc

    gnuc = _safe_solve(im, gnuc)
    # unit = hbar/mu_N, mu_N is nuclear magneton
    unit = 2 * nist.PROTON_MASS_AU
    return .5 * unit * gnuc

def _safe_solve(a, b):
    '''Solve x * a = b where a is a symmetric matrix'''
    w, v = numpy.linalg.eigh(a)
    v = v[:,w>1e-12]
    a_inv = numpy.dot(v/w[w>1e-12], v.T)
    return b.dot(a_inv)


class RotationalGTensor(rhf_mag.Magnetizability):
    '''HF rotational g-tensors'''

    def dump_flags(self, verbose=None):
        rhf_mag.Magnetizability.dump_flags(self, verbose)
        if self.gauge_orig is not None:
            logger.warn(self, 'Rotational g-tensor with '
                        'perturbation-independent basis is in testing.\n'
                        'Results do not fully agree with those in '
                        'JCP, 105, 2804.')
        return self

    def kernel(self):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        mag_dia = self.dia(self.gauge_orig)
        mag_para = self.para(self.gauge_orig)
        e_nuc = nuc(self.mol)
        e2 = mag_para + mag_dia + e_nuc

        logger.timer(self, 'Rotational g-tensors', *cput0)
        if self.verbose >= logger.NOTE:
            _write = rhf_nmr._write
            _write(self.stdout, e2, '\nRotational g-tensors (au)')
            #FIXME: Dia-, para-magnetic parts are partitioned in different way
            #See JCP, 105, 2804
            #_write(self.stdout, mag_dia, 'dia-magnetic contributions (au)')
            #_write(self.stdout, mag_para, 'para-magnetic contributions (au)')
            _write(self.stdout, e_nuc, 'nuclear contributions (au)')
        return e2

    dia = dia
    para = para

from pyscf import scf
scf.hf.RHF.RotationalGTensor = lib.class_as_method(RotationalGTensor)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   .917
                  F  ,  0.   0.   0.
                  '''
    mol.basis = 'ccpvdz'
    mol.build()

    mf = scf.RHF(mol).run()
    rotg = mf.RotationalGTensor()
    m = rotg.kernel()
    print(m[0,0] - 0.740149929639848)

    rotg.gauge_orig = (0,0,.1)
    m = rotg.kernel()
    print(m[0,0] - 0.8323151749078354)

    mol.atom = '''C  ,  0.   0.   0.
                  O  ,  0.   0.   1.1283
                  '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(m[0,0] - -0.2805925799038227)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(lib.finger(m) - 0.09396805421224698)
