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
Non-relativistic rotational g-tensor for UHF
'''


import numpy
from pyscf import lib
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.magnetizability import uhf as uhf_mag
from pyscf.prop.rotational_gtensor import rhf as rhf_g
from pyscf.data import nist


def dia(magobj, gauge_orig=None):
    '''Part of rotational g-tensors. It is the direct second derivatives of
    the Lagrangian (corresponding to the zeroth order wavefunction).  Unit
    hbar/mu_N is not included.  This part may be different to the conventional
    dia-magnetic contributions of rotational g-tensors.
    '''
    mol = magobj.mol
    im, mass_center = rhf_g.inertia_tensor(mol)
    if gauge_orig is None:
        # Eq. (35) of JCP 105, 2804 (1996); DOI:10.1063/1.472143
        e2 = uhf_mag.dia(magobj, gauge_orig)
        e2 -= uhf_mag.dia(magobj, mass_center)
        e2 = rhf_g._safe_solve(im, e2)
        return -4 * nist.PROTON_MASS_AU * e2
    else:
        dm0a, dm0b = magobj._scf.make_rdm1()
        dm0 = dm0a + dm0b
        with mol.with_common_origin(gauge_orig):
            int_r = mol.intor('int1e_r', comp=3)
        cm = mass_center - gauge_orig
        e2 = numpy.einsum('xpq,qp,y->xy', int_r, dm0, cm)

        e2 = numpy.eye(3) * e2.trace() - e2
        e2 *= .5
        e2 = rhf_g._safe_solve(im, e2)
        return -2 * nist.PROTON_MASS_AU * e2


# Note mo10 is the imaginary part of MO^1
def para(magobj, gauge_orig=None, h1=None, s1=None, with_cphf=None):
    '''Part of rotational g-tensors from the first order wavefunctions. Unit
    hbar/mu_N is not included.  This part may be different to the conventional
    para-magnetic contributions of rotational g-tensors.
    '''
    mol = magobj.mol
    im, mass_center = rhf_g.inertia_tensor(mol)

    if gauge_orig is None:
        # The first order Hamiltonian for rotation part is the same to the
        # first order Hamiltonian for magnetic field except a factor of 2. It can
        # be computed using the magnetizability code.
        mag_para = uhf_mag.para(magobj, gauge_orig, h1, s1, with_cphf) * 2

    else:
        mf = magobj._scf
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orboa = mo_coeff[0][:,mo_occ[0]>0]
        orbob = mo_coeff[1][:,mo_occ[1]>0]

        # for magnetic field
        with mol.with_common_origin(mass_center):
            h10 = .5 * mol.intor('int1e_cg_irxp', 3)
            h10a = lib.einsum('xpq,pi,qj->xij', h10, mo_coeff[0].conj(), orboa)
            h10b = lib.einsum('xpq,pi,qj->xij', h10, mo_coeff[1].conj(), orbob)

        # for rotation part
        with mol.with_common_origin(gauge_orig):
            h01 = -mol.intor('int1e_cg_irxp', 3)
            h01a = lib.einsum('xpq,pi,qj->xij', h01, mo_coeff[0].conj(), orboa)
            h01b = lib.einsum('xpq,pi,qj->xij', h01, mo_coeff[1].conj(), orbob)

        s10a = numpy.zeros_like(h10a)
        s10b = numpy.zeros_like(h10b)
        mo10 = uhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ,
                                            (h10a,h10b), (s10a,s10b))[0]

        mag_para = numpy.einsum('xji,yji->xy', mo10[0].conj(), h01a)
        mag_para+= numpy.einsum('xji,yji->xy', mo10[1].conj(), h01b)
        mag_para = mag_para + mag_para.conj()

    mag_para = rhf_g._safe_solve(im, mag_para)
    # unit = hbar/mu_N, mu_N is nuclear magneton
    unit = -2 * nist.PROTON_MASS_AU
    return mag_para * unit


class RotationalGTensor(rhf_g.RotationalGTensor):
    '''Rotational g-tensors for UHF'''
    dia = dia
    para = para
    get_fock = uhf_nmr.get_fock

from pyscf import scf
scf.uhf.UHF.RotationalGTensor = lib.class_as_method(RotationalGTensor)


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

    mf = scf.UHF(mol).run()
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
    mf = scf.UHF(mol).run()
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(m[0,0] - -0.2805925799038227)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.UHF(mol).run()
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(lib.finger(m) - 0.09396805421224698)
