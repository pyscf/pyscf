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
Non-relativistic rotational g-tensor for DFT
'''


import time
from pyscf.prop.nmr import rks as rks_nmr
from pyscf.prop.rotational_gtensor import rhf as rhf_g
from pyscf.prop.magnetizability import rks as rks_mag
from pyscf.data import nist


def dia(magobj, gauge_orig=None):
    '''Part of rotational g-tensors. It is the direct second derivatives of
    the Lagrangian (corresponding to the zeroth order wavefunction).  Unit
    hbar/mu_N is not included.  This part may be different to the conventional
    dia-magnetic contributions of rotational g-tensors.
    '''
    if gauge_orig is None:
        mol = magobj.mol
        im, mass_center = rhf_g.inertia_tensor(mol)
        # Eq. (35) of JCP 105, 2804 (1996); DOI:10.1063/1.472143
        e2 = rks_mag.dia(magobj, gauge_orig)
        e2 -= rks_mag.dia(magobj, mass_center)
        e2 = rhf_g._safe_solve(im, e2)
        return -4 * nist.PROTON_MASS_AU * e2
    else:
        return rhf_g.dia(magobj, gauge_orig)


class RotationalGTensor(rhf_g.RotationalGTensor):
    '''Rotational g-tensors for RKS'''
    dia = dia
    get_fock = rks_nmr.get_fock
    solve_mo1 = rks_nmr.solve_mo1

from pyscf import lib
from pyscf import dft
dft.rks.RKS.RotationalGTensor = dft.rks_symm.RKS.RotationalGTensor = lib.class_as_method(RotationalGTensor)


if __name__ == '__main__':
    from pyscf import lib
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   .917
                  F  ,  0.   0.   0.
                  '''
    mol.basis = 'ccpvdz'
    mol.build()

    mf = dft.RKS(mol).run(xc='b3lyp')
    rotg = mf.RotationalGTensor()
    m = rotg.kernel()
    print(m[0,0] - 0.6944660741142765)

    rotg.gauge_orig = (0,0,.1)
    m = rotg.kernel()
    print(m[0,0] - 0.7362841753490392)

    mol.atom = '''C  ,  0.   0.   0.
                  O  ,  0.   0.   1.1283
                  '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = dft.RKS(mol).run(xc='bp86')
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(m[0,0] - -0.28690127897458007)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = dft.RKS(mol).run()
    rotg = RotationalGTensor(mf)
    m = rotg.kernel()
    print(lib.finger(m) - 0.08807921511593972)
