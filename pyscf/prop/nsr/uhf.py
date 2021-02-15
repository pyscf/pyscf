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
Non-relativistic nuclear spin-rotation tensors for UHF
'''

import numpy
from pyscf.prop.nsr import rhf as rhf_nsr
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.data import nist


def dia(nsrobj, gauge_orig=None, shielding_nuc=None, dm0=None):
    '''Diamagnetic part of NSR tensors.
    '''
    if dm0 is None: dm0 = nsrobj._scf.make_rdm1()
    if not (isinstance(dm0, numpy.ndarray) and dm0.ndim == 2):
        dm0 = dm0[0] + dm0[1]
    return rhf_nsr.dia(nsrobj, gauge_orig, shielding_nuc, dm0)


def para(nsrobj, mo10=None, mo_coeff=None, mo_occ=None, shielding_nuc=None):
    '''Paramagnetic part of NSR shielding tensors.
    '''
    if shielding_nuc is None: shielding_nuc = nsrobj.shielding_nuc

    # The first order Hamiltonian for rotation part is the same to the
    # first order Hamiltonian for magnetic field except a factor of 2.
    nsr_para = uhf_nmr.para(nsrobj, mo10, mo_coeff, mo_occ,
                            shielding_nuc)[0] * 2

    mol = nsrobj.mol
    im, mass_center = rhf_nsr.inertia_tensor(mol)
    nsr_para = rhf_nsr._safe_solve(im, nsr_para)
    unit = rhf_nsr._atom_gyro_list(mol)[shielding_nuc] * nist.ALPHA**2
    return numpy.einsum('ixy,i->ixy', nsr_para, unit)


class NSR(rhf_nsr.NSR):
    '''Nuclear-spin rotation tensors for UHF'''
    dia = dia
    para = para
    get_fock = uhf_nmr.get_fock
    solve_mo1 = uhf_nmr.solve_mo1

from pyscf import lib
from pyscf import scf
scf.uhf.UHF.NSR = lib.class_as_method(NSR)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import lib
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   0.917
                  f  ,  0.   0.   0.
                  '''
    mol.basis = 'dzp'
    mol.build()

    mf = scf.UHF(mol).run()
    rotg = mf.NSR()
    m = rotg.kernel()
    print(m[1,0,0] - -274.44236312671563)
    print(lib.finger(m) - 26.68194604747653)

    rotg.gauge_orig = (0,0,.917/lib.param.BOHR)
    m = rotg.kernel()
    print(m[0,0,0] - 274.9496549281562)
    print(lib.finger(m) - 96.22450556074686)

    mol.atom = '''C  ,  0.   0.   0.
                  O  ,  0.   0.   1.1283
                  '''
    mol.basis = 'ccpvdz'
    mol.nucprop = {'C': {'mass': 13}}
    mol.build()
    mf = scf.UHF(mol).run()
    rotg = NSR(mf)
    m = rotg.kernel()
    print(m[0,0,0] - -34.6383402509185)
    print(lib.finger(m) - -12.115914020554674)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.UHF(mol).run()
    rotg = NSR(mf)
    m = rotg.kernel()
    print(lib.finger(m) - -66.68587704814556)

