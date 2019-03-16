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
Non-relativistic nuclear spin-rotation tensors for UKS
'''

from pyscf.prop.nsr import uhf as uhf_nsr
from pyscf.prop.nmr import uks as uks_nmr

class NSR(uhf_nsr.NSR):
    '''Nuclear-spin rotation tensors for UKS'''
    get_fock = uks_nmr.get_fock
    solve_mo1 = uks_nmr.solve_mo1

from pyscf import lib
from pyscf import dft
dft.uks.UKS.NSR = dft.uks_symm.UKS.NSR = lib.class_as_method(NSR)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    from pyscf import lib
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   0.917
                  f  ,  0.   0.   0.
                  '''
    mol.basis = 'dzp'
    mol.build()

    mf = dft.UKS(mol).run(xc='b3lyp')
    rotg = mf.NSR()
    m = rotg.kernel()
    print(m[1,0,0] - -301.49652448221707)
    print(lib.finger(m) - 28.57893850199683)

    rotg.gauge_orig = (0,0,.917/lib.param.BOHR)
    m = rotg.kernel()
    print(m[0,0,0] - 277.173892536396)
    print(lib.finger(m) - 96.92616726791988)

    mol.atom = '''C  ,  0.   0.   0.
                  O  ,  0.   0.   1.1283
                  '''
    mol.basis = 'ccpvdz'
    mol.nucprop = {'C': {'mass': 13}}
    mol.build()
    mf = dft.UKS(mol).run(xc='bp86')
    rotg = NSR(mf)
    m = rotg.kernel()
    print(m[0,0,0] - -32.23298865237305)
    print(lib.finger(m) - -11.278686427378966)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = dft.UKS(mol).run()
    rotg = NSR(mf)
    m = rotg.kernel()
    print(lib.finger(m) - -66.94250282318671)

