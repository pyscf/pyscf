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
See also pyscf/hessian/uhf.py
'''

from pyscf.hessian import uhf as uhf_hess
from pyscf.prop.freq.rhf import kernel


class Frequency(uhf_hess.Hessian):
    def __init__(self, mf):
        self.nroots = None
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-3
        uhf_hess.Hessian.__init__(self, mf)

    kernel = kernel

Freq = Frequency


if __name__ == '__main__':
    import numpy
    from pyscf import gto
    from pyscf import scf
    from pyscf import lib

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.spin = 2
    mol.unit = 'B'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = uhf_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.075282233847343283)
    hop = Freq(mf).gen_hop()[0]
    print(lib.finger(hop(x)) - -0.075282233847343283)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).set(nroots=1).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
