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
Non-relativistic RKS spin-spin coupling (SSC) constants
'''

# RHF and RKS have the same code for SSC tensors
from pyscf.prop.ssc.rhf import SpinSpinCoupling, SSC
from pyscf.prop.ssc.rhf import make_dso, make_pso

if __name__ == '__main__':
    from pyscf import gto, lib, dft
    mol = gto.M(atom='''
                O 0 0      0
                H 0 -0.757 0.587
                H 0  0.757 0.587''',
                basis='6-31g', verbose=3)

    mf = dft.RKS(mol).set(xc='b3lyp').run()
    ssc = mf.SSC()
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()
    print(lib.finger(jj)*1e8 - -0.33428832201108766)
