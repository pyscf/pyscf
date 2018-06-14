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

import numpy as np
import sys, re
import pyscf
import pyscf.dft
from  pyscf import gto, rt
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

def TestTDHF():
    """
    Tests Basic Propagation Functionality. TDDFT
    """
    prm = '''
    Model	TDHF
    Method	MMUT
    dt	0.02
    MaxIter	100
    ExDir	1.0
    EyDir	1.0
    EzDir	1.0
    FieldAmplitude	0.01
    FieldFreq	0.9202
    ApplyImpulse	1
    ApplyCw		0
    StatusEvery	10
    '''
    geom = """
    H 0. 0. 0.
    H 0. 0. 0.9
    H 2.0 0.  0
    H 2.0 0.9 0
    """
    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    ks = pyscf.dft.RKS(mol)
    ks.xc='HF'
    ks.kernel()
    aprop = rt.tdscf.RTTDSCF(ks,prm,output)
    return
TestTDHF()
