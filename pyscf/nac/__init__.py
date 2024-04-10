#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

"""
Analytical Nonadiabatic Coupling Vectors
============================

Simple usage::

    >>> from pyscf import gto, scf, mcscf, nac
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 2, 2).state_average([0.5, 0.5]).run()
    >>> mc_nac = nac.sacasscf.NonAdiabaticCouplings(mc)
    >>> mc_nac = mc.nac_method() # Also valid
    >>> mc_nac.kernel(state=(0,1), use_etfs=False)
"""

from . import sacasscf
