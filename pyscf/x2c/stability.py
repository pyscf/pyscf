#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

'''
Generate X2C-SCF response functions
'''

from pyscf.x2c import _response_functions  # noqa
from pyscf.scf.stability import dhf_stability

def x2chf_stability(mf, verbose=None, return_status=False):
    '''
    Stability analysis for X2C-HF/X2C-KS method.

    Args:
        mf : DHF or DKS object

    Kwargs:
        return_status: bool
            Whether to return `stable_i` and `stable_e`

    Returns:
        If return_status is False (default), the return value includes
        a new set of orbitals, which are more close to the stable condition.

        Else, another one boolean variable (indicating current status:
        stable or unstable) is returned.
    '''
    return dhf_stability(mf, verbose, return_status)
