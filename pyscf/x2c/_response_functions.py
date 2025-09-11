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

from pyscf.x2c import x2c
from pyscf.scf._response_functions import _gen_ghf_response

def _gen_x2chf_response(mf, mo_coeff=None, mo_occ=None,
                        with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of X2C-HF response function
    and density matrices.
    '''
    return _gen_ghf_response(mf, mo_coeff, mo_occ, with_j, hermi, max_memory,
                             with_nlc=with_nlc)

x2c.UHF.gen_response = _gen_x2chf_response
x2c.RHF.gen_response = _gen_x2chf_response
