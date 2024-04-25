#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
dispersion correction for HF and DFT
'''

from pyscf.dft.rks import KohnShamDFT
from pyscf.dft import dft_parser

def get_dispersion(mf, disp_version=None):
    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise
    mol = mf.mol
    if isinstance(mf, KohnShamDFT):
        method = mf.xc
    else:
        method = 'hf'
    method, disp, with_3body = dft_parser.parse_dft(method)[2]

    # priority: args > mf.disp > dft_parser
    if disp_version is None:
        disp_version = disp
        # dispersion version can be customized via mf.disp
        if hasattr(mf, 'disp') and mf.disp is not None:
            disp_version = mf.disp
    if disp_version is None:
        return 0.0

    # 3-body contribution can be disabled with mf.disp_with_3body
    if hasattr(mf, 'disp_with_3body') and mf.disp_with_3body is not None:
        with_3body = mf.disp_with_3body

    # for dftd3
    if disp_version[:2].upper() == 'D3':
        d3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        res = d3_model.get_dispersion()
        e_d3 = res.get('energy')
        mf.scf_summary['dispersion'] = e_d3
        return e_d3

    # for dftd4
    elif disp_version[:2].upper() == 'D4':
        d4_model = dftd4.DFTD4Dispersion(mol, xc=method, atm=with_3body)
        res = d4_model.get_dispersion()
        e_d4 = res.get('energy')
        mf.scf_summary['dispersion'] = e_d4
        return e_d4
    else:
        raise RuntimeError(f'dipersion correction: {disp_version} is not supported.')

# Inject to SCF class
from pyscf import scf
scf.hf.SCF.get_dispersion = get_dispersion
