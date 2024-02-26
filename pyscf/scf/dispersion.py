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


import numpy
from pyscf.scf.hf import KohnShamDFT

# dispersion correction for these xc functionals are replacing nlc
dftd3_xc_map = {
    'wb97m-v': 'wb97m',
    'b97m-v': 'b97m',
    'wb97x-v': 'wb97x'
}

def get_dispersion(mf, disp_version=None, with_3body=False):
    if disp_version is None:
        disp_version = mf.disp
    mol = mf.mol
    if disp_version is None:
        return 0.0
    if hasattr(mf, 'with_3body'):
        with_3body = mf.disp_with_3body
    if isinstance(mf, KohnShamDFT):
        method = mf.xc
    else:
        method = 'hf'

    # use xc name defined in dftd3 for special cases
    if method in dftd3_xc_map:
        method = dftd3_xc_map[method]

    # for dftd3
    if disp_version[:2].upper() == 'D3':
        try:
            import dftd3.pyscf as disp
        except ImportError:
            raise ImportError("\n \
cannot find dftd3 in the current environment.\n \
please install dftd3 via \n \
**************************************\n\
        pip3 install dftd3 \n \
**************************************")

        d3 = disp.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        e_d3, _ = d3.kernel()
        mf.scf_summary['dispersion'] = e_d3
        return e_d3

    # for dftd4
    elif disp_version[:2].upper() == 'D4':
        try:
            import dftd4.pyscf as disp
        except ImportError:
            raise ImportError("\n \
cannot find dftd4 in the current environment. \n \
please install dftd4 via \n \
***************************************\n \
        pip3 install dftd4 \n \
***************************************")

        d4 = disp.DFTD4Dispersion(mol, xc=method, atm=with_3body)
        e_d4, _ = d4.kernel()
        mf.scf_summary['dispersion'] = e_d4
        return e_d4
    else:
        raise RuntimeError(f'dipersion correction: {disp_version} is not supported.')

# Inject to SCF class
from pyscf import scf
scf.hf.SCF.get_dispersion = get_dispersion
