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

from pyscf.lib import logger

# supported dispersion corrections
DISP_VERSIONS = ['d3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4']

def parse_disp(disp):
    '''Decode the disp parameter for 3-body correction'''
    disp_lower = disp.lower()
    if disp_lower.endswith('2b'):
        return disp_lower.replace('2b', ''), False
    elif disp_lower.endswith('atm'):
        return disp_lower.replace('atm', ''), True
    else:
        return disp_lower, False

def get_dispersion(mf, disp_version=None, with_3body=None, verbose=None):
    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise
    mol = mf.mol

    # priority: args > mf.disp
    if disp_version is None:
        if hasattr(mf, 'disp'): disp_version = mf.disp

    if disp_version is None:
        return 0.0

    # The disp method for both HF and MCSCF is set to 'hf'
    method = getattr(mf, 'xc', 'hf')

    # overwrite method if method exists in disp_version
    if ',' in disp_version:
        disp_version, method = disp_version.split(',')

    disp_version, disp_with_3body = parse_disp(disp_version)

    if disp_version not in DISP_VERSIONS:
        raise NotImplementedError

    if with_3body is None:
        with_3body = disp_with_3body

    # for dftd3
    if disp_version[:2].upper() == 'D3':
        logger.info(mf, "Calc dispersion correction with DFTD3.")
        logger.info(mf, f"Parameters: xc={method}, version={disp_version}, atm={with_3body}")
        d3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        res = d3_model.get_dispersion()
        e_d3 = res.get('energy')
        mf.scf_summary['dispersion'] = e_d3
        return e_d3

    # for dftd4
    elif disp_version[:2].upper() == 'D4':
        logger.info(mf, "Calc dispersion correction with DFTD4.")
        logger.info(mf, f"Parameters: xc={method}, atm={with_3body}")
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
