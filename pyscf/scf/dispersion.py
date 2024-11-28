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

import warnings
from functools import lru_cache
from pyscf.lib import logger
from pyscf import scf

# supported dispersion corrections
DISP_VERSIONS = ['d3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4']
XC_MAP = {'wb97m-d3bj': 'wb97m',
          'b97m-d3bj': 'b97m',
          'wb97x-d3bj': 'wb97x',
          'wb97m-v': 'wb97m',
          'b97m-v': 'b97m',
          'wb97x-v': 'wb97x'
          }

# special cases:
# - wb97x-d is not supported yet
# - wb97*-d3bj is wb97*-v with d3bj
# - wb97x-d3 is not supported yet
# - 3c method is not supported yet

# These xc functionals need special treatments
_white_list = {
    'wb97m-d3bj': ('wb97m-v', False, 'd3bj'),
    'b97m-d3bj': ('b97m-v', False, 'd3bj'),
    'wb97x-d3bj': ('wb97x-v', False, 'd3bj'),
}

# These xc functionals are not supported yet
_black_list = {
    'wb97x-d', 'wb97x-d3', 'wb97x_d', 'wb97x_d3',
    'wb97m-d3bj2b', 'wb97m-d3bjatm',
    'b97m-d3bj2b', 'b97m-d3bjatm',
}

@lru_cache(128)
def parse_dft(xc_code):
    '''
    Extract (xc, nlc, disp) from xc_code
    '''
    if not isinstance(xc_code, str):
        return xc_code, '', None
    method_lower = xc_code.lower()

    if method_lower in _black_list:
        raise NotImplementedError(f'{method_lower} is not supported yet.')

    if method_lower in _white_list:
        return _white_list[method_lower]

    if method_lower.endswith('-3c'):
        raise NotImplementedError('*-3c methods are not supported yet.')

    if '-d3' in method_lower or '-d4' in method_lower:
        xc, disp = method_lower.split('-')
    else:
        xc, disp = method_lower, None

    return xc, '', disp

@lru_cache(128)
def parse_disp(dft_method):
    '''Decode the disp parameters based on the xc code.
    Returns xc_code_for_dftd3, disp_version, with_3body

    Example: b3lyp-d3bj2b -> (b3lyp, d3bj, False)
             wb97x-d3bj   -> (wb97x, d3bj, False)
    '''
    if dft_method == 'hf':
        return 'hf', None, False

    dft_lower = dft_method.lower()
    xc, nlc, disp = parse_dft(dft_lower)
    if dft_lower in XC_MAP:
        xc = XC_MAP[dft_lower]

    if disp is None:
        return xc, None, False
    disp_lower = disp.lower()
    if disp_lower.endswith('2b'):
        return xc, disp_lower.replace('2b', ''), False
    elif disp_lower.endswith('atm'):
        return xc, disp_lower.replace('atm', ''), True
    else:
        return xc, disp_lower, False

def check_disp(mf, disp=None):
    '''Check whether to apply dispersion correction based on the xc attribute.
    If dispersion is allowed, return the DFTD3 disp version, such as d3bj,
    d3zero, d4.
    '''
    if disp is None:
        disp = mf.disp
    if disp == 0: # disp = False
        return False

    # To prevent mf.do_disp() triggering the SCF.__getattr__ method, do not use
    # method = getattr(mf, 'xc', 'hf').
    if isinstance(mf, scf.hf.KohnShamDFT):
        method = mf.xc
    else:
        # Set the disp method for both HF and MCSCF to 'hf'
        method = 'hf'
    disp_version = parse_disp(method)[1]

    if disp is None: # Using the disp version decoded from the mf.xc attribute
        if disp_version is None:
            return False
    elif disp_version is None: # Using the version specified by mf.disp
        disp_version = disp
    elif disp != disp_version:
        raise RuntimeError(f'mf.disp {disp} conflicts with mf.xc {method}')

    if disp_version not in DISP_VERSIONS:
        raise NotImplementedError
    return disp_version

def get_dispersion(mf, disp=None, with_3body=None, verbose=None):
    disp_version = check_disp(mf, disp)
    if not disp_version:
        return 0.

    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise

    mol = mf.mol
    method = getattr(mf, 'xc', 'hf')
    method, _, disp_with_3body = parse_disp(method)

    if with_3body is not None:
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
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')
