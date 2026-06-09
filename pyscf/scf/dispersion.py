#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# modified by Jiashu Liang <jsliang25@gmail.com>
#

'''
dispersion correction for HF and DFT
'''

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
    'wb97x-3c': ('wb97x-v', False, 'd4:wb97x-3c'),
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
        if method_lower == "wb97x-3c":
            return _white_list[method_lower]
        raise NotImplementedError('Only wb97x-3c is supported for now. Other 3c methods are not supported yet.')

    if '-d3' in method_lower or '-d4' in method_lower:
        xc, disp = method_lower.split('-')
    else:
        xc, disp = method_lower, None

    return xc, '', disp

@lru_cache(128)
def parse_disp(dft_method=None, disp=None):
    '''Decode the disp parameters based on the xc code.

    The logic for determining the dispersion parameters is as follows:
    1. If `disp` is specified, it takes precedence.
       - If `disp` contains ':', it is parsed as `disp_version:method`.
       - Otherwise, the method is derived from `dft_method`.
    2. If `disp` is not specified, the dispersion settings are inferred from `dft_method`.

    The `with_3body` flag is determined by the dispersion version suffix:
    - '2b' suffix -> False (2-body only)
    - 'atm' suffix -> True (Axilrod-Teller-Muto 3-body term)
    - 'd4' -> True (D4 always includes 3-body)
    - 'd3' (without suffix) -> False

    Args:
        dft_method (str): The DFT method name (e.g., 'b3lyp', 'wb97x-d3bj').
        disp (str): Explicit dispersion version (e.g., 'd3bj', 'd3bjatm').

    Returns:
        tuple: (disp_method, disp_version, with_3body)

    Examples:
        >>> parse_disp('b3lyp-d3bj2b')
        ('b3lyp', 'd3bj', False)
        >>> parse_disp('b3lyp-d3bjatm')
        ('b3lyp', 'd3bj', True)
        >>> parse_disp('wb97x-d3bj')
        ('wb97x', 'd3bj', False)
        >>> parse_disp(None, 'd4:wb97x-3c')
        ('wb97x-3c', 'd4', True)
    '''

    # If anything not specified, return None
    if dft_method is None and disp is None:
        return None, None, False

    def process_3body(disp_version):
        if not disp_version:
            return disp_version, False
        if disp_version.endswith('2b'):
            return disp_version[:-2], False
        elif disp_version.endswith('atm'):
            return disp_version[:-3], True
        elif 'd4' in disp_version:
            return disp_version, True
        elif 'd3' in disp_version:
            return disp_version, False
        else:
            raise ValueError(f"Unknown dispersion version {disp_version} in parse_disp.")

    if dft_method is not None:
        dft_lower = dft_method.lower()
        xc, _, disp_from_dft = parse_dft(dft_lower)
        if xc in XC_MAP:
            xc = XC_MAP[xc]

    # Use disp if specified
    # returned method will be the latter part of disp if disp is a string with colon, otherwise, use xc
    if disp is not None:
        if ":" in disp:
            disp_version, method = disp.split(':')
            disp_version, with_3body = process_3body(disp_version)
            return method, disp_version, with_3body
        elif dft_method is not None:
            disp, with_3body = process_3body(disp)
            return xc, disp, with_3body
        else:
            raise ValueError(f"the method used in dispersion {disp} is not specified.")

    # otherwise, use disp_from_dft
    if disp_from_dft is None:
        return None, None, False

    if ":" in disp_from_dft:
        disp_version, method = disp_from_dft.split(':')
        disp_version, with_3body = process_3body(disp_version)
        return method, disp_version, with_3body

    disp_from_dft, with_3body = process_3body(disp_from_dft)
    return xc, disp_from_dft, with_3body


def check_disp(mf, disp=None):
    '''Check if dispersion correction should be applied and if the version is supported.

    The function determines the dispersion method from the SCF object (`mf`) or the
    explicit `disp` argument. It then verifies if the determined dispersion version
    is supported in `DISP_VERSIONS`.

    Args:
        mf (scf.hf.SCF): The SCF object (HF or DFT).
        disp (str or bool, optional): Dispersion version to check.
            If None, uses `mf.disp`.
            If False, returns False immediately.

    Returns:
        bool: True if dispersion is enabled and supported.
              False if dispersion is disabled (disp=False) or not specified/implied.

    Raises:
        ValueError: If the dispersion version is not supported.
    '''
    if disp is None:
        disp = getattr(mf, 'disp', None)
    if disp is False or disp == 0:
        return False

    # To prevent mf.do_disp() triggering the SCF.__getattr__ method, do not use
    # method = getattr(mf, 'xc', 'hf').
    if isinstance(mf, scf.hf.KohnShamDFT):
        method = mf.xc
    else:
        # Set the disp method for both HF and MCSCF to 'hf'
        method = 'hf'
    disp_version = parse_disp(method, disp)[1]

    if disp_version is None:
        return False

    if disp_version not in DISP_VERSIONS:
        raise ValueError(f"Unknown dispersion version {disp_version}.")
    return True

def get_dispersion(mf, disp=None, with_3body=None, verbose=None):
    '''
    Calculate the dispersion correction energy.

    Args:
        mf : SCF object
            The SCF object.
        disp : str, optional
            The dispersion correction version. Default is None.
            Format examples: "d3", "d3bj", "d4", "d3bj2b", "d3bjatm", "d4:wb97x-3c", etc.
            Note: In "d4:wb97x-3c", the latter part follows the method id of simple-dftd3 and dftd4 repo.
        with_3body : bool, optional
            Whether to include the 3-body term. Default is None.
        verbose : int, optional
            The verbose level. Default is None.

    Returns:
        float
            The dispersion correction energy.

    Note:
        Priority of `disp` and `with_3body`:
        1. Function arguments (disp, with_3body)
        2. mf.disp (if available)
        3. mf.xc (parsed from the functional name)
    '''
    if not check_disp(mf, disp):
        return 0.

    if disp is None:
        disp = getattr(mf, 'disp', None)

    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise

    dft_method = getattr(mf, 'xc', 'hf')
    method, disp_version, disp_with_3body = parse_disp(dft_method, disp)
    if with_3body is None:
        with_3body = disp_with_3body

    mol = mf.mol

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
