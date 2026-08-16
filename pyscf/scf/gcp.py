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

'''
short-range basis correction (gCP/SRB) for HF and DFT

The geometrical counterpoise correction (gCP) is a short-range basis
correction (SRB) used by composite methods like B97-3c and r2SCAN-3c.
The correction is evaluated with the GCP module of simple-dftd3, which is
shipped with the pyscf-dispersion package.
'''

from pyscf.lib import logger
from pyscf import scf

try:
    from pyscf.dispersion import gcp as _gcp
except ImportError:
    _gcp = None

# xc name -> (gcp method, gcp basis).  The gcp method and basis keys follow
# the parameter table of simple-dftd3.  The basis here is the *basis key* of
# the gCP parameter table, not the molecular basis set name.
GCP_METHODS = {
    'b97-3c': ('b973c', 'def2mtzvp'),
    'b97_3c': ('b973c', 'def2mtzvp'),
    'gga_xc_b97_3c': ('b973c', 'def2mtzvp'),
    'r2scan-3c': ('r2scan3c', 'def2mtzvpp'),
    'r2scan_3c': ('r2scan3c', 'def2mtzvpp'),
    # wB97X-3c is defined without the gCP correction (its gCP energy is zero)
    'wb97x-3c': ('wb97x3c', None),
    'wb97x_3c': ('wb97x3c', None),
}

def parse_gcp(xc_code):
    '''Extract the (gcp method, gcp basis) from the xc code.

    Returns None if the xc code is not a composite 3c method with a gCP
    correction.
    '''
    if not isinstance(xc_code, str):
        return None
    method_lower = xc_code.lower()
    if method_lower in GCP_METHODS:
        return GCP_METHODS[method_lower]
    return None

def check_gcp(mf, gcp=None):
    '''Check if the gCP (short-range basis) correction should be applied.

    Args:
        mf : SCF object
        gcp : bool or str, optional
            If None, uses mf.gcp.
            If False, the correction is disabled.

    Returns:
        bool: True if the gCP correction is enabled and the method supports it.
    '''
    if gcp is None:
        gcp = getattr(mf, 'gcp', None)
    if gcp is False or gcp == 0:
        return False

    # To prevent mf.do_gcp() triggering the SCF.__getattr__ method, do not use
    # method = getattr(mf, 'xc', 'hf').
    if isinstance(mf, scf.hf.KohnShamDFT):
        method = mf.xc
    else:
        # Set the gcp method for both HF and MCSCF to 'hf'
        method = 'hf'
    return parse_gcp(method) is not None

def get_gcp(mf, gcp=None, verbose=None):
    '''
    Calculate the gCP (short-range basis correction) energy.

    Args:
        mf : SCF object
        gcp : bool or str, optional
            If None, uses mf.gcp.
            If False, the correction is disabled.
        verbose : int, optional

    Returns:
        float: the gCP correction energy in Hartree.
    '''
    if not check_gcp(mf, gcp):
        return 0.

    if gcp is None:
        gcp = getattr(mf, 'gcp', None)

    method = getattr(mf, 'xc', 'hf')
    gcp_method, gcp_basis = parse_gcp(method)
    if gcp_method is None:
        return 0.

    if _gcp is None:
        raise RuntimeError('gCP not available. Install them with '
                           '`pip install pyscf-dispersion`')

    mol = mf.mol
    logger.info(mf, 'Calc gCP correction with method %s, basis %s.',
                gcp_method, gcp_basis)
    gcp_model = _gcp.GCP(mol, method=gcp_method, basis=gcp_basis)
    res = gcp_model.get_counterpoise()
    e_gcp = res.get('energy')
    mf.scf_summary['gcp'] = e_gcp
    return e_gcp
