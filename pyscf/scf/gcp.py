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
gCP/SRB correction for HF and DFT

The composite 3c methods (B97-3c, r2SCAN-3c) include a basis set
incompleteness correction.  Depending on the method parameters, the GCP
module of simple-dftd3 evaluates the original geometrical counterpoise
correction (gCP, Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012))
and/or the short-range basis correction (SRB) variant: B97-3c uses the
SRB variant (in addition to the gCP potential), while r2SCAN-3c uses the
gCP potential with refitted parameters.  Both are distinct corrections
exposed through the same interface by the pyscf-dispersion package.
Following that interface, the correction is referred to as gCP in the
attribute and method names below (mf.gcp, do_gcp, get_gcp).
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
}

def parse_gcp(xc_code):
    '''Extract the (gcp method, gcp basis) from the xc code.

    Returns None if the xc code is not a composite 3c method with a
    gCP/SRB correction.
    '''
    if not isinstance(xc_code, str):
        return None
    method_lower = xc_code.lower()
    if method_lower in GCP_METHODS:
        return GCP_METHODS[method_lower]
    return None

def _resolve_gcp(method, gcp):
    '''Resolve the (gcp method, gcp basis) for a gCP request.

    An explicit gCP request is given as a string, following the `disp`
    convention: a gCP method key of the simple-dftd3 parameter table
    (e.g. 'b973c', 'r2scan3c') or 'method:basis' to also select the basis
    key (e.g. 'b973c:def2mtzvp').  Otherwise the method is derived from
    the xc code `method`.  Returns None if no gCP/SRB correction applies.
    '''
    if isinstance(gcp, str):
        method_, _, basis = gcp.partition(':')
        method_ = method_.lower()
        if basis:
            basis = basis.lower()
        else:
            basis = None
        return (method_, basis) if method_ else None
    return parse_gcp(method)

def _method_of(mf):
    # To prevent mf.do_gcp() triggering the SCF.__getattr__ method, do not
    # use method = getattr(mf, 'xc', 'hf').
    if isinstance(mf, scf.hf.KohnShamDFT):
        return mf.xc
    else:
        # Set the gcp method for both HF and MCSCF to 'hf'
        return 'hf'

def check_gcp(mf, gcp=None):
    '''Check if the gCP/SRB correction should be applied.

    Args:
        mf : SCF object
        gcp : bool or str, optional
            If None, uses mf.gcp.
            If False, the correction is disabled.
            If a string, it is an explicit gCP method key of the
            simple-dftd3 parameter table (e.g. 'b973c', 'r2scan3c'), or
            'method:basis' to also select the basis key.

    Returns:
        bool: True if the gCP/SRB correction is enabled.
    '''
    if gcp is None:
        gcp = getattr(mf, 'gcp', None)
    if gcp is False or gcp == 0:
        return False
    if isinstance(gcp, str):
        # An explicit customized gCP request
        return bool(gcp)
    return _resolve_gcp(_method_of(mf), gcp) is not None

def get_gcp(mf, gcp=None, verbose=None):
    '''
    Calculate the gCP/SRB correction energy.

    Args:
        mf : SCF object
        gcp : bool or str, optional
            If None, uses mf.gcp.
            If False, the correction is disabled.
            If a string, it is an explicit gCP method key of the
            simple-dftd3 parameter table (e.g. 'b973c', 'r2scan3c'), or
            'method:basis' to also select the basis key.
        verbose : int, optional

    Returns:
        float: the gCP/SRB correction energy in Hartree.
    '''
    if not check_gcp(mf, gcp):
        return 0.

    if gcp is None:
        gcp = getattr(mf, 'gcp', None)

    resolved = _resolve_gcp(_method_of(mf), gcp)
    if resolved is None:
        return 0.
    gcp_method, gcp_basis = resolved

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
