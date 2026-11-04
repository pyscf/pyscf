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
# Author: Shirong Wang <srwang20@fudan.edu.cn>
#

'''
Hessian of the gCP/SRB correction for HF and DFT
'''

import numpy as np
from pyscf.lib import logger
from pyscf.scf.dispersion import _get_dispersion_mol
from pyscf.scf.gcp import check_gcp, _resolve_gcp, _method_of

def get_gcp(hessobj, gcp=None):
    mf = hessobj.base
    mol = mf.mol
    natm = mol.natm
    h_gcp = np.zeros([natm,natm,3,3])
    if not check_gcp(mf, gcp):
        return h_gcp

    if gcp is None:
        gcp = getattr(mf, 'gcp', None)

    resolved = _resolve_gcp(_method_of(mf), gcp)
    if resolved is None:
        return h_gcp
    gcp_method, gcp_basis = resolved

    try:
        from pyscf.dispersion import gcp as _gcp
    except ImportError:
        print('gCP not available. Install them with `pip install pyscf-dispersion`')
        raise

    mol, atom_idx = _get_dispersion_mol(mol)
    nreal = len(atom_idx)
    if nreal == 0:
        return h_gcp

    logger.info(mf, 'Calc gCP correction with method %s, basis %s.',
                gcp_method, gcp_basis)
    gcp_model = _gcp.GCP(mol, method=gcp_method, basis=gcp_basis)
    if not hasattr(gcp_model, 'get_hessian'):
        raise RuntimeError('gCP analytical Hessian requires pyscf-dispersion >= 1.6.0. '
                           'Upgrade with `pip install -U pyscf-dispersion`.')
    res = gcp_model.get_hessian()
    h_gcp[np.ix_(atom_idx,atom_idx)] = res['hessian'].reshape(nreal,3,nreal,3).transpose(0,2,1,3)
    return h_gcp

# Inject to Hessian class
from pyscf import hessian
hessian.rhf.HessianBase.get_gcp = get_gcp
