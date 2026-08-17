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
Hessian of the gCP/SRB correction for HF and DFT
'''

import numpy as np
from pyscf.lib import logger
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

    logger.info(mf, 'Calc gCP correction with method %s, basis %s.',
                gcp_method, gcp_basis)
    logger.warn(mf, 'gCP does not support analytical Hessian, using finite difference')
    coords = hessobj.mol.atom_coords()
    mol = mol.copy()
    eps = 1e-5
    for i in range(natm):
        for j in range(3):
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            gcp_model = _gcp.GCP(mol, method=gcp_method, basis=gcp_basis)
            res = gcp_model.get_counterpoise(grad=True)
            g1 = res.get('gradient')

            coords[i,j] -= 2.0*eps
            mol.set_geom_(coords, unit='Bohr')
            gcp_model = _gcp.GCP(mol, method=gcp_method, basis=gcp_basis)
            res = gcp_model.get_counterpoise(grad=True)
            g2 = res.get('gradient')

            coords[i,j] += eps
            h_gcp[i,:,j,:] = (g1 - g2)/(2.0*eps)
    return h_gcp

# Inject to Hessian class
from pyscf import hessian
hessian.rhf.HessianBase.get_gcp = get_gcp
