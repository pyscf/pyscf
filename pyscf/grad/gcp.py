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
gradient of the gCP/SRB correction for HF and DFT
'''

import numpy as np
from pyscf.lib import logger
from pyscf.scf.gcp import check_gcp, _resolve_gcp, _method_of

def get_gcp(mf_grad, gcp=None, verbose=None):
    '''gradient of the gCP/SRB correction'''
    mf = mf_grad.base
    mol = mf.mol
    if not check_gcp(mf, gcp):
        return np.zeros([mol.natm,3])

    if gcp is None:
        gcp = getattr(mf, 'gcp', None)

    resolved = _resolve_gcp(_method_of(mf), gcp)
    if resolved is None:
        return np.zeros([mol.natm,3])
    gcp_method, gcp_basis = resolved

    try:
        from pyscf.dispersion import gcp as _gcp
    except ImportError:
        print('gCP not available. Install them with `pip install pyscf-dispersion`')
        raise

    logger.info(mf, 'Calc gCP correction with method %s, basis %s.',
                gcp_method, gcp_basis)
    gcp_model = _gcp.GCP(mol, method=gcp_method, basis=gcp_basis)
    res = gcp_model.get_counterpoise(grad=True)
    return res.get('gradient')

# Inject to Gradient
from pyscf import grad
grad.rhf.GradientsBase.get_gcp = get_gcp
