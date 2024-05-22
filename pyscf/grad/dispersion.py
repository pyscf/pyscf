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
gradient of dispersion correction for HF and DFT
'''

import numpy
from pyscf.lib import logger
from pyscf.scf.dispersion import parse_disp, DISP_VERSIONS

def get_dispersion(mf_grad, disp=None, with_3body=None, verbose=None):
    '''gradient of DFTD3/DFTD4 dispersion correction'''
    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise
    mf = mf_grad.base
    mol = mf.mol

    # The disp method for both HF and MCSCF is set to 'hf'
    method = getattr(mf, 'xc', 'hf')
    method, disp_version, disp_with_3body = parse_disp(method)

    # Check conflicts
    if mf.disp is not None and disp_version is not None:
        if mf.disp != disp_version:
            raise RuntimeError('disp is conflict with xc')
    if mf.disp is not None:
        disp_version = mf.disp
    if disp is not None:
        disp_version = disp
    if with_3body is not None:
        with_3body = disp_with_3body

    if disp_version is None:
        return 0.0

    if disp_version not in DISP_VERSIONS:
        raise NotImplementedError

    if disp_version[:2].upper() == 'D3':
        logger.info(mf, "Calc dispersion correction with DFTD3.")
        logger.info(mf, f"Parameters: xc={method}, version={disp_version}, atm={with_3body}")
        d3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        res = d3_model.get_dispersion(grad=True)
        g_d3 = res.get('gradient')
        return g_d3
    elif disp_version[:2].upper() == 'D4':
        logger.info(mf, "Calc dispersion correction with DFTD4.")
        logger.info(mf, f"Parameters: xc={method}, atm={with_3body}")
        d4_model = dftd4.DFTD4Dispersion(mol, xc=method, atm=with_3body)
        res = d4_model.get_dispersion(grad=True)
        g_d4 = res.get('gradient')
        return g_d4
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to Gradient
from pyscf import grad
grad.rhf.GradientsBase.get_dispersion = get_dispersion
