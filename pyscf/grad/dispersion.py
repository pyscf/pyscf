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
from pyscf.dft.rks import KohnShamDFT

def get_dispersion(mf_grad, disp_version=None, with_3body=False):
    '''gradient of dispersion correction for RHF/RKS'''
    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise
    mf = mf_grad.base
    mol = mf.mol
    # priority: args > mf.disp
    if disp_version is None:
        if hasattr(mf, 'disp'): disp_version = mf.disp

    if disp_version is None:
        return numpy.zeros([mol.natm,3])

    if isinstance(mf, KohnShamDFT):
        method = mf.xc
    else:
        method = 'hf'

    # overwrite method if method exists in disp_version
    if ',' in disp_version:
        disp_version, method = disp_version.split(',')

    if with_3body is None:
        # 3-body contribution can be disabled with mf.disp_with_3body
        if hasattr(mf, 'disp_with_3body'):
            with_3body = mf.disp_with_3body
        else:
            with_3body = False

    if hasattr(mf, 'nlc') and mf.nlc not in [False, '']:
        import warnings
        warnings.warn('NLC is incompatiable with dispersion correction.')

    if disp_version[:2].upper() == 'D3':
        d3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        res = d3_model.get_dispersion(grad=True)
        g_d3 = res.get('gradient')
        return g_d3
    elif disp_version[:2].upper() == 'D4':
        d4_model = dftd4.DFTD4Dispersion(mol, xc=method, atm=with_3body)
        res = d4_model.get_dispersion(grad=True)
        g_d4 = res.get('gradient')
        return g_d4
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to Gradient
from pyscf import grad
grad.rhf.GradientsBase.get_dispersion = get_dispersion
