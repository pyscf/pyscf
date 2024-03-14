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
from pyscf.scf.hf import KohnShamDFT

def get_dispersion(mf_grad, disp_version=None):
    '''gradient of dispersion correction for RHF/RKS'''
    if disp_version is None:
        disp_version = mf_grad.base.disp
    mol = mf_grad.base.mol
    disp_version = mf_grad.base.disp
    if disp_version is None:
        return numpy.zeros([mol.natm,3])

    if isinstance(mf_grad.base, KohnShamDFT):
        method = mf_grad.base.xc
    else:
        method = 'hf'

    if disp_version[:2].upper() == 'D3':
        # raised error in SCF module, assuming dftd3 installed
        import dftd3.pyscf as disp
        d3 = disp.DFTD3Dispersion(mol, xc=method, version=disp_version)
        _, g_d3 = d3.kernel()
        return g_d3
    elif disp_version[:2].upper() == 'D4':
        from pyscf.data.elements import charge
        atoms = numpy.array([ charge(a[0]) for a in mol._atom])
        coords = mol.atom_coords()
        from dftd4.interface import DampingParam, DispersionModel
        model = DispersionModel(atoms, coords)
        res = model.get_dispersion(DampingParam(method=method), grad=True)
        return res.get("gradient")
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to Gradient
from pyscf import grad
grad.rhf.GradientsBase.get_dispersion = get_dispersion
