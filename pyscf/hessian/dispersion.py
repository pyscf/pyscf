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
Hessian of dispersion correction for HF and DFT
'''


import numpy
from pyscf.dft.rks import KohnShamDFT
from pyscf.dft import dft_parser

def get_dispersion(hessobj, disp_version=None, with_3body=False):
    mf = hessobj.base
    mol = mf.mol
    if isinstance(mf, KohnShamDFT):
        method = mf.xc
    else:
        method = 'hf'
    method, disp, with_3body = dft_parser.parse_dft(method)[2]

    # priority: args > mf.disp > dft_parser
    if disp_version is None:
        disp_version = disp
        # dispersion version can be customized via mf.disp
        if hasattr(mf, 'disp') and mf.disp is not None:
            disp_version = mf.disp

    natm = mol.natm
    h_disp = numpy.zeros([natm,natm,3,3])
    if disp_version is None:
        return h_disp

    # 3-body contribution can be disabled with mf.disp_with_3body
    if hasattr(mf, 'disp_with_3body') and mf.disp_with_3body is not None:
        with_3body = mf.disp_with_3body

    if mf.disp[:2].upper() == 'D3':
        import dftd3.pyscf as disp
        coords = hessobj.mol.atom_coords()
        mol = mol.copy()
        eps = 1e-5
        for i in range(natm):
            for j in range(3):
                coords[i,j] += eps
                mol.set_geom_(coords, unit='Bohr')
                d3 = disp.DFTD3Dispersion(mol, xc=method, version=mf.disp, atm=with_3body)
                _, g1 = d3.kernel()

                coords[i,j] -= 2.0*eps
                mol.set_geom_(coords, unit='Bohr')
                d3 = disp.DFTD3Dispersion(mol, xc=method, version=mf.disp, atm=with_3body)
                _, g2 = d3.kernel()

                coords[i,j] += eps
                h_disp[i,:,j,:] = (g1 - g2)/(2.0*eps)
            return h_disp

    elif mf.disp[:2].upper() == 'D4':
        import dftd4.pyscf as disp
        coords = hessobj.mol.atom_coords()
        mol = mol.copy()
        eps = 1e-5
        for i in range(natm):
            for j in range(3):
                coords[i,j] += eps
                mol.set_geom_(coords, unit='Bohr')
                d4 = disp.DFTD4Dispersion(mol, xc=method, atm=with_3body)
                _, g1 = d4.kernel()

                coords[i,j] -= 2.0*eps
                mol.set_geom_(coords, unit='Bohr')
                d4 = disp.DFTD4Dispersion(mol, xc=method, atm=with_3body)
                _, g2 = d4.kernel()

                coords[i,j] += eps
                h_disp[i,:,j,:] = (g1 - g2)/(2.0*eps)

        return h_disp
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to SCF class
from pyscf import hessian
hessian.rhf.HessianBase.get_dispersion = get_dispersion
