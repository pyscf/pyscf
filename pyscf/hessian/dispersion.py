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


import numpy as np
from pyscf.lib import logger
from pyscf.scf.dispersion import check_disp, parse_disp, _make_d4_model, _get_dispersion_mol

def get_dispersion(hessobj, disp=None, with_3body=None):
    mf = hessobj.base
    mol = mf.mol
    natm = mol.natm
    h_disp = np.zeros([natm,natm,3,3])
    if not check_disp(mf, disp):
        return h_disp

    if disp is None:
        disp = getattr(mf, 'disp', None)

    try:
        from pyscf.dispersion import dftd3, dftd4
    except ImportError:
        print('dftd3 and dftd4 not available. Install them with `pip install pyscf-dispersion`')
        raise

    method = getattr(mf, 'xc', 'hf')
    method, disp_version, disp_with_3body = parse_disp(method, disp)

    if with_3body is None:
        with_3body = disp_with_3body

    mol, atom_idx = _get_dispersion_mol(mol)
    nreal = len(atom_idx)
    if nreal == 0:
        return h_disp

    if disp_version[:2].upper() == 'D3':
        logger.info(mf, "Calc dispersion correction with DFTD3.")
        logger.info(mf, f"Parameters: xc={method}, version={disp_version}, atm={with_3body}")
        d3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version, atm=with_3body)
        if not hasattr(d3_model, 'get_hessian'):
            raise RuntimeError('DFTD3 analytical Hessian requires pyscf-dispersion >= 1.7.0. '
                               'Upgrade with `pip install -U pyscf-dispersion`.')
        res = d3_model.get_hessian()
        h_disp[np.ix_(atom_idx,atom_idx)] = res['hessian'].reshape(nreal,3,nreal,3).transpose(0,2,1,3)
        return h_disp

    elif disp_version[:2].upper() == 'D4':
        logger.info(mf, "Calc dispersion correction with DFTD4.")
        logger.info(mf, f"Parameters: xc={method}, atm={with_3body}")
        logger.warn(mf, "DFTD4 does not support analytical Hessian, using finite difference.")
        coords = mol.atom_coords()
        mol = mol.copy()
        eps = 1e-5
        for i in range(nreal):
            for j in range(3):
                coords[i,j] += eps
                mol.set_geom_(coords, unit='Bohr')
                d4_model = _make_d4_model(mol, method, with_3body)
                res = d4_model.get_dispersion(grad=True)
                g1 = res.get('gradient')

                coords[i,j] -= 2.0*eps
                mol.set_geom_(coords, unit='Bohr')
                d4_model = _make_d4_model(mol, method, with_3body)
                res = d4_model.get_dispersion(grad=True)
                g2 = res.get('gradient')

                coords[i,j] += eps
                h_disp[atom_idx[i],atom_idx,j,:] = (g1 - g2)/(2.0*eps)

        return h_disp
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to SCF class
from pyscf import hessian
hessian.rhf.HessianBase.get_dispersion = get_dispersion
