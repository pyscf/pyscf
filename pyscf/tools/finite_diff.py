#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Finite difference driver
'''

import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad.rhf import GradientsBase
from pyscf.hessian.rhf import HessianBase

def kernel(method, displacement=1e-2):
    '''
    Evaluate gradients or Hessians for a given method using finite difference approximation.

    Args:
        method (callable):
            The function for which the gradient or Hessian is to be computed.

    Kwargs:
        displacement:
            The small change for finite difference calculations. Default is 1e-2.

    Returns:
        An (n, 3) array for gradients or (n, n, 3, 3) array for hessian,
        depending on the given method.
    '''
    assert isinstance(method, lib.StreamObject)

    mol = method.mol
    original_coords = mol.atom_coords()
    natm = mol.natm
    if isinstance(method, GradientsBase):
        logger.info(mol, 'Computing finite-difference Hessian for {method}')
        if method.base.conv_tol > displacement**3:
            logger.warn(mol, 'conv_tol %g with displacement %g might be insufficinet',
                        method.base.conv_tol, displacement)
        de = np.empty((natm,3,natm,3))
    else:
        logger.info(mol, 'Computing finite-difference Gradients for {method}')
        if method.conv_tol > displacement**2:
            logger.warn(mol, 'conv_tol %g with displacement %g might be insufficinet',
                        method.conv_tol, displacement)
        de = np.empty((natm,3))

    scan = None
    if isinstance(method, (lib.SinglePointScanner, lib.GradScanner)):
        scan = method
    elif hasattr(method, 'as_scanner'):
        logger.info(mol, 'Apply {method}.scanner')
        scan = method.as_scanner()
    else:
        method = method.copy()
        if isinstance(method, GradientsBase):
            method.base = method.base.copy()

    if scan is not None:
        def evaluate(r):
            mol.set_geom_(r, unit='Bohr')
            if scan:
                if isinstance(method, GradientsBase):
                    res = scan(mol)[1]
                else:
                    res = scan(mol)
                if not scan.converged:
                    raise RuntimeError('{scan} not converged')
                return res
    else:
        logger.info(mol, '{method}.scanner not found. '
                    'Initial guess may not be utilized among different geometries')
        def evaluate(r):
            if isinstance(method, GradientsBase):
                method.base.mol.set_geom_(r, unit='Bohr')
                method.base.run()
                if not method.base.converged:
                    raise RuntimeError('{method.base} not converged')
                method.mol.set_geom_(r, unit='Bohr')
                return method.kernel()
            else:
                method.mol.set_geom_(r, unit='Bohr')
                if not method.converged:
                    raise RuntimeError('{method} not converged')
                res = method.kernel()
            return res

    try:
        atom_coords = original_coords.copy()
        for i in range(natm):
            for x in range(3):
                atom_coords[i,x] += displacement
                e1 = evaluate(atom_coords)
                atom_coords[i,x] -= 2*displacement
                e2 = evaluate(atom_coords)
                de[i,x] = (e1 - e2) / (2*displacement)
                atom_coords[i,x] = original_coords[i,x]
    finally:
        mol.set_geom_(original_coords, unit='Bohr')

    if isinstance(method, GradientsBase):
        # Hessian is stored as (N,N,3,3)
        de = de.transpose(0,2,1,3)
    return de

class Gradients(GradientsBase):
    displacement = 1e-2

    def __init__(self, method):
        assert isinstance(method, lib.StreamObject)
        assert not isinstance(method, GradientsBase)
        self.base = method
        self.mol = mol = method.mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.de = None

    def kernel(self):
        self.de = kernel(self.base, self.displacement)
        return self.de

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self

        logger.info(self, 'Create Gradient scanner for %s', self.base.__class__)
        name = 'FiniteDiffGrad' + GradScanner.__name_mixin__
        return lib.set_class(GradScanner(self),
                             (GradScanner, self.__class__), name)

class GradScanner(lib.GradScanner):
    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.base(mol)
        e_tot = self.base.e_tot
        de = self.kernel()
        return e_tot, de

class Hessian(HessianBase):
    displacement = 1e-2

    def __init__(self, method):
        assert isinstance(method, lib.StreamObject)
        assert isinstance(method, GradientsBase)
        self.base = method.base
        self._method = method
        self.mol = mol = method.mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.de = None

    def kernel(self):
        self.de = kernel(self._method, self.displacement)
        return self.de

    def as_scanner(self):
        return self
