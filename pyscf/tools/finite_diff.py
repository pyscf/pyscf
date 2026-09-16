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

Differencing energies gives gradients, differencing analytic gradients gives
Hessians.  The latter is the only route to a Hessian for the many methods that
have an analytic gradient but no analytic second derivative: TDHF and TDDFT,
MP2, CISD, CCSD, CCSD(T) and CASSCF among them.

Excited states are differentiated at a fixed root.  Build the gradient scanner
to pin the state and hand it over:

    >>> H = finite_diff.kernel(td.nuc_grad_method().as_scanner(state=2))
'''

import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad.rhf import GradientsBase
from pyscf.hessian.rhf import HessianBase

# Near the crossover between truncation error and the noise in the quantity
# being differenced. Larger steps are dominated by truncation: on H2O/STO-3G
# the RHF Hessian is off by 5.5e-05 at 1e-2 and 9.1e-07 at 1e-3.
DISPLACEMENT = 1e-3

# Coarser DFT grids leave an error that does not shrink with the displacement
# and that exceeds the finite difference error itself.
GRID_LEVEL = 5


def _mean_field(method):
    '''The mean-field object underneath a gradients, post-SCF or excited-state method'''
    mf = method
    for _ in range(8):
        if isinstance(mf, GradientsBase):
            mf = mf.base
        elif hasattr(mf, '_scf'):
            mf = mf._scf
        else:
            break
    return mf

def _displace(mol, coords):
    '''A copy of mol at the given coordinates, in the same frame.

    Point group detection has a finite tolerance that the displacement falls
    inside, so a displaced molecule is still assigned the point group of the
    reference and the wavefunction is symmetry-adapted to a group the geometry
    has lost.  Pinning the group to C1 avoids that, and avoids the
    reorientation that comes with re-detection, while keeping mol.symmetry
    truthy for symmetry-adapted SCF classes, which reject a symmetry-free Mole.
    '''
    if mol.symmetry:
        return mol.set_geom_(coords, symmetry='C1', inplace=False)
    return mol.set_geom_(coords, inplace=False)

def _check_sanity(method, displacement, hessian):
    mol = method.mol
    mf = _mean_field(method)

    # Differencing amplifies the error of the quantity being differenced by
    # 1/(2*displacement). For a Hessian that error is itself set by how well
    # the gradient, and so the wavefunction, is converged.
    tol = displacement**4 if hessian else displacement**3
    conv_tol = getattr(mf, 'conv_tol', None)
    if conv_tol is not None and conv_tol > tol:
        logger.warn(mol, 'conv_tol %g of %s is too loose for displacement %g. '
                    'Errors are amplified by 1/(2*displacement) = %.0f here. '
                    'Set conv_tol <= %g.', conv_tol, mf.__class__.__name__,
                    displacement, .5/displacement, tol)

    base = method.base if isinstance(method, GradientsBase) else method
    if base is not mf:
        conv_tol = getattr(base, 'conv_tol', None)
        if conv_tol is not None and conv_tol > tol:
            logger.warn(mol, 'conv_tol %g of %s is too loose for displacement '
                        '%g. Set it to <= %g.', conv_tol,
                        base.__class__.__name__, displacement, tol)

    grids = getattr(mf, 'grids', None)
    if grids is not None and (grids.level < GRID_LEVEL or grids.prune is not None):
        logger.warn(mol, 'DFT grid (level %d, pruned %s) is coarse for finite '
                    'differences. The grid error does not shrink with the '
                    'displacement. Use grids.level >= %d and grids.prune = None.',
                    grids.level, grids.prune is not None, GRID_LEVEL)

def _flatten_xy(xy):
    '''Excitation amplitudes of one state as a normalised vector'''
    x, y = xy
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if y.size == x.size:
        x = np.concatenate([x, y])
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x

def _reference_state(scan):
    '''Amplitudes of the root being differentiated, for tracking it'''
    state = getattr(scan, 'state', None)
    xy = getattr(getattr(scan, 'base', None), 'xy', None)
    if state is None or not xy:
        return None
    return _flatten_xy(xy[state - 1])

def _track_state(scan, ref_xy, mol):
    '''Warn when the tuned root is no longer the one being differentiated.

    Excited states can change order under displacement.  Amplitudes carry an
    arbitrary sign and are expressed in the displaced MO basis, so compare
    their magnitudes; over these displacements the MO basis barely moves.
    '''
    if ref_xy is None:
        return
    xy = getattr(scan.base, 'xy', None)
    if not xy:
        return
    ovlp = [abs(np.dot(ref_xy, _flatten_xy(i))) for i in xy]
    best = int(np.argmax(ovlp))
    state = scan.state - 1
    if best != state:
        logger.warn(mol, 'Root %d now overlaps state %d more strongly (%.3f vs '
                    '%.3f). The states have swapped order and the result '
                    'differences across the crossing.',
                    state + 1, best + 1, ovlp[best], ovlp[state])


def kernel(method, displacement=DISPLACEMENT):
    '''
    Evaluate gradients or Hessians for a given method using finite difference approximation.

    Args:
        method (callable):
            The function for which the gradient or Hessian is to be computed.

    Kwargs:
        displacement:
            The small change for finite difference calculations. Default is 1e-3.

    Returns:
        An (n, 3) array for gradients or (n, n, 3, 3) array for hessian,
        depending on the given method.
    '''
    assert isinstance(method, lib.StreamObject)

    mol = method.mol
    original_coords = mol.atom_coords()
    natm = mol.natm
    hessian = isinstance(method, GradientsBase)
    if hessian:
        logger.info(mol, 'Computing finite-difference Hessian for %s', method)
        de = np.empty((natm,3,natm,3))
    else:
        logger.info(mol, 'Computing finite-difference gradients for %s', method)
        de = np.empty((natm,3))
    _check_sanity(method, displacement, hessian)

    # Mole.atom_coords is in Bohr; a template in Bohr keeps set_geom_ from
    # announcing a unit change on every displacement.
    work = mol.copy()
    work.unit = 'Bohr'

    scan = None
    if isinstance(method, (lib.SinglePointScanner, lib.GradScanner)):
        scan = method
    elif hasattr(method, 'as_scanner'):
        logger.info(mol, 'Apply %s.as_scanner', method)
        scan = method.as_scanner()
    else:
        method = method.copy()
        if hessian:
            method.base = method.base.copy()

    if scan is not None:
        ref_xy = _reference_state(scan)
        def evaluate(r):
            res = scan(_displace(work, r))
            if not scan.converged:
                raise RuntimeError('%s not converged' % scan)
            _track_state(scan, ref_xy, mol)
            return res[1] if hessian else res
    else:
        logger.info(mol, '%s.as_scanner not found. Initial guess may not be '
                    'utilized among different geometries', method)
        def evaluate(r):
            dmol = _displace(work, r)
            if hessian:
                method.base.reset(dmol)
                method.base.run()
                if not method.base.converged:
                    raise RuntimeError('%s not converged' % method.base)
                method.mol = dmol
                return method.kernel()
            method.reset(dmol)
            res = method.kernel()
            if not method.converged:
                raise RuntimeError('%s not converged' % method)
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

    if hessian:
        # Hessian is stored as (N,N,3,3)
        de = de.transpose(0,2,1,3)
        # The exact Hessian is symmetric; the differencing error is not
        n3 = natm * 3
        h = de.transpose(0,2,1,3).reshape(n3, n3)
        de = ((h + h.T) * .5).reshape(natm,3,natm,3).transpose(0,2,1,3)
    return de

class Gradients(GradientsBase):
    displacement = DISPLACEMENT

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
    displacement = DISPLACEMENT

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
