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
# and that exceeds the finite difference error itself. On H2O/STO-3G PBE0,
# against a level 9 unpruned grid, the RKS Hessian differenced from gradients
# is off by 1.1e-04 on the default grid and 2.6e-05 at level 5 unpruned, where
# the error of the displacement is 3e-06. Grid response on the gradients
# removes part of it, leaving 3.2e-05 on the default grid.
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
        if getattr(method, 'grid_response', None) is False:
            logger.warn(mol, 'Setting grid_response = True on the gradients '
                        'also reduces the grid error.')

def _spin_blocks(a, nspin):
    '''Amplitudes of one state split into spin channels.  TDA has y = 0.'''
    if nspin == 1:
        a = [a]
    return [None if np.isscalar(i) else np.asarray(i) for i in a]

def _reference_state(scan):
    '''The root being differentiated, and the MOs its amplitudes are in'''
    state = getattr(scan, 'state', None)
    td = getattr(scan, 'base', None)
    xy = getattr(td, 'xy', None)
    mf = getattr(td, '_scf', None)
    if state is None or not xy or mf is None:
        return None
    mo_coeff = np.asarray(mf.mo_coeff)
    if mo_coeff.shape[-2] != mf.mol.nao:
        return None  # e.g. GHF, whose MOs span both spins
    return mf.mol.copy(), mo_coeff.copy(), np.asarray(mf.mo_occ).copy(), xy[state-1]

def _state_overlaps(ref, td):
    '''|<ref|I>| for every root I of td at the displaced geometry.

    The amplitudes of the two geometries are in different MO bases, which can
    differ by phases and, for degenerate orbitals, arbitrary rotations.  The
    reference amplitudes are carried over through the overlap of the occupied
    and of the virtual MOs across the two geometries.
    '''
    mol0, c0, occ0, (x0, y0) = ref
    mf = td._scf
    c1 = np.asarray(mf.mo_coeff)
    occ1 = np.asarray(mf.mo_occ)
    s = gto.intor_cross('int1e_ovlp', mol0, mf.mol)
    nspin = 1 if c0.ndim == 2 else 2
    if nspin == 1:
        c0, c1, occ0, occ1 = [c0], [c1], [occ0], [occ1]

    def dot(xa, ya, xb, yb):
        v = sum(np.vdot(a, b) for a, b in zip(xa, xb))
        # The RPA metric
        v -= sum(np.vdot(a, b) for a, b in zip(ya, yb)
                 if a is not None and b is not None)
        return v

    x0 = _spin_blocks(x0, nspin)
    y0 = _spin_blocks(y0, nspin)
    n0 = abs(dot(x0, y0, x0, y0))
    for i in range(nspin):
        o0, v0 = occ0[i] > 0, occ0[i] == 0
        o1, v1 = occ1[i] > 0, occ1[i] == 0
        socc = c0[i][:,o0].T.dot(s).dot(c1[i][:,o1])
        svir = c0[i][:,v0].T.dot(s).dot(c1[i][:,v1])
        x0[i] = socc.T.dot(x0[i]).dot(svir)
        if y0[i] is not None:
            y0[i] = socc.T.dot(y0[i]).dot(svir)

    ovlp = []
    for x, y in td.xy:
        x = _spin_blocks(x, nspin)
        y = _spin_blocks(y, nspin)
        ovlp.append(abs(dot(x0, y0, x, y)) / np.sqrt(n0 * abs(dot(x, y, x, y))))
    return ovlp

def _track_state(scan, ref, mol):
    '''Warn when the root being differentiated is no longer the reference state'''
    if ref is None or not getattr(scan.base, 'xy', None):
        return
    ovlp = _state_overlaps(ref, scan.base)
    best = int(np.argmax(ovlp))
    state = scan.state - 1
    if best != state:
        logger.warn(mol, 'The reference state now best matches root %d rather '
                    'than root %d (overlap %.3f vs %.3f). The states have '
                    'crossed, or a degenerate pair has been split, and the '
                    'result differences across it.',
                    best + 1, state + 1, ovlp[best], ovlp[state])


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
    caller_mf = _mean_field(method)

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
        ref_state = _reference_state(scan)
        def evaluate(r):
            res = scan(_displace(work, r))
            if not scan.converged:
                raise RuntimeError('%s not converged' % scan)
            _track_state(scan, ref_state, mol)
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
        # Scanners share grids and density fitting objects with the method
        # they were made from, leaving them built for the last displacement
        if hasattr(caller_mf, 'reset'):
            caller_mf.reset(caller_mf.mol)

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
