#!/usr/bin/env python
# Copyright 2019 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Mole class for MM particles
'''

import numpy
from pyscf import gto
from pyscf.data.elements import charge

class Mole(gto.Mole):
    '''Mole class for MM particles.

    Args:
        atoms : geometry of MM particles (unit Bohr).

            | [[atom1, (x, y, z)],
            |  [atom2, (x, y, z)],
            |  ...
            |  [atomN, (x, y, z)]]

    Kwargs:
        charges : fractional charges of MM particles

    '''
    def __init__(self, atoms, charges=None, expnts=None):
        gto.Mole.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'

        # Initialize ._atm and ._env to save the coordinates and charges and
        # other info of MM particles
        natm = len(atoms)
        _atm = numpy.zeros((natm,6), dtype=numpy.int32)
        _atm[:,gto.CHARGE_OF] = [charge(a[0]) for a in atoms]
        coords = numpy.asarray([a[1] for a in atoms], dtype=numpy.double)
        if charges is None:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_POINT
            charges = _atm[:,gto.CHARGE_OF:gto.CHARGE_OF+1]
        else:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
            charges = numpy.asarray(charges)[:,numpy.newaxis]

        # enable gaussians for MM charges; could placed in _env and setting _bas
        # but I'd rather keep this simple interface
        if expnts is None:
            self._expnts = [1e16] * natm
        else:
            assert len(expnts) == natm
            self._expnts = expnts

        self._env = numpy.append(numpy.zeros(gto.PTR_ENV_START),
                                 numpy.hstack((coords, charges)).ravel())
        _atm[:,gto.PTR_COORD] = gto.PTR_ENV_START + numpy.arange(natm) * 4
        _atm[:,gto.PTR_FRAC_CHARGE] = gto.PTR_ENV_START + numpy.arange(natm) * 4 + 3
        self._atm = _atm

        self._built = True

def create_mm_mol(atoms_or_coords, charges=None, radii=None, unit='Angstrom'):
    '''Create an MM object based on the given coordinates and charges of MM
    particles.
    '''
    if isinstance(atoms_or_coords, numpy.ndarray):
        # atoms_or_coords == np.array([(xx, xx, xx)])
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    elif (isinstance(atoms_or_coords, (list, tuple)) and
          atoms_or_coords and
          isinstance(atoms_or_coords[0][1], (int, float))):
        # atoms_or_coords == [(xx, xx, xx)]
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    else:
        atoms = atoms_or_coords
    atoms = gto.format_atom(atoms, unit=unit)
    if radii is None:
        expnts = None
    else:
        # unit conversion
        if not unit.upper().startswith(('B', 'AU')):
            from pyscf.lib import param
            radii = [r / param.BOHR for r in radii]
        expnts = [1/r**2 for r in radii]
    return Mole(atoms, charges, expnts)

