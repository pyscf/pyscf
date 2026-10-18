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

"""CHGCAR grid coordinates, file ordering, and sampled fields."""
import tempfile
import unittest

import numpy as np

from pyscf import gto, lib
from pyscf.pbc import gto as pbcgto
from pyscf.tools import chgcar


def read_field(path, natm):
    with open(path) as stream:
        lines = stream.read().splitlines()
    shape = tuple(map(int, lines[9 + natm].split()))
    values = np.fromstring(' '.join(lines[10 + natm:]), sep=' ')
    # VASP stores the x index fastest.
    return lines, values.reshape(shape, order='F')


class KnownValues(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = tmp.name + '/CHGCAR'

    def test_periodic_grid_and_file(self):
        cell = pbcgto.M(atom='He 0 0 0; H 1 0 0; H 0 1 0', basis='sto-3g',
                        unit='Bohr', a=[[4, 0, 0], [1, 5, 0], [0, 0, 6]], verbose=0)
        grid = chgcar.CHGCAR(cell, nx=2, ny=3, nz=4, resolution=None)
        self.assertEqual(grid.get_ngrids(), 24)
        coords = grid.get_coords()
        fractional = coords @ np.linalg.inv(cell.lattice_vectors())
        expected = np.array([[x / 2, y / 3, z / 4]
                             for x in range(2) for y in range(3) for z in range(4)])
        np.testing.assert_allclose(fractional, expected, atol=1e-14)
        field = np.arange(24.).reshape(2, 3, 4) / 10
        grid.write(field, self.path, comment='grid test ')
        lines, restored = read_field(self.path, cell.natm)
        self.assertTrue(lines[0].startswith('grid test '))
        self.assertEqual(lines[5].split(), ['H', 'He'])
        self.assertEqual(list(map(int, lines[6].split())), [2, 1])
        np.testing.assert_allclose(np.loadtxt(lines[2:5]), cell.lattice_vectors() * lib.param.BOHR,
                                   atol=1e-8)
        np.testing.assert_allclose(np.loadtxt(lines[8:11]),
                                   cell.atom_coords()[[1, 2, 0]] * lib.param.BOHR,
                                   atol=1e-8)
        np.testing.assert_allclose(restored / cell.vol, field, atol=1e-9)
        np.testing.assert_array_equal(field, np.arange(24.).reshape(2, 3, 4) / 10)

    def test_molecular_grid_does_not_move_input_atoms(self):
        for unit in ('Bohr', 'Angstrom'):
            with self.subTest(unit=unit):
                mol = gto.M(atom='He 1 2 3', unit=unit, verbose=0)
                before = mol.atom_coords().copy()
                grid = chgcar.CHGCAR(mol, resolution=1., margin=2.)
                self.assertEqual((grid.nx, grid.ny, grid.nz), (4, 4, 4))
                np.testing.assert_allclose(grid.box, np.eye(3) * 4, atol=1e-14)
                np.testing.assert_allclose(grid.cell.atom_coords(), [[2, 2, 2]], atol=1e-14)
                np.testing.assert_array_equal(mol.atom_coords(), before)

    def test_density_and_orbital(self):
        for periodic in (False, True):
            for resolution in (None, 2.):
                with self.subTest(periodic=periodic, resolution=resolution):
                    if periodic:
                        mol = pbcgto.M(atom='He 0 0 0', a=np.eye(3) * 5,
                                       unit='Bohr', basis='sto-3g', verbose=0)
                    else:
                        mol = gto.M(atom='He 1 2 3', basis='sto-3g',
                                    unit='Bohr', verbose=0)
                    before = mol.atom_coords().copy()
                    options = dict(nx=2, ny=3, nz=4, resolution=resolution)
                    grid = chgcar.CHGCAR(mol, **options)
                    coords = grid.get_coords()
                    if periodic:
                        ao = mol.pbc_eval_gto('GTOval', coords)
                    else:
                        # Translate the sampling points back to the original
                        # molecular frame; the exported lattice starts at 0.
                        offset = before.min(axis=0) - chgcar.BOX_MARGIN
                        ao = mol.eval_gto('GTOval', coords + offset)
                    chgcar.orbital(mol, self.path, np.array([1.]), **options)
                    lines, orbital = read_field(self.path, mol.natm)
                    np.testing.assert_allclose(orbital.ravel() / grid.vol, ao[:, 0],
                                               atol=1e-9, rtol=1e-8)
                    chgcar.density(mol, self.path, np.array([[2.]]), **options)
                    lines, density = read_field(self.path, mol.natm)
                    np.testing.assert_allclose(density.ravel() / grid.vol, 2 * ao[:, 0]**2,
                                               atol=1e-9, rtol=1e-8)
                    np.testing.assert_array_equal(mol.atom_coords(), before)

    def test_field_shape_validation(self):
        mol = gto.M(atom='He', verbose=0)
        grid = chgcar.CHGCAR(mol, nx=2, ny=3, nz=4, resolution=None)
        for shape in ((24,), (2, 3, 5)):
            with self.subTest(shape=shape):
                with self.assertRaises(AssertionError):
                    grid.write(np.zeros(shape), self.path)


if __name__ == '__main__':
    unittest.main()
