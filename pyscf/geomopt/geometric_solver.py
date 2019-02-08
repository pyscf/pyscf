#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
Interface to geomeTRIC library https://github.com/leeping/geomeTRIC
'''

import tempfile
import numpy
import geometric
import geometric.molecule
#from geometric import molecule
from pyscf import lib
from pyscf.geomopt.addons import as_pyscf_method, dump_mol_geometry
from pyscf import __config__

INCLUDE_GHOST = getattr(__config__, 'geomopt_berny_solver_optimize_include_ghost', True)
ASSERT_CONV = getattr(__config__, 'geomopt_berny_solver_optimize_assert_convergence', True)

class PySCFEngine(geometric.engine.Engine):
    def __init__(self, scanner):
        molecule = geometric.molecule.Molecule()
        mol = scanner.mol
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        # Molecule is the geometry parser for a bunch of formats which use
        # Angstrom for Cartesian coordinates by default.
        molecule.xyzs = [mol.atom_coords()*lib.param.BOHR]  # In Angstrom

        super(PySCFEngine, self).__init__(molecule)
        self.scanner = scanner
        self.cycle = 0

    def calc_new(self, coords, dirname):
        scanner = self.scanner
        mol = scanner.mol
        lib.logger.note(scanner, '\nGeometry optimization step %d', self.cycle)
        self.cycle += 1
        # geomeTRIC handles coords and gradients in atomic unit
        coords = coords.reshape(-1,3)
        if scanner.verbose >= lib.logger.NOTE:
            dump_mol_geometry(self.scanner.mol, coords*lib.param.BOHR)
        mol.set_geom_(coords, unit='Bohr')
        energy, gradient = scanner(mol)
        if scanner.assert_convergence and not scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % scanner.base)
        return energy, gradient.ravel()

def kernel(method, assert_convergence=ASSERT_CONV,
           include_ghost=INCLUDE_GHOST, constraints=None):
    if isinstance(method, lib.GradScanner):
        g_scanner = method
    elif getattr(method, 'nuc_grad_method', None):
        g_scanner = method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)
    if not include_ghost:
        g_scanner.atmlst = numpy.where(method.mol.atom_charges() != 0)[0]
    g_scanner.assert_convergence = assert_convergence

    tmpf = tempfile.mktemp(dir=lib.param.TMPDIR)
    m = geometric.optimize.run_optimizer(customengine=PySCFEngine(g_scanner),
                                         input=tmpf, constraints=constraints)

    #FIXME: geomeTRIC library keeps running until converged. We need a function
    # to terminate the program even not converged.
    conv = True

    #return conv, method.mol.copy().set_geom_(m.xyzs[-1], unit='Bohr')
    return method.mol.copy().set_geom_(m.xyzs[-1], unit='Angstrom')

optimize = kernel

del(INCLUDE_GHOST, ASSERT_CONV)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, dft, cc, mp
    mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
                ''',
                basis='3-21g')

    mf = scf.RHF(mol)
    mol1 = optimize(mf)
    print(mf.kernel() - -153.219208484874)
    print(scf.RHF(mol1).kernel() - -153.222680852335)

    mf = dft.RKS(mol)
    mf.xc = 'pbe,'
    mf.conv_tol = 1e-7
    mol1 = optimize(mf)

    mymp2 = mp.MP2(scf.RHF(mol))
    mol1 = optimize(mymp2)

    mycc = cc.CCSD(scf.RHF(mol))
    mol1 = optimize(mycc)
