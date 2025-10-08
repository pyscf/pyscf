# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Hessian of SMD solvent model, copied from GPU4PySCF with modification for CPU
'''
# pylint: disable=C0103

import numpy as np
from pyscf import scf, lib
from pyscf.solvent import smd
from pyscf.solvent.grad import smd as smd_grad
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.solvent.hessian import pcm as pcm_hess
from pyscf.lib import logger

def get_cds(smdobj):
    mol = smdobj.mol.copy()
    smdobj_tmp = smdobj.copy()
    def smd_grad_scanner(mol):
        smdobj_tmp.reset(mol)
        return smd_grad.get_cds(smdobj_tmp)

    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    log.warn("Using finite difference scheme for CDS contribution.")
    coords = mol.atom_coords(unit='B')
    coords_backup = coords.copy()
    eps = 1e-4
    natm = mol.natm
    hess_cds = np.zeros([natm,natm,3,3])
    for ia in range(mol.natm):
        for j in range(3):
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            grad0_cds = smd_grad_scanner(mol)

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            grad1_cds = smd_grad_scanner(mol)
            hess_cds[ia,:,j] = (grad0_cds - grad1_cds) / (2.0 * eps)
            coords[ia,j] = coords_backup[ia,j]
    t1 = log.timer_debug1('solvent energy', *t1)
    return hess_cds # hartree
