#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import scipy.special
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import df
from pyscf.pbc import scf
from pyscf.pbc.prop.efg import EFG
from pyscf.gto import PTR_COORD

def ewald_deriv1(cell, atm_id):
    ew_eta = cell.ew_eta
    ew_cut = cell.ew_cut*4
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[atm_id,:] - coords + Lall[:,None,:]
    r = np.sqrt(np.einsum('Ljx,Ljx->Lj', rLij, rLij))
    r[r<1e-16] = 1e100
    ewovrl = -chargs[atm_id] * np.einsum('Ljx,j,Lj->x', rLij, chargs,
                                         scipy.special.erfc(ew_eta * r) / r**3)
    ewovrl -= (2./np.sqrt(np.pi) * ew_eta * chargs[atm_id] *
               np.einsum('Ljx,j,Lj->x', rLij, chargs,
                         np.exp(-ew_eta**2 * r**2) / r**2))

    mesh = gto.cell._cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200

    coulG = 4*np.pi / absG2
    coulG *= weights
    SI = cell.get_SI(Gv)
    ZSI = np.einsum("i,ij->j", chargs, SI)
    ZexpG2 = ZSI * np.exp(-absG2/(4*ew_eta**2))
    ewg = np.einsum('ix,i,i,i->x',-Gv, chargs[atm_id]*SI[atm_id].conj(), ZexpG2, coulG).imag
    return ewovrl + ewg

def ewald_deriv2(cell, atm_id):
    ew_eta = cell.ew_eta
    ew_cut = cell.ew_cut
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[atm_id,:] - coords + Lall[:,None,:]
    rr = np.einsum('Ljx,Ljy->Ljxy', rLij, rLij)
    r = np.sqrt(np.einsum('Ljxx->Lj', rr))
    r[r<1e-16] = 1e60
    r[:,atm_id] = 1e60
    idx = np.arange(3)
    erfc_part = scipy.special.erfc(ew_eta * r) / r**5
    ewovrl = 3 * chargs[atm_id] * np.einsum('Ljxy,j,Lj->xy', rr, chargs, erfc_part)
    ewovrl[idx,idx] -= ewovrl.trace() / 3

    exp_part = np.exp(-ew_eta**2 * r**2) / r**4
    ewovrl_part = (2./np.sqrt(np.pi) * ew_eta * chargs[atm_id] *
                   np.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))
    ewovrl += ewovrl_part

    ewovrl += 2*ewovrl_part
    ewovrl[idx,idx] -= ewovrl_part.trace()

    exp_part = np.exp(-ew_eta**2 * r**2) / r**2
    ewovrl += (4./np.sqrt(np.pi) * ew_eta**3 * chargs[atm_id] *
               np.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))

    mesh = gto.cell._cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    GG = np.einsum('gi,gj->gij', Gv, Gv)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200

    coulG = 4*np.pi / absG2
    coulG *= weights
    SI = cell.get_SI(Gv)
    ZSI = np.einsum("i,ij->j", chargs, SI)
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    coulG *= expG2
    #:ewg =-.5*chargs[atm_id] * np.einsum('ixy,i,i,i->xy', GG, SI[atm_id], ZSI.conj(), coulG).real
    #:ewg+= chargs[atm_id]**2 * np.einsum('ixy,i,i,i->xy', GG, SI[atm_id], SI[atm_id].conj(), coulG).real
    #:ewg-= .5*chargs[atm_id] * np.einsum('ixy,i,i,i->xy', GG, SI[atm_id].conj(), ZSI, coulG).real
    ewg = chargs[atm_id]**2 * np.einsum('ixy,i,i,i->xy', GG, SI[atm_id], SI[atm_id].conj(), coulG).real
    ewg-= chargs[atm_id]    * np.einsum('ixy,i,i,i->xy', GG, SI[atm_id].conj(), ZSI, coulG).real
    return ewovrl + ewg

class KnownValues(unittest.TestCase):
    def test_quad_nuc(self):
        np.random.seed(2)
        cell = gto.M(atom='''H .0 0 0
                    He 0 0.1 1
                    He 1 0.1 1
                    H 0 1.1 1
                    ''',
                    a = np.eye(3)*2 + np.random.rand(3,3)*.1,
                    basis = [[0, (1, 1)]],
                    unit='B')

        def deriv1(cell, d1, atm_id=0):
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] += .0001
            e1 = cell.energy_nuc()
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] -= .0002
            e2 = cell.energy_nuc()
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] += .0001
            de = (e1 - e2) / .0002
            return de

        def deriv2(cell, d1, atm_id=0):
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] += .0001
            e1 = ewald_deriv1(cell, atm_id)
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] -= .0002
            e2 = ewald_deriv1(cell, atm_id)
            cell._env[cell._atm[atm_id,PTR_COORD]+d1] += .0001
            de = (e1 - e2) / .0002
            return de

        for at in range(4):
            g0 = ewald_deriv1(cell, at)
            g1 = (deriv1(cell, 0, at),
                  deriv1(cell, 1, at),
                  deriv1(cell, 2, at))
            self.assertAlmostEqual(abs(g0-g1).max(), 0, 7)

        for at in range(4):
            h0 = ewald_deriv2(cell, at)
            h1 = (deriv2(cell, 0, at),
                  deriv2(cell, 1, at),
                  deriv2(cell, 2, at))
            self.assertAlmostEqual(abs(g0-g1).max(), 0, 6)

    def test_efg_kernel(self):
        cell = gto.Cell()
        cell.atom = 'H 1 0.8 0; H 0. 1.3 0'
        cell.a = np.eye(3) * 3
        cell.basis = [[0, (2.0, 1.)], [0, (0.5, 1.)]]
        cell.mesh = [20]*3
        cell.build()
        np.random.seed(12)
        kpts = cell.make_kpts([2]*3)
        nao = cell.nao
        dm = np.random.random((len(kpts),nao,nao))
        mf = scf.RHF(cell)
        mf.make_rdm1 = lambda *args, **kwargs: dm

        vref = EFG(mf)
        self.assertAlmostEqual(lib.finger(vref), 0.67090557110411564, 9)

        mf.with_df = df.AFTDF(cell)
        mf.with_df.eta = 0
        v = EFG(mf)
        self.assertAlmostEqual(abs(v-vref).max(), 0, 2)

        mf.with_df = df.AFTDF(cell)
        mf.with_df.mesh = [8]*3
        v = EFG(mf)
        self.assertAlmostEqual(abs(v-vref).max(), 0, 2)


if __name__ == "__main__":
    print("Full Tests for efg integrals")
    unittest.main()

