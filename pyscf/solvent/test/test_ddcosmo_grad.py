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
from functools import reduce
import numpy
from pyscf import gto, scf, lib
from pyscf.solvent import ddcosmo
from pyscf.solvent import ddcosmo_grad
from pyscf.symm import sph


dx = 0.0001
mol0 = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B')
mol1 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%(-dx), unit='B')
mol2 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%dx, unit='B')
dx = dx * 2
nao = mol0.nao_nr()
dm = numpy.random.random((nao,nao))
dm = dm + dm.T

class KnownValues(unittest.TestCase):

    def test_e_psi1(self):
        def get_e_psi1(pcmobj):
            pcmobj.grids.build()
            mol = pcmobj.mol
            natm = mol.natm
            lmax = pcmobj.lmax

            r_vdw = ddcosmo.get_atomic_radii(pcmobj)
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

            fi = ddcosmo.make_fi(pcmobj, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            nexposed = numpy.count_nonzero(ui==1)
            nbury = numpy.count_nonzero(ui==0)
            on_shell = numpy.count_nonzero(ui>0) - nexposed

            nlm = (lmax+1)**2
            Lmat = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
            Lmat = Lmat.reshape(natm*nlm,-1)

            cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

            phi = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui)
            L_X = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
            psi, vmat, L_S = \
                    ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, pcmobj.grids, ylm_1sph,
                                          cached_pol, L_X, Lmat)
            psi1 = ddcosmo_grad.make_e_psi1(pcmobj, dm, r_vdw, ui, pcmobj.grids, ylm_1sph,
                                            cached_pol, L_X, Lmat)
            return L_X, psi, psi1

        pcmobj = ddcosmo.DDCOSMO(mol0)
        L_X, psi0, psi1 = get_e_psi1(pcmobj)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        L_X1, psi = get_e_psi1(pcmobj)[:2]
        e1 = numpy.einsum('jx,jx', psi, L_X)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        L_X2, psi = get_e_psi1(pcmobj)[:2]
        e2 = numpy.einsum('jx,jx', psi, L_X)
        self.assertAlmostEqual(abs((e2-e1)/dx - psi1[0,2]).max(), 0, 7)

    def test_phi(self):
        def get_phi1(pcmojb):
            pcmobj.grids.build()
            mol = pcmobj.mol
            natm = mol.natm
            lmax = pcmobj.lmax

            r_vdw = pcmobj.get_atomic_radii()
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

            fi = ddcosmo.make_fi(pcmobj, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            nexposed = numpy.count_nonzero(ui==1)
            nbury = numpy.count_nonzero(ui==0)
            on_shell = numpy.count_nonzero(ui>0) - nexposed

            nlm = (lmax+1)**2
            Lmat = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
            Lmat = Lmat.reshape(natm*nlm,-1)

            cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

            phi = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui)
            L_X = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
            psi, vmat, L_S = \
                    ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, pcmobj.grids, ylm_1sph,
                                          cached_pol, L_X, Lmat)
            phi1 = ddcosmo_grad.make_phi1(pcmobj, dm, r_vdw, ui)
            phi1 = numpy.einsum('izjx,jx->iz', phi1, L_S)
            return L_S, phi, phi1

        pcmobj = ddcosmo.DDCOSMO(mol0)
        L_S, phi0, phi1 = get_phi1(pcmobj)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        L_S1, phi = get_phi1(pcmobj)[:2]
        e1 = numpy.einsum('jx,jx', phi, L_S)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        L_S2, phi = get_phi1(pcmobj)[:2]
        e2 = numpy.einsum('jx,jx', phi, L_S)
        self.assertAlmostEqual(abs((e2-e1)/dx - phi1[0,2]).max(), 0, 7)

    def test_fi(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
        ui1 = -fi1
        fi = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui = 1 - fi
        ui1[:,:,ui<0] = 0

        pcmobj = ddcosmo.DDCOSMO(mol1)
        fi_1 = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui_1 = 1 - fi_1
        ui_1[ui_1<0] = 0

        pcmobj = ddcosmo.DDCOSMO(mol2)
        fi_2 = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui_2 = 1 - fi_2
        ui_2[ui_2<0] = 0
        self.assertAlmostEqual(abs((fi_2-fi_1)/dx - fi1[0,2]).max(), 0, 6)
        self.assertAlmostEqual(abs((ui_2-ui_1)/dx - ui1[0,2]).max(), 0, 6)

    def test_L1(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        r_vdw = pcmobj.get_atomic_radii()
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcmobj.lmax, True))

        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L_1 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L_2 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
        self.assertAlmostEqual(abs((L_2-L_1)/dx - L1[0,2]).max(), 0, 7)

    def test_e_cosmo_grad(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        de = ddcosmo_grad.kernel(pcmobj, dm)
        pcmobj = ddcosmo.DDCOSMO(mol1)
        e1 = pcmobj.energy(dm)
        pcmobj = ddcosmo.DDCOSMO(mol2)
        e2 = pcmobj.energy(dm)
        self.assertAlmostEqual(abs((e2-e1)/dx - de[0,2]).max(), 0, 7)

    def test_scf_grad(self):
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol0)).run()
        de_cosmo = ddcosmo_grad.kernel(mf._solvent, mf.make_rdm1())
        de = mf.nuc_grad_method().kernel()
        dm1 = mf.make_rdm1()

        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol1)).run()
        e1 = mf.e_tot
        e1_cosmo = mf._solvent.energy(dm1)

        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol2)).run()
        e2 = mf.e_tot
        e2_cosmo = mf._solvent.energy(dm1)
        self.assertAlmostEqual(abs((e2-e1)/dx - de[0,2]).max(), 0, 7)
        self.assertAlmostEqual(abs((e2_cosmo-e1_cosmo)/dx - de_cosmo[0,2]).max(), 0, 7)


if __name__ == "__main__":
    print("Full Tests for ddcosmo gradients")
    unittest.main()

