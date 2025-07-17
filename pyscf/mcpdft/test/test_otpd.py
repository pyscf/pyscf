#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf import mcpdft
from pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from pyscf.mcpdft.otpd import density_orbital_derivative
import unittest

def vector_error (test, ref):
    err = test - ref
    norm_test = linalg.norm (test)
    norm_ref = linalg.norm (ref)
    norm_err = linalg.norm (err)
    if norm_ref > 0: err = norm_err / norm_ref
    elif norm_test > 0: err = norm_err / norm_test
    else: err = norm_err
    return err

h2 = scf.RHF (gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g', 
    output='/dev/null', verbose=0)).run ()
lih = scf.RHF (gto.M (atom = 'Li 0 0 0; H 1.2 0 0', basis = 'sto-3g',
    output='/dev/null', verbose=0)).run ()

def get_dm2_ao (mc, mo_coeff, casdm1, casdm2):
    i, ncas = mc.ncore, mc.ncas
    j = i + ncas
    mo_occ = mo_coeff[:,:j]
    dm1 = 2*np.eye (j)
    dm1[i:j,i:j] = casdm1
    dm2 = np.multiply.outer (dm1, dm1)
    dm2 -= 0.5*np.multiply.outer (dm1, dm1).transpose (0,3,2,1)
    dm2[i:j,i:j,i:j,i:j] = casdm2
    return np.einsum ('pqrs,ip,jq,kr,ls->ijkl', dm2, mo_occ, mo_occ,
                      mo_occ, mo_occ)

def get_rho_ref (dm1s, ao):
    rho = np.einsum ('sjk,caj,ak->sca', dm1s, ao[:4], ao[0])
    rho[:,1:4] += np.einsum ('sjk,cak,aj->sca', dm1s, ao[1:4], ao[0])
    return rho 

def get_Pi_ref (dm2, ao):
    nderiv, ngrid, nao = ao.shape
    Pi = np.zeros ((5,ngrid))
    Pi[:4]   = np.einsum ('ijkl,cai,aj,ak,al->ca', dm2,
                          ao[:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,caj,ai,ak,al->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,cak,ai,aj,al->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,cal,ai,aj,ak->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    X, Y, Z, XX, YY, ZZ = 1,2,3,4,7,9
    Pi[4]  = np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[XX], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[YY], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[ZZ], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[X], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[Y], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[Z], ao[0], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[0], ao[X], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[0], ao[Y], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[0], ao[Z], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[0], ao[0], ao[X]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[0], ao[0], ao[Y]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[0], ao[0], ao[Z]) / 2
    return Pi

def num_Drho_DPi (mc, x, dm1s_mo, cascm2, ao, mask):
    ot = mc.otfnal
    ni = ot._numint
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore+ncas
    mo = mc.mo_coeff.dot (mc.update_rotate_matrix (x))
    dm1s = np.dot (mo, dm1s_mo).transpose (1,0,2)
    dm1s = np.dot (dm1s, mo.conj ().T)
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i], 1) for i in range (2))
    Drho = np.array ([m[0] (0, ao, mask, 'MGGA') for m in make_rho])
    DPi = get_ontop_pair_density (ot, Drho, ao, cascm2, mo[:,ncore:nocc],
                                  deriv=2, non0tab=mask)
    return Drho, DPi

def an_Drho_DPi (mc, x, drho, dPi, mo):
    mo1 = mo.dot (mc.unpack_uniq_var (x))*2
    Drho = np.einsum ('sdgi,gi->sdg', drho[:,:4], mo1[0])
    Drho[:,1:4] += np.einsum ('sgi,dgi->sdg', drho[:,0], mo1[1:4])
    DPi = np.einsum ('dgi,gi->dg', dPi[:4], mo1[0])
    DPi[1:4] += np.einsum ('gi,dgi->dg', dPi[0], mo1[1:4])
    return Drho, DPi

def convergence_table_Drho_DPi (mc, x, make_rho, casdm1s, cascm2, ao, mask):
    ot = mc.otfnal
    ni = ot._numint
    nao, nmo = mc.mo_coeff.shape 
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore+ncas
    dm1s_mo = np.stack ([np.eye (nmo),]*2, axis=0)
    dm1s_mo[:,nocc:,:] = 0
    dm1s_mo[:,:,nocc:] = 0
    dm1s_mo[:,ncore:nocc,ncore:nocc] = casdm1s
    mo_cas = mc.mo_coeff[:,ncore:nocc]
    rho = np.array ([m[0] (0, ao, mask, 'MGGA') for m in make_rho])
    Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas, deriv=2, non0tab=mask)
    mo = _grid_ao2mo (ot.mol, ao, mc.mo_coeff, non0tab=mask)
    drho, dPi = density_orbital_derivative (
        ot, ncore, ncas, casdm1s, cascm2, rho, mo, deriv=1,
        non0tab=mask)
    err_tab = np.zeros ((4, 5))
    for ix, p in enumerate (range (16,20)):
        x1 = x / 2**p
        Drho_an, DPi_an = an_Drho_DPi (mc, x1, drho, dPi, mo)
        Drho_num, DPi_num = num_Drho_DPi (mc, x1, dm1s_mo, cascm2, ao, mask)
        Drho_num -= rho
        DPi_num -= Pi
        err_tab[ix,0] = 1/2**p
        err_tab[ix,1] = vector_error (Drho_an[:,0], Drho_num[:,0])
        err_tab[ix,2] = vector_error (Drho_an[:,1:4], Drho_num[:,1:4])
        err_tab[ix,3] = vector_error (DPi_an[0], DPi_num[0])
        err_tab[ix,4] = vector_error (DPi_an[1:4], DPi_num[1:4])
    denom_tab = err_tab[:-1].copy ()
    err_tab[1:][denom_tab==0] = 0.5
    denom_tab[denom_tab==0] = 1
    conv_tab = err_tab[1:] / denom_tab
    return conv_tab

def tearDownModule():
    global h2, lih
    h2.mol.stdout.close ()
    lih.mol.stdout.close ()
    del h2, lih

class KnownValues(unittest.TestCase):

    def test_otpd (self):
        for mol, mf in zip (('H2', 'LiH'), (h2, lih)):
            for state, nel in zip (('Singlet', 'Triplet'), (2, (2,0))):
                mc = mcpdft.CASSCF (mf, 'tLDA,VWN3', 2, nel, grids_attr={'atom_grid':(2,14)}).run ()
                ncore, ncas = mc.ncore, mc.ncas
                nocc = ncore+ncas
                dm1s = np.array (mc.make_rdm1s ())
                casdm1s, casdm2s = mc.fcisolver.make_rdm12s (mc.ci, mc.ncas, mc.nelecas)
                casdm1 = casdm1s[0] + casdm1s[1]
                casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
                cascm2 = casdm2 - np.multiply.outer (casdm1, casdm1)
                cascm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
                cascm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
                mo_cas = mc.mo_coeff[:,ncore:nocc]
                nao, ncas = mo_cas.shape
                with self.subTest (mol=mol, state=state):
                    ot, ni = mc.otfnal, mc.otfnal._numint
                    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i], 1) for i in range (2))
                    dm2_ao = get_dm2_ao (mc, mc.mo_coeff, casdm1, casdm2)
                    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, nao, 2, 2000):
                        rho = np.array ([m[0] (0, ao, mask, 'MGGA') for m in make_rho])
                        Pi_test = get_ontop_pair_density (
                            ot, rho, ao, cascm2, mo_cas, deriv=2,
                            non0tab=mask)
                        Pi_ref = get_Pi_ref (dm2_ao, ao)
                        self.assertAlmostEqual (lib.fp (Pi_test), lib.fp (Pi_ref), 10)
                        mo = _grid_ao2mo (ot.mol, ao, mc.mo_coeff, non0tab=mask)
                        drho, dPi = density_orbital_derivative (
                            ot, ncore, ncas, casdm1s, cascm2, rho, mo, deriv=1,
                        non0tab=mask)
                        rho_test = np.einsum ('sdgi,gi->sdg', drho[:,:4], mo[0])
                        rho_test[:,1:4] += np.einsum ('sgi,dgi->sdg', drho[:,0], mo[1:4])
                        self.assertAlmostEqual (lib.fp (rho_test), lib.fp (rho[:,:4]), 10)
                        Pi_test = np.einsum ('dgi,gi->dg', dPi[:4], mo[0]) / 2
                        Pi_test[1:4] += np.einsum ('gi,dgi->dg', dPi[0], mo[1:4]) / 2
                        self.assertAlmostEqual (lib.fp (Pi_test), lib.fp (Pi_ref[:4]), 10)

    def test_otpd_orbital_deriv (self):
        for mol, mf in zip (('H2', 'LiH'), (h2, lih)):
            for state, nel in zip (('Singlet', 'Triplet'), (2, (2,0))):
                mc = mcpdft.CASSCF (mf, 'tLDA,VWN3', 2, nel, grids_attr={'atom_grid':(2,14)}).run ()
                ncore, ncas = mc.ncore, mc.ncas
                nocc = ncore+ncas
                dm1s = np.array (mc.make_rdm1s ())
                casdm1s, casdm2s = mc.fcisolver.make_rdm12s (mc.ci, mc.ncas, mc.nelecas)
                casdm1 = casdm1s[0] + casdm1s[1]
                casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
                cascm2 = casdm2 - np.multiply.outer (casdm1, casdm1)
                cascm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
                cascm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
                mo_cas = mc.mo_coeff[:,ncore:nocc]
                nao, nmo = mc.mo_coeff.shape
                x = 2*(1-np.random.rand (nmo, nmo))
                x = mc.pack_uniq_var (x-x.T)
                ot, ni = mc.otfnal, mc.otfnal._numint
                make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i], 1) for i in range (2))
                for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, nao, 2, 2000):
                    conv_tab=convergence_table_Drho_DPi (mc, x, make_rho, casdm1s, cascm2, ao, mask)
                    conv_tab=conv_tab[-3:].sum (0)/3
                    with self.subTest (mol=mol, state=state, quantity="rho"):
                        self.assertAlmostEqual (conv_tab[1], .5, 3)
                    with self.subTest (mol=mol, state=state, quantity="rho'"):
                        self.assertAlmostEqual (conv_tab[2], .5, 3)
                    with self.subTest (mol=mol, state=state, quantity="Pi"):
                        self.assertAlmostEqual (conv_tab[3], .5, 3)
                    with self.subTest (mol=mol, state=state, quantity="Pi'"):
                        self.assertAlmostEqual (conv_tab[4], .5, 3)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT on-top pair density construction")
    unittest.main()






