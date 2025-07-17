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
from pyscf import gto, scf, df, mcscf, lib, fci
from pyscf.fci.addons import fix_spin_, initguess_triplet
from pyscf import mcpdft
#from pyscf.fci import csf_solver
from pyscf.grad.mspdft import mspdft_heff_response, mspdft_heff_HellmanFeynman
#from pyscf.df.grad import dfsacasscf
import unittest, math

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, output='/dev/null', verbose = 0)
def random_si ():
    phi = math.pi * (np.random.rand (1)[0] - 0.5)
    cp, sp = math.cos (phi), math.sin (phi)
    si = np.array ([[cp,-sp],[sp,cp]])
    return si
si = random_si ()
def get_mc_ref (mol, ri=False, sam=False):
    mf = scf.RHF (mol)
    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcscf.CASSCF (mf.run (), 6, 6)
    mo = None
    ci0 = None
    if sam:
        fcisolvers = [fci.solver (mol), fci.solver (mol)]
        if mol.symmetry:
            fcisolvers[0].wfnsym = 'A1'
            fcisolvers[1].wfnsym = 'A2'
        else:
            h1, h0 = mc.get_h1cas ()
            h2 = mc.get_h2cas ()
            hdiag = mc.fcisolver.make_hdiag (h1, h2, 6, 6)
            ci0 = [mc.fcisolver.get_init_guess (6, 6, 1, hdiag),
                   initguess_triplet (6, 6, '1011')]
        mc = mcscf.addons.state_average_mix (mc, fcisolvers, [0.5,0.5])
    else:
        if mol.symmetry:
            mc.fcisolver.wfnsym = 'A1'
        mc = mc.state_average ([0.5,0.5])
        mc.fix_spin_(ss=0)
    mc.conv_tol = 1e-12
    #mc.kernel (ci0=ci0)
    #print (mc.e_states[1]-mc.e_states[0])
    #return mc
    return mc.run (mo, ci0)
#mc_list = [[[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)] for m in (mol_nosymm, mol_symm)]
mc_list = [] # Crunch within unittest.main for accurate clock
def get_mc_list ():
    if len (mc_list) == 0:
        for m in [mol_nosymm, mol_symm]:
            mc_list.append ([[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)])
    return mc_list

def tearDownModule():
    global mol_nosymm, mol_symm, mc_list, si
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mc_list, si

class KnownValues(unittest.TestCase):

    def test_offdiag_response_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
            for mca, atype in zip (mcs, ('nomix','mix')):
                if 'no' not in atype:
                    continue
                # TODO: redesign this test case. MS-PDFT "_mix" is undefined except
                # for L-PDFT and XMS-PDFT, whose gradients aren't implemented yet
                for mc, itype in zip (mca, ('conv', 'DF')):
                    ci_arr = np.asarray (mc.ci)
                    if itype == 'conv': mc_grad = mc.nuc_grad_method ()
                    else: continue #mc_grad = dfsacasscf.Gradients (mc)
                    # TODO: proper DF functionality
                    ngorb = mc_grad.ngorb
                    dw_ref = np.stack ([mc_grad.get_wfn_response (state=i) for i in (0,1)], axis=0)
                    dworb_ref, dwci_ref = dw_ref[:,:ngorb], dw_ref[:,ngorb:]
                    with self.subTest (symm=stype, solver=atype, eri=itype, check='energy convergence'):
                        self.assertTrue (mc.converged)
                    with self.subTest (symm=stype, solver=atype, eri=itype, check='ref CI d.f. zero'):
                        self.assertLessEqual (linalg.norm (dwci_ref), 1e-4)
                    ham_si = np.diag (mc.e_states)
                    ham_si = si @ ham_si @ si.T
                    eris = mc.ao2mo (mc.mo_coeff)
                    ci = list (np.tensordot (si, ci_arr, axes=1))
                    ci_arr = np.asarray (ci)
                    si_diag = si * si
                    dw_diag = np.stack ([mc_grad.get_wfn_response (state=i, ci=ci) for i in (0,1)], axis=0)
                    dworb_diag, dwci_ref = dw_diag[:,:ngorb], dw_diag[:,ngorb:]
                    dworb_ref -= np.einsum ('sc,sr->rc', dworb_diag, si_diag)
                    dwci_ref = -np.einsum ('rpab,qab->rpq', dwci_ref.reshape (2,2,20,20), ci_arr)
                    dwci_ref -= dwci_ref.transpose (0,2,1)
                    dwci_ref = np.einsum ('spq,sr->rpq', dwci_ref, si_diag)
                    dwci_ref = dwci_ref[:,1,0]
                    for r in (0,1):
                        dworb_test, dwci_test = mspdft_heff_response (mc_grad, ci=ci, state=r, eris=eris,
                            si_bra=si[:,r], si_ket=si[:,r], heff_mcscf=ham_si)
                        dworb_test = mc.pack_uniq_var (dworb_test)
                        with self.subTest (symm=stype, solver=atype, eri=itype, root=r, check='orb'):
                            self.assertAlmostEqual (lib.fp (dworb_test), lib.fp (dworb_ref[r]), 8)
                        with self.subTest (symm=stype, solver=atype, eri=itype, root=r, check='CI'):
                            self.assertAlmostEqual (lib.fp (dwci_test), lib.fp (dwci_ref[r]), 8)

    def test_offdiag_grad_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
            for mca, atype in zip (mcs, ('nomix','mix')):
                if 'no' not in atype:
                    continue
                    # TODO: redesign this test case. MS-PDFT "_mix" is undefined except
                    # for L-PDFT and XMS-PDFT, whose gradients aren't implemented yet
                for mc, itype in zip (mca, ('conv', 'DF')):
                    ci_arr = np.asarray (mc.ci)
                    if itype == 'conv': mc_grad = mc.nuc_grad_method ()
                    else: continue #mc_grad = dfsacasscf.Gradients (mc)
                    # TODO: proper DF functionality
                    de_ref = np.stack ([mc_grad.get_ham_response (state=i) for i in (0,1)], axis=0)
                    eris = mc.ao2mo (mc.mo_coeff)
                    ci = list (np.tensordot (si, ci_arr, axes=1))
                    ci_arr = np.asarray (ci)
                    si_diag = si * si
                    de_diag = np.stack ([mc_grad.get_ham_response (state=i, ci=ci) for i in (0,1)], axis=0)
                    de_ref -= np.einsum ('sac,sr->rac', de_diag, si_diag)
                    mf_grad = mc._scf.nuc_grad_method ()
                    for r in (0,1):
                        de_test = mspdft_heff_HellmanFeynman (mc_grad, ci=ci, state=r,
                            si_bra=si[:,r], si_ket=si[:,r], eris=eris, mf_grad=mf_grad)
                        with self.subTest (symm=stype, solver=atype, eri=itype, root=r):
                            self.assertAlmostEqual (lib.fp (de_test), lib.fp (de_ref[r]), 8)

    def test_scanner (self):
        def get_lih (r):
            mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                         output='/dev/null', verbose=0)
            mf = scf.RHF (mol).run ()
            mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
            mc.fix_spin_(ss=0)
            mc = mc.multi_state ([0.5,0.5], 'cms').run (conv_tol=1e-8)
            return mol, mc.nuc_grad_method ()
        mol1, mc_grad1 = get_lih (1.5)
        mol2, mc_grad2 = get_lih (1.55)
        mc_grad2 = mc_grad2.as_scanner ()
        for state in 0,1:
            de1 = mc_grad1.kernel (state=state)
            e1 = mc_grad1.base.e_states[state]
            e2, de2 = mc_grad2 (mol1, state=state)
            self.assertTrue(mc_grad1.converged)
            self.assertTrue(mc_grad2.converged)
            self.assertAlmostEqual (e1, e2, 6)
            self.assertAlmostEqual (lib.fp (de1), lib.fp (de2), 6)


if __name__ == "__main__":
    print("Full Tests for MS-PDFT gradient off-diagonal heff fns")
    unittest.main()






