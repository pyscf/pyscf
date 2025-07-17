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
from pyscf.data.nist import BOHR
from pyscf import mcpdft
#from pyscf.fci import csf_solver
from pyscf.grad.cmspdft import diab_response, diab_grad, diab_response_o0, diab_grad_o0
from pyscf.grad import mspdft as mspdft_grad
#from pyscf.df.grad import dfsacasscf, dfmspdft
import unittest, math

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, output='/dev/null', verbose = 0)
Lis = math.pi * (np.random.rand (1) - 0.5)
def get_mc_ref (mol, ri=False, sam=False):
    mf = scf.RHF (mol)
    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcscf.CASSCF (mf.run (), 6, 6)
    mo = None
    ci0 = None
    if sam:
        #fcisolvers = [csf_solver (mol, smult=((2*i)+1)) for i in (0,1)]
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
        #mc.fcisolver = csf_solver (mol, smult=1)
        if mol.symmetry:
            mc.fcisolver.wfnsym = 'A1'
        mc = mc.state_average ([0.5,0.5])
        mc.fix_spin_(ss=0)
    mc.conv_tol = 1e-12
    return mc.run (mo, ci0)
#mc_list = [[[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)] for m in (mol_nosymm, mol_symm)]
mc_list = [] # Crunch within unittest.main for accurate clock
def get_mc_list ():
    if len (mc_list) == 0:
        for m in [mol_nosymm, mol_symm]:
            mc_list.append ([[get_mc_ref (m, ri=i, sam=j) for i in (0,1)] for j in (0,1)])
    return mc_list

def tearDownModule():
    global mol_nosymm, mol_symm, mc_list, Lis, get_mc_list
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mc_list, Lis, get_mc_list

class KnownValues(unittest.TestCase):

    def test_diab_response_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
            for mca, atype in zip (mcs, ('nomix','mix')):
                if atype == 'mix': continue # TODO: enable state-average-mix
                for mc, itype in zip (mca, ('conv', 'DF')):
                    ci_arr = np.asarray (mc.ci)
                    if itype == 'conv': mc_grad = mc.nuc_grad_method ()
                    else: continue #mc_grad = dfsacasscf.Gradients (mc)
                    eris = mc.ao2mo (mc.mo_coeff)
                    with self.subTest (symm=stype, solver=atype, eri=itype, check='energy convergence'):
                        self.assertTrue (mc.converged)
                    def _crunch (fn):
                        dw = fn (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
                        dworb, dwci = mc_grad.unpack_uniq_var (dw)
                        return dworb, dwci
                    with self.subTest (symm=stype, solver=atype, eri=itype):
                        dworb_test, dwci_test = _crunch (diab_response)
                        dworb_ref, dwci_ref = _crunch (diab_response_o0)
                    with self.subTest (symm=stype, solver=atype, eri=itype, check='orb'):
                        self.assertAlmostEqual (lib.fp (dworb_test), lib.fp (dworb_ref), 8)
                    with self.subTest (symm=stype, solver=atype, eri=itype, check='CI'):
                        self.assertAlmostEqual (lib.fp (dwci_test), lib.fp (dwci_ref), 8)

    def test_diab_grad_sanity (self):
        for mcs, stype in zip (get_mc_list (), ('nosymm','symm')):
            for mca, atype in zip (mcs, ('nomix','mix')):
                for mc, itype in zip (mca, ('conv', 'DF')):
                    ci_arr = np.asarray (mc.ci)
                    if itype == 'conv': mc_grad = mc.nuc_grad_method ()
                    else: continue #mc_grad = dfsacasscf.Gradients (mc)
                    # TODO: proper DF functionality
                    eris = mc.ao2mo (mc.mo_coeff)
                    mf_grad = mc._scf.nuc_grad_method ()
                    with self.subTest (symm=stype, solver=atype, eri=itype):
                        dh_test = diab_grad (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris, mf_grad=mf_grad)
                        dh_ref = diab_grad_o0 (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris, mf_grad=mf_grad)
                        self.assertAlmostEqual (lib.fp (dh_test), lib.fp (dh_ref), 8)

if __name__ == "__main__":
    print("Full Tests for CMS-PDFT gradient objective fn derivatives")
    unittest.main()






