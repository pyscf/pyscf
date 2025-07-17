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
from pyscf import gto, scf
from pyscf.dft.libxc import XC_KEYS, XC_ALIAS, hybrid_coeff, rsh_coeff
from pyscf.dft.libxc import parse_xc, is_nlc, needs_laplacian
from pyscf import mcpdft
from pyscf.mcpdft.otfnal import make_hybrid_fnal
import unittest
from itertools import product

h2 = scf.RHF (gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = 'sto-3g', 
                     output='/dev/null', verbose=0)).run ()
mc = mcpdft.CASSCF (h2, 'tPBE', 2, 2, grids_level=1).run ()
LIBXC_KILLS = ['GGA_X_LB','GGA_X_LBM','LDA_XC_TIH']

def _test_hybrid_and_decomp (kv, xc):
    txc = 't'+xc
    e_pdft, e_ot = mc.energy_tot (otxc=txc)
    decomp = mc.get_energy_decomposition (otxc=txc)
    e_nuc, e_1e, e_coul, e_otx, e_otc, e_mcwfn = decomp
    kv.assertAlmostEqual (e_otx+e_otc, e_ot, 10)
    kv.assertAlmostEqual (e_nuc+e_1e+e_coul, e_pdft-e_ot, 10)
    htxc = 't'+make_hybrid_fnal (xc, 0.25)
    e_hyb = mc.energy_tot (otxc=htxc)[0]
    kv.assertAlmostEqual (e_hyb, 0.75*e_pdft+0.25*mc.e_mcscf, 10)

# _itrf.LIBXC_is_meta_gga returns an error code (-1) if the functional isn't found
# Abuse this to screen broken or unavailable KS-DFT functionals
# Cf. PySCF issue #1692
# TODO: revisit after _itrf or libxc.py is improved to handle error codes
def is_laplacian_meta_gga_or_broken(xc_code):
    if xc_code is None:
        return None
    elif isinstance(xc_code, str):
        if is_nlc(xc_code):
            return 'NLC'
        hyb, fn_facs = parse_xc(xc_code)
    else:
        fn_facs = [(xc_code, 1)]  # mimic fn_facs
    if not fn_facs:
        return False

    else:
        return any(needs_laplacian(xid) for xid, fac in fn_facs)

def _skip_key (key):
    xc = str (key)
    return (xc in LIBXC_KILLS
            or '_XC_' in xc
            or '_K_' in xc
            or xc.startswith ('HYB_')
            or is_laplacian_meta_gga_or_broken (xc)
            or hybrid_coeff (xc)
            or rsh_coeff (xc)[0])

def setUpModule ():
    global h2, mc, XC_TEST_KEYS
    XC_KEYS1 = {key for key in XC_KEYS if not _skip_key (key)}
    XC_KEYS2 = {key for key, val in XC_ALIAS.items () if not _skip_key (val)}
    XC_TEST_KEYS = XC_KEYS1.union (XC_KEYS2)

def tearDownModule():
    global h2, mc, XC_TEST_KEYS
    h2.mol.stdout.close ()
    del h2, mc, XC_TEST_KEYS

class KnownValues(unittest.TestCase):

    def test_combo_fnals (self):
        # just a sanity test for the string parsing
        x_list = ["", "LDA", "0.4*LDA+0.6*B88", "0.5*LDA+0.7*B88-0.2*MPW91"]
        c_list = ["", "VWN3", "0.33*VWN3+0.67*LYP", "0.43*VWN3-0.77*LYP-0.2*P86"]
        for x, c in product(x_list, c_list):
            xc = x + "," + c
            if xc == ",":
                continue
            with self.subTest(x=x, c=c):
                _test_hybrid_and_decomp(self, xc)

    def test_many_fnals (self):
        # sanity test for built-in functionals
        # many functionals are expected to fail and must be skipped
        for xc in XC_TEST_KEYS:
            with self.subTest (xc=xc):
                _test_hybrid_and_decomp (self, xc)

    def test_null_fnal (self):
        _test_hybrid_and_decomp (self, '')

if __name__ == "__main__":
    print("Full Tests for MC-PDFT on-top functional class API")
    unittest.main()






