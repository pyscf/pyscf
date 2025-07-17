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
from pyscf.data.nist import HARTREE2EV
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf import mcpdft
import unittest


##############################################################################
# Inline CC-PVTZ basis set for Be as implemented in OpenMolcas v21.10        #
##############################################################################

Bebasis = gto.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (11s,5p,2d,1f) -> [4s,3p,2d,1f]
Be    S
      6.863000E+03           2.360000E-04           0.000000E+00          -4.300000E-05           0.000000E+00
      1.030000E+03           1.826000E-03           0.000000E+00          -3.330000E-04           0.000000E+00
      2.347000E+02           9.452000E-03           0.000000E+00          -1.736000E-03           0.000000E+00
      6.656000E+01           3.795700E-02           0.000000E+00          -7.012000E-03           0.000000E+00
      2.169000E+01           1.199650E-01           0.000000E+00          -2.312600E-02           0.000000E+00
      7.734000E+00           2.821620E-01           0.000000E+00          -5.813800E-02           0.000000E+00
      2.916000E+00           4.274040E-01           0.000000E+00          -1.145560E-01           0.000000E+00
      1.130000E+00           2.662780E-01           0.000000E+00          -1.359080E-01           0.000000E+00
      2.577000E-01           1.819300E-02           1.000000E+00           2.280260E-01           0.000000E+00
      1.101000E-01          -7.275000E-03           0.000000E+00           5.774410E-01           0.000000E+00
      4.409000E-02           1.903000E-03           0.000000E+00           3.178730E-01           1.000000E+00
Be    P
      7.436000E+00           0.000000E+00           1.073600E-02           0.000000E+00
      1.577000E+00           0.000000E+00           6.285400E-02           0.000000E+00
      4.352000E-01           0.000000E+00           2.481800E-01           0.000000E+00
      1.438000E-01           1.000000E+00           5.236990E-01           0.000000E+00
      4.994000E-02           0.000000E+00           3.534250E-01           1.000000E+00
Be    D
      3.493000E-01           1.000000E+00           0.000000E+00
      1.724000E-01           0.000000E+00           1.000000E+00
Be    F
      3.423000E-01           1.0000000
END
                    ''')

##############################################################################
# Inline definition of quadrature grid that can be implemented in OpenMolcas #
# v21.10. The corresponding input to the SEWARD module in OpenMolcas is      #
#   grid input                                                               #
#   nr=100                                                                   #
#   lmax=41                                                                  #
#   rquad=ta                                                                 #
#   nopr                                                                     #
#   noro                                                                     #
#   end of grid input                                                        #
##############################################################################

om_ta_alpha = [0.8, 0.9, # H, He
    1.8, 1.4, # Li, Be
        1.3, 1.1, 0.9, 0.9, 0.9, 0.9, # B - Ne
    1.4, 1.3, # Na, Mg
        1.3, 1.2, 1.1, 1.0, 1.0, 1.0, # Al - Ar
    1.5, 1.4, # K, Ca
            1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, # Sc - Zn
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9] # Ga - Kr
def om_treutler_ahlrichs(n, chg, *args, **kwargs):
    '''
    "Treutler-Ahlrichs" as implemented in OpenMolcas
    '''
    r = np.empty(n)
    dr = np.empty(n)
    alpha = om_ta_alpha[chg-1]
    step = 2.0 / (n+1) # = numpy.pi / (n+1)
    ln2 = alpha / np.log(2)
    for i in range(n):
        x = (i+1)*step - 1 # = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * np.log((1-x)/2)
        dr[i] = (step #* numpy.sin((i+1)*step) 
                * ln2*(1+x)**.6 *(-.6/(1+x)*np.log((1-x)/2)+1/(1-x)))
    return r[::-1], dr[::-1]

my_grids = {'atom_grid': (99,590),
    'radi_method': om_treutler_ahlrichs,
    'prune': False,
    'radii_adjust': None}

##############################################################################
# Reference from OpenMolcas v21.10. tPBE and tBLYP values are consistent w/  #
# JCTC 10, 3669 (2014) (after erratum!)                                      #
##############################################################################

ref = {'N': {'tPBE': 2.05911249,
             'tBLYP': 1.97701243,
             'ftPBE': 1.50034408,
             'ftBLYP': 1.31520533},
       'Be': {'tPBE': -2.56743750,
              'tBLYP': -2.56821329,
              'ftPBE': -2.59816134,
              'ftBLYP': -2.58502831}}

Natom = scf.RHF (gto.M (atom = 'N 0 0 0', basis='cc-pvtz', spin=3, symmetry='D2h', output='/dev/null'))
Beatom = scf.RHF (gto.M (atom = 'Be 0 0 0', basis=Bebasis, spin=2, symmetry='D2h', output='/dev/null'))
Natom_hs = [Natom, (4,1), 'Au', None]
Natom_ls = [Natom, (3,2), 'B3u', None]
Beatom_ls = [Beatom, (1,1), 'Ag', None]
Beatom_hs = [Beatom, (2,0), 'B3u', None]

calcs = {'N': (Natom_hs, Natom_ls),
         'Be': (Beatom_hs, Beatom_ls)}

def check_calc (calc):
    if not calc[0].converged: calc[0].kernel ()
    if calc[-1] is None:
        mf = calc[0]
        mol = mf.mol
        nelecas = calc[1]
        s = (nelecas[1]-nelecas[0])*0.5
        ss = s*(s+1)
        mc = mcscf.CASSCF (mf, 4, nelecas).set (conv_tol=1e-10)
        mc.fix_spin_(ss=ss)
        mc.fcisolver.wfnsym = calc[2]
        calc[-1] = mc
    if not calc[-1].converged:
        calc[-1].kernel ()
    return calc[-1]

def get_gap (gs, es, fnal):
    gs = check_calc (gs)
    es = check_calc (es)
    e0 = mcpdft.CASSCF (gs._scf, fnal, gs.ncas, gs.nelecas, grids_attr=my_grids).set (
        fcisolver = gs.fcisolver, conv_tol=1e-10).kernel (
        gs.mo_coeff, gs.ci)[0]
    e1 = mcpdft.CASSCF (es._scf, fnal, es.ncas, es.nelecas, grids_attr=my_grids).set (
        fcisolver = es.fcisolver, conv_tol=1e-10).kernel (
        es.mo_coeff, es.ci)[0]
    return (e1-e0)*HARTREE2EV

def tearDownModule():
    global Natom, Natom_hs, Natom_ls, Beatom, Beatom_ls, Beatom_hs
    Natom.mol.stdout.close ()
    Beatom.mol.stdout.close ()
    del Natom, Natom_hs, Natom_ls, Beatom, Beatom_ls, Beatom_hs

class KnownValues(unittest.TestCase):

    def test_gaps (self):
        for atom in 'N', 'Be':
            for fnal in 'tPBE','tBLYP','ftPBE','ftBLYP':
                my_ref = ref[atom][fnal]
                my_test = get_gap (*calcs[atom], fnal)
                with self.subTest (atom=atom, fnal=fnal):
                    self.assertAlmostEqual (my_test, my_ref, 5)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT energies of N and Be atom spin states")
    unittest.main()






