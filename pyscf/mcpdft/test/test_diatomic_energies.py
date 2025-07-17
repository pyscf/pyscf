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
from pyscf import gto, scf, df, fci
from pyscf.fci.addons import fix_spin_
from pyscf import mcpdft
#from pyscf.dft.openmolcas_grids import quasi_ultrafine
#from pyscf.fci import csf_solver
import unittest

# Need to use custom grids to get consistent agreement w/ the other program
# particularly for ftPBE test below.
# TODO: type in orbital initialization from OpenMolcas converged orbitals,
# which should result in agreement to 1e-8.

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

def diatomic (atom1, atom2, r, fnal, basis, ncas, nelecas, nstates, charge=None, spin=None,
              symmetry=False, cas_irrep=None, density_fit=False):
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format (atom1, atom2, r)
    mol = gto.M (atom=xyz, basis=basis, charge=charge, spin=spin, symmetry=symmetry, verbose=0, output='/dev/null')
    mf = scf.RHF (mol)
    if density_fit: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mf.kernel ()
    mc = mcpdft.CASSCF (mf, fnal, ncas, nelecas, grids_attr=my_grids)
    #if spin is not None: smult = spin+1
    #else: smult = (mol.nelectron % 2) + 1
    #mc.fcisolver = csf_solver (mol, smult=smult)
    if spin is None: spin = mol.nelectron%2
    ss = spin*(spin+2)*0.25
    mc = mc.multi_state ([1.0/float(nstates),]*nstates, 'cms')
    mc.fix_spin_(ss=ss, shift=1)
    mc.conv_tol = mc.conv_tol_sarot = 1e-12
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep (cas_irrep)
    mc.kernel (mo)
    return mc.e_states

def tearDownModule():
    global diatomic
    del diatomic

class KnownValues(unittest.TestCase):

    def test_h2_cms3ftlda22_sto3g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        e_ref = [-1.02544144, -0.44985771, -0.23390995]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_h2_cms2ftlda22_sto3g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        e_ref = [-1.11342858, -0.50064433]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_h2_cms3ftlda22_631g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 3)
        e_ref = [-1.08553117, -0.69136123, -0.49602992]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 4)

    def test_h2_cms2ftlda22_631g (self):
        e = diatomic ('H', 'H', 1.3, 'ftLDA,VWN3', '6-31G', 2, 2, 2)
        e_ref = [-1.13120015, -0.71600911]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms2ftlda44_sto3g (self):
        e = diatomic ('Li', 'H', 1.8, 'ftLDA,VWN3', 'STO-3G', 4, 4, 2, symmetry=True, cas_irrep={'A1': 4})
        e_ref = [-7.86001566, -7.71804507]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms2ftlda22_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2)
        e_ref = [-7.77572652, -7.68950326]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_lih_cms2ftpbe22_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftPBE', 'STO-3G', 2, 2, 2)
        e_ref = [-7.83953187, -7.75506453]
        # Reference values obtained with OpenMolcas
        #   version: 22.02
        #   tag: 277-gd1f6a7392
        #   commit: c3bdc83f9213a511233096e94715be3bbc73fb94
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 6)

    def test_lih_cms2ftlda22_sto3g_df (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 2, density_fit=True)
        e_ref = [-7.776307, -7.689764]
        # Reference values from this program
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 5)

    def test_lih_cms3ftlda22_sto3g (self):
        e = diatomic ('Li', 'H', 2.5, 'ftLDA,VWN3', 'STO-3G', 2, 2, 3)
        e_ref = [-7.79692534, -7.64435032, -7.35033371]
        # Reference values obtained with OpenMolcas
        #   version: 21.06 
        #   tag: 109-gbd596f6ca-dirty
        #   commit: bd596f6cabd6da0301f3623af2de6a14082b34b5
        for i in range (3):
            with self.subTest (state=i):
                self.assertAlmostEqual (e[i], e_ref[i], 5)


if __name__ == "__main__":
    print("Full Tests for CMS-PDFT energies of diatomic molecules")
    unittest.main()






