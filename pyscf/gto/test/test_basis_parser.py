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
import tempfile
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib

class KnownValues(unittest.TestCase):
    def test_parse_pople(self):
        self.assertEqual(gto.basis._parse_pople_basis('631g(d)', 'C'),
                         ('pople-basis/6-31G.dat', 'pople-basis/6-31G-polarization-d.dat'))
        self.assertEqual(gto.basis._parse_pople_basis('631g**', 'C'),
                         ('pople-basis/6-31Gss.dat',))
        self.assertEqual(gto.basis._parse_pople_basis('631++g**', 'C'),
                         ('pople-basis/6-31++Gss.dat',))
        self.assertRaises(KeyError, gto.basis._parse_pople_basis, '631g++', 'C')

    def test_basis_load(self):
        self.assertEqual(gto.basis.load(__file__, 'H'), [])
        self.assertRaises(KeyError, gto.basis.load, 'abas', 'H')
        #self.assertRaises(RuntimeError, gto.basis.load(__file__, 'C'), [])

        self.assertEqual(len(gto.basis.load('631++g**', 'C')), 8)
        self.assertEqual(len(gto.basis.load('ccpcvdz', 'C')), 7)

        basdat = gto.basis.load('minao', 'C') + gto.basis.load('sto3g', 'C')
        basdat1 = gto.basis.parse_nwchem.parse(
            gto.basis.parse_nwchem.convert_basis_to_nwchem('C', basdat), 'C')
        bas = []
        for b in sorted(basdat, reverse=True):
            b1 = b[:1]
            for x in b[1:]:
                b1.append(list(x))
            bas.append(b1)
        bas = [b for b in bas if b[0]==0] + [b for b in bas if b[0]==1]
        self.assertEqual(bas, basdat1)

    def test_basis_load_ecp(self):
        self.assertEqual(gto.basis.load_ecp(__file__, 'H'), [])

    def test_parse_basis(self):
        basis_str = '''
#BASIS SET: (6s,3p) -> [2s,1p]
C    S
     71.6168370              0.15432897       
     13.0450960              0.53532814       
#
      3.5305122              0.44463454       
C    SP
      2.9412494             -0.09996723             0.15591627       
      0.6834831              0.39951283             0.60768372       
      0.2222899              0.70011547             0.39195739       '''
        self.assertRaises(KeyError, gto.basis.parse_nwchem.parse, basis_str, 'O')
        basis_dat = gto.basis.parse_nwchem.parse(basis_str)
        self.assertEqual(len(basis_dat), 3)

    def test_parse_ecp(self):
        ecp_str = '''
#
Na nelec 10
Na ul
1    175.5502590            -10.0000000        
2     35.0516791            -47.4902024        
#
2      7.9060270            -17.2283007        

Na S
0    243.3605846              3.0000000        
1     41.5764759             36.2847626        
2     13.2649167             72.9304880        
Na P
0   1257.2650682              5.0000000        
1    189.6248810            117.4495683        
2     54.5247759            423.3986704        
'''
        ecpdat = gto.basis.parse_nwchem.parse_ecp(ecp_str, 'Na')
        self.assertEqual(ecpdat[0], 10)
        self.assertEqual(len(ecpdat[1]), 3)
        ecpdat1 = gto.basis.parse_nwchem.parse_ecp(ecp_str)
        self.assertEqual(ecpdat, ecpdat1)

        ecpdat1 = gto.basis.parse_nwchem.parse_ecp(
            gto.basis.parse_nwchem.convert_ecp_to_nwchem('Na', ecpdat), 'Na')
        self.assertEqual(ecpdat, ecpdat1)

    def test_optimize_contraction(self):
        bas = gto.parse(r'''
#BASIS SET: (6s,3p) -> [2s,1p]
        C    S
              2.9412494             -0.09996723
              0.6834831              0.39951283
              0.2222899              0.70011547
        C    S
              2.9412494             0.15591627
              0.6834831             0.60768372
              0.2222899             0.39195739
                                    ''', optimize=True)
        self.assertEqual(len(bas), 1)

        bas = [[1, 0,
                [2.9412494, -0.09996723],
                [0.6834831,  0.39951283],
                [0.2222899,  0.70011547]],
               [1, 1,
                [2.9412494, -0.09996723],
                [0.6834831,  0.39951283],
                [0.2222899,  0.70011547]],
               [1, 1,
                [2.9412494,  0.15591627],
                [0.6834831,  0.60768372],
                [0.2222899,  0.39195739]]]
        bas = gto.basis.parse_nwchem.optimize_contraction(bas)
        self.assertEqual(len(bas), 2)

    def test_remove_zero(self):
        bas = gto.parse(r'''
        C    S
        7.2610457926   0.0000000000   0.0000000000
        2.1056583087   0.0000000000   0.0000000000
        0.6439906571   1.0000000000   0.0000000000
        0.0797152017   0.0000000000   1.0000000000
        0.0294029590   0.0000000000   0.0000000000
                                    ''')
        self.assertEqual(len(bas[0]), 3)

        bas = [[0, 0,
                [7.2610457926,  0.0000000000,  0.0000000000],
                [2.1056583087,  0.0000000000,  0.0000000000],
                [0.6439906571,  1.0000000000,  0.0000000000],
                [0.0797152017,  0.0000000000,  1.0000000000],
                [0.0294029590,  0.0000000000,  0.0000000000]]]
        bas = gto.basis.parse_nwchem.remove_zero(bas)
        self.assertEqual(len(bas[0]), 4)


if __name__ == "__main__":
    print("test basis module")
    unittest.main()
