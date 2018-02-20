#!/usr/bin/env python

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

        self.assertEqual(len(gto.basis.load('631++g**', 'C')), 8)
        self.assertEqual(len(gto.basis.load('ccpcvdz', 'C')), 7)

        basdat = gto.basis.load('minao', 'C')
        basdat1 = gto.basis.parse_nwchem.parse(
            gto.basis.parse_nwchem.convert_basis_to_nwchem('C', basdat), 'C')
        bas = []
        for b in basdat:
            b1 = b[:1]
            for x in b[1:]:
                b1.append(list(x))
            bas.append(b1)
        self.assertEqual(bas, basdat1)

    def test_basis_load_ecp(self):
        self.assertEqual(gto.basis.load(__file__, 'H'), [])

    def test_parse_ecp(self):
        ecpdat = gto.basis.parse_nwchem.parse_ecp('''
#
Na nelec 10
Na ul
1    175.5502590            -10.0000000        
2     35.0516791            -47.4902024        
2      7.9060270            -17.2283007        

Na S
0    243.3605846              3.0000000        
1     41.5764759             36.2847626        
2     13.2649167             72.9304880        
Na P
0   1257.2650682              5.0000000        
1    189.6248810            117.4495683        
2     54.5247759            423.3986704        
                                         ''', 'Na')
        self.assertEqual(ecpdat[0], 10)
        self.assertEqual(len(ecpdat[1]), 3)
        ecpdat1 = gto.basis.parse_nwchem.parse_ecp('''
#
Na nelec 10
Na ul
1    175.5502590            -10.0000000        
2     35.0516791            -47.4902024        
2      7.9060270            -17.2283007        

Na S
0    243.3605846              3.0000000        
1     41.5764759             36.2847626        
2     13.2649167             72.9304880        
Na P
0   1257.2650682              5.0000000        
1    189.6248810            117.4495683        
2     54.5247759            423.3986704        
                                         ''')
        self.assertEqual(ecpdat, ecpdat1)

        ecpdat1 = gto.basis.parse_nwchem.parse_ecp(
            gto.basis.parse_nwchem.convert_ecp_to_nwchem('Na', ecpdat), 'Na')
        self.assertEqual(ecpdat, ecpdat1)


if __name__ == "__main__":
    print("test basis module")
    unittest.main()
