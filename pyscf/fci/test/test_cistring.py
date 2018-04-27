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
import numpy
from pyscf.fci import cistring


class KnownValues(unittest.TestCase):
    def test_strings4orblist(self):
        ref = ['0b1010', '0b100010', '0b101000', '0b10000010', '0b10001000',
               '0b10100000']
        for i, x in enumerate(cistring.gen_strings4orblist([1,3,5,7], 2)):
            self.assertEqual(bin(x), ref[i])
        ref = ['0b11', '0b101', '0b110', '0b1001', '0b1010', '0b1100',
               '0b10001', '0b10010', '0b10100', '0b11000']
        for i, x in enumerate(cistring.gen_strings4orblist(range(5), 2)):
            self.assertEqual(bin(x), ref[i])

    def test_linkstr_index(self):
        idx1 = cistring.gen_linkstr_index_o0(range(4), 2)
        idx2 = cistring.gen_linkstr_index(range(4), 2)
        idx23 = numpy.array([[0, 0, 3, 1],
                             [3, 3, 3, 1],
                             [1, 0, 4, 1],
                             [2, 0, 5, 1],
                             [1, 3, 0, 1],
                             [2, 3, 1, 1],])
        self.assertTrue(numpy.all(idx1[:,:,2:] == idx2[:,:,2:]))
        self.assertTrue(numpy.all(idx23 == idx2[3]))

        idx1 = cistring.gen_linkstr_index(range(7), 3)
        idx2 = cistring.reform_linkstr_index(idx1)
        idx3 = cistring.gen_linkstr_index_trilidx(range(7), 3)
        idx3[:,:,1] = 0
        self.assertTrue(numpy.all(idx2 == idx3))

    def test_addr2str(self):
        self.assertEqual(bin(cistring.addr2str(6, 3, 7)), '0b11001')
        self.assertEqual(bin(cistring.addr2str(6, 3, 8)), '0b11010')
        self.assertEqual(bin(cistring.addr2str(7, 4, 9)), '0b110011')

    def test_str2addr(self):
        self.assertEqual(cistring.str2addr(6, 3, int('0b11001' ,2)), 7)
        self.assertEqual(cistring.str2addr(6, 3, int('0b11010' ,2)), 8)
        self.assertEqual(cistring.str2addr(7, 4, int('0b110011',2)), 9)

    def test_gen_cre_str_index(self):
        idx = cistring.gen_cre_str_index(range(4), 2)
        idx0 = [[[ 2, 0, 0, 1], [ 3, 0, 1, 1]],
                [[ 1, 0, 0,-1], [ 3, 0, 2, 1]],
                [[ 0, 0, 0, 1], [ 3, 0, 3, 1]],
                [[ 1, 0, 1,-1], [ 2, 0, 2,-1]],
                [[ 0, 0, 1, 1], [ 2, 0, 3,-1]],
                [[ 0, 0, 2, 1], [ 1, 0, 3, 1]]]
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_gen_des_str_index(self):
        idx = cistring.gen_des_str_index(range(4), 2)
        idx0 = [[[ 0, 0, 1,-1], [ 0, 1, 0, 1]],
                [[ 0, 0, 2,-1], [ 0, 2, 0, 1]],
                [[ 0, 1, 2,-1], [ 0, 2, 1, 1]],
                [[ 0, 0, 3,-1], [ 0, 3, 0, 1]],
                [[ 0, 1, 3,-1], [ 0, 3, 1, 1]],
                [[ 0, 2, 3,-1], [ 0, 3, 2, 1]]],
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_tn_strs(self):
        self.assertEqual(t1strs(7, 3), cistring.tn_strs(7, 3, 1).tolist())
        self.assertEqual(t2strs(7, 3), cistring.tn_strs(7, 3, 2).tolist())

    def test_sub_addrs(self):
        addrs = cistring.sub_addrs(6, 3, (0,2,3,5))
        self.assertEqual([bin(x) for x in cistring.addrs2str(6, 3, addrs)],
                         ['0b1101', '0b100101', '0b101001', '0b101100'])

        addrs = cistring.sub_addrs(6, 3, (3,0,5,2))
        self.assertEqual([bin(x) for x in cistring.addrs2str(6, 3, addrs)],
                         ['0b101001', '0b1101', '0b101100', '0b100101'])

        addrs = cistring.sub_addrs(6, 3, (3,0,5,2), 2)
        self.assertEqual([bin(x) for x in cistring.addrs2str(6, 3, addrs)],
                         ['0b111', '0b1011', '0b1110', '0b10101', '0b11001', '0b11100',
                          '0b100011', '0b100110', '0b101010', '0b110001', '0b110100', '0b111000'])

        addrs = cistring.sub_addrs(6, 3, (0,2,3,5), 2)
        self.assertEqual([bin(x) for x in cistring.addrs2str(6, 3, addrs)],
                         ['0b111', '0b1011', '0b1110', '0b10101', '0b11001', '0b11100',
                          '0b100011', '0b100110', '0b101010', '0b110001', '0b110100', '0b111000'])

        addrs = cistring.sub_addrs(6, 3, (0,2,3,5), 1)
        self.assertEqual([bin(x) for x in cistring.addrs2str(6, 3, addrs)],
                         ['0b10011', '0b10110', '0b11010', '0b110010'])

def t1strs(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    t1s = []
    for a in range(nocc, norb):
        for i in reversed(range(nocc)):
            t1s.append(hf_str ^ (1 << i) | (1 << a))
    return t1s

def t2strs(norb, nelec):
    nocc = nelec
    nvir = norb - nocc
    hf_str = int('1'*nocc, 2)
    ij_allow = [(1<<i)^(1<<j) for i in reversed(range(nocc)) for j in reversed(range(i))]
    ab_allow = [(1<<i)^(1<<j) for i in range(nocc,norb) for j in range(nocc,i)]
    t2s = []
    for ab in ab_allow:
        for ij in ij_allow:
            t2s.append(hf_str ^ ij | ab)
    return t2s

if __name__ == "__main__":
    print("Full Tests for CI string")
    unittest.main()

