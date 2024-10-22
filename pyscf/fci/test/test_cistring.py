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


def gen_cre_str_index(orb_list, nelec):
    '''Slow version of gen_cre_str_index function'''
    cre_strs = cistring.make_strings(orb_list, nelec+1)
    credic = dict(zip(cre_strs,range(cre_strs.__len__())))
    def progate1e(str0):
        linktab = []
        for i in orb_list:
            if not str0 & (1 << i):
                str1 = str0 | (1 << i)
                linktab.append((i, 0, credic[str1], cistring.cre_sign(i, str0)))
        return linktab

    strs = cistring.make_strings(orb_list, nelec)
    t = [progate1e(s) for s in strs.astype(numpy.int64)]
    return numpy.array(t, dtype=numpy.int32)

def gen_des_str_index(orb_list, nelec):
    '''Slow version of gen_des_str_index function'''
    des_strs = cistring.make_strings(orb_list, nelec-1)
    desdic = dict(zip(des_strs,range(des_strs.__len__())))
    def progate1e(str0):
        linktab = []
        for i in orb_list:
            if str0 & (1 << i):
                str1 = str0 ^ (1 << i)
                linktab.append((0, i, desdic[str1], cistring.des_sign(i, str0)))
        return linktab

    strs = cistring.make_strings(orb_list, nelec)
    t = [progate1e(s) for s in strs.astype(numpy.int64)]
    return numpy.array(t, dtype=numpy.int32)

def gen_linkstr_index(orb_list, nelec, strs=None):
    if strs is None:
        strs = cistring.make_strings(orb_list, nelec)
    strdic = dict(zip(strs,range(strs.__len__())))
    def propagate1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        linktab = []
        for i in occ:
            linktab.append((i, i, strdic[str0], 1))
        for i in occ:
            for a in vir:
                str1 = str0 ^ (1 << i) | (1 << a)
                # [cre, des, target_address, parity]
                linktab.append((a, i, strdic[str1], cistring.cre_des_sign(a, i, str0)))
        return linktab

    t = [propagate1e(s) for s in strs.astype(numpy.int64)]
    return numpy.array(t, dtype=numpy.int32)

def addr2str(norb, nelec, addr):
    assert cistring.num_strings(norb, nelec) > addr
    if norb > 64:
        raise NotImplementedError('norb > 64')
    if addr == 0 or nelec == norb or nelec == 0:
        return (1 << nelec) - 1   # ..0011..11
    else:
        for i in reversed(range(norb)):
            addrcum = cistring.num_strings(i, nelec)
            if addrcum <= addr:
                return (1 << i) | addr2str(i, nelec-1, addr-addrcum)

def str2addr(norb, nelec, string):
    if norb <= nelec or nelec == 0:
        return 0
    elif (1<<(norb-1)) & string:  # remove the first bit
        return cistring.num_strings(norb-1, nelec) \
                + str2addr(norb-1, nelec-1, string^(1<<(norb-1)))
    else:
        return str2addr(norb-1, nelec, string)

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

        strs = cistring.gen_strings4orblist(range(8), 4)
        occlst = cistring.gen_occslst(range(8), 4)
        self.assertAlmostEqual(abs(occlst - cistring._strs2occslst(strs, 8)).sum(), 0, 12)
        self.assertAlmostEqual(abs(strs - cistring._occslst2strs(occlst)).sum(), 0, 12)

    def test_linkstr_index(self):
        idx1 = gen_linkstr_index(range(4), 2)
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

        tab1 = gen_cre_str_index(range(8), 4)
        tab2 = cistring.gen_cre_str_index(range(8), 4)
        self.assertAlmostEqual(abs(tab1 - tab2).max(), 0, 12)
        tab1 = gen_des_str_index(range(8), 4)
        tab2 = cistring.gen_des_str_index(range(8), 4)
        self.assertAlmostEqual(abs(tab1 - tab2).max(), 0, 12)

        tab1 = gen_linkstr_index(range(8), 4)
        tab2 = cistring.gen_linkstr_index(range(8), 4)
        self.assertAlmostEqual(abs(tab1 - tab2).sum(), 0, 12)
        tab3 = cistring.gen_linkstr_index_o1(range(8), 4)
        self.assertAlmostEqual(abs(tab1 - tab3).sum(), 0, 12)

    def test_addr2str(self):
        self.assertEqual(bin(cistring.addr2str(6, 3, 7)), '0b11001')
        self.assertEqual(bin(cistring.addr2str(6, 3, 8)), '0b11010')
        self.assertEqual(bin(cistring.addr2str(7, 4, 9)), '0b110011')

        self.assertEqual(addr2str(6, 3, 7), cistring.addr2str(6, 3, 7))
        self.assertEqual(addr2str(6, 3, 8), cistring.addr2str(6, 3, 8))
        self.assertEqual(addr2str(7, 4, 9), cistring.addr2str(7, 4, 9))

        self.assertEqual(bin(cistring.addr2str(6, 3, 7)), '0b11001')
        self.assertEqual(bin(cistring.addr2str(6, 3, 8)), '0b11010')
        self.assertEqual(bin(cistring.addr2str(7, 4, 9)), '0b110011')

        # Test large address
        string = 0b101101111101101111001110111111110111111100
        address = cistring.str2addr(norb=63, nelec=32, string=string)
        string2 = cistring.addr2str(norb=63, nelec=32, addr=address)
        self.assertEqual(string, string2)

    def test_str2addr(self):
        self.assertEqual(str2addr(6, 3, int('0b11001' ,2)), cistring.str2addr(6, 3, int('0b11001' ,2)))
        self.assertEqual(str2addr(6, 3, int('0b11010' ,2)), cistring.str2addr(6, 3, int('0b11010' ,2)))
        self.assertEqual(str2addr(7, 4, int('0b110011',2)), cistring.str2addr(7, 4, int('0b110011',2)))

        self.assertEqual(cistring.str2addr(6, 3, int('0b11001' ,2)), 7)
        self.assertEqual(cistring.str2addr(6, 3, int('0b11010' ,2)), 8)
        self.assertEqual(cistring.str2addr(7, 4, int('0b110011',2)), 9)
        self.assertEqual(cistring.str2addr(6, 3, cistring.addr2str(6, 3, 7)), 7)
        self.assertEqual(cistring.str2addr(6, 3, cistring.addr2str(6, 3, 8)), 8)
        self.assertEqual(cistring.str2addr(7, 4, cistring.addr2str(7, 4, 9)), 9)

        self.assertTrue(all(numpy.arange(20) ==
                            cistring.strs2addr(6, 3, cistring.addrs2str(6, 3, range(20)))))

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

    def test_parity(self):
        strs = cistring.gen_strings4orblist(range(5), 3)
        links = cistring.gen_linkstr_index(range(5), 3)
        parity = []
        for addr0, link in enumerate(links):
            parity.append([cistring.parity(strs[addr0], strs[addr1])
                           for addr1 in link[:,2]])
        self.assertEqual(parity, links[:,:,3].tolist())

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
