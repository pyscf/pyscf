#!/usr/bin/env python

import unittest
import numpy
from pyscf import fci


class KnowValues(unittest.TestCase):
    def test_strings4orblist(self):
        ref = ['0b1010', '0b100010', '0b101000', '0b10000010', '0b10001000',
               '0b10100000']
        self.assertEqual(fci.cistring.gen_strings4orblist([1,3,5,7], 2),
                         [int(x,2) for x in ref])
        ref = ['0b11', '0b101', '0b110', '0b1001', '0b1010', '0b1100',
               '0b10001', '0b10010', '0b10100', '0b11000']
        self.assertEqual(fci.cistring.gen_strings4orblist(range(5), 2),
                         [int(x,2) for x in ref])

    def test_linkstr_index(self):
        idx1 = fci.cistring.gen_linkstr_index_o0(range(4), 2)
        idx2 = fci.cistring.gen_linkstr_index(range(4), 2)
        idx23 = numpy.array([[0, 0, 3, 1],
                             [3, 3, 3, 1],
                             [1, 0, 4, 1],
                             [2, 0, 5, 1],
                             [1, 3, 0, 1],
                             [2, 3, 1, 1],])
        self.assertTrue(numpy.all(idx1[:,:,2:] == idx2[:,:,2:]))
        self.assertTrue(numpy.all(idx23 == idx2[3]))

    def test_addr2str(self):
        self.assertEqual(bin(fci.cistring.addr2str(6, 3, 7)), '0b11001')
        self.assertEqual(bin(fci.cistring.addr2str(6, 3, 8)), '0b11010')
        self.assertEqual(bin(fci.cistring.addr2str(7, 4, 9)), '0b110011')

    def test_str2addr(self):
        self.assertEqual(fci.cistring.str2addr(6, 3, int('0b11001' ,2)), 7)
        self.assertEqual(fci.cistring.str2addr(6, 3, int('0b11010' ,2)), 8)
        self.assertEqual(fci.cistring.str2addr(7, 4, int('0b110011',2)), 9)

    def test_gen_cre_str_index(self):
        idx = fci.cistring.gen_cre_str_index(range(4), 2)
        idx0 = [[[ 2, 3, 0, 1], [ 3, 6, 1, 1]],
                [[ 1, 1, 0,-1], [ 3, 6, 2, 1]],
                [[ 0, 0, 0, 1], [ 3, 6, 3, 1]],
                [[ 1, 1, 1,-1], [ 2, 3, 2,-1]],
                [[ 0, 0, 1, 1], [ 2, 3, 3,-1]],
                [[ 0, 0, 2, 1], [ 1, 1, 3, 1]]]
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_gen_des_str_index(self):
        idx = fci.cistring.gen_des_str_index(range(4), 2)
        idx0 = [[[ 0, 0, 1,-1], [ 1, 1, 0, 1]],
                [[ 0, 0, 2,-1], [ 2, 3, 0, 1]],
                [[ 1, 1, 2,-1], [ 2, 3, 1, 1]],
                [[ 0, 0, 3,-1], [ 3, 6, 0, 1]],
                [[ 1, 1, 3,-1], [ 3, 6, 1, 1]],
                [[ 2, 3, 3,-1], [ 3, 6, 2, 1]]],
        self.assertTrue(numpy.allclose(idx, idx0))


if __name__ == "__main__":
    print("Full Tests for CI string")
    unittest.main()

