#!/usr/bin/env python

from pyscf import gto
from pyscf import lib
import pyscf.lib.parameters as param

print("test contracted GTO")
mol = gto.Mole()
mol.atom = [
    [1  , (0.,1.,1.)],
    ["O", (0.,0.,0.)],
    [1  , (1.,1.,0.)], ]
mol.nucmod = { "O":param.MI_NUC_GAUSS, 3:param.MI_NUC_GAUSS }
mol.basis = {
    "O": [(0, 0, (15, 1)), ],
    "H": [(0, 0, (1, 1, 0), (3, 3, 1), (5, 1, 0)),
          (1, 0, (1, 1)), ]}
mol.basis['O'].extend(gto.mole.expand_etbs(((0, 4, 1, 1.8),
                                            (1, 3, 2, 1.8),
                                            (2, 2, 1, 1.8),)))

#TODO: test self.nucmod

mol.verbose = 4
mol.output = None
mol.build()


def test_num_basis(mol):
    print(mol.nao_nr())
    if mol.num_NR_function() == 34 and mol.num_2C_function() == 68:
        print("test_num_basis pass")
    else:
        print("test_num_basis fail")
test_num_basis(mol)


tao = [-2, 1, -4, 3, 6, -5, 10, -9, 8, -7, -12, 11, -14, 13, -16, 15, -18, 17,
       -20, 19, 22, -21, 26, -25, 24, -23, 28, -27, 32, -31, 30, -29, 34, -33,
       38, -37, 36, -35, -42, 41, -40, 39, -48, 47, -46, 45, -44, 43, -52, 51,
       -50, 49, -58, 57, -56, 55, -54, 53, -60, 59, -62, 61, 64, -63, 68, -67,
       66, -65]

if mol.time_reversal_map() == tao:
    print("time_reversal_map pass")
else:
    print("time_reversal_map fail")

mol.x = None
mol.check_sanity(mol)
