import cistring
import direct_ms0
import direct_spin0
import direct_spin1
import direct_uhf
import direct_ms0_symm
import direct_spin0_symm
import direct_spin1_symm
import addons
from cistring import num_strings
from rdm import reorder_rdm

def solver(mol, singlet=True):
    if mol.symmetry:
        if singlet:
            return direct_spin0_symm.FCISolver(mol)
        else:
            return direct_spin1_symm.FCISolver(mol)
    else:
        if singlet:
            return direct_spin0.FCISolver(mol)
        else:
            return direct_spin1.FCISolver(mol)

