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

from ase.build import bulk

A2B = 1.889725989

# All lattice constants in this file are taken from 
# Jochen Heyd, Juan E. Peralta, Gustavo E. Scuseria, and Richard L. Martin, J. Chem. Phys. 123, 174101 (2005)
# except LiH, LiF, LiCl, and ZnO (taken from ./lattice.py)

def get_ase_atom(formula):
    formula = formula.lower()
    assert formula in ['c', 'si', 'ge', 'sic', 'bn', 'bp', 'bas', 'bsb', 'aln', 'alp', 'alas', 'alsb', 'gan', 'beta-gan', 'gap', 'gaas', 'gasb', 'inn', 'inp', 'inas', 'insb', 'zns', 'znse', 'znte', 'cds', 'cdse', 'cdte', 'mgo', 'mgs', 'mgse', 'mgte', 'cas', 'case', 'cate', 'srs', 'srse', 'srte', 'bas', 'base', 'bate', 'lih', 'lif', 'licl', 'zno']
    if formula == "c":
        ase_atom = get_ase_diamond_primitive("C")
    elif formula == "si":
        ase_atom = get_ase_diamond_primitive("Si")
    elif formula == "ge":
        ase_atom = get_ase_diamond_primitive("Ge")
    elif formula == "sic":
        ase_atom = get_ase_zincblende("Si", "C")
    elif formula == "bn":
        ase_atom = get_ase_zincblende("B", "N")
    elif formula == "bp":
        ase_atom = get_ase_zincblende("B", "P")
    elif formula == "bas":
        ase_atom = get_ase_zincblende("B", "As")
    elif formula == "bsb":
        ase_atom = get_ase_zincblende("B", "Sb")
    elif formula == "alp":
        ase_atom = get_ase_zincblende("Al", "P")
    elif formula == "alas":
        ase_atom = get_ase_zincblende("Al", "As")
    elif formula == "alsb":
        ase_atom = get_ase_zincblende("Al", "Sb")
    elif formula == "beta-gan":
        ase_atom = get_ase_zincblende("Ga", "N")
    elif formula == "gap":
        ase_atom = get_ase_zincblende("Ga", "P")
    elif formula == "gaas":
        ase_atom = get_ase_zincblende("Ga", "As")
    elif formula == "gasb":
        ase_atom = get_ase_zincblende("Ga", "Sb")
    elif formula == "inp":
        ase_atom = get_ase_zincblende("In", "P")
    elif formula == "inas":
        ase_atom = get_ase_zincblende("In", "As")
    elif formula == "insb":
        ase_atom = get_ase_zincblende("In", "Sb")
    elif formula == "zns":
        ase_atom = get_ase_zincblende("Zn", "S")
    elif formula == "znse":
        ase_atom = get_ase_zincblende("Zn", "Se")
    elif formula == "znte":
        ase_atom = get_ase_zincblende("Zn", "Te")
    elif formula == "cds":
        ase_atom = get_ase_zincblende("Cd", "S")
    elif formula == "cdse":
        ase_atom = get_ase_zincblende("Cd", "Se")
    elif formula == "cdte":
        ase_atom = get_ase_zincblende("Cd", "Te")
    elif formula == "mgs":
        ase_atom = get_ase_zincblende("Mg", "S")
    elif formula == "mgte":
        ase_atom = get_ase_zincblende("Mg", "Te")
    elif formula == "aln":
        ase_atom = get_ase_wurtzite("Al", "N")
    elif formula == "gan":
        ase_atom = get_ase_wurtzite("Ga", "N")
    elif formula == "inn":
        ase_atom = get_ase_wurtzite("In", "N")
    elif formula == "zno":
        ase_atom = get_ase_wurtzite("Zn", "O")
    elif formula == "mgo":
        ase_atom = get_ase_rocksalt("Mg", "O")
    elif formula == "mgse":
        ase_atom = get_ase_rocksalt("Mg", "Se")
    elif formula == "cas":
        ase_atom = get_ase_rocksalt("Ca", "S")
    elif formula == "case":
        ase_atom = get_ase_rocksalt("Ca", "Se")
    elif formula == "cate":
        ase_atom = get_ase_rocksalt("Ca", "Te")
    elif formula == "srs":
        ase_atom = get_ase_rocksalt("Sr", "S")
    elif formula == "srse":
        ase_atom = get_ase_rocksalt("Sr", "Se")
    elif formula == "srte":
        ase_atom = get_ase_rocksalt("Sr", "Te")
    elif formula == "bas":
        ase_atom = get_ase_rocksalt("Ba", "S")
    elif formula == "base":
        ase_atom = get_ase_rocksalt("Ba", "Se")
    elif formula == "bate":
        ase_atom = get_ase_rocksalt("Ba", "Te")
    elif formula == "lih":
        ase_atom = get_ase_rocksalt("Li","H")
    elif formula == "lif":
        ase_atom = get_ase_rocksalt("Li","F")
    elif formula == "licl":
        ase_atom = get_ase_rocksalt("Li","Cl")
    else:
        raise ValueError("Unknown formula {}".format(formula))
    return ase_atom

def get_ase_rocksalt(A="Li", B="Cl"):
    assert A in ["Mg", "Ca", "Sr", "Ba", "Li"]
    assert B in ["O", "Se", "S", "Te", "H", "F", "Cl"]
    if A == "Mg":
        if B == "O":
            ase_atom = bulk("MgO", "rocksalt", a=4.207*A2B)
        elif B == "Se":
            ase_atom = bulk("MgSe", "rocksalt", a=5.40*A2B)
    elif A == "Ca":
        if B == "S":
            ase_atom = bulk("CaS", "rocksalt", a=5.689*A2B)
        elif B == "Se":
            ase_atom = bulk("CaSe", "rocksalt", a=5.916*A2B)
        elif B == "Te":
            ase_atom = bulk("CaTe", "rocksalt", a=6.348*A2B)
    elif A == "Sr":
        if B == "S":
            ase_atom = bulk("SrS", "rocksalt", a=5.99*A2B)
        elif B == "Se":
            ase_atom = bulk("SrSe", "rocksalt", a=6.234*A2B)
        elif B == "Te":
            ase_atom = bulk("SrTe", "rocksalt", a=6.64*A2B)
    elif A == "Ba":
        if B == "S":
            ase_atom = bulk("BaS", "rocksalt", a=6.389*A2B)
        elif B == "Se":
            ase_atom = bulk("BaSe", "rocksalt", a=6.595*A2B)
        elif B == "Te":
            ase_atom = bulk("BaTe", "rocksalt", a=7.007*A2B)
    elif A == "Li":
        if B == "H":
            ase_atom = bulk("LiH", "rocksalt", a=4.0834*A2B)
        elif B == "F":
            ase_atom = bulk("LiF", "rocksalt", a=4.0351*A2B)
        elif B == "Cl":
            ase_atom = bulk("LiCl", "rocksalt", a=5.13*A2B)
    else:
        raise NotImplementedError('No formula found for system ',
            A, B, '.  Choose a different system?  Or add it to the list!')    
    return ase_atom

def get_ase_wurtzite(A="Zn", B="O"):
    assert A in ["Al", "Ga", "In", "Zn"]
    assert B in ["N", "O"]
    if B == "N":
        if A == "Al":
            ase_atom = bulk("AlN", "wurtzite", a=3.111*A2B, c=4.981*A2B)
        elif A == "Ga":
            ase_atom = bulk("GaN", "wurtzite", a=3.189*A2B, c=5.185*A2B)
        elif A == "In":
            ase_atom = bulk("InN", "wurtzite", a=3.537*A2B, c=5.704*A2B)
    elif B == "O" and A == "Zn":
        ase_atom = bulk("ZnO", "wurtzite", a=3.25*A2B, c=5.2*A2B)
    else:
        raise NotImplementedError('No formula found for system ',
            A, B, '.  Choose a different system?  Or add it to the list!')         
    return ase_atom

def get_ase_zincblende(A="Ga", B="As"):
    assert A in ["Si", "B", "Al", "Ga", "In", "Zn", "Cd", "Mg"]
    assert B in ["C", "N", "P", "As", "Sb", "S", "Se", "Te"]
    if A == "Si" and B == "C":
        ase_atom = bulk("SiC", "zincblende", a=4.358*A2B)
    elif A == "B":
        if B == "N":
            ase_atom = bulk("BN", "zincblende", a=3.616*A2B)
        elif B == "P":
            ase_atom = bulk("BP", "zincblende", a=4.538*A2B)
        elif B == "As":
            ase_atom = bulk("BAs", "zincblende", a=4.777*A2B)
        elif B == "Sb":
            print("WARNING: No experimental value found for BSb. Use computed lattice constant from HSE instead.")
            ase_atom = bulk("BSb", "zincblende", a=5.251*A2B)
    elif A == "Al":
        if B == "P":
            ase_atom = bulk("AlP", "zincblende", a=5.463*A2B)
        elif B == "As":
            ase_atom = bulk("AlAs", "zincblende", a=5.661*A2B)
        elif B == "Sb":
            ase_atom = bulk("AlSb", "zincblende", a=6.136*A2B)
    elif A == "Ga":
        if B == "N":
            ase_atom = bulk("GaN", "zincblende", a=4.523*A2B)
        elif B == "P":
            ase_atom = bulk("GaP", "zincblende", a=5.451*A2B)
        elif B == "As":
            ase_atom = bulk("GaAs", "zincblende", a=5.648*A2B)
        elif B == "Sb":
            ase_atom = bulk("GaSb", "zincblende", a=6.096*A2B)
    elif A == "In":
        if B == "P":
            ase_atom = bulk("InP", "zincblende", a=5.869*A2B)
        elif B == "As":
            ase_atom = bulk("InAs", "zincblende", a=6.058*A2B)
        elif B == "Sb":
            ase_atom = bulk("InSb", "zincblende", a=6.479*A2B)
    elif A == "Zn":
        if B == "S":
            ase_atom = bulk("ZnS", "zincblende", a=5.409*A2B)
        elif B == "Se":
            ase_atom = bulk("ZnSe", "zincblende", a=5.668*A2B)
        elif B == "Te":
            ase_atom = bulk("ZnTe", "zincblende", a=6.089*A2B)
    elif A == "Cd":
        if B == "S":
            ase_atom = bulk("CdS", "zincblende", a=5.818*A2B)
        elif B == "Se":
            ase_atom = bulk("CdSe", "zincblende", a=6.052*A2B)
        elif B == "Te":
            ase_atom = bulk("CdTe", "zincblende", a=6.480*A2B)
    elif A == "Mg":
        if B == "S":
            ase_atom = bulk("MgS", "zincblende", a=5.622*A2B)
        elif B == "Te":
            ase_atom = bulk("MgTe", "zincblende", a=6.42*A2B)
    else:
        raise NotImplementedError('No formula found for system ',
            A, B, '.  Choose a different system?  Or add it to the list!')
    return ase_atom

def get_ase_diamond_primitive(atom="C"):
    if atom == 'C':
        ase_atom = bulk('C', 'diamond', a=3.567*A2B)
    elif atom == 'Si':
        ase_atom = bulk('Si', 'diamond', a=5.430*A2B)
    elif atom == 'Ge':
        ase_atom = bulk('Ge', 'diamond', a=5.658*A2B)
    else:
        raise NotImplementedError('No formula found for system ',            atom, '.  Choose a different system?  Or add it to the list!')
    return ase_atom
            

if __name__ == '__main__':
    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
    from pyscf.pbc import gto
    import numpy as np
    import sys

    sc40_plus_4 = ['c', 'si', 'ge', 'sic', 'bn', 'bp', 'bas', 'bsb', 'aln', 'alp', 'alas', 'alsb', 'gan', 'beta-gan', 'gap', 'gaas', 'gasb', 'inn', 'inp', 'inas', 'insb', 'zns', 'znse', 'znte', 'cds', 'cdse', 'cdte', 'mgo', 'mgs', 'mgse', 'mgte', 'cas', 'case', 'cate', 'srs', 'srse', 'srte', 'bas', 'base', 'bate', 'lih', 'lif', 'licl', 'zno']

    for formula in sc40_plus_4:
        cell = gto.Cell()

        ase_atom = get_ase_atom(formula)
        cell.atom = ase_atoms_to_pyscf(ase_atom)        
        cell.a = np.array(ase_atom.cell)

        cell.unit = "B"
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
        cell.verbose = 0
        try:
            cell.build()
            print("{}: cell.build() succeeded".format(formula))
        except RuntimeError as err:
            print("{}: cell.build() failed because {}".format(formula, err))
        except:
            print("{}: cell.build() failed because {}".format(formula, sys.exc_info()[0]))
            raise
