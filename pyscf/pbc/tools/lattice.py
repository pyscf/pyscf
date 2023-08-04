# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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

A2B = 1.889725989

def get_ase_atom(formula):
    formula = formula.lower()
    assert formula in ['lih','lif','licl','mgo',
                       'c','si','ge','sic','gaas','gan','cds',
                       'zns','zno','bn','alp']
    if formula == 'lih':
        ase_atom = get_ase_rocksalt('Li','H')
    elif formula == 'lif':
        ase_atom = get_ase_rocksalt('Li','F')
    elif formula == 'licl':
        ase_atom = get_ase_rocksalt('Li','Cl')
    elif formula == 'mgo':
        ase_atom = get_ase_rocksalt('Mg','O')
    elif formula == 'c':
        ase_atom = get_ase_diamond_primitive('C')
    elif formula == 'si':
        ase_atom = get_ase_diamond_primitive('Si')
    elif formula == 'ge':
        ase_atom = get_ase_diamond_primitive('Ge')
    elif formula == 'sic':
        ase_atom = get_ase_zincblende('Si','C')
    elif formula == 'gaas':
        ase_atom = get_ase_zincblende('Ga','As')
    elif formula == 'gan':
        ase_atom = get_ase_zincblende('Ga','N')
    elif formula == 'bn':
        ase_atom = get_ase_zincblende('B','N')
    elif formula == 'alp':
        ase_atom = get_ase_zincblende('Al','P')
    elif formula == 'zno':
        ase_atom = get_ase_wurtzite('Zn','O')
    elif formula == 'cds':
        ase_atom = get_ase_zincblende('Cd','S')
    elif formula == 'zns':
        ase_atom = get_ase_zincblende('Zn','S')

    return ase_atom

def get_ase_wurtzite(A='Zn', B='O'):
    # Lattice constants taken from wikipedia (TODO: is wikipedia a valid
    # citation at this point? en.wikipedia.org/wiki/Lattice_constant)
    assert A in ['Zn']
    assert B in ['O']
    from ase.lattice import bulk
    if A=='Zn' and B=='O':
        ase_atom = bulk('ZnO', 'wurtzite', a=3.25*A2B, c=5.2*A2B)
    else:
        raise NotImplementedError('No formula found for system %s %s. '
                                  'Choose a different system?  Or add it to the list!' % (A, B))
    return ase_atom

def get_bandpath_fcc(ase_atom, npoints=30):
    # Set-up the band-path via special points
    from ase.dft.kpoints import ibz_points, kpoint_convert, get_bandpath
    points = ibz_points['fcc']
    G = points['Gamma']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    kpts_reduced, kpath, sp_points = get_bandpath([L, G, X, W, K, G],
                                                  ase_atom.cell, npoints=npoints)
    kpts_cartes = kpoint_convert(ase_atom.cell, skpts_kc=kpts_reduced)

    return kpts_reduced, kpts_cartes, kpath, sp_points

def get_ase_zincblende(A='Ga', B='As'):
    # Lattice constants from Shishkin and Kresse, PRB 75, 235102 (2007)
    assert A in ['Si', 'Ga', 'Cd', 'Zn', 'B', 'Al']
    assert B in ['C', 'As', 'S', 'O', 'N', 'P']
    from ase.lattice import bulk
    if A=='Si' and B=='C':
        ase_atom = bulk('SiC', 'zincblende', a=4.350*A2B)
    elif A=='Ga' and B=='As':
        ase_atom = bulk('GaAs', 'zincblende', a=5.648*A2B)
    elif A=='Ga' and B=='N':
        ase_atom = bulk('GaN', 'zincblende', a=4.520*A2B)
    elif A=='Cd' and B=='S':
        ase_atom = bulk('CdS', 'zincblende', a=5.832*A2B)
    elif A=='Zn' and B=='S':
        ase_atom = bulk('ZnS', 'zincblende', a=5.420*A2B)
    elif A=='Zn' and B=='O':
        ase_atom = bulk('ZnO', 'zincblende', a=4.580*A2B)
    elif A=='B' and B=='N':
        ase_atom = bulk('BN', 'zincblende', a=3.615*A2B)
    elif A=='Al' and B=='P':
        ase_atom = bulk('AlP', 'zincblende', a=5.451*A2B)
    else:
        raise NotImplementedError('No formula found for system %s %s. '
                                  'Choose a different system?  Or add it to the list!' % (A, B))

    return ase_atom

def get_ase_rocksalt(A='Li', B='Cl'):
    assert A in ['Li', 'Mg']
    # Add Na, K
    assert B in ['H', 'F', 'Cl', 'O']
    # Add Br, I
    from ase.lattice import bulk
    if A=='Li':
        if B=='H':
            ase_atom = bulk('LiH', 'rocksalt', a=4.0834*A2B)
        elif B=='F':
            ase_atom = bulk('LiF', 'rocksalt', a=4.0351*A2B)
        elif B=='Cl':
            ase_atom = bulk('LiCl', 'rocksalt', a=5.13*A2B)
    elif A=='Mg' and B=='O':
        ase_atom = bulk('MgO', 'rocksalt', a=4.213*A2B)
    else:
        raise NotImplementedError('No formula found for system %s %s. '
                                  'Choose a different system?  Or add it to the list!' % (A, B))

    return ase_atom

def get_ase_alkali_halide(A='Li', B='Cl'):
    return get_ase_rocksalt(A,B)

def get_ase_diamond_primitive(atom='C'):
    """Get the ASE atoms for primitive (2-atom) diamond unit cell."""
    from ase.build import bulk
    if atom == 'C':
        ase_atom = bulk('C', 'diamond', a=3.5668*A2B)
    elif atom == 'Si':
        ase_atom = bulk('Si', 'diamond', a=5.431*A2B)
    elif atom == 'Ge':
        ase_atom = bulk('Ge', 'diamond', a=5.658*A2B)
    else:
        raise NotImplementedError('No formula found for system %s. '
                                  'Choose a different system?  Or add it to the list!' % atom)
    return ase_atom

def get_ase_diamond_cubic(atom='C'):
    """Get the ASE atoms for cubic (8-atom) diamond unit cell."""
    from ase.lattice.cubic import Diamond
    if atom == 'C':
        ase_atom = Diamond(symbol='C', latticeconstant=3.5668*A2B)
    elif atom == 'Si':
        ase_atom = Diamond(symbol='Si', latticeconstant=5.431*A2B)
    else:
        raise NotImplementedError('No formula found for system %s. '
                                  'Choose a different system?  Or add it to the list!' % atom)
    return ase_atom

def get_ase_graphene(vacuum=5.0):
    """Get the ASE atoms for primitive (2-atom) graphene unit cell."""
    from ase.lattice.hexagonal import Graphene
    ase_atom = Graphene('C', latticeconstant={'a':2.46*A2B,'c':vacuum*A2B})
    return ase_atom

def get_ase_graphene_xxx(vacuum=5.0):
    """Get the ASE atoms for primitive (2-atom) graphene unit cell."""
    from ase.lattice import bulk
    ase_atom = bulk('C', 'hcp', a=2.46*A2B, c=vacuum*A2B)
    ase_atom.positions[1,2] = 0.0
    return ase_atom
