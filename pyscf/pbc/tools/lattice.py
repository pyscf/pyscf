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

    return ase_atom

def get_ase_alkali_halide(A='Li', B='Cl'):
    return get_ase_rocksalt(A,B)

def get_ase_diamond_primitive(atom='C'):
    """Get the ASE atoms for primitive (2-atom) diamond unit cell."""
    from ase.build import bulk
    if atom == 'C':
        ase_atom = bulk('C', 'diamond', a=3.5668*A2B)
    else:
        ase_atom = bulk('Si', 'diamond', a=5.431*A2B)
    return ase_atom

def get_ase_diamond_cubic(atom='C'):
    """Get the ASE atoms for cubic (8-atom) diamond unit cell."""
    from ase.lattice.cubic import Diamond
    if atom == 'C':
        ase_atom = Diamond(symbol='C', latticeconstant=3.5668*A2B)
    else:
        ase_atom = Diamond(symbol='Si', latticeconstant=5.431*A2B)
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
