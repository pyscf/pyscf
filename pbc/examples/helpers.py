import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

def get_ase_alkali_halide(A='Li', B='Cl'):
    assert A in ['Li']
    # Na, K
    assert B in ['H', 'F', 'Cl']
    # Br, I
    from ase.lattice import bulk
    if A=='Li':
        if B=='H':
            ase_atom = bulk('LiH', 'rocksalt', a=4.0834)
        elif B=='F':
            ase_atom = bulk('LiF', 'rocksalt', a=4.0351)
        elif B=='Cl':
            ase_atom = bulk('LiCl', 'rocksalt', a=5.13)
    return ase_atom

def get_ase_alkali_halide_xxx(A='Li', B='Cl'):
    assert A in ['Li']
    # Na, K
    assert B in ['H', 'F', 'Cl']
    # Br, I
    from ase.lattice.compounds import Rocksalt
    if A=='Li':
        if B=='H':
            ase_atom = Rocksalt(symbol=("Li", "H"), latticeconstant=4.0834)
        elif B=='F':
            ase_atom = Rocksalt(symbol=("Li", "F"), latticeconstant=4.0351)
        elif B=='Cl':
            ase_atom = Rocksalt(symbol=("Li", "Cl"), latticeconstant=5.13)
    return ase_atom

def get_ase_graphene(vacuum=5.0):
    """Get the ASE atoms for primitive (2-atom) graphene unit cell."""
    from ase.lattice.hexagonal import Graphene
    ase_atom = Graphene('C', latticeconstant={'a':2.46,'c':vacuum})
    return ase_atom

def get_ase_graphene_xxx(vacuum=5.0):
    """Get the ASE atoms for primitive (2-atom) graphene unit cell."""
    from ase.lattice import bulk
    ase_atom = bulk('C', 'hcp', a=2.46, c=vacuum)
    ase_atom.positions[1,2] = 0.0
    return ase_atom

def get_ase_diamond_primitive(atom='C'):
    """Get the ASE atoms for primitive (2-atom) diamond unit cell."""
    from ase.lattice import bulk
    if atom == 'C':
        ase_atom = bulk('C', 'diamond', a=3.5668)
    else:
        ase_atom = bulk('Si', 'diamond', a=5.431)
    return ase_atom

def get_ase_diamond_cubic(atom='C'):
    """Get the ASE atoms for cubic (8-atom) diamond unit cell."""
    from ase.lattice.cubic import Diamond
    if atom == 'C':
        ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
    else:
        ase_atom = Diamond(symbol='Si', latticeconstant=5.431)
    return ase_atom

def build_cell(ase_atom, ke=20.0, gsmax=None, basis='gth-szv', pseudo='gth-pade',
               incore_anyway=False):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell.T

    cell.basis = basis
    cell.pseudo = pseudo
    
    if incore_anyway == True:
        cell.incore_anyway = True

    if gsmax is not None:
        cell.gs = np.array([gsmax,gsmax,gsmax])
    else:
        # Will be over-written after build
        cell.gs = np.array([1,1,1])

    #cell.verbose = 4
    cell.build()
    if gsmax is None:
        cell.gs = pbctools.cutoff_to_gs(cell._h, ke)
    cell.build()
    return cell

