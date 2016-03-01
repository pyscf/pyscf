import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

def get_ase_diamond_primitive():
    """Get the ASE atoms for primitive (2-atom) diamond unit cell."""
    from ase.lattice import bulk
    ase_atom = bulk('C', 'diamond', a=3.5668)
    return ase_atom

def get_ase_diamond_cubic():
    """Get the ASE atoms for cubic (8-atom) diamond unit cell."""
    from ase.lattice.cubic import Diamond
    ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
    return ase_atom

def build_cell(ase_atom, ke=20.0, gsmax=None, basis='gth-szv'):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell

    cell.basis = basis
    cell.pseudo = 'gth-pade'
    
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

