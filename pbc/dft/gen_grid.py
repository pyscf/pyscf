import numpy as np
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf import dft
from pyscf.pbc import tools


def gen_uniform_grids(cell):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    ngs = 2*np.asarray(cell.gs)+1
    qv = cartesian_prod([np.arange(x) for x in ngs])
    invN = np.diag(1./ngs)
    coords = np.dot(qv, np.dot(cell._h, invN).T)
    return coords


class UniformGrids(object):
    '''Uniform Grid class.'''

    def __init__(self, cell):
        self.cell = cell
        self.coords = None
        self.weights = None
        self.stdout = cell.stdout
        self.verbose = cell.verbose

    def build_(self, cell=None):
        return self.setup_grids_(cell)
    def setup_grids_(self, cell=None):
        if cell == None: cell = self.cell

        self.coords = gen_uniform_grids(self.cell)
        self.weights = np.ones(self.coords.shape[0])
        self.weights *= cell.vol/self.weights.shape[0]

        return self.coords, self.weights

    def dump_flags(self):
        logger.info(self, 'Uniform grid')

    def kernel(self, cell=None):
        self.dump_flags()
        return self.setup_grids_(cell)


def gen_becke_grids(cell, atom_grid={}, radi_method=dft.radi.gauss_chebyshev,
                    level=3, prune=dft.gen_grid.nwchem_prune):
    '''real-space grids using Becke scheme

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.
        weights : (ngx*ngy*ngz) ndarray
    '''
    scell = tools.pbc.cell_plus_imgs(cell, [min(x,2) for x in cell.nimgs])
    coords = np.asarray([scell.atom_coord(ia) for ia in range(scell.natm)])
# Generating grids for the entire super cell is slow.  We don't need generate
# grids for the super cell because out of certain region the weights obtained
# from Becke partitioning are no longer important.  The region is controlled
# by r_cutoff
    #r_cutoff = pyscf.lib.norm(pyscf.lib.norm(cell._h, axis=1))
    r_cutoff = max(pyscf.lib.norm(cell._h, axis=1)) * 1.25
    logger.debug1(cell, 'r_cutoff %g', r_cutoff)
# Filter important atoms. Atoms close to the unicell if they are close to any
# of the atoms in the unit cell
    mask = np.zeros(scell.natm, dtype=bool)
    for ia in range(cell.natm):
        c0 = cell.atom_coord(ia)
        dr = coords[cell.natm:] - c0
        rr = np.einsum('ix,ix->i', dr, dr)
        mask[cell.natm:] |= rr < r_cutoff**2
    scell._atm = scell._atm[mask]
    scell.natm = len(scell._atm)

    atom_grids_tab = dft.gen_grid.gen_atomic_grids(scell, atom_grid, radi_method,
                                                   level, prune)
    coords, weights = dft.gen_grid.gen_partition(scell, atom_grids_tab)

    # search for grids in unit cell
    #b1,b2,b3 = np.linalg.inv(h)  # reciprocal lattice
    #np.einsum('kj,ij->ki', coords, (b1,b2,b3))
    c = np.dot(coords, np.linalg.inv(cell._h.T))
    mask = ((c[:,0]>=0) & (c[:,1]>=0) & (c[:,2]>=0) &
            (c[:,0]< 1) & (c[:,1]< 1) & (c[:,2]< 1))
    return coords[mask], weights[mask]


class BeckeGrids(dft.gen_grid.Grids):
    '''Becke, JCP, 88, 2547 (1988)'''
    def __init__(self, cell):
        self.cell = cell
        pyscf.dft.gen_grid.Grids.__init__(self, cell)

    def build_(self, cell=None):
        if cell is None: cell = self.cell
        self.coords, self.weights = gen_becke_grids(self.cell, self.atom_grid,
                                                    radi_method=self.radi_method,
                                                    level=self.level,
                                                    prune=self.prune)
        logger.info(self, 'tot grids = %d', len(self.weights))
        return self.coords, self.weights


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto

    n = 30
    cell = pgto.Cell()
    cell.h = '''
    4   0   0
    0   4   0
    0   0   4
    '''
    cell.gs = [n,n,n]

    cell.atom = '''He     0.    0.       1.
                   He     1.    0.       1.'''
    cell.basis = {'He': [[0, (1.0, 1.0)]]}
    cell.build()
    g = BeckeGrids(cell)
    g.build_()
