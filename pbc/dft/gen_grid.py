#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy as np
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf import dft
from pyscf.pbc import tools


def gen_uniform_grids(cell, gs=None):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    if gs is None:
        gs = cell.gs
    ngs = 2*np.asarray(gs)+1
    qv = cartesian_prod([np.arange(x) for x in ngs])
    invN = np.diag(1./ngs)
    a_frac = np.einsum('i,ij->ij', 1./ngs, cell.lattice_vectors())
    coords = np.dot(qv, a_frac)
    return coords


class UniformGrids(object):
    '''Uniform Grid class.'''

    def __init__(self, cell):
        self.cell = cell
        self.coords = None
        self.weights = None
        self.gs = None
        self.stdout = cell.stdout
        self.verbose = cell.verbose

    def build(self, cell=None):
        if cell == None: cell = self.cell

        self.coords = gen_uniform_grids(self.cell, self.gs)
        self.weights = np.empty(self.coords.shape[0])
        self.weights[:] = cell.vol/self.weights.shape[0]

        return self.coords, self.weights

    def dump_flags(self):
        if self.gs is None:
            logger.info(self, 'Uniform grid, gs = %s', self.cell.gs)
        else:
            logger.info(self, 'Uniform grid, gs = %s', self.gs)

    def kernel(self, cell=None):
        self.dump_flags()
        return self.build(cell)


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
    scell = tools.pbc.cell_plus_imgs(cell, cell.nimgs)
    coords = scell.atom_coords()
# Generating grids for the entire super cell is slow.  We don't need generate
# grids for the super cell because out of certain region the weights obtained
# from Becke partitioning are no longer important.  The region is controlled
# by r_cutoff
    #r_cutoff = pyscf.lib.norm(pyscf.lib.norm(cell.lattice_vectors(), axis=0))
    r_cutoff = max(pyscf.lib.norm(cell.lattice_vectors(), axis=0)) * 1.25
# Filter important atoms. Atoms close to the unicell if they are close to any
# of the atoms in the unit cell
    mask = np.zeros(scell.natm, dtype=bool)
    for ia in range(cell.natm):
        c0 = cell.atom_coord(ia)
        dr = coords - c0
        rr = np.einsum('ix,ix->i', dr, dr)
        mask |= rr < r_cutoff**2
    scell._atm = scell._atm[mask]
    logger.debug(cell, 'r_cutoff %.9g  natm = %d', r_cutoff, scell.natm)

    atom_grids_tab = dft.gen_grid.gen_atomic_grids(scell, atom_grid, radi_method,
                                                   level, prune)
    coords, weights = dft.gen_grid.gen_partition(scell, atom_grids_tab)

    # search for grids in unit cell
    b = cell.reciprocal_vectors(norm_to=1)
    c = np.dot(coords, b.T)
    mask = ((c[:,0]>=0) & (c[:,1]>=0) & (c[:,2]>=0) &
            (c[:,0]< 1) & (c[:,1]< 1) & (c[:,2]< 1))
    return coords[mask], weights[mask]


class BeckeGrids(dft.gen_grid.Grids):
    '''Becke, JCP, 88, 2547 (1988)'''
    def __init__(self, cell):
        self.cell = cell
        pyscf.dft.gen_grid.Grids.__init__(self, cell)

    def build(self, cell=None):
        if cell is None: cell = self.cell
        self.coords, self.weights = gen_becke_grids(self.cell, self.atom_grid,
                                                    radi_method=self.radi_method,
                                                    level=self.level,
                                                    prune=self.prune)
        logger.info(self, 'tot grids = %d', len(self.weights))
        logger.info(self, 'cell vol = %.9g  sum(weights) = %.9g',
                    cell.vol, self.weights.sum())
        return self.coords, self.weights


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto

    n = 3
    cell = pgto.Cell()
    cell.a = '''
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
    g.build()
    print g.weights.sum()
    print cell.vol
