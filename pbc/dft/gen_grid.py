#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy as np
import pyscf.lib
from pyscf.lib import logger
from pyscf import dft
from pyscf.pbc import tools
from pyscf.pbc.gto.cell import gen_uniform_grids


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
# modified from pyscf.dft.gen_grid.gen_partition
    Ls = cell.get_lattice_Ls()
    atm_coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    atom_grids_tab = dft.gen_grid.gen_atomic_grids(cell, atom_grid, radi_method,
                                                   level, prune)
    coords_all = []
    weights_all = []
    b = cell.reciprocal_vectors(norm_to=1)
    supatm_idx = []
    k = 0
    for iL, L in enumerate(Ls):
        for ia in range(cell.natm):
            coords, vol = atom_grids_tab[cell.atom_symbol(ia)]
            coords = coords + atm_coords[iL,ia]
            # search for grids in unit cell
            c = b.dot(coords.T).round(8)
            mask = ((c[0]>=0) & (c[1]>=0) & (c[2]>=0) &
                    (c[0]<=1) & (c[1]<=1) & (c[2]<=1))
            vol = vol[mask]
            if vol.size > 8:
                c = c[:,mask]
                vol[c[0]==0] *= .5
                vol[c[1]==0] *= .5
                vol[c[2]==0] *= .5
                vol[c[0]==1] *= .5
                vol[c[1]==1] *= .5
                vol[c[2]==1] *= .5
                coords = coords[mask]
                coords_all.append(coords)
                weights_all.append(vol)
                supatm_idx.append(k)
            k += 1
    offs = np.append(0, np.cumsum([w.size for w in weights_all]))
    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)

    atm_coords = np.asarray(atm_coords.reshape(-1,3)[supatm_idx], order='C')
    sup_natm = len(atm_coords)
    ngrids = len(coords_all)
    pbecke = np.empty((sup_natm,ngrids))
    coords = np.asarray(coords_all, order='F')
    p_radii_table = pyscf.lib.c_null_ptr()
    fn = dft.gen_grid.libdft.VXCgen_grid
    fn(pbecke.ctypes.data_as(ctypes.c_void_p),
       coords.ctypes.data_as(ctypes.c_void_p),
       atm_coords.ctypes.data_as(ctypes.c_void_p),
       p_radii_table, ctypes.c_int(sup_natm), ctypes.c_int(ngrids))

    weights_all /= pbecke.sum(axis=0)
    for ia in range(sup_natm):
        p0, p1 = offs[ia], offs[ia+1]
        weights_all[p0:p1] *= pbecke[ia,p0:p1]
    return coords_all, weights_all


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
    print(g.weights.sum())
    print(cell.vol)
