#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import dft
from pyscf.pbc.gto.cell import gen_uniform_grids
from pyscf.dft.gen_grid import (sg1_prune, nwchem_prune, treutler_prune,
                                stratmann, original_becke, gen_atomic_grids,
                                BLKSIZE)

libpbc = lib.load_library('libpbc')

def make_mask(cell, coords, relativity=0, shls_slice=None, verbose=None):
    '''Mask to indicate whether a shell is zero on grid.
    The resultant mask array is an extension to the mask array used in
    molecular code (see also pyscf.dft.numint.make_mask function).
    For given shell ID and block ID, the value of the extended mask array
    means the number of images in Ls that does not vanish.
    '''
    coords = np.asarray(coords, order='F')
    natm = ctypes.c_int(cell._atm.shape[0])
    nbas = ctypes.c_int(cell.nbas)
    ngrids = len(coords)
    if shls_slice is None:
        shls_slice = (0, cell.nbas)
    assert(shls_slice == (0, cell.nbas))

    Ls = cell.get_lattice_Ls()
    Ls = Ls[np.argsort(lib.norm(Ls, axis=1))]

    non0tab = np.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                          dtype=np.uint8)
    libpbc.PBCnr_ao_screen(non0tab.ctypes.data_as(ctypes.c_void_p),
                           coords.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(ngrids),
                           Ls.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(len(Ls)),
                           cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                           cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                           cell._env.ctypes.data_as(ctypes.c_void_p))
    return non0tab

class UniformGrids(object):
    '''Uniform Grid class.'''

    def __init__(self, cell):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.coords = None
        self.weights = None
        self.gs = None
        self.non0tab = None

    def build(self, cell=None, with_non0tab=False):
        if cell is None: cell = self.cell

        self.coords = gen_uniform_grids(self.cell, self.gs)
        self.weights = np.empty(self.coords.shape[0])
        self.weights[:] = cell.vol/self.weights.shape[0]

        if with_non0tab:
            self.non0tab = self.make_mask(cell, self.coords)
        else:
            self.non0tab = None
        return self.coords, self.weights

    def dump_flags(self):
        if self.gs is None:
            logger.info(self, 'Uniform grid, gs = %s', self.cell.gs)
        else:
            logger.info(self, 'Uniform grid, gs = %s', self.gs)

    def kernel(self, cell=None, with_non0tab=False):
        self.dump_flags()
        return self.build(cell, with_non0tab)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell=None, coords=None, relativity=0, shls_slice=None,
                  verbose=None):
        if cell is None: cell = self.cell
        if coords is None: coords = self.coords
        return make_mask(cell, coords, relativity, shls_slice, verbose)


def gen_becke_grids(cell, atom_grid={}, radi_method=dft.radi.gauss_chebyshev,
                    level=3, prune=nwchem_prune):
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
    atom_grids_tab = gen_atomic_grids(cell, atom_grid, radi_method, level, prune)
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

            mask = np.ones(c.shape[1], dtype=bool)
            if cell.dimension >= 1:
                mask &= (c[0]>=0) & (c[0]<=1)
            if cell.dimension >= 2:
                mask &= (c[1]>=0) & (c[1]<=1)
            if cell.dimension == 3:
                mask &= (c[2]>=0) & (c[2]<=1)

            vol = vol[mask]
            if vol.size > 8:
                c = c[:,mask]
                if cell.dimension >= 1:
                    vol[c[0]==0] *= .5
                    vol[c[0]==1] *= .5
                if cell.dimension >= 2:
                    vol[c[1]==0] *= .5
                    vol[c[1]==1] *= .5
                if cell.dimension == 3:
                    vol[c[2]==0] *= .5
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
    p_radii_table = lib.c_null_ptr()
    fn = dft.gen_grid.libdft.VXCgen_grid
    ngrids = weights_all.size

    max_memory = cell.max_memory - lib.current_memory()[0]
    blocksize = min(ngrids, max(2000, int(max_memory*1e6/8 / sup_natm)))
    displs = lib.misc._blocksize_partition(offs, blocksize)
    for n0, n1 in zip(displs[:-1], displs[1:]):
        p0, p1 = offs[n0], offs[n1]
        pbecke = np.empty((sup_natm,p1-p0))
        coords = np.asarray(coords_all[p0:p1], order='F')
        fn(pbecke.ctypes.data_as(ctypes.c_void_p),
           coords.ctypes.data_as(ctypes.c_void_p),
           atm_coords.ctypes.data_as(ctypes.c_void_p),
           p_radii_table, ctypes.c_int(sup_natm), ctypes.c_int(p1-p0))

        weights_all[p0:p1] /= pbecke.sum(axis=0)
        for ia in range(n0, n1):
            i0, i1 = offs[ia], offs[ia+1]
            weights_all[i0:i1] *= pbecke[ia,i0-p0:i1-p0]

    return coords_all, weights_all


class BeckeGrids(dft.gen_grid.Grids):
    '''Atomic grids for all-electron calculation.'''
    def __init__(self, cell):
        self.cell = cell
        dft.gen_grid.Grids.__init__(self, cell)

    def build(self, cell=None, with_non0tab=False):
        if cell is None: cell = self.cell
        self.coords, self.weights = gen_becke_grids(self.cell, self.atom_grid,
                                                    radi_method=self.radi_method,
                                                    level=self.level,
                                                    prune=self.prune)
        if with_non0tab:
            self.non0tab = self.make_mask(cell, self.coords)
        else:
            self.non0tab = None
        logger.info(self, 'tot grids = %d', len(self.weights))
        logger.info(self, 'cell vol = %.9g  sum(weights) = %.9g',
                    cell.vol, self.weights.sum())
        return self.coords, self.weights

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell=None, coords=None, relativity=0, shls_slice=None,
                  verbose=None):
        if cell is None: cell = self.cell
        if coords is None: coords = self.coords
        return make_mask(cell, coords, relativity, shls_slice, verbose)

AtomicGrids = BeckeGrids


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
