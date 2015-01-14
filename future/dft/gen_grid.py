#!/usr/bin/env python
# File: gen_grid.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate DFT grids and weights, based on the code provided by Gerald Knizia <>
'''


import os
import ctypes
from functools import reduce
import numpy
import pyscf.lib
from pyscf import gto
from pyscf.dft import radi

libdft = pyscf.lib.load_library('libdft')

# Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996), eq.11
def stratmann(g):
    '''Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996)'''
    a = .64 # comment after eq. 14, 
    if isinstance(g, numpy.ndarray):
        ma = g/a
        ma2 = ma * ma
        g1 = (1/16.)*(ma*(35 + ma2*(-35 + ma2*(21 - 5 *ma2))))
        g1[g<=-a] = -1
        g1[g>= a] =  1
        return g1
    else:
        if g <= -a:
            g = -1
        elif g >= a:
            g = 1
        else:
            ma = g/a
            ma2 = ma*ma
            g = (1/16.)*(ma*(35 + ma2*(-35 + ma2*(21 - 5 *ma2))))
        return g

def original_becke(g):
    '''Treutler & Ahlrichs, JCP, 102, 346 (1995)'''
    g = (3 - g*g) * g * .5
    g = (3 - g*g) * g * .5
    g = (3 - g*g) * g * .5
    return g

def gen_atomic_grids(mol, radi_method=radi.gauss_chebeshev, level=3):
    atom_grids_tab = {}
    for atm in mol.basis.keys():
        if atm in mol.grids:
            n_rad, n_ang = mol.grids[atm]
        else:
            chg = gto.mole._charge(atm)
            n_rad = _num_radpt(chg, level)
            n_ang = _num_angpt(chg, level)
        rad, rad_weight = radi_method(n_rad)
        # from FDftGridGenerator::GetAtomGridParams
        # atomic_scale = 1
        # rad *= atomic_scale
        # rad_weight *= atomic_scale

        #TODO: reduce grid size for inner shells, i.e. for inner most
        # radial points, use smaller n_ang
        grid = numpy.empty((n_ang,4))
        libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(n_ang))
        ang, ang_weight = grid[:,:3], grid[:,3]
        atom_grids_tab[atm] = (rad, rad_weight, ang, ang_weight)
    return atom_grids_tab

def count_girds(mol, atom_grids_tab):
    ngrid = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        _, rad_wt, _, ang_wt = atom_grids_tab[symb]
        ngrid += rad_wt.size * ang_wt.size
    return ngrid

def gen_partition(mol, atom_grids_tab, atomic_radii=None,
                  becke_scheme=original_becke):
    #TODO: reduce grid size for inner shells

# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
    tril_idx = numpy.tril_indices(mol.natm, -1)
    atm_coords = numpy.array([mol.atom_coord(i) for i in range(mol.natm)])
    atm_dist_inv = 1/_inter_distance(mol)[tril_idx]
    if atomic_radii is not None:
        rad = numpy.array([atomic_radii[mol.atom_charge(ia)-1] \
                           for ia in range(mol.natm)])
        rr = rad.reshape(-1,1)/rad
        a = .25 * (rr.T - rr)
        a = a[tril_idx]
        a[a<-.5] = -.5
        a[a>0.5] = 0.5
        fn_adjust = lambda g: g + numpy.einsum('i,ij->ij', a, (1-g*g))
    else:
        fn_adjust = lambda g: g

    # for a given grid, fractions on each atom
    def gen_grid_partition(coords):
        ngrid = coords.shape[0]
        grid_dist = numpy.empty((mol.natm,ngrid,3))
        for ia in range(mol.natm):
            grid_dist[ia] = coords - atm_coords[ia]
        grid_dist = numpy.sqrt(numpy.einsum('ijk,ijk->ij',grid_dist,grid_dist))
        grid_diff = grid_dist[tril_idx[0]] - grid_dist[tril_idx[1]]
        mu_tab = numpy.einsum('i,ij->ij', atm_dist_inv, grid_diff)
        g = fn_adjust(mu_tab)
        g = becke_scheme(g)
        gplus = .5 * (1-g)
        gminus = .5 * (1+g)
        pbecke = numpy.ones((mol.natm,ngrid))
        for k,(i,j) in enumerate(zip(*tril_idx)):
            pbecke[i] *= gplus[k]
            pbecke[j] *= gminus[k]

        return pbecke

    ip = 0
    ngrid = count_girds(mol, atom_grids_tab)
    coords_all = []
    weights_all = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        rad, rad_weight, ang, ang_weight = atom_grids_tab[symb]
        coords = numpy.einsum('i,jk->ijk',rad,ang).reshape(-1,3) \
                + atm_coords[ia]
        vol = numpy.einsum('i,j->ij', rad_weight, ang_weight).ravel()
        pbecke = gen_grid_partition(coords)
        weights = vol * pbecke[ia] / pbecke.sum(axis=0)
        coords_all.append(coords)
        weights_all.append(weights)
    return numpy.vstack(coords_all), numpy.hstack(weights_all)


class Grids(object):
    def __init__(self, mol):
        self.mol = mol
        self.atomic_radii = radi.BRAGG_RADII
        #self.atomic_radii = radi.COVALENT_RADII
        #self.atomic_radii = None # to switch off atomic radii adjustment
        self.radi_method = radi.gauss_chebeshev
        #self.becke_scheme = stratmann
        self.becke_scheme = original_becke
        self.level = 3

        self.coords  = None
        self.weights = None
        #TODO:self.blocks = None
        for atm,(rad,ang) in mol.grids.items():
            if not _allow_ang_grids(ang):
                raise ValueError('angular grids %d for %s' % (ang, atm))


    def setup_grids(self, mol=None):
        if mol is None:
            mol = self.mol
        atom_grids_tab = self.gen_atomic_grids(mol, level=self.level)
        self.coords, self.weights = \
                self.gen_partition(mol, atom_grids_tab, self.atomic_radii,
                                   self.becke_scheme)
        return self.coords, self.weights

    def gen_atomic_grids(self, mol, level=3):
        return gen_atomic_grids(mol, self.radi_method, level)

    def count_girds(self, mol, atom_grids_tab):
        return count_girds(mol, atom_grids_tab)

    def gen_partition(self, mol, atom_grids_tab, atomic_radii=None,
                      becke_scheme=original_becke):
        return gen_partition(mol, atom_grids_tab, atomic_radii,
                             becke_scheme)


#TODO: OPTIMIZE ME ACCORDING TO JCP 102, 346

#TODO: screen out insiginficant grids
#TODO:

def _num_radpt(charge, level=3):
    n = (0.75 + (level-1)*0.2) * 14 * (charge+2.)**(1/3.)
    return int(n)


def _num_angpt(charge, level=3):
    atom_period = 0
    tab = (2,8,8,18,18,32,32,50)
    for i in range(7):
        if charge < sum(tab[:i]):
            break
    atom_period = i

    DefaultAngularGridLs = (9,11,17,23,29,35,47,59,71,89)
    l = DefaultAngularGridLs[level-1 + atom_period-1]
    return SPHERICAL_POINTS_ORDER[l]

def _allow_ang_grids(n):
    return n in SPHERICAL_POINTS_ORDER.values()

SPHERICAL_POINTS_ORDER = {
      0:    1,
      3:    6,
      5:   14,
      7:   26,
      9:   38,
     11:   50,
     13:   74,
     15:   86,
     17:  110,
     19:  146,
     21:  170,
     23:  194,
     25:  230,
     27:  266,
     29:  302,
     31:  350,
     35:  434,
     41:  590,
     47:  770,
     53:  974,
     59: 1202,
     65: 1454,
     71: 1730,
     77: 2030,
     83: 2354,
     89: 2702,
     95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810
}

def _inter_distance(mol):
# see gto.mole.energy_nuc
    chargs = numpy.array([mol.atom_charge(i) for i in range(len(mol._atm))])
    coords = numpy.array([mol.atom_coord(i) for i in range(len(mol._atm))])
    rr = numpy.dot(coords, coords.T)
    rd = rr.diagonal()
    rr = rd[:,None] + rd - rr*2
    rr[numpy.diag_indices_from(rr)] = 0
    return numpy.sqrt(rr)


if __name__ == '__main__':
    import gto
    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#"out_h2o"
    h2o.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ])

    h2o.basis = {"H": '6-31g',
                 "O": '6-31g',}
    h2o.grids = {"H": (50, 302),
                 "O": (50, 302),}
    h2o.build()
    import time
    t0 = time.clock()
    g = Grids(h2o)
    g.setup_grids()
    print(g.coords.shape)
    print(time.clock() - t0)
