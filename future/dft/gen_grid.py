#!/usr/bin/env python
#
# File: gen_grid.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate DFT grids and weights, based on the code provided by Gerald Knizia <>
'''


import numpy
import radi
from pyscf.lib import _dft


#TODO: OPTIMIZE ME ACCORDING TO JCP 102, 346

class Grids(object):
    def __init__(self, mol):
        self.mol = mol
        self.atomic_radii = radi.BRAGG_RADII
        #self.atomic_radii = radi.COVALENT_RADII
        self.radi_method = radi.gauss_chebeshev
        self.adjust_atomic_radii = True
        #self.becke_scheme = stratmann
        self.becke_scheme = original_becke

        self.coords  = None
        self.weights = None
        #TODO:self.blocks = None
        for atm,(rad,ang) in mol.grids.items():
            if not _allow_ang_grids(ang):
                raise ValueError('angular grids %d for %s' % (ang, atm))

        self._tril_idx = numpy.tril_indices(mol.natm, -1)
        self._atm_dist_inv = 1/mol.inter_distance()[self._tril_idx]
        self._atm_coords = numpy.array([mol.coord_of_atm(ia) \
                                        for ia in range(mol.natm)])


    def setup_grids(self):
        mol = self.mol
        atom_grids_tab = self.gen_atomic_grids(mol)
        self.coords, self.weights = self.gen_partition(mol, atom_grids_tab)
        return self.coords, self.weights

    def gen_atomic_grids(self, mol):
        atom_grids_tab = {}
        for atm in mol.basis.keys():
            if mol.grids.has_key(atm):
                n_rad, n_ang = mol.grids[atm]
            else:
                chg = gto.mole._charge(atm)
                n_rad = default_num_radpt(chg)
                n_ang = default_num_angpt(chg)
            rad, rad_weight = self.radi_method(n_rad)
            # from FDftGridGenerator::GetAtomGridParams
            # atomic_scale = 1
            # rad *= atomic_scale
            # rad_weight *= atomic_scale

            #TODO: reduce grid size for inner shells, i.e. for inner most
            # radial points, use smaller n_ang
            ang, ang_weight = _dft.make_angular_grid(n_ang)
            atom_grids_tab[atm] = (rad, rad_weight, ang, ang_weight)
        return atom_grids_tab

    def count_tot_girds(self, mol, atom_grids_tab):
        ngrid = 0
        for ia in range(mol.natm):
            symb = mol.symbol_of_atm(ia)
            _, rad_wt, _, ang_wt = atom_grids_tab[symb]
            ngrid += rad_wt.size * ang_wt.size
        return ngrid

    # for a given grid, fractions on each atom
    def gen_grid_partition(self, mol, coords, adjust_atomic_radii):
        ngrid = coords.shape[0]
        grid_dist = numpy.empty((mol.natm,ngrid,3))
        for ia in range(mol.natm):
            grid_dist[ia] = coords - self._atm_coords[ia]
        grid_dist = numpy.sqrt(numpy.einsum('ijk,ijk->ij',grid_dist,grid_dist))
        grid_diff = grid_dist[self._tril_idx[0]] \
                  - grid_dist[self._tril_idx[1]]
        mu_tab = numpy.einsum('i,ij->ij', self._atm_dist_inv, grid_diff)
        g = adjust_atomic_radii(mu_tab)
        g = self.becke_scheme(g)
        gplus = .5 * (1-g)
        gminus = .5 * (1+g)
        pbecke = numpy.ones((mol.natm,ngrid))
        for k,(i,j) in enumerate(zip(*self._tril_idx)):
            pbecke[i] *= gplus[k]
            pbecke[j] *= gminus[k]

        return pbecke

    def gen_partition(self, mol, atom_grids_tab):
        #TODO: reduce grid size for inner shells

# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
        if self.adjust_atomic_radii:
            rad = numpy.array([self.atomic_radii[mol.charge_of_atm(ia)-1] \
                               for ia in range(mol.natm)])
            rr = rad.reshape(-1,1)/rad
            a = .25 * (rr.T - rr)
            a = a[self._tril_idx]
            a[a<-.5] = -.5
            a[a>0.5] = 0.5
            fn_adjust = lambda g: g + numpy.einsum('i,ij->ij', a, (1-g*g))
        else:
            fn_adjust = lambda g: g

        ip = 0
        ngrid = self.count_tot_girds(mol, atom_grids_tab)
        coords_all = []
        weights_all = []
        for ia in range(mol.natm):
            symb = mol.symbol_of_atm(ia)
            rad, rad_weight, ang, ang_weight = atom_grids_tab[symb]
            coords = numpy.einsum('i,jk->ijk',rad,ang).reshape(-1,3) \
                    + self._atm_coords[ia]
            vol = numpy.einsum('i,j->ij', rad_weight, ang_weight).reshape(-1)
            pbecke = self.gen_grid_partition(mol, coords, fn_adjust)
            weights = vol * pbecke[ia] / pbecke.sum(axis=0)
            coords_all.append(coords)
            weights_all.append(weights)
        return numpy.vstack(coords_all), numpy.hstack(weights_all)


#TODO: screen out insiginficant grids

#TODO:
NPOINTS_PER_BLOCK = 128
def generate_blocks():
    # FDftGridGenerator::BlockifyGridR
    return block_list

LEVEL = 3
def default_num_radpt(charge):
    n = (0.75 + (LEVEL-1)*0.2) * 14 * (charge+2.)**(1/3.)
    return int(n)

def default_num_angpt(charge):
    atom_period = 0
    tab = (2,8,8,18,18,32,32,50)
    for i in range(7):
        if charge < sum(tab[:i]):
            break
    atom_period = i

    DefaultAngularGridLs = (9,11,17,23,29,35,47,59,71,89)
    l = DefaultAngularGridLs[LEVEL-1 + atom_period-1]
    l2npt = {
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
        131: 5810}
    return l2npt[l]

def _allow_ang_grids(n):
    return n in (1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266,
                 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354,
                 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810)

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


if __name__ == '__main__':
    from pyscf import gto
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
    print g.coords.shape
    print time.clock() - t0
