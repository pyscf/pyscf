#!/usr/bin/env python
# File: gen_grid.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate DFT grids and weights, based on the code provided by Gerald Knizia <>
'''


import ctypes
import numpy
import pyscf.lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import radi

libdft = pyscf.lib.load_library('libdft')

# ~= (L+1)**2/3
LEBEDEV_ORDER = {
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
LEBEDEV_NGRID = numpy.asarray((
    1   , 6   , 14  , 26  , 38  , 50  , 74  , 86  , 110 , 146 ,
    170 , 194 , 230 , 266 , 302 , 350 , 434 , 590 , 770 , 974 ,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810))

# SG0
# S. Chien and P. Gill,  J. Comput. Chem. 27 (2006) 730-739.

# P.M.W. Gill, B.G. Johnson, J.A. Pople, Chem. Phys. Letters 209 (1993) 506-512
SG1RADII = numpy.array((
    0,
    1.0000,                                                 0.5882,
    3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
    4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333))


def sg1_prune(nuc, rads, n_ang):
    '''SG1, CPL, 209, 506

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
# In SG1 the ang grids for the five regions
#            6  38 86  194 86
    leb_ngrid = numpy.array([6, 38, 86, 194, 86])
    alphas = numpy.array((
        (0.25  , 0.5, 1.0, 4.5),
        (0.1667, 0.5, 0.9, 3.5),
        (0.1   , 0.4, 0.8, 2.5)))
    if nuc <= 2:  # H, He
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[0]).sum(axis=1)
    elif nuc <= 10:  # Li - Ne
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[1]).sum(axis=1)
    else:
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[2]).sum(axis=1)
    return leb_ngrid[place]

def nwchem_prune(nuc, rads, n_ang):
    '''NWChem

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
    alphas = numpy.array((
        (0.25  , 0.5, 1.0, 4.5),
        (0.1667, 0.5, 0.9, 3.5),
        (0.1   , 0.4, 0.8, 2.5)))
    leb_ngrid = LEBEDEV_NGRID[4:]  # [38, 50, 74, 86, ...]
    if n_ang < 50:
        angs = numpy.empty(len(rads), dtype=int)
        angs[:] = n_ang
        return angs
    elif n_ang == 50:
        leb_l = numpy.array([1, 2, 2, 2, 1])
    else:
        idx = numpy.where(leb_ngrid==n_ang)[0][0]
        leb_l = numpy.array([1, 3, idx-1, idx, idx-1])

    if nuc <= 2:  # H, He
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[0]).sum(axis=1)
    elif nuc <= 10:  # Li - Ne
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[1]).sum(axis=1)
    else:
        place = ((rads/SG1RADII[nuc]).reshape(-1,1) > alphas[2]).sum(axis=1)
    angs = leb_l[place]
    angs = leb_ngrid[angs]
    return angs

# Prune scheme JCP 102, 346
def treutler_prune(nuc, rads, n_ang):
    '''Treutler-Ahlrichs

    Args:
        nuc : int
            Nuclear charge.

        rads : 1D array
            Grid coordinates on radical axis.

        n_ang : int
            Max number of grids over angular part.

    Returns:
        A list has the same length as rads. The list element is the number of
        grids over angular part for each radial grid.
    '''
    nr = len(rads)
    leb_ngrid = numpy.empty(nr, dtype=int)
    leb_ngrid[:nr//3] = 14 # l=5
    leb_ngrid[nr//3:nr//2] = 50 # l=11
    leb_ngrid[nr//2:] = n_ang
    return leb_ngrid



###########################################################
# Becke partitioning

# Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996), eq.11
def stratmann(g):
    '''Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996)'''
    a = .64  # comment after eq. 14
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
    '''Becke, JCP, 88, 2547 (1988)'''
    # g = (3 - g**2) * g * .5
    # g = (3 - g**2) * g * .5
    # g = (3 - g**2) * g * .5
    # return g
    g1 = numpy.empty_like(g)
    libdft.VXCoriginal_becke(g1.ctypes.data_as(ctypes.c_void_p),
                             g.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(g.size))
    return g1

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, prune=nwchem_prune):
    '''Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = mol.atom_charge(ia)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    if n_ang in LEBEDEV_ORDER:
                        n_ang = LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = _default_rad(chg, level)
                n_ang = _default_ang(chg, level)
            rad, dr = radi_method(n_rad)
            rad_weight = 4*numpy.pi * rad*rad * dr
            # atomic_scale = 1
            # rad *= atomic_scale
            # rad_weight *= atomic_scale

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad
            pyscf.lib.logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s',
                                   symb, n_rad, angs)

            angs = numpy.array(angs)
            coords = []
            vol = []
            for n in set(angs):
                grid = numpy.empty((n,4))
                libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(n))
                idx = numpy.where(angs==n)[0]
                for i0, i1 in prange(0, len(idx), 12):  # 12 radi-grids as a group
                    coords.append(numpy.einsum('i,jk->jik',rad[idx[i0:i1]],
                                               grid[:,:3]).reshape(-1,3))
                    vol.append(numpy.einsum('i,j->ji', rad_weight[idx[i0:i1]],
                                            grid[:,3]).ravel())
            atom_grids_tab[symb] = (numpy.vstack(coords), numpy.hstack(vol))
    return atom_grids_tab


def gen_partition(mol, atom_grids_tab, atomic_radii_adjust=None,
                  becke_scheme=original_becke):
    '''Generate the mesh grid coordinates and weights for DFT numerical integration.
    We can change atomic_radii_adjust becke_scheme to generate different meshgrid.

    Returns:
        grid_coord and grid_weight arrays.  grid_coord array has shape (N,3);
        weight 1D array has N elements.
    '''
    atm_coords = numpy.array([mol.atom_coord(i) for i in range(mol.natm)])
    atm_dist = radi._inter_distance(mol)
    def gen_grid_partition(coords):
        ngrid = coords.shape[0]
        grid_dist = numpy.empty((mol.natm,ngrid))
        for ia in range(mol.natm):
            dc = coords - atm_coords[ia]
            grid_dist[ia] = numpy.sqrt(numpy.einsum('ij,ij->i',dc,dc))
        pbecke = numpy.ones((mol.natm,ngrid))
        for i in range(mol.natm):
            for j in range(i):
                g = 1/atm_dist[i,j] * (grid_dist[i]-grid_dist[j])
                if callable(atomic_radii_adjust):
                    g = atomic_radii_adjust(i, j, g)
                g = becke_scheme(g)
                pbecke[i] *= .5 * (1-g)
                pbecke[j] *= .5 * (1+g)

        return pbecke

    coords_all = []
    weights_all = []
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke = gen_grid_partition(coords)
        weights = vol * pbecke[ia] / pbecke.sum(axis=0)
        coords_all.append(coords)
        weights_all.append(weights)
    return numpy.vstack(coords_all), numpy.hstack(weights_all)



class Grids(pyscf.lib.StreamObject):
    '''DFT mesh grids

    Attributes for Grids:
        level : int (0 - 6)
            big number for large mesh grids, default is 3

        atomic_radii : function or None
            can be one of
            | radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
            | radi.treutler_atomic_radii_adjust(mol, radi.COVALENT_RADII)
            | radi.becke_atomic_radii_adjust(mol, radi.BRAGG_RADII)
            | radi.becke_atomic_radii_adjust(mol, radi.COVALENT_RADII)
            | None,          to switch off atomic radii adjustment

        radi_method : function(n) => (rad_grids, rad_weights)
            scheme for radial grids, can be one of
            | radi.treutler
            | radi.gauss_chebyshev

        becke_scheme : function(v) => array_like_v
            weight partition function, can be one of
            | gen_grid.stratmann
            | gen_grid.original_becke

        prune : function(nuc, rad_grids, n_ang) => list_n_ang_for_each_rad_grid
            scheme to reduce number of grids, can be one of
            | gen_grid.sg1_prune
            | gen_grid.nwchem_prune
            | gen_grid.treutler_prune
            | None  (to switch off grid pruning)

        symmetry : bool
            whether to symmetrize mesh grids (TODO)

        atom_grid : dict
            Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.

        Examples:

        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
        >>> grids = dft.gen_grid.Grids(mol)
        >>> grids.level = 4
        >>> grids.build_()
        '''
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.atomic_radii = radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
        #self.atomic_radii = radi.becke_atomic_radii_adjust(mol, radi.BRAGG_RADII)
        #self.atomic_radii = radi.becke_atomic_radii_adjust(mol, radi.COVALENT_RADII)
        #self.atomic_radii = None # to switch off atomic radii adjustment
        self.radi_method = radi.treutler
        #self.radi_method = radi.gauss_chebyshev
        #self.becke_scheme = stratmann
        self.becke_scheme = original_becke
        self.level = 3
        self.prune = nwchem_prune
        self.symmetry = mol.symmetry
        self.atom_grid = {}

##################################################
# don't modify the following attributes, they are not input options
        self.coords  = None
        self.weights = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        logger.info(self, 'radial grids: %s', self.radi_method.__doc__)
        logger.info(self, 'becke partition: %s', self.becke_scheme.__doc__)
        logger.info(self, 'pruning grids: %s', self.prune)
        logger.info(self, 'grids dens level: %d', self.level)
        logger.info(self, 'symmetrized grids: %d', self.symmetry)
        if self.atomic_radii is not None:
            logger.info(self, 'atom radii adjust function: %s',
                        self.atomic_radii.__doc__)
        if self.atom_grid:
            logger.info(self, 'User specified grid scheme %s', str(self.atom_grid))
        return self

    def build_(self, mol=None):
        if mol is None: mol = self.mol
        self.check_sanity()
        atom_grids_tab = self.gen_atomic_grids(mol, atom_grid=self.atom_grid,
                                               radi_method=self.radi_method,
                                               level=self.level,
                                               prune=self.prune)
        self.coords, self.weights = \
                self.gen_partition(mol, atom_grids_tab, self.atomic_radii,
                                   self.becke_scheme)
        pyscf.lib.logger.info(self, 'tot grids = %d', len(self.weights))
        return self.coords, self.weights
    build = build_
    setup_grids = build_
    setup_grids_ = build_

    def kernel(self, mol=None):
        self.dump_flags()
        return self.build_(mol)

    def gen_atomic_grids(self, mol, atom_grid=None, radi_method=None,
                         level=None, prune=None):
        ''' See gen_grid.gen_atomic_grids function'''
        if atom_grid is None: atom_grid = self.atom_grid
        if radi_method is None: radi_method = mol.radi_method
        if level is None: level = self.level
        if prune is None: prune = self.prune
        return gen_atomic_grids(mol, atom_grid, self.radi_method, level, prune)

    def gen_partition(self, mol, atom_grids_tab, atomic_radii=None,
                      becke_scheme=original_becke):
        ''' See gen_grid.gen_partition function'''
        return gen_partition(mol, atom_grids_tab, atomic_radii,
                             becke_scheme)

    @property
    def prune_scheme(self):
        sys.stderr.write('WARN: Attribute .prune_scheme will be removed in PySCF v1.1. '
                         'Please use .prune instead\n')
        return self.prune



def _default_rad(nuc, level=3):
    '''Number of radial grids '''
    tab   = numpy.array( (2 , 10, 18, 36, 54, 86, 118))
    #           Period    1   2   3   4   5   6   7         # level
    grids = numpy.array((( 20, 30, 35, 45, 50, 55, 60),     # 0
                         ( 30, 45, 50, 60, 65, 70, 75),     # 1
                         ( 40, 60, 65, 75, 80, 85, 90),     # 2
                         ( 50, 75, 80, 90, 95,100,105),     # 3
                         ( 60, 90, 95,105,110,115,120),     # 4
                         ( 70,105,110,120,125,130,135),     # 5
                         ( 80,120,125,135,140,145,150),     # 6
                         ( 90,135,140,150,155,160,165),     # 7
                         (100,150,155,165,170,175,180),     # 8
                         (200,200,200,200,200,200,200),))   # 9
    period = (nuc > tab).sum()
    return grids[level,period]

def _default_ang(nuc, level=3):
    '''Order of angular grids. See LEBEDEV_ORDER for the mapping of
    the order and the number of angular grids'''
    tab   = numpy.array( (2 , 10, 18, 36, 54, 86, 118))
    #           Period    1   2   3   4   5   6   7         # level
    order = numpy.array(((15, 17, 17, 17, 17, 17, 17 ),     # 0
                         (17, 23, 23, 23, 23, 23, 23 ),     # 1
                         (23, 29, 29, 29, 29, 29, 29 ),     # 2
                         (29, 29, 35, 35, 35, 35, 35 ),     # 3
                         (35, 41, 41, 41, 41, 41, 41 ),     # 4
                         (41, 47, 47, 47, 47, 47, 47 ),     # 5
                         (47, 53, 53, 53, 53, 53, 53 ),     # 6
                         (53, 59, 59, 59, 59, 59, 59 ),     # 7
                         (59, 59, 59, 59, 59, 59, 59 ),     # 8
                         (65, 65, 65, 65, 65, 65, 65 ),))   # 9
    period = (nuc > tab).sum()
    return LEBEDEV_ORDER[order[level,period]]

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)




if __name__ == '__main__':
    import gto
    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#"out_h2o"
    h2o.atom = [
        ['O' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)] ]
    h2o.build()
    import time
    t0 = time.clock()
    g = Grids(h2o)
    g.setup_grids()
    print(g.coords.shape)
    print(time.clock() - t0)

