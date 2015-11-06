#!/usr/bin/env python
# -*- coding: utf-8
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import gc
import numpy as np
import scipy.linalg
import scipy.optimize
import pyscf
import pyscf.lib.parameters as param
from pyscf import lib
import pyscf.gto.mole
from pyscf.gto.mole import format_atom, _symbol, _rm_digit, _std_symbol, _charge
from pyscf.pbc.gto import basis
from pyscf.pbc.gto import pseudo

def format_pseudo(pseudo_tab):
    '''Convert the input :attr:`Cell.pseudo` (dict) to the internal data format.

    ``{ atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }``

    Args:
        pseudo_tab : dict
            Similar to :attr:`Cell.pseudo` (a dict), it **cannot** be a str

    Returns:
        Formatted :attr:`~Cell.pseudo`

    Examples:

    >>> pbc.format_pseudo({'H':'gth-blyp', 'He': 'gth-pade'})
    {'H': [[1],
        0.2, 2, [-4.19596147, 0.73049821], 0],
     'He': [[2],
        0.2, 2, [-9.1120234, 1.69836797], 0]}
    '''
    fmt_pseudo = {}
    for atom in pseudo_tab.keys():
        symb = _symbol(atom)
        rawsymb = _rm_digit(symb)
        stdsymb = _std_symbol(rawsymb)
        symb = symb.replace(rawsymb, stdsymb)

        if isinstance(pseudo_tab[atom], str):
            fmt_pseudo[symb] = pseudo.load(pseudo_tab[atom], stdsymb)
        else:
            fmt_pseudo[symb] = pseudo_tab[atom]
    return fmt_pseudo

def make_pseudo_env(cell, _atm, _pseudo, pre_env=[]):
    for ia, atom in enumerate(cell._atom):
        symb = atom[0]
        _atm[ia,0] = sum(_pseudo[symb][0])
    _pseudobas = None
    return _atm, _pseudobas, pre_env

def format_basis(basis_tab):
    '''Convert the input :attr:`Cell.basis` to the internal data format.

    ``{ atom: (l, kappa, ((-exp, c_1, c_2, ..), nprim, nctr, ptr-exps, ptr-contraction-coeff)), ... }``

    Args:
        basis_tab : list
            Similar to :attr:`Cell.basis`, it **cannot** be a str

    Returns:
        Formated :attr:`~Cell.basis`

    Examples:

    >>> pbc.format_basis({'H':'gth-szv'})
    {'H': [[0,
        (8.3744350009, -0.0283380461),
        (1.8058681460, -0.1333810052),
        (0.4852528328, -0.3995676063),
        (0.1658236932, -0.5531027541)]]}
    '''
    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _symbol(atom)
        rawsymb = _rm_digit(symb)
        stdsymb = _std_symbol(rawsymb)
        symb = symb.replace(rawsymb, stdsymb)

        if isinstance(basis_tab[atom], str):
            fmt_basis[symb] = basis.load(basis_tab[atom], stdsymb)
        else:
            fmt_basis[symb] = basis_tab[atom]
    return fmt_basis

def copy(cell):
    '''Deepcopy of the given :class:`Cell` object
    '''
    import copy
    newcell = pyscf.gto.mole.copy(cell)
    newcell._pseudo = copy.deepcopy(cell._pseudo)
    return newcell

def pack(cell):
    '''Pack the given :class:`Cell` to a dict, which can be serialized with :mod:`pickle`
    '''
    cldic = pyscf.gto.mole.pack(cell)
    cldic['h'] = cell.h
    cldic['gs'] = cell.gs
    cldic['precision'] = cell.precision
    cldic['nimgs'] = cell.nimgs
    cldic['ew_eta'] = cell.ew_eta
    cldic['ew_cut'] = cell.ew_cut
    return cldic

def unpack(celldic):
    '''Convert the packed dict to a :class:`Cell` object.
    '''
    cl = Cell()
    cl.__dict__.update(celldic)
    return cl


class Cell(pyscf.gto.Mole):
    '''A Cell object holds the basic information of a crystal.

    Attributes:
        h : (3,3) ndarray
            The real-space unit cell lattice vector, three *columns*: array((a1,a2,a3))
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.
        pseudo : dict or str
            To define pseudopotential.
        precision : float
            To control Ewald sums and lattice sums accuracy
        ke_cutoff : float
            If set, defines a spherical cutoff of fourier components, with .5 * G**2 < ke_cutoff

        ** Following attributes (for experts) are automatically generated. **

        nimgs : (3,) ndarray of ints
            Number of basis function images in lattice sums, It affects the
            accuracy of integrals.  See :func:`get_nimgs`.
        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.  See :func:`get_ewald_params`

    (See other attributes in :class:`Mole`)

    Examples:

    >>> mol = Mole(atom='H^2 0 0 0; H 0 0 1.1', basis='sto3g')
    >>> cl = Cell()
    >>> cl.build(h='3 0 0; 0 3 0; 0 0 3', gs=[8,8,8], atom='C 1 1 1', basis='sto3g')
    >>> print(cl.atom_symbol(0))
    C
    >>> print(cl.get_nimgs(1e-9))
    [3,3,3]
    '''
    def __init__(self, **kwargs):
        pyscf.gto.Mole.__init__(self, **kwargs)
        self.h = None # lattice vectors, three *columns*: array((a1,a2,a3))
        self.gs = None
        self.precision = 1.e-8
        self.nimgs = None
        self.ew_eta = None
        self.ew_cut = None
        self.pseudo = None
        self.ke_cutoff = None # if set, defines a spherical cutoff
                              # of fourier components, with .5 * G**2 < ke_cutoff

##################################################
# don't modify the following variables, they are not input arguments
        self.Gv = None
        self.vol = None
        self._h = None # lattice vectors, three *columns*: array((a1,a2,a3))
                       # See defs. in MH 3.1
                       # scaled coordinates r = numpy.dot(_h, [x,y,z])
                       # reciprocal lattice array((b1,b2,b3)) = inv(_h).T
        self._pseudo = None
        self._keys = set(self.__dict__.keys())


    def build(self, *args, **kwargs):
        return self.build_(*args, **kwargs)

#Note: Exculde dump_input, parse_arg, basis from kwargs to avoid parsing twice
    def build_(self, dump_input=True, parse_arg=True,
               h=None, gs=None, precision=None, nimgs=None,
               ew_eta=None, ew_cut=None, pseudo=None, basis=None,
               *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.
        Whenever you change the value of the attributes of :class:`Cell`,
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            h : (3,3) ndarray
                The real-space unit cell lattice vectors.
            gs : (3,) ndarray of ints
                The number of *positive* G-vectors along each direction.
            pseudo : dict or str
                To define pseudopotential.  If given, overwrite :attr:`Cell.pseudo`
        '''
        if h is not None: self.h = h
        if gs is not None: self.gs = gs
        if nimgs is not None: self.nimgs = nimgs
        if ew_eta is not None: self.ew_eta = ew_eta
        if ew_cut is not None: self.ew_cut = ew_cut
        if pseudo is not None: self.pseudo = pseudo
        if basis is not None: self.basis = basis

        assert(self.h is not None)
        assert(self.gs is not None)

        if 'unit' in kwargs:
            self.unit = kwargs['unit']

        if 'atom' in kwargs:
            self.atom = kwargs['atom']
        _atom = format_atom(self.atom, unit=self.unit)

        # Set-up pseudopotential if it exists
        # This must happen before build() because it affects
        # tot_electrons() via atom_charge()
        if self.pseudo is not None:
            if isinstance(self.pseudo, str):
                # specify global pseudo for whole molecule
                uniq_atoms = set([a[0] for a in _atom])
                self._pseudo = self.format_pseudo(dict([(a, self.pseudo)
                                                      for a in uniq_atoms]))
            else:
                self._pseudo = self.format_pseudo(self.pseudo)
#            self.nelectron = self.tot_electrons()
#            if (self.nelectron+self.spin) % 2 != 0:
#                raise RuntimeError('Electron number %d and spin %d are not consistent\n' %
#                                   (self.nelectron, self.spin))

        # Check if we're using a GTH basis
        # This must happen before build() because it prepares self.basis
        if isinstance(self.basis, str):
            basis_name = self.basis.lower().replace(' ', '').replace('-', '').replace('_', '')
            if 'gth' in basis_name:
                # specify global basis for whole molecule
                uniq_atoms = set([a[0] for a in _atom])
                self.basis = self.format_basis(dict([(a, basis_name)
                                                     for a in uniq_atoms]))
            # This sets self.basis to be internal format, and will
            # be parsed appropriately by Mole.build

        # Do regular Mole.build_ with usual kwargs
        pyscf.gto.Mole.build_(self, dump_input, parse_arg, *args, **kwargs)

        if self.nimgs is None:
            self.nimgs = self.get_nimgs(self.precision)

        if self.ew_eta is None or self.ew_cut is None:
            self.ew_eta, self.ew_cut = self.get_ewald_params(self.precision)

        self.Gv = self.get_Gv()
        self.vol = scipy.linalg.det(self.lattice_vectors())
        self._h = self.lattice_vectors()

    def format_pseudo(self, pseudo_tab):
        return format_pseudo(pseudo_tab)

    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    def make_ecp_env(self, _atm, xxx, pre_env=[]):
        if self._pseudo:
            _atm, _ecpbas, _env = make_pseudo_env(self, _atm, self._pseudo, pre_env)
        else:
            _atm, _ecpbas, _env = _atm, None, pre_env
        return _atm, _ecpbas, _env

#    def atom_charge(self, atm_id):
#        '''Return the atom charge, accounting for pseudopotential.'''
#        if self.pseudo is None:
#            # This is what the original Mole.atom_charge() returns
#            CHARGE_OF  = 0
#            return self._atm[atm_id,CHARGE_OF]
#        else:
#            # Remember, _pseudo is a dict
#            nelecs = self._pseudo[ self.atom_symbol(atm_id) ][0]
#            return sum(nelecs)

    def get_nimgs(self, precision):
        r'''Choose number of basis function images in lattice sums
        to include for given precision in overlap, using

        precision ~ 4 * pi * r^2 * e^{-\alpha r^2}

        where \alpha is the smallest exponent in the basis. Note
        that assumes an isolated exponent in the middle of the box, so
        it adds one additional lattice vector to be safe.
        '''
        min_exp = np.min([np.min(self.bas_exp(ib)) for ib in range(self.nbas)])

        def fn(r):
            return (np.log(4*np.pi*r**2)-min_exp*r**2-np.log(precision))**2

        rcut = np.sqrt(-(np.log(precision)/min_exp)) # guess
        rcut = scipy.optimize.fsolve(fn, rcut)[0]
        rlengths = lib.norm(self.lattice_vectors(), axis=1) + 1e-200
        nimgs = np.ceil(np.reshape(rcut/rlengths, rlengths.shape[0])).astype(int)

        return nimgs+1 # additional lattice vector to take into account
                       # case where there are functions on the edges of the box.

    def get_ewald_params(self, precision):
        r'''Choose a reasonable value of Ewald 'eta' and 'cut' parameters.

        Choice is based on largest G vector and desired relative precision.

        The relative error in the G-space sum is given by (keeping only
        exponential factors)
            precision ~ e^{(-Gmax^2)/(4 \eta^2)}
        which determines eta. Then, real-space cutoff is determined by (exp.
        factors only)
            precision ~ erfc(eta*rcut) / rcut ~ e^{(-eta**2 rcut*2)}

        Returns:
            ew_eta, ew_cut : float
                The Ewald 'eta' and 'cut' parameters.
        '''
        #  See Martin, p. 85 
        _h = self.lattice_vectors()
        Gmax = min([ 2.*np.pi*self.gs[i]/lib.norm(_h[i,:]) for i in range(3) ])

        log_precision = np.log(precision)
        ew_eta = np.sqrt(-Gmax**2/(4*log_precision))

        rcut = np.sqrt(-log_precision)/ew_eta
        ew_cut = self.get_bounding_sphere(rcut)
        return ew_eta, ew_cut

    def get_bounding_sphere(self, rcut):
        '''Finds all the lattice points within a sphere of radius rcut.  

        Defines a parallelipiped given by -N_x <= n_x <= N_x, with x in [1,3]
        See Martin p. 85

        Args:
            rcut : number
                real space cut-off for interaction

        Returns:
            cut : ndarray of 3 ints defining N_x
        '''
        invhT = scipy.linalg.inv(self.lattice_vectors().T)
        Gmat = invhT.T
        n1 = np.ceil(lib.norm(Gmat[0,:])*rcut).astype(int)
        n2 = np.ceil(lib.norm(Gmat[1,:])*rcut).astype(int)
        n3 = np.ceil(lib.norm(Gmat[2,:])*rcut).astype(int)
        cut = np.array([n1, n2, n3])
        return cut

    def get_Gv(self):
        '''Calculate three-dimensional G-vectors for the cell; see MH (3.8).

        Indices along each direction go as [0...self.gs, -self.gs...-1]
        to follow FFT convention. Note that, for each direction, ngs = 2*self.gs+1.

        Args:
            self : instance of :class:`Cell`

        Returns:
            Gv : (ngs, 3) ndarray of floats
                The array of G-vectors.
        '''
        invh = scipy.linalg.inv(self.lattice_vectors())

        gxrange = range(self.gs[0]+1)+range(-self.gs[0],0)
        gyrange = range(self.gs[1]+1)+range(-self.gs[1],0)
        gzrange = range(self.gs[2]+1)+range(-self.gs[2],0)
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))

        Gv = 2*np.pi* np.dot(gxyz, invh)
        return Gv

    def get_SI(self):
        '''Calculate the structure factor for all atoms; see MH (3.34).

        Args:
            self : instance of :class:`Cell`

        Returns:
            SI : (natm, ngs) ndarray, dtype=np.complex128
                The structure factor for each atom at each G-vector.
        '''
        coords = [self.atom_coord(ia) for ia in range(self.natm)]
        SI = np.exp(-1j*np.dot(coords, self.Gv.T))
        return SI

    def lattice_vectors(self):
        if isinstance(self.h, str):
            h = self.h.replace(';',' ').replace(',',' ').replace('\n',' ')
            h = np.asarray([float(x) for x in h.split()]).reshape(3,3)
        else:
            h = np.asarray(self.h)
        if self.unit.startswith(('B','b','au','AU')):
            return h
        elif self.unit.startswith(('A','a')):
            return h * (1./param.BOHR)
        else:
            return h * (1./self.unit)

    def get_abs_kpts(self, scaled_kpts):
        '''Get absolute kpts (in inverse Bohr), given "scaled" kpts in fractions 
        of lattice vectors.

        Args:
            scaled_kpts : (nkpts, 3) ndarray of floats

        Returns:
            abs_kpts : (nkpts, 3) ndarray of floats 
        '''
        # inv_h has reciprocal vectors as rows
        return 2*np.pi*np.dot(scaled_kpts, scipy.linalg.inv(self._h))

    def copy(self):
        return copy(self)

    def pack(self):
        return pack(self)

    def unpack(self):
        return unpack(self)

