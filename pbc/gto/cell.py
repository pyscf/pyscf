#!/usr/bin/env python
# -*- coding: utf-8
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import sys
import json
import numpy as np
import scipy.linalg
import scipy.optimize
import pyscf.lib.parameters as param
import pyscf.gto.mole
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto.mole import format_atom, _symbol, _rm_digit, _std_symbol, _charge
from pyscf.pbc.gto import basis
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pbctools

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
        basis_tab : dict
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
    '''Pack the input args of :class:`Cell` to a dict, which can be serialized
    with :mod:`pickle`
    '''
    cldic = pyscf.gto.mole.pack(cell)
    cldic['h'] = cell.h
    cldic['gs'] = cell.gs
    cldic['precision'] = cell.precision
    cldic['pseudo'] = cell.pseudo
    cldic['ke_cutoff'] = cell.ke_cutoff
    cldic['nimgs'] = cell.nimgs
    cldic['ew_eta'] = cell.ew_eta
    cldic['ew_cut'] = cell.ew_cut
    return cldic

def unpack(celldic):
    '''Convert the packed dict to a :class:`Cell` object, to generate the
    input arguments for :class:`Cell` object.
    '''
    cl = Cell()
    cl.__dict__.update(celldic)
    return cl


def dumps(cell):
    '''Serialize Cell object to a JSON formatted str.
    '''
    exclude_keys = set(('output', 'stdout', '_keys'))

    celldic = dict(cell.__dict__)
    for k in exclude_keys:
        del(celldic[k])
    for k in celldic:
        if isinstance(celldic[k], np.ndarray):
            celldic[k] = celldic[k].tolist()
    celldic['atom'] = repr(cell.atom)
    celldic['basis']= repr(cell.basis)
    celldic['pseudo' ] = repr(cell.pseudo)

    try:
        return json.dumps(celldic)
    except TypeError:
        import warnings
        def skip_value(dic):
            dic1 = {}
            for k,v in dic.items():
                if (v is None or
                    isinstance(v, (basestring, bool, int, long, float))):
                    dic1[k] = v
                elif isinstance(v, (list, tuple)):
                    dic1[k] = v   # Should I recursively skip_vaule?
                elif isinstance(v, set):
                    dic1[k] = list(v)
                elif isinstance(v, dict):
                    dic1[k] = skip_value(v)
                else:
                    msg =('Function cell.dumps drops attribute %s because '
                          'it is not JSON-serializable' % k)
                    warnings.warn(msg)
            return dic1
        return json.dumps(skip_value(celldic), skipkeys=True)

def loads(cellstr):
    '''Deserialize a str containing a JSON document to a Cell object.
    '''
    from numpy import array  # for eval function
    celldic = json.loads(cellstr)
    if sys.version_info < (3,):
# Convert to utf8 because JSON loads fucntion returns unicode.
        def byteify(inp):
            if isinstance(inp, dict):
                return dict([(byteify(k), byteify(v)) for k, v in inp.iteritems()])
            elif isinstance(inp, (tuple, list)):
                return [byteify(x) for x in inp]
            elif isinstance(inp, unicode):
                return inp.encode('utf-8')
            else:
                return inp
        celldic = byteify(celldic)
    cell = Cell()
    cell.__dict__.update(celldic)
    cell.atom = eval(cell.atom)
    cell.basis= eval(cell.basis)
    cell.pseudo  = eval(cell.pseudo)
    cell._atm = np.array(cell._atm, dtype=np.int32)
    cell._bas = np.array(cell._bas, dtype=np.int32)
    cell._env = np.array(cell._env, dtype=np.double)
    cell._ecpbas = np.array(cell._ecpbas, dtype=np.int32)
    cell._h = np.asarray(cell._h)

    return cell

def intor_cross(intor, cell1, cell2, comp=1, hermi=0, kpt=None):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    nimgs = np.max((cell1.nimgs, cell2.nimgs), axis=0)
    Ls = get_lattice_Ls(cell1, nimgs)
# Change the basis position only, keep all other envrionments
    cellL = cell2.copy()
    ptr_coord = cellL._atm[:,pyscf.gto.PTR_COORD]
    _envL = cellL._env
    int1e = 0
    for L in Ls:
        _envL[ptr_coord+0] = cell2._env[ptr_coord+0] + L[0]
        _envL[ptr_coord+1] = cell2._env[ptr_coord+1] + L[1]
        _envL[ptr_coord+2] = cell2._env[ptr_coord+2] + L[2]
        if kpt is None:
            int1e += pyscf.gto.mole.intor_cross(intor, cell1, cellL, comp)
        else:
            factor = np.exp(1j*np.dot(kpt, L))
            int1e += pyscf.gto.mole.intor_cross(intor, cell1, cellL, comp) * factor
    return int1e


def get_lattice_Ls(cell, nimgs=None):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.'''
    if nimgs is None:
        nimgs = cell.nimgs
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell._h, Ts.T).T
    return Ls


class Cell(pyscf.gto.Mole):
    '''A Cell object holds the basic information of a crystal.

    Attributes:
        h : (3,3) ndarray
            The real-space unit cell lattice vectors, a "three-column" array [a1|a2|a3]
            (Note a1 = h[:,0]; a2 = h[:,1]; a2 = h[:,2]).  See defs. in MH Sec. 3.1
            Convert from relative or "scaled" coordinates `s` to "absolute"
            cartesian coordinates `r` via `r = np.dot(_h, s)`.
            Reciprocal lattice vectors are given by [b1|b2|b3] = 2 pi inv(_h.T).
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
        self.ke_cutoff = None # if set, defines a spherical cutoff
                              # of fourier components, with .5 * G**2 < ke_cutoff
        self.precision = 1.e-8
        self.pseudo = None
        self.dimension = None

##################################################
# These attributes are initialized by build function if not given
        self.nimgs = None
        self.ew_eta = None
        self.ew_cut = None

##################################################
# don't modify the following variables, they are not input arguments
        self.vol = None
        self._h = None
        self._pseudo = None
        self._keys = set(self.__dict__.keys())

    def build(self, *args, **kwargs):
        return self.build_(*args, **kwargs)

#Note: Exculde dump_input, parse_arg, basis from kwargs to avoid parsing twice
    def build_(self, dump_input=True, parse_arg=True,
               h=None, gs=None, ke_cutoff=None, precision=None, nimgs=None,
               ew_eta=None, ew_cut=None, pseudo=None, basis=None,
               dimension=None,
               *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.
        Whenever you change the value of the attributes of :class:`Cell`,
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            h : (3,3) ndarray
                The real-space unit cell lattice vectors, a "three-column" array [a1|a2|a3]
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
        if dimension is not None: self.dimension = dimension

        assert(self.h is not None)
        assert(self.gs is not None or self.ke_cutoff is not None)

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

        self.vol = float(scipy.linalg.det(self.lattice_vectors()))
        self._h = self.lattice_vectors()
        if self.ke_cutoff is not None:
            self.gs = pbctools.cutoff_to_gs(self._h, self.ke_cutoff)

        if self.ew_eta is None or self.ew_cut is None:
            self.ew_eta, self.ew_cut = self.get_ewald_params(self.precision)

        if dump_input and self.verbose >= logger.INFO:
            logger.info(self, 'lattice vector [a1        | a2        | a3       ]')
            logger.info(self, '               [%.9f | %.9f | %.9f]', *self._h[0])
            logger.info(self, '               [%.9f | %.9f | %.9f]', *self._h[1])
            logger.info(self, '               [%.9f | %.9f | %.9f]', *self._h[2])
            logger.info(self, 'Cell volume = %g', self.vol)
            logger.info(self, 'nimgs = %s', self.nimgs)
            logger.info(self, 'precision = %g', self.precision)
            logger.info(self, 'gs = %s', self.gs)
            logger.info(self, 'pseudo = %s', self.pseudo)
            logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
            logger.info(self, 'ew_eta = %g', self.ew_eta)
            logger.info(self, 'ew_cut = %s', self.ew_cut)


    @property
    def Gv(self):
        return self.get_Gv()

    @lib.with_doc(format_pseudo.__doc__)
    def format_pseudo(self, pseudo_tab):
        return format_pseudo(pseudo_tab)

    @lib.with_doc(format_basis.__doc__)
    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    def make_ecp_env(self, _atm, xxx, pre_env=[]):
        if self._pseudo:
            _atm, _ecpbas, _env = make_pseudo_env(self, _atm, self._pseudo, pre_env)
        else:
            _atm, _ecpbas, _env = _atm, [], pre_env
        return _atm, _ecpbas, _env

    def get_nimgs(self, precision=None):
        r'''Choose number of basis function images in lattice sums
        to include for given precision in overlap, using

        precision ~ 4 * pi * r^2 * e^{-\alpha r^2}

        where \alpha is the smallest exponent in the basis. Note
        that assumes an isolated exponent in the middle of the box, so
        it adds one additional lattice vector to be safe.
        '''
        if precision is None:
            precision = self.precision
        min_exp = np.min([np.min(self.bas_exp(ib)) for ib in range(self.nbas)])

        def fn(r):
            return np.log(4*np.pi*r**2)-min_exp*r**2-np.log(precision)

        guess = np.sqrt((5-np.log(precision))/min_exp)
        rcut = scipy.optimize.fsolve(fn, guess, xtol=1e-4)[0]
        rlengths = lib.norm(self.lattice_vectors(), axis=1) + 1e-200
        nimgs = np.ceil(np.reshape(rcut/rlengths, rlengths.shape[0])).astype(int)

        return nimgs+1 # additional lattice vector to take into account
                       # case where there are functions on the edges of the box.

    def get_ewald_params(self, precision=None):
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
        if precision is None:
            precision = self.precision

        #  See Martin, p. 85 
        _h = self.lattice_vectors()
        Gmax = min([ 2.*np.pi*self.gs[i]/lib.norm(_h[i,:]) for i in range(3) ])

        log_precision = np.log(precision)
        ew_eta = float(np.sqrt(-Gmax**2/(4*log_precision)))

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
        gxrange = range(self.gs[0]+1)+range(-self.gs[0],0)
        gyrange = range(self.gs[1]+1)+range(-self.gs[1],0)
        gzrange = range(self.gs[2]+1)+range(-self.gs[2],0)
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))

        invh = scipy.linalg.inv(self.lattice_vectors())
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
        '''Get absolute k-points (in 1/Bohr), given "scaled" k-points in
        fractions of lattice vectors.

        Args:
            scaled_kpts : (nkpts, 3) ndarray of floats

        Returns:
            abs_kpts : (nkpts, 3) ndarray of floats 
        '''
        # inv_h has reciprocal vectors as rows
        return 2*np.pi*np.dot(scaled_kpts, scipy.linalg.inv(self._h))

    def get_scaled_kpts(self, abs_kpts):
        '''Get scaled k-points, given absolute k-points in 1/Bohr.

        Args:
            abs_kpts : (nkpts, 3) ndarray of floats 

        Returns:
            scaled_kpts : (nkpts, 3) ndarray of floats
        '''
        return 1./(2*np.pi)*np.dot(abs_kpts, self._h)

    def copy(self):
        return copy(self)

    pack = pack
    @lib.with_doc(unpack.__doc__)
    def unpack(self, moldic):
        return unpack(moldic)
    def unpack_(self, moldic):
        self.__dict__.update(moldic)
        return self

    dumps = dumps
    @lib.with_doc(loads.__doc__)
    def loads(self, molstr):
        return loads(molstr)
    def loads_(self, molstr):
        self.__dict__.update(loads(molstr).__dict__)
        return self

    get_lattice_Ls = get_lattice_Ls

    def from_ase_(self, ase_atom):
        '''Update cell based on given ase atom object

        Examples:

        >>> from ase.lattice import bulk
        >>> cell.from_ase_(bulk('C', 'diamond', a=LATTICE_CONST))
        '''
        from pyscf.pbc.tools import pyscf_ase
        self.h = ase_atom.cell
        self.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        return self

    def pbc_intor(self, intor, comp=1, hermi=0, kpt=None):
        assert('2e' not in intor)
        return intor_cross(intor, self, self, comp, hermi, kpt)

