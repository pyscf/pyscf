#!/usr/bin/env python
# -*- coding: utf-8
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import sys
import json
import ctypes
import warnings
import numpy as np
import scipy.linalg
import scipy.misc
import scipy.special
import scipy.optimize
import pyscf.lib.parameters as param
from pyscf import lib
from pyscf import dft
from pyscf.lib import logger
from pyscf.gto import mole
from pyscf.gto import moleintor
from pyscf.gto.mole import _symbol, _rm_digit, _atom_symbol, _std_symbol, _charge
from pyscf.gto.mole import conc_env, uncontract
from pyscf.pbc.gto import basis
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pbctools
from pyscf.gto.basis import ALIAS as MOLE_ALIAS

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

libpbc = lib.load_library('libpbc')

def M(**kwargs):
    r'''This is a shortcut to build up Cell object.

    Examples:

    >>> from pyscf.pbc import gto
    >>> cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g', gs=[10]*3)
    '''
    cell = Cell()
    cell.build(**kwargs)
    return cell
C = M


def format_pseudo(pseudo_tab):
    r'''Convert the input :attr:`Cell.pseudo` (dict) to the internal data format::

       { atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }

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
    for atom in pseudo_tab:
        symb = _symbol(atom)
        rawsymb = _rm_digit(symb)
        stdsymb = _std_symbol(rawsymb)
        symb = symb.replace(rawsymb, stdsymb)

        if isinstance(pseudo_tab[atom], (str, unicode)):
            fmt_pseudo[symb] = pseudo.load(str(pseudo_tab[atom]), stdsymb)
        else:
            fmt_pseudo[symb] = pseudo_tab[atom]
    return fmt_pseudo

def make_pseudo_env(cell, _atm, _pseudo, pre_env=[]):
    for ia, atom in enumerate(cell._atom):
        symb = atom[0]
        if symb in _pseudo:
            _atm[ia,0] = sum(_pseudo[symb][0])
    _pseudobas = None
    return _atm, _pseudobas, pre_env

def format_basis(basis_tab):
    '''Convert the input :attr:`Cell.basis` to the internal data format::

      { atom: (l, kappa, ((-exp, c_1, c_2, ..), nprim, nctr, ptr-exps, ptr-contraction-coeff)), ... }

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
    def convert(basis_name, symb):
        if basis_name.lower().startswith('unc'):
            return uncontract(basis.load(basis_name[3:], symb))
        else:
            return basis.load(basis_name, symb)

    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _atom_symbol(atom)
        stdsymb = _std_symbol(symb)
        if stdsymb.startswith('GHOST-'):
            stdsymb = stdsymb[6:]
        atom_basis = basis_tab[atom]
        if isinstance(atom_basis, (str, unicode)):
            if 'gth' in atom_basis:
                bset = convert(str(atom_basis), symb)
            else:
                bset = atom_basis
        else:
            bset = []
            for rawb in atom_basis:
                if isinstance(rawb, (str, unicode)) and 'gth' in rawb:
                    bset.append(convert(str(rawb), stdsymb))
                else:
                    bset.append(rawb)
        fmt_basis[symb] = bset
    return mole.format_basis(fmt_basis)

def copy(cell):
    '''Deepcopy of the given :class:`Cell` object
    '''
    import copy
    newcell = mole.copy(cell)
    newcell._pseudo = copy.deepcopy(cell._pseudo)
    return newcell

def pack(cell):
    '''Pack the input args of :class:`Cell` to a dict, which can be serialized
    with :mod:`pickle`
    '''
    cldic = mole.pack(cell)
    cldic['a'] = cell.a
    cldic['gs'] = cell.gs
    cldic['precision'] = cell.precision
    cldic['pseudo'] = cell.pseudo
    cldic['ke_cutoff'] = cell.ke_cutoff
    cldic['rcut'] = cell.rcut
    cldic['ew_eta'] = cell.ew_eta
    cldic['ew_cut'] = cell.ew_cut
    cldic['dimension'] = cell.dimension
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
    celldic['pseudo'] = repr(cell.pseudo)
    celldic['ecp'] = repr(cell.ecp)

    try:
        return json.dumps(celldic)
    except TypeError:
        def skip_value(dic):
            dic1 = {}
            for k,v in dic.items():
                if (v is None or
                    isinstance(v, (str, unicode, bool, int, long, float))):
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
    cell.basis = eval(cell.basis)
    cell.pseudo = eval(cell.pseudo)
    cell.pseudo = eval(cell.ecp)
    cell._atm = np.array(cell._atm, dtype=np.int32)
    cell._bas = np.array(cell._bas, dtype=np.int32)
    cell._env = np.array(cell._env, dtype=np.double)
    cell._ecpbas = np.array(cell._ecpbas, dtype=np.int32)

    return cell

def intor_cross(intor, cell1, cell2, comp=1, hermi=0, kpts=None, kpt=None):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    intor = cell1._add_suffix(intor)
    intor = moleintor.ascint3(intor)
    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    atm, bas, env = conc_env(cell1._atm, cell1._bas, cell1._env,
                             cell2._atm, cell2._bas, cell2._env)
    atm = np.asarray(atm, dtype=np.int32)
    bas = np.asarray(bas, dtype=np.int32)
    env = np.asarray(env, dtype=np.double)
    natm = len(atm)
    nbas = len(bas)
    shls_slice = (0, cell1.nbas, cell1.nbas, nbas)
    ao_loc = moleintor.make_loc(bas, intor)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    out = np.empty((nkpts,comp,ni,nj), dtype=np.complex128)

    if hermi == 0:
        aosym = 's1'
    else:
        aosym = 's2'
    fill = getattr(libpbc, 'PBCnr2c_fill_k'+aosym)
    fintor = getattr(moleintor.libcgto, intor)
    intopt = lib.c_null_ptr()

    Ls = cell1.get_lattice_Ls(rcut=max(cell1.rcut, cell2.rcut))
    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCnr2c_drv
    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(*(shls_slice[:4])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), intopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = out[k]
        if hermi != 0:
            for ic in range(comp):
                lib.hermi_triu(v[ic], hermi=hermi, inplace=True)
        if comp == 1:
            v = v[0]
        if abs(kpt).sum() < 1e-9:  # gamma_point
            v = v.real
        mat.append(v)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        mat = mat[0]
    return mat


def get_nimgs(cell, precision=None):
    r'''Choose number of basis function images in lattice sums
    to include for given precision in overlap, using

    precision ~ \int r^l e^{-\alpha r^2} (r-rcut)^l e^{-\alpha (r-rcut)^2}
    ~ (rcut^2/(2\alpha))^l e^{\alpha/2 rcut^2}

    where \alpha is the smallest exponent in the basis. Note
    that assumes an isolated exponent in the middle of the box, so
    it adds one additional lattice vector to be safe.
    '''
    if precision is None:
        precision = cell.precision

    rcut = max([cell.bas_rcut(ib, precision) for ib in range(cell.nbas)])

    # nimgs determines the supercell size
    nimgs = cell.get_bounding_sphere(rcut)
    return nimgs

def _estimate_rcut(alpha, l, c, precision=1e-8):
    C = (c**2+1e-200)*(2*l+1)*alpha / precision
    r0 = 20
    r0 = np.sqrt(max(0, 2*np.log(C*(r0**2*alpha)**(l+1)).max()) / alpha)
    rcut = np.sqrt(max(0, 2*np.log(C*(r0**2*alpha)**(l+1)).max()) / alpha)
    return rcut

def bas_rcut(cell, bas_id, precision=1e-8):
    r'''Estimate the largest distance between the function and its image to
    reach the precision in overlap

    precision ~ \int g(r-0) g(r-R)
    '''
    l = cell.bas_angular(bas_id)
    es = cell.bas_exp(bas_id)
    cs = abs(cell.bas_ctr_coeff(bas_id)).max(axis=1)
    rcut = _estimate_rcut(es, l, cs, precision)
    return rcut.max()

def _estimate_ke_cutoff(alpha, l, c, precision=1e-8, weight=1.):
    '''Energy cutoff estimation'''
    log_k0 = 2.5 + np.log(alpha) / 2
    l2fac2 = scipy.misc.factorial2(l*2+1)
    log_rest = np.log(precision*l2fac2**2*(4*alpha)**(l*2+1) / (32*np.pi**2*c**4*weight))
    Ecut = 2*alpha * (log_k0*(4*l+3) - log_rest)
    Ecut[Ecut<0] = 1e-10
    log_k0 = .5 * np.log(Ecut*2)
    Ecut = 2*alpha * (log_k0*(4*l+3) - log_rest)
    return Ecut.max()

def estimate_ke_cutoff(cell, precision=1e-8):
    '''Energy cutoff estimation'''
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = np.linalg.norm(b[0]) / (2*np.pi)
    elif cell.dimension == 2:
        w = np.linalg.norm(np.cross(b[0], b[1])) / (2*np.pi)**2
    else:
        w = abs(np.linalg.det(b)) / (2*np.pi)**3

    Ecut_max = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i)
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        Ecut_max = max(Ecut_max, _estimate_ke_cutoff(es, l, cs, precision, w))
    return Ecut_max

def error_for_ke_cutoff(cell, ke_cutoff):
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = np.linalg.norm(b[0]) / (2*np.pi)
    elif cell.dimension == 2:
        w = np.linalg.norm(np.cross(b[0], b[1])) / (2*np.pi)**2
    else:
        w = abs(np.linalg.det(b)) / (2*np.pi)**3

    kmax = np.sqrt(ke_cutoff*2)
    errmax = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i)
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        fac = 64*np.pi**2*cs**4*w / scipy.misc.factorial2(l*2+1)**2
        efac = np.exp(-ke_cutoff/(2*es))
        if 0:
            ka = scipy.misc.factorial2(l*4+3) * (2*es)**(2*l+2)
            err0 = np.sqrt(np.pi*es)*scipy.special.erfc(kmax/np.sqrt(4*es)) * ka
            ka *= efac * kmax
            for m in range(l*2+2):
                err0 += ka
                ka *= ke_cutoff / (es * (2*m+3))
            errmax = max(errmax, (fac/(4*es)**(2*l+2)*err0).max())
        else:
            err1 = .5*fac/(4*es)**(2*l+1) * kmax**(4*l+3) * efac
            errmax = max(errmax, err1.max())
            if np.any(ke_cutoff < 5*es):
                err2 = (1.41*efac+2.51)*fac/(4*es)**(2*l+2) * kmax**(4*l+5)
                errmax = max(errmax, err2[ke_cutoff<5*es].max())
            if np.any(ke_cutoff < es):
                err2 = (1.41*efac+2.51)*fac/2**(2*l+2) * np.sqrt(2*es)
                errmax = max(errmax, err2[ke_cutoff<es].max())
    return errmax

def get_bounding_sphere(cell, rcut):
    '''Finds all the lattice points within a sphere of radius rcut.  

    Defines a parallelipiped given by -N_x <= n_x <= N_x, with x in [1,3]
    See Martin p. 85

    Args:
        rcut : number
            real space cut-off for interaction

    Returns:
        cut : ndarray of 3 ints defining N_x
    '''
    #Gmat = cell.reciprocal_vectors(norm_to=1)
    #n1 = np.ceil(lib.norm(Gmat[0,:])*rcut)
    #n2 = np.ceil(lib.norm(Gmat[1,:])*rcut)
    #n3 = np.ceil(lib.norm(Gmat[2,:])*rcut)
    #cut = np.array([n1, n2, n3]).astype(int)
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = lib.norm(b, axis=1)
    nimgs = np.ceil(rcut*heights_inv).astype(int)

    for i in range(cell.dimension, 3):
        nimgs[i] = 1
    return nimgs

def get_Gv(cell, gs=None):
    '''Calculate three-dimensional G-vectors for the cell; see MH (3.8).

    Indices along each direction go as [0...cell.gs, -cell.gs...-1]
    to follow FFT convention. Note that, for each direction, ngs = 2*cell.gs+1.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        Gv : (ngs, 3) ndarray of floats
            The array of G-vectors.
    '''
    if gs is None:
        gs = cell.gs
    gxrange = np.append(range(gs[0]+1), range(-gs[0],0))
    gyrange = np.append(range(gs[1]+1), range(-gs[1],0))
    gzrange = np.append(range(gs[2]+1), range(-gs[2],0))
    gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))

    b = cell.reciprocal_vectors()
    Gv = np.dot(gxyz, b)
    return Gv

def get_Gv_weights(cell, gs=None):
    '''Calculate G-vectors and weights.

    Returns:
        Gv : (ngs, 3) ndarray of floats
            The array of G-vectors.
    '''
    if gs is None:
        gs = cell.gs
    def plus_minus(n):
        #rs, ws = dft.delley(n)
        #rs, ws = dft.treutler_ahlrichs(n)
        #rs, ws = dft.mura_knowles(n)
        rs, ws = dft.gauss_chebyshev(n)
        #return np.hstack((0,rs,-rs[::-1])), np.hstack((0,ws,ws[::-1]))
        return np.hstack((rs,-rs[::-1])), np.hstack((ws,ws[::-1]))

    # Default, the 3D uniform grids
    b = cell.reciprocal_vectors()
    rx = np.append(np.arange(gs[0]+1.), np.arange(-gs[0],0.))
    ry = np.append(np.arange(gs[1]+1.), np.arange(-gs[1],0.))
    rz = np.append(np.arange(gs[2]+1.), np.arange(-gs[2],0.))

    ngs = [i*2+1 for i in gs]
    if cell.dimension == 0:
        rx, wx = plus_minus(gs[0])
        ry, wy = plus_minus(gs[1])
        rz, wz = plus_minus(gs[2])
        rx /= np.linalg.norm(b[0])
        ry /= np.linalg.norm(b[1])
        rz /= np.linalg.norm(b[2])
        weights = np.einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)
    elif cell.dimension == 1:
        wx = np.repeat(np.linalg.norm(b[0]), ngs[0])
        ry, wy = plus_minus(gs[1])
        rz, wz = plus_minus(gs[2])
        ry /= np.linalg.norm(b[1])
        rz /= np.linalg.norm(b[2])
        weights = np.einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)
    elif cell.dimension == 2:
        area = np.linalg.norm(np.cross(b[0], b[1]))
        wxy = np.repeat(area, ngs[0]*ngs[1])
        rz, wz = plus_minus(gs[2])
        rz /= np.linalg.norm(b[2])
        weights = np.einsum('i,k->ik', wxy, wz).reshape(-1)
    else:
        weights = abs(np.linalg.det(b))
    Gvbase = (rx, ry, rz)
    Gv = np.dot(lib.cartesian_prod(Gvbase), b)
    # 1/cell.vol == det(b)/(2pi)^3
    weights *= 1/(2*np.pi)**3
    return Gv, Gvbase, weights

def get_SI(cell, Gv=None):
    '''Calculate the structure factor for all atoms; see MH (3.34).

    Args:
        cell : instance of :class:`Cell`

        Gv : (N,3) array
            G vectors

    Returns:
        SI : (natm, ngs) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    '''
    if Gv is None:
        Gv = cell.get_Gv()
    coords = cell.atom_coords()
    SI = np.exp(-1j*np.dot(coords, Gv.T))
    return SI

def get_ewald_params(cell, precision=1e-8, gs=None):
    r'''Choose a reasonable value of Ewald 'eta' and 'cut' parameters.

    Choice is based on largest G vector and desired relative precision.

    The relative error in the G-space sum is given by

        precision ~ 4\pi Gmax^2 e^{(-Gmax^2)/(4 \eta^2)}

    which determines eta. Then, real-space cutoff is determined by (exp.
    factors only)

        precision ~ erfc(eta*rcut) / rcut ~ e^{(-eta**2 rcut*2)}

    Returns:
        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.
    '''
    if cell.natm == 0:
        return 0, 0
    elif cell.dimension == 3:
        if gs is None:
            gs = cell.gs
        gs = _cut_gs_for_ewald(cell, gs)
        Gmax = min(np.asarray(gs) * lib.norm(cell.reciprocal_vectors(), axis=1))
        log_precision = np.log(precision/(4*np.pi*Gmax**2))
        ew_eta = np.sqrt(-Gmax**2/(4*log_precision))
        ew_cut = _estimate_rcut(ew_eta**2, 0, 1., precision)
    else:
# Non-uniform PW grids are used for low-dimensional ewald summation.  The cutoff
# estimation for long range part based on exp(G^2/(4*eta^2)) does not work for
# non-uniform grids.  Smooth model density is preferred.
        ew_cut = cell.rcut
        ew_eta = np.sqrt(max(np.log(4*np.pi*ew_cut**2/precision)/ew_cut**2, .1))
    return ew_eta, ew_cut

# roughly 4 grids per axis
def _cut_gs_for_ewald(cell, gs):
    gs = np.copy(gs)
    gs_max = np.asarray(np.linalg.norm(cell.lattice_vectors(), axis=1) * 2,
                        dtype=int)
    gs_max[cell.dimension:] = gs[cell.dimension:]
    gs_max[gs_max<40] = 40
    gs[gs>gs_max] = gs_max[gs>gs_max]
    return gs

def ewald(cell, ew_eta=None, ew_cut=None):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
    if cell.natm == 0:
        return 0
    if ew_eta is None: ew_eta = cell.ew_eta
    if ew_cut is None: ew_cut = cell.ew_cut
    chargs = cell.atom_charges()
    coords = cell.atom_coords()

    Lall = cell.get_lattice_Ls(rcut=ew_cut)
    ewovrl = 0.
    for i, qi in enumerate(chargs):
        ri = coords[i]
        for j in range(i):
            qj = chargs[j]
            rj = coords[j]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            ewovrl += (qi * qj / r * scipy.special.erfc(ew_eta * r)).sum()
    # exclude the point where Lall == 0
    r = lib.norm(Lall, axis=1)
    r[r<1e-16] = 1e200
    ewovrl += .5 * (chargs**2).sum() * (1./r * scipy.special.erfc(ew_eta * r)).sum()

    # last line of Eq. (F.5) in Martin
    ewself  = -.5 * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
    if cell.dimension == 3:
        ewself += -.5 * np.sum(chargs)**2 * np.pi/(ew_eta**2 * cell.vol)

    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf

    gs = _cut_gs_for_ewald(cell, cell.gs)
    Gv, Gvbase, weights = cell.get_Gv_weights(gs)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200
    coulG = 4*np.pi / absG2
    coulG *= weights
    JexpG2 = np.exp(-absG2/(4*ew_eta**2)) * coulG

    ZSI = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
    ZSIG2 = np.abs(ZSI)**2
    ewg = .5 * np.dot(ZSIG2, JexpG2)

    logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

energy_nuc = ewald

def make_kpts(cell, nks, wrap_around=False, with_gamma_point=True, scaled_center=None):
    '''Given number of kpoints along x,y,z , generate kpoints

    Args:
        nks : (3,) ndarray

    Kwargs:
        wrap_around : bool
            To ensure all kpts are in first Brillouin zone.
        with_gamma_point : bool
            Whether to shift Monkhorst-pack grid to include gamma-point.
        scaled_center : (3,) array
            Shift all points in the Monkhorst-pack grid to be centered on
            scaled_center, given as the zeroth index of the returned kpts.
            Scaled meaning that the k-points are scaled to a grid from 
            [-1,1] x [-1,1] x [-1,1]

    Returns:
        kpts in absolute value (unit 1/Bohr).  Gamma point is placed at the
        first place in the k-points list

    Examples:

    >>> cell.make_kpts((4,4,4))
    '''
    ks_each_axis = []
    for n in nks:
        if with_gamma_point or scaled_center is not None:
            ks = np.arange(n, dtype=float) / n
        else:
            ks = (np.arange(n)+.5)/n-.5
        if wrap_around:
            ks[ks>=.5] -= 1
        ks_each_axis.append(ks)
    if scaled_center is None:
        scaled_center = [0.0,0.0,0.0]
    scaled_kpts = lib.cartesian_prod(ks_each_axis)
    scaled_kpts += np.array(scaled_center)
    kpts = cell.get_abs_kpts(scaled_kpts)
    return kpts

def gen_uniform_grids(cell, gs=None):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    if gs is None: gs = cell.gs
    ngs = 2*np.asarray(gs)+1
    qv = lib.cartesian_prod([np.arange(x) for x in ngs])
    a_frac = np.einsum('i,ij->ij', 1./ngs, cell.lattice_vectors())
    coords = np.dot(qv, a_frac)
    return coords

# Check whether ecp keywords are presented in pp and whether pp keywords are
# presented in ecp.  The return (ecp, pp) should have only the ecp keywords and
# pp keywords in each dict.
# The "misplaced" ecp/pp keywords have lowest priority, ie if the atom is
# defined in ecp, the misplaced ecp atom found in pp does NOT replace the
# definition in ecp, and versa vise.
def classify_ecp_pseudo(cell, ecp, pp):
    def classify(ecp, pp_alias):
        if isinstance(ecp, (str, unicode)):
            if pseudo._format_pseudo_name(ecp)[0] in pp_alias:
                return {}, str(ecp)
        elif isinstance(ecp, dict):
            ecp_as_pp = {}
            for atom in ecp:
                key = ecp[atom]
                if (isinstance(key, (str, unicode)) and
                    pseudo._format_pseudo_name(key)[0] in pp_alias):
                    ecp_as_pp[atom] = str(key)
            if ecp_as_pp:
                ecp_left = dict(ecp)
                for atom in ecp_as_pp:
                    ecp_left.pop(atom)
                return ecp_left, ecp_as_pp
        return ecp, {}
    ecp_left, ecp_as_pp = classify(ecp, pseudo.ALIAS)
    pp_left , pp_as_ecp = classify(pp, MOLE_ALIAS)

    # ecp = ecp_left + pp_as_ecp
    # pp = pp_left + ecp_as_pp
    ecp = ecp_left
    if pp_as_ecp and not isinstance(ecp_left, (str, unicode)):
        # If ecp is a str, all atoms have ecp definition.  The misplaced ecp has no effects.
        logger.info(cell, 'PBC pseudo-potentials keywords for %s found in .ecp',
                    pp_as_ecp.keys())
        if ecp_left:
            pp_as_ecp.update(ecp_left)
        ecp = pp_as_ecp
    pp = pp_left
    if ecp_as_pp and not isinstance(pp_left, (str, unicode)):
        logger.info(cell, 'ECP keywords for %s found in PBC .pseudo',
                    ecp_as_pp.keys())
        if pp_left:
            ecp_as_pp.update(pp_left)
        pp = ecp_as_pp
    return ecp, pp


class Cell(mole.Mole):
    '''A Cell object holds the basic information of a crystal.

    Attributes:
        a : (3,3) ndarray
            Lattice primitive vectors. Each row represents a lattice vector
            Reciprocal lattice vectors are given by  b1,b2,b3 = 2 pi inv(a).T
        gs : (3,) list of ints
            The number of *positive* G-vectors along each direction.
            The default value is estimated based on :attr:`precision`
        pseudo : dict or str
            To define pseudopotential.
        precision : float
            To control Ewald sums and lattice sums accuracy
        rcut : float
            Cutoff radius (unit Bohr) in lattice summation. The default value
            is estimated based on the required :attr:`precision`.
        ke_cutoff : float
            If set, defines a spherical cutoff of planewaves, with .5 * G**2 < ke_cutoff
            The default value is estimated based on :attr:`precision`
        dimension : int
            Default is 3

        ** Following attributes (for experts) are automatically generated. **

        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.  See :func:`get_ewald_params`

    (See other attributes in :class:`Mole`)

    Examples:

    >>> mol = Mole(atom='H^2 0 0 0; H 0 0 1.1', basis='sto3g')
    >>> cl = Cell()
    >>> cl.build(a='3 0 0; 0 3 0; 0 0 3', gs=[8,8,8], atom='C 1 1 1', basis='sto3g')
    >>> print(cl.atom_symbol(0))
    C
    '''
    def __init__(self, **kwargs):
        mole.Mole.__init__(self, **kwargs)
        self.a = None # lattice vectors, (a1,a2,a3)
        self.gs = None
        self.ke_cutoff = None # if set, defines a spherical cutoff
                              # of fourier components, with .5 * G**2 < ke_cutoff
        self.precision = 1.e-8
        self.pseudo = None
        self.dimension = 3

##################################################
# These attributes are initialized by build function if not given
        self.ew_eta = None
        self.ew_cut = None
        self.rcut = None

##################################################
# don't modify the following variables, they are not input arguments
        self._pseudo = {}
        self._keys = set(self.__dict__.keys())

    @property
    def nelec(self):
        ne = self.nelectron
        nalpha = (ne + self.spin) // 2
        nbeta = nalpha - self.spin
        if nalpha + nbeta != ne:
            warnings.warn('Electron number %d and spin %d are not consistent '
                          'in unit cell\n' % (ne, self.spin))
        return nalpha, nbeta

#Note: Exculde dump_input, parse_arg, basis from kwargs to avoid parsing twice
    def build(self, dump_input=True, parse_arg=True,
              a=None, gs=None, ke_cutoff=None, precision=None, nimgs=None,
              ew_eta=None, ew_cut=None, pseudo=None, basis=None, h=None,
              dimension=None, rcut= None, ecp=None,
              *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.
        Whenever you change the value of the attributes of :class:`Cell`,
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            a : (3,3) ndarray
                The real-space unit cell lattice vectors. Each row represents
                a lattice vector.
            gs : (3,) ndarray of ints
                The number of *positive* G-vectors along each direction.
            pseudo : dict or str
                To define pseudopotential.  If given, overwrite :attr:`Cell.pseudo`
        '''
        if h is not None: self.h = h
        if a is not None: self.a = a
        if gs is not None: self.gs = gs
        if nimgs is not None: self.nimgs = nimgs
        if ew_eta is not None: self.ew_eta = ew_eta
        if ew_cut is not None: self.ew_cut = ew_cut
        if pseudo is not None: self.pseudo = pseudo
        if basis is not None: self.basis = basis
        if dimension is not None: self.dimension = dimension
        if precision is not None: self.precision = precision
        if rcut is not None: self.rcut = rcut
        if ecp is not None: self.ecp = ecp
        if ke_cutoff is not None: self.ke_cutoff = ke_cutoff

        assert(self.a is not None)

        if 'unit' in kwargs:
            self.unit = kwargs['unit']

        if 'atom' in kwargs:
            self.atom = kwargs['atom']

        # Set-up pseudopotential if it exists
        # This must happen before build() because it affects
        # tot_electrons() via atom_charge()

        self.ecp, self.pseudo = classify_ecp_pseudo(self, self.ecp, self.pseudo)
        if self.pseudo is not None:
            if isinstance(self.pseudo, (str, unicode)):
                # specify global pseudo for whole molecule
                _atom = self.format_atom(self.atom, unit=self.unit)
                uniq_atoms = set([a[0] for a in _atom])
                self._pseudo = self.format_pseudo(dict([(a, str(self.pseudo))
                                                      for a in uniq_atoms]))
            else:
                self._pseudo = self.format_pseudo(self.pseudo)

        # Do regular Mole.build with usual kwargs
        _built = self._built
        mole.Mole.build(self, False, parse_arg, *args, **kwargs)

        if self.rcut is None:
            self.rcut = max([self.bas_rcut(ib, self.precision)
                             for ib in range(self.nbas)] + [0])

        _a = self.lattice_vectors()
        if np.linalg.det(_a) < 0 and self.dimension == 3:
            sys.stderr.write('''WARNING!
  Lattice are not in right-handed coordinate system. This can cause wrong value for some integrals.
  It's recommended to resort the lattice vectors to\na = %s\n\n''' % _a[[0,2,1]])

        if self.gs is None:
            if self.ke_cutoff is None:
                ke_cutoff = estimate_ke_cutoff(self, self.precision)
            else:
                ke_cutoff = self.ke_cutoff
            self.gs = pbctools.cutoff_to_gs(_a, ke_cutoff)
            if self.dimension < 3:
                #prec ~ exp(-0.87278467*gs -2.99944305)*nelec
                gz = np.log(self.nelectron/self.precision)/0.8727847-3.4366358
                self.gs[self.dimension:] = int(gz)

        if self.ew_eta is None or self.ew_cut is None:
            self.ew_eta, self.ew_cut = self.get_ewald_params(self.precision, self.gs)

        if dump_input and not _built and self.verbose > logger.NOTE:
            self.dump_input()
            logger.info(self, 'lattice vectors  a1 [%.9f, %.9f, %.9f]', *_a[0])
            logger.info(self, '                 a2 [%.9f, %.9f, %.9f]', *_a[1])
            logger.info(self, '                 a3 [%.9f, %.9f, %.9f]', *_a[2])
            logger.info(self, 'dimension = %s', self.dimension)
            logger.info(self, 'Cell volume = %g', self.vol)
            logger.info(self, 'rcut = %s (nimgs = %s)', self.rcut, self.nimgs)
            logger.info(self, 'lattice sum = %d cells', len(self.get_lattice_Ls()))
            logger.info(self, 'precision = %g', self.precision)
            logger.info(self, 'pseudo = %s', self.pseudo)
            if ke_cutoff is not None:
                logger.info(self, 'ke_cutoff = %s', ke_cutoff)
                logger.info(self, '    = gs (FFT-mesh) %s', self.gs)
            else:
                logger.info(self, 'gs (FFT-mesh) = %s', self.gs)
                Ecut = pbctools.gs_to_cutoff(self.lattice_vectors(), self.gs)
                logger.info(self, '    = ke_cutoff %s', Ecut[:self.dimension])
            logger.info(self, 'ew_eta = %g', self.ew_eta)
            logger.info(self, 'ew_cut = %s (nimgs = %s)', self.ew_cut,
                        self.get_bounding_sphere(self.ew_cut))
        return self
    kernel = build

    @property
    def h(self):
        return np.asarray(self.a).T
    @h.setter
    def h(self, x):
        sys.stderr.write('cell.h is deprecated.  It is replaced by the '
                         '(row-based) lattice vectors cell.a:  cell.a = cell.h.T\n')
        if isinstance(x, (str, unicode)):
            x = x.replace(';',' ').replace(',',' ').replace('\n',' ')
            self.a = np.asarray([float(z) for z in x.split()]).reshape(3,3).T
        else:
            self.a = np.asarray(x).T

    @property
    def _h(self):
        return self.lattice_vectors().T

    @property
    def vol(self):
        return abs(np.linalg.det(self.lattice_vectors()))

    @property
    def Gv(self):
        return self.get_Gv(self.gs)

    @lib.with_doc(format_pseudo.__doc__)
    def format_pseudo(self, pseudo_tab):
        return format_pseudo(pseudo_tab)

    @lib.with_doc(format_basis.__doc__)
    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    @property
    def nimgs(self):
        return self.get_bounding_sphere(self.rcut)
    @nimgs.setter
    def nimgs(self, x):
        b = self.reciprocal_vectors(norm_to=1)
        heights_inv = lib.norm(b, axis=1)
        self.rcut = max(np.asarray(x) / heights_inv)

        if self.nbas == 0:
            rcut_guess = _estimate_rcut(.05, 0, 1, 1e-8)
        else:
            rcut_guess = max([self.bas_rcut(ib, self.precision)
                              for ib in range(self.nbas)])
        if self.rcut > rcut_guess*1.5:
            msg = ('.nimgs is a deprecated attribute.  It is replaced by .rcut '
                   'attribute for lattic sum cutoff radius.  The given nimgs '
                   '%s is far over the estimated cutoff radius %s. ' %
                   (x, rcut_guess))
            warnings.warn(msg)

    def make_ecp_env(self, _atm, _ecp, pre_env=[]):
        if _ecp and self._pseudo:
            conflicts = set(self._pseudo.keys()).intersection(set(_ecp.keys()))
            if conflicts:
                raise RuntimeError('Pseudo potential for atoms %s are defined '
                                   'in both .ecp and .pseudo.' % list(conflicts))

        _ecpbas, _env = np.zeros((0,8)), pre_env
        if _ecp:
            _atm, _ecpbas, _env = mole.make_ecp_env(self, _atm, _ecp, _env)
        if self._pseudo:
            _atm, _, _env = make_pseudo_env(self, _atm, self._pseudo, _env)
        return _atm, _ecpbas, _env

    def lattice_vectors(self):
        '''Convert the primitive lattice vectors.

        Return 3x3 array in which each row represents one direction of the
        lattice vectors (unit in Bohr)
        '''
        if isinstance(self.a, (str, unicode)):
            a = self.a.replace(';',' ').replace(',',' ').replace('\n',' ')
            a = np.asarray([float(x) for x in a.split()]).reshape(3,3)
        else:
            a = np.asarray(self.a, dtype=np.double)
        if isinstance(self.unit, (str, unicode)):
            if self.unit.startswith(('B','b','au','AU')):
                return a
            else:
                return a/param.BOHR
        else:
            return a/self.unit

    def reciprocal_vectors(self, norm_to=2*np.pi):
        r'''
        .. math::

            \begin{align}
            \mathbf{b_1} &= 2\pi \frac{\mathbf{a_2} \times \mathbf{a_3}}{\mathbf{a_1} \cdot (\mathbf{a_2} \times \mathbf{a_3})} \\
            \mathbf{b_2} &= 2\pi \frac{\mathbf{a_3} \times \mathbf{a_1}}{\mathbf{a_2} \cdot (\mathbf{a_3} \times \mathbf{a_1})} \\
            \mathbf{b_3} &= 2\pi \frac{\mathbf{a_1} \times \mathbf{a_2}}{\mathbf{a_3} \cdot (\mathbf{a_1} \times \mathbf{a_2})}
            \end{align}

        '''
        a = self.lattice_vectors()
        if self.dimension == 1:
            assert(abs(np.dot(a[0], a[1])) < 1e-9 and
                   abs(np.dot(a[0], a[2])) < 1e-9 and
                   abs(np.dot(a[1], a[2])) < 1e-9)
        elif self.dimension == 2:
            assert(abs(np.dot(a[0], a[2])) < 1e-9 and
                   abs(np.dot(a[1], a[2])) < 1e-9)
        b = np.linalg.inv(a.T)
        return norm_to * b

    def get_abs_kpts(self, scaled_kpts):
        '''Get absolute k-points (in 1/Bohr), given "scaled" k-points in
        fractions of lattice vectors.

        Args:
            scaled_kpts : (nkpts, 3) ndarray of floats

        Returns:
            abs_kpts : (nkpts, 3) ndarray of floats 
        '''
        return np.dot(scaled_kpts, self.reciprocal_vectors())

    def get_scaled_kpts(self, abs_kpts):
        '''Get scaled k-points, given absolute k-points in 1/Bohr.

        Args:
            abs_kpts : (nkpts, 3) ndarray of floats 

        Returns:
            scaled_kpts : (nkpts, 3) ndarray of floats
        '''
        return 1./(2*np.pi)*np.dot(abs_kpts, self.lattice_vectors().T)

    make_kpts = make_kpts

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

    bas_rcut = bas_rcut

    get_lattice_Ls = pbctools.get_lattice_Ls

    get_nimgs = get_nimgs

    get_ewald_params = get_ewald_params

    get_bounding_sphere = get_bounding_sphere

    get_Gv = get_Gv
    get_Gv_weights = get_Gv_weights

    get_SI = get_SI

    ewald = ewald
    energy_nuc = ewald

    gen_uniform_grids = gen_uniform_grids

    def pbc_intor(self, intor, comp=1, hermi=0, kpts=None, kpt=None):
        '''One-electron integrals with PBC. See also Mole.intor'''
        return intor_cross(intor, self, self, comp, hermi, kpts, kpt)

    def from_ase(self, ase_atom):
        '''Update cell based on given ase atom object

        Examples:

        >>> from ase.lattice import bulk
        >>> cell.from_ase(bulk('C', 'diamond', a=LATTICE_CONST))
        '''
        from pyscf.pbc.tools import pyscf_ase
        self.a = ase_atom.cell
        self.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        return self

    def to_mol(self):
        '''Return a Mole object using the same atoms and basis functions as
        the Cell object.
        '''
        mol = mole.Mole()
        cell_dic = [(key, getattr(self, key)) for key in mol.__dict__.keys()]
        mol.__dict__.update(cell_dic)
        return mol

    def has_ecp(self):
        '''Whether pesudo potential is used in the system.'''
        return self.pseudo or self._pseudo or (len(self._ecpbas) > 0)

