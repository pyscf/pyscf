#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
try:
  from scipy.special import factorial2
except:
  from scipy.misc import factorial2
from scipy.special import erf, erfc, erfcx
import scipy.optimize
import pyscf.lib.parameters as param
from pyscf import lib
from pyscf.dft import radi
from pyscf.lib import logger
from pyscf.gto import mole
from pyscf.gto import moleintor
from pyscf.gto.mole import _symbol, _rm_digit, _atom_symbol, _std_symbol, charge
from pyscf.gto.mole import conc_env, uncontract
from pyscf.pbc.gto import basis
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.gto.eval_gto import eval_gto as pbc_eval_gto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.gto.basis import ALIAS as MOLE_ALIAS
from pyscf import __config__

INTEGRAL_PRECISION = getattr(__config__, 'pbc_gto_cell_Cell_precision', 1e-8)
WRAP_AROUND = getattr(__config__, 'pbc_gto_cell_make_kpts_wrap_around', False)
WITH_GAMMA = getattr(__config__, 'pbc_gto_cell_make_kpts_with_gamma', True)
EXP_DELIMITER = getattr(__config__, 'pbc_gto_cell_split_basis_exp_delimiter',
                        [1.0, 0.5, 0.25, 0.1, 0])


# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str
    long = int

libpbc = _pbcintor.libpbc

def M(**kwargs):
    r'''This is a shortcut to build up Cell object.

    Examples:

    >>> from pyscf.pbc import gto
    >>> cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g')
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
        if symb in _pseudo and _atm[ia,0] != 0:  # pass ghost atoms.
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
    cldic['precision'] = cell.precision
    cldic['pseudo'] = cell.pseudo
    cldic['ke_cutoff'] = cell.ke_cutoff
    cldic['exp_to_discard'] = cell.exp_to_discard
    cldic['_mesh'] = cell._mesh
    cldic['_rcut'] = cell._rcut
    cldic['_ew_eta'] = cell._ew_eta
    cldic['_ew_cut'] = cell._ew_cut
    cldic['dimension'] = cell.dimension
    cldic['low_dim_ft_type'] = cell.low_dim_ft_type
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
    cell.ecp = eval(cell.ecp)
    cell._atm = np.array(cell._atm, dtype=np.int32)
    cell._bas = np.array(cell._bas, dtype=np.int32)
    cell._env = np.array(cell._env, dtype=np.double)
    cell._ecpbas = np.array(cell._ecpbas, dtype=np.int32)

    return cell

def conc_cell(cell1, cell2):
    '''Concatenate two Cell objects.
    '''
    cell3 = Cell()
    cell3._atm, cell3._bas, cell3._env = \
            conc_env(cell1._atm, cell1._bas, cell1._env,
                     cell2._atm, cell2._bas, cell2._env)
    off = len(cell1._env)
    natm_off = len(cell1._atm)
    if len(cell2._ecpbas) == 0:
        cell3._ecpbas = cell1._ecpbas
    else:
        ecpbas2 = np.copy(cell2._ecpbas)
        ecpbas2[:,mole.ATOM_OF  ] += natm_off
        ecpbas2[:,mole.PTR_EXP  ] += off
        ecpbas2[:,mole.PTR_COEFF] += off
        if len(cell1._ecpbas) == 0:
            cell3._ecpbas = ecpbas2
        else:
            cell3._ecpbas = np.hstack((cell1._ecpbas, ecpbas2))

    cell3.verbose = cell1.verbose
    cell3.output = cell1.output
    cell3.max_memory = cell1.max_memory
    cell3.charge = cell1.charge + cell2.charge
    cell3.spin = cell1.spin + cell2.spin
    cell3.cart = cell1.cart and cell2.cart
    cell3._atom = cell1._atom + cell2._atom
    cell3.unit = cell1.unit
    cell3._basis = dict(cell2._basis)
    cell3._basis.update(cell1._basis)
    # Whether to update the lattice_vectors?
    cell3.a = cell1.a
    cell3.mesh = np.max((cell1.mesh, cell2.mesh), axis=0)

    ke_cutoff1 = cell1.ke_cutoff
    ke_cutoff2 = cell2.ke_cutoff
    if ke_cutoff1 is None and ke_cutoff2 is None:
        cell3.ke_cutoff = None
    else:
        if ke_cutoff1 is None:
            ke_cutoff1 = estimate_ke_cutoff(cell1, cell1.precision)
        if ke_cutoff2 is None:
            ke_cutoff2 = estimate_ke_cutoff(cell2, cell2.precision)
        cell3.ke_cutoff = max(ke_cutoff1, ke_cutoff2)

    cell3.precision = min(cell1.precision, cell2.precision)
    cell3.dimension = max(cell1.dimension, cell2.dimension)
    cell3.low_dim_ft_type = cell1.low_dim_ft_type or cell2.low_dim_ft_type
    cell3.ew_eta = min(cell1.ew_eta, cell2.ew_eta)
    cell3.ew_cut = max(cell1.ew_cut, cell2.ew_cut)
    cell3.rcut = max(cell1.rcut, cell2.rcut)

    cell3._pseudo.update(cell1._pseudo)
    cell3._pseudo.update(cell2._pseudo)
    cell3._ecp.update(cell1._ecp)
    cell3._ecp.update(cell2._ecp)

    cell3.nucprop.update(cell1.nucprop)
    cell3.nucprop.update(cell2.nucprop)

    if not cell1._built:
        logger.warn(cell1, 'Warning: intor envs of %s not initialized.', cell1)
    if not cell2._built:
        logger.warn(cell2, 'Warning: intor envs of %s not initialized.', cell2)
    cell3._built = cell1._built or cell2._built
    return cell3

def intor_cross(intor, cell1, cell2, comp=None, hermi=0, kpts=None, kpt=None,
                shls_slice=None, **kwargs):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    import copy
    intor, comp = moleintor._get_intor_and_comp(cell1._add_suffix(intor), comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    pcell = copy.copy(cell1)
    pcell.precision = min(cell1.precision, cell2.precision)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = conc_env(cell1._atm, cell1._bas, cell1._env,
                             cell2._atm, cell2._bas, cell2._env)
    if shls_slice is None:
        shls_slice = (0, cell1.nbas, 0, cell2.nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    j0 += cell1.nbas
    j1 += cell1.nbas
    ao_loc = moleintor.make_loc(bas, intor)
    ni = ao_loc[i1] - ao_loc[i0]
    nj = ao_loc[j1] - ao_loc[j0]
    out = np.empty((nkpts,comp,ni,nj), dtype=np.complex128)

    if hermi == 0:
        aosym = 's1'
    else:
        aosym = 's2'
    fill = getattr(libpbc, 'PBCnr2c_fill_k'+aosym)
    fintor = getattr(moleintor.libcgto, intor)
    cintopt = lib.c_null_ptr()
    pbcopt = kwargs.get('pbcopt', None)
    if pbcopt is None:
        pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell)
    if isinstance(pbcopt, _pbcintor.PBCOpt):
        cpbcopt = pbcopt._this
    else:
        cpbcopt = lib.c_null_ptr()

    Ls = cell1.get_lattice_Ls(rcut=max(cell1.rcut, cell2.rcut))
    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCnr2c_drv
    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cpbcopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

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

def _estimate_rcut(alpha, l, c, precision=INTEGRAL_PRECISION):
    C = (c**2+1e-200)*(2*l+1)*alpha / precision
    r0 = 20
    # +1. to ensure np.log returning positive value
    r0 = np.sqrt(2.*np.log(C*(r0**2*alpha)**(l+1)+1.) / alpha)
    rcut = np.sqrt( 2.*np.log(C*(r0**2*alpha)**(l+1)+1.) / alpha)
    return rcut

def bas_rcut(cell, bas_id, precision=INTEGRAL_PRECISION):
    r'''Estimate the largest distance between the function and its image to
    reach the precision in overlap

    precision ~ \int g(r-0) g(r-R)
    '''
    l = cell.bas_angular(bas_id)
    es = cell.bas_exp(bas_id)
    cs = abs(cell.bas_ctr_coeff(bas_id)).max(axis=1)
    rcut = _estimate_rcut(es, l, cs, precision)
    return rcut.max()

def _estimate_ke_cutoff(alpha, l, c, precision=INTEGRAL_PRECISION, weight=1.):
    '''Energy cutoff estimation based on cubic lattice'''
    # This function estimates the energy cutoff for (ii|ii) type of electron
    # repulsion integrals. The energy cutoff for nuclear attraction is larger
    # than the energy cutoff for ERIs.  The estimated error is roughly
    #     error ~ 64 pi^3 c^2 /((2l+1)!!(4a)^l) (2Ecut)^{l+.5} e^{-Ecut/4a}
    # log_k0 = 3 + np.log(alpha) / 2
    # l2fac2 = factorial2(l*2+1)
    # log_rest = np.log(precision*l2fac2*(4*alpha)**l / (16*np.pi**2*c**2))
    # Enuc_cut = 4*alpha * (log_k0*(2*l+1) - log_rest)
    # Enuc_cut[Enuc_cut <= 0] = .5
    # log_k0 = .5 * np.log(Ecut*2)
    # Enuc_cut = 4*alpha * (log_k0*(2*l+1) - log_rest)
    # Enuc_cut[Enuc_cut <= 0] = .5
    #
    # However, nuclear attraction can be evaluated with the trick of Ewald
    # summation which largely reduces the requirements to the energy cutoff.
    # In practice, the cutoff estimation for ERIs as below should be enough.
    log_k0 = 3 + np.log(alpha) / 2
    l2fac2 = factorial2(l*2+1)
    log_rest = np.log(precision*l2fac2**2*(4*alpha)**(l*2+1) / (128*np.pi**4*c**4))
    Ecut = 2*alpha * (log_k0*(4*l+3) - log_rest)
    Ecut[Ecut <= 0] = .5
    log_k0 = .5 * np.log(Ecut*2)
    Ecut = 2*alpha * (log_k0*(4*l+3) - log_rest)
    Ecut[Ecut <= 0] = .5
    return Ecut

def estimate_ke_cutoff(cell, precision=INTEGRAL_PRECISION):
    '''Energy cutoff estimation'''
    Ecut_max = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i)
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        ke_guess = _estimate_ke_cutoff(es, l, cs, precision)
        Ecut_max = max(Ecut_max, ke_guess.max())
    return Ecut_max

def error_for_ke_cutoff(cell, ke_cutoff):
    b = cell.reciprocal_vectors()
    kmax = np.sqrt(ke_cutoff*2)
    errmax = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i)
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        fac = (256*np.pi**4*cs**4 * factorial2(l*4+3)
               / factorial2(l*2+1)**2)
        efac = np.exp(-ke_cutoff/(2*es))
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

def get_Gv(cell, mesh=None, **kwargs):
    '''Calculate three-dimensional G-vectors for the cell; see MH (3.8).

    Indices along each direction go as [0...N-1, -N...-1] to follow FFT convention.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        Gv : (ngrids, 3) ndarray of floats
            The array of G-vectors.
    '''
    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]

    gx = np.fft.fftfreq(mesh[0], 1./mesh[0])
    gy = np.fft.fftfreq(mesh[1], 1./mesh[1])
    gz = np.fft.fftfreq(mesh[2], 1./mesh[2])
    gxyz = lib.cartesian_prod((gx, gy, gz))

    b = cell.reciprocal_vectors()
    Gv = lib.ddot(gxyz, b)
    return Gv

def get_Gv_weights(cell, mesh=None, **kwargs):
    '''Calculate G-vectors and weights.

    Returns:
        Gv : (ngris, 3) ndarray of floats
            The array of G-vectors.
    '''
    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]

    # Default, the 3D uniform grids
    rx = np.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = np.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = np.fft.fftfreq(mesh[2], 1./mesh[2])
    b = cell.reciprocal_vectors()
    weights = abs(np.linalg.det(b))

    if (cell.dimension < 2 or
        (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        if cell.dimension == 0:
            rx, wx = _non_uniform_Gv_base(mesh[0]//2)
            ry, wy = _non_uniform_Gv_base(mesh[1]//2)
            rz, wz = _non_uniform_Gv_base(mesh[2]//2)
            rx /= np.linalg.norm(b[0])
            ry /= np.linalg.norm(b[1])
            rz /= np.linalg.norm(b[2])
            weights = np.einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)
        elif cell.dimension == 1:
            wx = np.repeat(np.linalg.norm(b[0]), mesh[0])
            ry, wy = _non_uniform_Gv_base(mesh[1]//2)
            rz, wz = _non_uniform_Gv_base(mesh[2]//2)
            ry /= np.linalg.norm(b[1])
            rz /= np.linalg.norm(b[2])
            weights = np.einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)
        elif cell.dimension == 2:
            area = np.linalg.norm(np.cross(b[0], b[1]))
            wxy = np.repeat(area, mesh[0]*mesh[1])
            rz, wz = _non_uniform_Gv_base(mesh[2]//2)
            rz /= np.linalg.norm(b[2])
            weights = np.einsum('i,k->ik', wxy, wz).reshape(-1)

    Gvbase = (rx, ry, rz)
    Gv = np.dot(lib.cartesian_prod(Gvbase), b)
    # 1/cell.vol == det(b)/(2pi)^3
    weights *= 1/(2*np.pi)**3
    return Gv, Gvbase, weights

def _non_uniform_Gv_base(n):
    #rs, ws = radi.delley(n)
    #rs, ws = radi.treutler_ahlrichs(n)
    #rs, ws = radi.mura_knowles(n)
    rs, ws = radi.gauss_chebyshev(n)
    #return np.hstack((0,rs,-rs[::-1])), np.hstack((0,ws,ws[::-1]))
    return np.hstack((rs,-rs[::-1])), np.hstack((ws,ws[::-1]))

def get_SI(cell, Gv=None):
    '''Calculate the structure factor (0D, 1D, 2D, 3D) for all atoms; see MH (3.34).

    Args:
        cell : instance of :class:`Cell`

        Gv : (N,3) array
            G vectors

    Returns:
        SI : (natm, ngrids) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    '''
    coords = cell.atom_coords()
    ngrids = np.prod(cell.mesh)
    if Gv is None or Gv.shape[0] == ngrids:
        basex, basey, basez = cell.get_Gv_weights(cell.mesh)[1]
        b = cell.reciprocal_vectors()
        rb = np.dot(coords, b.T)
        SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
        SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
        SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
        SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
        SI = SI.reshape(-1,ngrids)
    else:
        SI = np.exp(-1j*np.dot(coords, Gv.T))
    return SI

def get_ewald_params(cell, precision=INTEGRAL_PRECISION, mesh=None):
    r'''Choose a reasonable value of Ewald 'eta' and 'cut' parameters.
    eta^2 is the exponent coefficient of the model Gaussian charge for nucleus
    at R:  \frac{eta^3}{pi^1.5} e^{-eta^2 (r-R)^2}

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
    elif (cell.dimension < 2 or
          (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
# Non-uniform PW grids are used for low-dimensional ewald summation.  The cutoff
# estimation for long range part based on exp(G^2/(4*eta^2)) does not work for
# non-uniform grids.  Smooth model density is preferred.
        ew_cut = cell.rcut
        ew_eta = np.sqrt(max(np.log(4*np.pi*ew_cut**2/precision)/ew_cut**2, .1))
    else:
        if mesh is None:
            mesh = cell.mesh
        mesh = _cut_mesh_for_ewald(cell, mesh)
        Gmax = min(np.asarray(mesh)//2 * lib.norm(cell.reciprocal_vectors(), axis=1))
        log_precision = np.log(precision/(4*np.pi*(Gmax+1e-100)**2))
        ew_eta = np.sqrt(-Gmax**2/(4*log_precision)) + 1e-100
        ew_cut = _estimate_rcut(ew_eta**2, 0, 1., precision)
    return ew_eta, ew_cut

def _cut_mesh_for_ewald(cell, mesh):
    mesh = np.copy(mesh)
    mesh_max = np.asarray(np.linalg.norm(cell.lattice_vectors(), axis=1) * 2,
                          dtype=int)  # roughly 2 grids per bohr
    if (cell.dimension < 2 or
        (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        mesh_max[cell.dimension:] = mesh[cell.dimension:]

    mesh_max[mesh_max<80] = 80
    mesh[mesh>mesh_max] = mesh_max[mesh>mesh_max]
    return mesh

def ewald(cell, ew_eta=None, ew_cut=None):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
    # If lattice parameter is not set, the cell object is treated as a mole
    # object. The nuclear repulsion energy is computed.
    if cell.a is None:
        return mole.energy_nuc(cell)

    if cell.natm == 0:
        return 0

    if ew_eta is None: ew_eta = cell.ew_eta
    if ew_cut is None: ew_cut = cell.ew_cut
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    r = np.sqrt(np.einsum('Lijx,Lijx->Lij', rLij, rLij))
    rLij = None
    r[r<1e-16] = 1e200
    ewovrl = .5 * np.einsum('i,j,Lij->', chargs, chargs, erfc(ew_eta * r) / r)

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
    mesh = _cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200
    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        coulG = 4*np.pi / absG2
        coulG *= weights
        ZSI = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
        ZexpG2 = ZSI * np.exp(-absG2/(4*ew_eta**2))
        ewg = .5 * np.einsum('i,i,i', ZSI.conj(), ZexpG2, coulG).real

    elif cell.dimension == 2:  # Truncated Coulomb
        # The following 2D ewald summation is taken from:
        # R. Sundararaman and T. Arias PRB 87, 2013
        def fn(eta,Gnorm,z):
            Gnorm_z = Gnorm*z
            large_idx = Gnorm_z > 20.0
            ret = np.zeros_like(Gnorm_z)
            x = Gnorm/2./eta + eta*z
            with np.errstate(over='ignore'):
                erfcx = erfc(x)
                ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                ret[ large_idx] = np.exp((Gnorm*z-x**2)[large_idx]) * erfcx[large_idx]
            return ret
        def gn(eta,Gnorm,z):
            return np.pi/Gnorm*(fn(eta,Gnorm,z) + fn(eta,Gnorm,-z))
        def gn0(eta,z):
            return -2*np.pi*(z*erf(eta*z) + np.exp(-(eta*z)**2)/eta/np.sqrt(np.pi))
        b = cell.reciprocal_vectors()
        inv_area = np.linalg.norm(np.cross(b[0], b[1]))/(2*np.pi)**2
        # Perform the reciprocal space summation over  all reciprocal vectors
        # within the x,y plane.
        planarG2_idx = np.logical_and(Gv[:,2] == 0, absG2 > 0.0)
        Gv = Gv[planarG2_idx]
        absG2 = absG2[planarG2_idx]
        absG = absG2**(0.5)
        # Performing the G != 0 summation.
        rij = coords[:,None,:] - coords[None,:,:]
        Gdotr = np.einsum('ijx,gx->ijg', rij, Gv)
        ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                        gn(ew_eta,absG,rij[:,:,2:3]))
        # Performing the G == 0 summation.
        ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta,rij[:,:,2]))
        ewg *= inv_area*0.5

    else:
        logger.warn(cell, 'No method for PBC dimension %s, dim-type %s.'
                    '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                    cell.dimension, cell.low_dim_ft_type)
        raise NotImplementedError

    logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

energy_nuc = ewald

def make_kpts(cell, nks, wrap_around=WRAP_AROUND, with_gamma_point=WITH_GAMMA,
              scaled_center=None):
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

def get_uniform_grids(cell, mesh=None, **kwargs):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    if mesh is None: mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    mesh = np.asarray(mesh, dtype=np.double)
    qv = lib.cartesian_prod([np.arange(x) for x in mesh])
    a_frac = np.einsum('i,ij->ij', 1./mesh, cell.lattice_vectors())
    coords = np.dot(qv, a_frac)
    return coords
gen_uniform_grids = get_uniform_grids

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
                return {}, {'default': str(ecp)}
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

def _split_basis(cell, delimiter=EXP_DELIMITER):
    '''
    Split the contracted basis to small segmant.  The new basis has more
    shells.  Each shell has less primitive basis and thus is more local.
    '''
    import copy
    _bas = []
    _env = cell._env.copy()
    contr_coeff = []
    for ib in range(cell.nbas):
        pexp = cell._bas[ib,mole.PTR_EXP]
        pcoeff1 = cell._bas[ib,mole.PTR_COEFF]
        nc = cell.bas_nctr(ib)
        es = cell.bas_exp(ib)
        cs = cell._libcint_ctr_coeff(ib)
        l = cell.bas_angular(ib)
        if cell.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = l * 2 + 1

        mask = np.ones(es.size, dtype=bool)
        count = 0
        for thr in delimiter:
            idx = np.where(mask & (es >= thr))[0]
            np1 = len(idx)
            if np1 > 0:
                pcoeff0, pcoeff1 = pcoeff1, pcoeff1 + np1 * nc
                cs1 = cs[idx]
                _env[pcoeff0:pcoeff1] = cs1.T.ravel()
                btemp = cell._bas[ib].copy()
                btemp[mole.NPRIM_OF] = np1
                btemp[mole.PTR_COEFF] = pcoeff0
                btemp[mole.PTR_EXP] = pexp
                _bas.append(btemp)
                mask[idx] = False
                pexp += np1
                count += 1
        contr_coeff.append(np.vstack([np.eye(degen*nc)] * count))

    pcell = copy.copy(cell)
    pcell._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pcell._env = _env
    return pcell, scipy.linalg.block_diag(*contr_coeff)

def tot_electrons(cell, nkpts=1):
    '''Total number of electrons
    '''
    if cell._nelectron is None:
        nelectron = cell.atom_charges().sum() * nkpts - cell.charge
    else: # Custom cell.nelectron stands for num. electrons per unit cell
        nelectron = cell._nelectron * nkpts
    # Round off to the nearest integer
    nelectron = int(nelectron+0.5)
    return nelectron

def _mesh_inf_vaccum(cell):
    #prec ~ exp(-0.436392335*mesh -2.99944305)*nelec
    meshz = (np.log(cell.nelectron/cell.precision)-2.99944305)/0.436392335
    # meshz has to be even number due to the symmetry on z+ and z-
    return int(meshz*.5 + .999) * 2


class Cell(mole.Mole):
    '''A Cell object holds the basic information of a crystal.

    Attributes:
        a : (3,3) ndarray
            Lattice primitive vectors. Each row represents a lattice vector
            Reciprocal lattice vectors are given by  b1,b2,b3 = 2 pi inv(a).T
        mesh : (3,) list of ints
            The number G-vectors along each direction.
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
            Periodic dimensions. Default is 3
        low_dim_ft_type : str
            For semi-empirical periodic systems, whether to calculate
            integrals at the non-PBC dimension using the sampled mesh grids in
            infinity vacuum (inf_vacuum) or truncated Coulomb potential
            (analytic_2d_1). Unless explicitly specified, analytic_2d_1 is
            used for 2D system and inf_vacuum is assumed for 1D and 0D.

        ** Following attributes (for experts) are automatically generated. **

        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.  See :func:`get_ewald_params`

    (See other attributes in :class:`Mole`)

    Examples:

    >>> mol = Mole(atom='H^2 0 0 0; H 0 0 1.1', basis='sto3g')
    >>> cl = Cell()
    >>> cl.build(a='3 0 0; 0 3 0; 0 0 3', atom='C 1 1 1', basis='sto3g')
    >>> print(cl.atom_symbol(0))
    C
    '''

    precision = getattr(__config__, 'pbc_gto_cell_Cell_precision', 1e-8)
    exp_to_discard = getattr(__config__, 'pbc_gto_cell_Cell_exp_to_discard', None)

    def __init__(self, **kwargs):
        mole.Mole.__init__(self, **kwargs)
        self.a = None # lattice vectors, (a1,a2,a3)
        self.ke_cutoff = None # if set, defines a spherical cutoff
                              # of fourier components, with .5 * G**2 < ke_cutoff
        self.pseudo = None
        self.dimension = 3
        # TODO: Simple hack for now; the implementation of ewald depends on the
        #       density-fitting class.  This determines how the ewald produces
        #       its energy.
        self.low_dim_ft_type = None

##################################################
# These attributes are initialized by build function if not given
        self.mesh = None
        self.ew_eta = None
        self.ew_cut = None
        self.rcut = None

##################################################
# don't modify the following variables, they are not input arguments
        keys = ('precision', 'exp_to_discard')
        self._keys = self._keys.union(self.__dict__).union(keys)

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, x):
        self._mesh = x
        self._mesh_from_build = False

    @property
    def rcut(self):
        return self._rcut
    @rcut.setter
    def rcut(self, x):
        self._rcut = x
        self._rcut_from_build = False

    @property
    def ew_eta(self):
        return self._ew_eta
    @ew_eta.setter
    def ew_eta(self, val):
        self._ew_eta = val
        self._ew_from_build = False

    @property
    def ew_cut(self):
        return self._ew_cut
    @ew_cut.setter
    def ew_cut(self, val):
        self._ew_cut = val
        self._ew_from_build = False

    if not getattr(__config__, 'pbc_gto_cell_Cell_verify_nelec', False):
# nelec method defined in Mole class raises error when the attributes .spin
# and .nelectron are inconsistent.  In PBC, when the system has even number of
# k-points, it is valid that .spin is odd while .nelectron is even.
# Overwriting nelec method to avoid this check.
        @property
        def nelec(self):
            ne = self.nelectron
            nalpha = (ne + self.spin) // 2
            nbeta = nalpha - self.spin
            if nalpha + nbeta != ne:
                warnings.warn('Electron number %d and spin %d are not consistent '
                              'in unit cell\n' % (ne, self.spin))
            return nalpha, nbeta

    def __getattr__(self, key):
        '''To support accessing methods (cell.HF, cell.KKS, cell.KUCCSD, ...)
        from Cell object.
        '''
        if key[:2] == '__':  # Skip Python builtins
            raise AttributeError('Cell object has no attribute %s' % key)
        elif key in ('_ipython_canary_method_should_not_exist_',
                   '_repr_mimebundle_'):
            # https://github.com/mewwts/addict/issues/26
            # https://github.com/jupyter/notebook/issues/2014
            raise AttributeError

        # Import all available modules. Some methods are registered to other
        # classes/modules when importing modules in __all__.
        from pyscf.pbc import __all__
        from pyscf.pbc import scf, dft
        from pyscf.dft import XC
        for mod in (scf, dft):
            method = getattr(mod, key, None)
            if callable(method):
                return method(self)

        if key[0] == 'K':  # with k-point sampling
            if 'TD' in key[:4]:
                if key in ('KTDHF', 'KTDA'):
                    mf = scf.KHF(self)
                else:
                    mf = dft.KKS(self)
                    xc = key.split('TD', 1)[1]
                    if xc in XC:
                        mf.xc = xc
                        key = 'KTDDFT'
            else:
                mf = scf.KHF(self)
            # Remove prefix 'K' because methods are registered without the leading 'K'
            key = key[1:]
        else:
            if 'TD' in key[:3]:
                if key in ('TDHF', 'TDA'):
                    mf = scf.HF(self)
                else:
                    mf = dft.KS(self)
                    xc = key.split('TD', 1)[1]
                    if xc in XC:
                        mf.xc = xc
                        key = 'TDDFT'
            else:
                mf = scf.HF(self)

        method = getattr(mf, key, None)
        if method is None:
            raise AttributeError('Cell object has no attribute %s' % key)

        mf.run()
        return method

    tot_electrons = tot_electrons

#Note: Exculde dump_input, parse_arg, basis from kwargs to avoid parsing twice
    def build(self, dump_input=True, parse_arg=True,
              a=None, mesh=None, ke_cutoff=None, precision=None, nimgs=None,
              ew_eta=None, ew_cut=None, pseudo=None, basis=None, h=None,
              dimension=None, rcut= None, ecp=None, low_dim_ft_type=None,
              *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.
        Whenever you change the value of the attributes of :class:`Cell`,
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            a : (3,3) ndarray
                The real-space unit cell lattice vectors. Each row represents
                a lattice vector.
            mesh : (3,) ndarray of ints
                The number of *positive* G-vectors along each direction.
            ke_cutoff : float
                If set, defines a spherical cutoff of planewaves, with .5 * G**2 < ke_cutoff
                The default value is estimated based on :attr:`precision`
            precision : float
                To control Ewald sums and lattice sums accuracy
            nimgs : (3,) ndarray of ints
                Number of repeated images in lattice summation to produce
                periodicity. This value can be estimated based on the required
                precision.  It's recommended NOT making changes to this value.
            rcut : float
                Cutoff radius (unit Bohr) in lattice summation to produce
                periodicity. The value can be estimated based on the required
                precision.  It's recommended NOT making changes to this value.
            ew_eta, ew_cut : float
                Parameters eta and cut to converge Ewald summation.
                See :func:`get_ewald_params`
            pseudo : dict or str
                To define pseudopotential.
            ecp : dict or str
                To define ECP type pseudopotential.
            h : (3,3) ndarray
                a.T. Deprecated
            dimension : int
                Default is 3
            low_dim_ft_type : str
                For semi-empirical periodic systems, whether to calculate
                integrals at the non-PBC dimension using the sampled mesh grids in
                infinity vacuum (inf_vacuum) or truncated Coulomb potential
                (analytic_2d_1). Unless explicitly specified, analytic_2d_1 is
                used for 2D system and inf_vacuum is assumed for 1D and 0D.
        '''
        if h is not None: self.h = h
        if a is not None: self.a = a
        if mesh is not None: self.mesh = mesh
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
        if low_dim_ft_type is not None: self.low_dim_ft_type = low_dim_ft_type

        if 'unit' in kwargs:
            self.unit = kwargs['unit']

        if 'atom' in kwargs:
            self.atom = kwargs['atom']

        if 'gs' in kwargs:
            self.gs = kwargs['gs']

        # Set-up pseudopotential if it exists
        # This must be done before build() because it affects
        # tot_electrons() via the call to .atom_charge()

        self.ecp, self.pseudo = classify_ecp_pseudo(self, self.ecp, self.pseudo)
        if self.pseudo is not None:
            _atom = self.format_atom(self.atom)
            uniq_atoms = set([a[0] for a in _atom])
            if isinstance(self.pseudo, (str, unicode)):
                # specify global pseudo for whole molecule
                _pseudo = dict([(a, str(self.pseudo)) for a in uniq_atoms])
            elif 'default' in self.pseudo:
                default_pseudo = self.pseudo['default']
                _pseudo = dict(((a, default_pseudo) for a in uniq_atoms))
                _pseudo.update(self.pseudo)
                del(_pseudo['default'])
            else:
                _pseudo = self.pseudo
            self._pseudo = self.format_pseudo(_pseudo)

        # Do regular Mole.build
        _built = self._built
        mole.Mole.build(self, False, parse_arg, *args, **kwargs)

        exp_min = np.array([self.bas_exp(ib).min() for ib in range(self.nbas)])
        if self.exp_to_discard is None:
            if np.any(exp_min < 0.1):
                sys.stderr.write('''WARNING!
  Very diffused basis functions are found in the basis set. They may lead to severe
  linear dependence and numerical instability.  You can set  cell.exp_to_discard=0.1
  to remove the diffused Gaussians whose exponents are less than 0.1.\n\n''')
        elif np.any(exp_min < self.exp_to_discard):
            # Discard functions of small exponents in basis
            _basis = {}
            for symb, basis_now in self._basis.items():
                basis_add = []
                for b in basis_now:
                    l = b[0]
                    if isinstance(b[1], int):
                        kappa = b[1]
                        b_coeff = np.array(b[2:])
                    else:
                        kappa = 0
                        b_coeff = np.array(b[1:])
                    es = b_coeff[:,0]
                    if np.any(es < self.exp_to_discard):
                        b_coeff = b_coeff[es>=self.exp_to_discard]
# contraction coefficients may be completely zero after removing one primitive
# basis. Removing the zero-coefficient basis.
                        b_coeff = b_coeff[:,np.all(b_coeff!=0, axis=0)]
                        if b_coeff.size > 0:
                            if kappa == 0:
                                basis_add.append([l] + b_coeff.tolist())
                            else:
                                basis_add.append([l,kappa] + b_coeff.tolist())
                    else:
                        basis_add.append(b)
                _basis[symb] = basis_add
            self._basis = _basis

            steep_shls = []
            nprim_drop = 0
            nctr_drop = 0
            for ib in range(len(self._bas)):
                l = self.bas_angular(ib)
                nprim = self.bas_nprim(ib)
                nc = self.bas_nctr(ib)
                es = self.bas_exp(ib)
                ptr = self._bas[ib,mole.PTR_COEFF]
                cs = self._env[ptr:ptr+nprim*nc].reshape(nc,nprim).T

                if np.any(es < self.exp_to_discard):
                    cs = cs[es>=self.exp_to_discard]
                    es = es[es>=self.exp_to_discard]
                    nprim_old, nc_old = nprim, nc

# contraction coefficients may be completely zero after removing one primitive
# basis. Removing the zero-coefficient basis.
                    cs = cs[:,np.all(cs!=0, axis=0)]
                    nprim, nc = cs.shape
                    self._bas[ib,mole.NPRIM_OF] = nprim
                    self._bas[ib,mole.NCTR_OF] = nc

                    nprim_drop = nprim_old - nprim + nprim_drop
                    nctr_drop = nc_old - nc + nctr_drop
                    if cs.size > 0:
                        pe = self._bas[ib,mole.PTR_EXP]
                        self._env[pe:pe+nprim] = es
                        cs = mole._nomalize_contracted_ao(l, es, cs)
                        self._env[ptr:ptr+nprim*nc] = cs.T.reshape(-1)
                if nprim > 0:
                    steep_shls.append(ib)
            self._bas = np.asarray(self._bas[steep_shls], order='C')
            logger.info(self, 'Discarded %d diffused primitive functions, '
                        '%d contracted functions', nprim_drop, nctr_drop)
            #logger.debug1(self, 'Old shells %s', steep_shls)

        # The rest initialization requires lattice parameters.  If .a is not
        # set, pass the rest initialization.
        if self.a is None:
            if dump_input and not _built and self.verbose > logger.NOTE:
                self.dump_input()
            return self

        if self.rcut is None or self._rcut_from_build:
            self._rcut = max([self.bas_rcut(ib, self.precision)
                              for ib in range(self.nbas)] + [0])
            self._rcut_from_build = True

        _a = self.lattice_vectors()
        if np.linalg.det(_a) < 0:
            sys.stderr.write('''WARNING!
  Lattice are not in right-handed coordinate system. This can cause wrong value for some integrals.
  It's recommended to resort the lattice vectors to\na = %s\n\n''' % _a[[0,2,1]])

        if self.dimension == 2 and self.low_dim_ft_type != 'inf_vacuum':
            # check vacuum size. See Fig 1 of PRB, 73, 2015119
            #Lz_guess = self.rcut*(1+np.sqrt(2))
            Lz_guess = self.rcut * 2
            if np.linalg.norm(_a[2]) < 0.7 * Lz_guess:
                sys.stderr.write('''WARNING!
  Size of vacuum may not be enough. The recommended vacuum size is %s AA (%s Bohr)\n\n'''
                                 % (Lz_guess*param.BOHR, Lz_guess))

        if self.mesh is None or self._mesh_from_build:
            if self.ke_cutoff is None:
                ke_cutoff = estimate_ke_cutoff(self, self.precision)
            else:
                ke_cutoff = self.ke_cutoff
            self._mesh = pbctools.cutoff_to_mesh(_a, ke_cutoff)

            if (self.dimension < 2 or
                (self.dimension == 2 and self.low_dim_ft_type == 'inf_vacuum')):
                self._mesh[self.dimension:] = _mesh_inf_vaccum(self)
            self._mesh_from_build = True

            # Set minimal mesh grids to handle the case mesh==0. since Madelung
            # constant may be computed even if the unit cell has 0 atoms. In this
            # system, cell.mesh was initialized to 0.
            self._mesh[self._mesh == 0] = 30

        if self.ew_eta is None or self.ew_cut is None or self._ew_from_build:
            self._ew_eta, self._ew_cut = self.get_ewald_params(self.precision, self.mesh)
            self._ew_from_build = True

        if dump_input and not _built and self.verbose > logger.NOTE:
            self.dump_input()
            logger.info(self, 'lattice vectors  a1 [%.9f, %.9f, %.9f]', *_a[0])
            logger.info(self, '                 a2 [%.9f, %.9f, %.9f]', *_a[1])
            logger.info(self, '                 a3 [%.9f, %.9f, %.9f]', *_a[2])
            logger.info(self, 'dimension = %s', self.dimension)
            logger.info(self, 'low_dim_ft_type = %s', self.low_dim_ft_type)
            logger.info(self, 'Cell volume = %g', self.vol)
            if self.exp_to_discard is not None:
                logger.info(self, 'exp_to_discard = %s', self.exp_to_discard)
            logger.info(self, 'rcut = %s (nimgs = %s)', self.rcut, self.nimgs)
            logger.info(self, 'lattice sum = %d cells', len(self.get_lattice_Ls()))
            logger.info(self, 'precision = %g', self.precision)
            logger.info(self, 'pseudo = %s', self.pseudo)
            if ke_cutoff is not None:
                logger.info(self, 'ke_cutoff = %s', ke_cutoff)
                logger.info(self, '    = %s mesh (%d PWs)',
                            self.mesh, np.prod(self.mesh))
            else:
                logger.info(self, 'mesh = %s (%d PWs)',
                            self.mesh, np.prod(self.mesh))
                Ecut = pbctools.mesh_to_cutoff(self.lattice_vectors(), self.mesh)
                logger.info(self, '    = ke_cutoff %s', Ecut)
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
        warnings.warn('cell.h is deprecated.  It is replaced by the '
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
        return self.get_Gv()

    @lib.with_doc(format_pseudo.__doc__)
    def format_pseudo(self, pseudo_tab):
        return format_pseudo(pseudo_tab)

    @lib.with_doc(format_basis.__doc__)
    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    @property
    def gs(self):
        return [n//2 for n in self.mesh]
    @gs.setter
    def gs(self, x):
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        self.mesh = [2*n+1 for n in x]

    @property
    def drop_exponent(self):
        return self.exp_to_discard
    @drop_exponent.setter
    def drop_exponent(self, x):
        self.exp_to_discard = x

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

    make_kpts = get_kpts = make_kpts

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

    gen_uniform_grids = get_uniform_grids = get_uniform_grids

    __add__ = conc_cell

    def pbc_intor(self, intor, comp=None, hermi=0, kpts=None, kpt=None,
                  shls_slice=None, **kwargs):
        r'''One-electron integrals with PBC.

        .. math::

            \sum_T \int \mu(r) * [intor] * \nu (r-T) dr

        See also Mole.intor
        '''
        if not self._built:
            logger.warn(self, 'Warning: intor envs of %s not initialized.', self)
            # FIXME: Whether to check _built and call build?  ._bas and .basis
            # may not be consistent. calling .build() may leads to wrong intor env.
            #self.build(False, False)
        return intor_cross(intor, self, self, comp, hermi, kpts, kpt,
                           shls_slice, **kwargs)

    @lib.with_doc(pbc_eval_gto.__doc__)
    def pbc_eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None,
                     shls_slice=None, non0tab=None, ao_loc=None, out=None):
        return pbc_eval_gto(self, eval_name, coords, comp, kpts, kpt,
                            shls_slice, non0tab, ao_loc, out)
    pbc_eval_ao = pbc_eval_gto

    @lib.with_doc(pbc_eval_gto.__doc__)
    def eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None,
                 shls_slice=None, non0tab=None, ao_loc=None, out=None):
        if eval_name[:3] == 'PBC':
            return self.pbc_eval_gto(eval_name, coords, comp, kpts, kpt,
                                     shls_slice, non0tab, ao_loc, out)
        else:
            return mole.eval_gto(self, eval_name, coords, comp,
                                 shls_slice, non0tab, ao_loc, out)
    eval_ao = eval_gto

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
        #FIXME: should cell be converted to mole object?  If cell is converted
        # and a mole object is returned, many attributes (e.g. the GTH basis,
        # gth-PP) will not be recognized by mole.build function.
        mol = self.view(mole.Mole)
        delattr(mol, 'a')
        delattr(mol, '_mesh')
        return mol

    def has_ecp(self):
        '''Whether pseudo potential is used in the system.'''
        return self.pseudo or self._pseudo or (len(self._ecpbas) > 0)

    def ao2mo(self, mo_coeffs, intor='int2e', erifile=None, dataname='eri_mo',
              **kwargs):
        raise NotImplementedError

del(INTEGRAL_PRECISION, WRAP_AROUND, WITH_GAMMA, EXP_DELIMITER)
