#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

import os
import sys
import json
import ctypes
import warnings
import numpy as np
import scipy.linalg
from scipy.special import erf, erfc
import pyscf.lib.parameters as param
from pyscf import lib
from pyscf.dft import radi
from pyscf.lib import logger
from pyscf.gto import mole
from pyscf.gto import moleintor
from pyscf.gto.mole import conc_env, is_au # noqa
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.gto.eval_gto import eval_gto as pbc_eval_gto
from pyscf.pbc.tools import pbc as pbctools
from pyscf import __config__

INTEGRAL_PRECISION = getattr(__config__, 'pbc_gto_cell_Cell_precision', 1e-8)
WRAP_AROUND = getattr(__config__, 'pbc_gto_cell_make_kpts_wrap_around', False)
WITH_GAMMA = getattr(__config__, 'pbc_gto_cell_make_kpts_with_gamma', True)
EXP_DELIMITER = getattr(__config__, 'pbc_gto_cell_split_basis_exp_delimiter',
                        [1.0, 0.5, 0.25, 0.1, 0])
# defined in lib/pbc/cell.h
RCUT_EPS = 1e-3
RCUT_MAX_CYCLE = 10

libpbc = _pbcintor.libpbc

def M(*args, **kwargs):
    r'''This is a shortcut to build up Cell object.

    Examples:

    >>> from pyscf.pbc import gto
    >>> cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g')
    '''
    cell = Cell()
    cell.build(*args, **kwargs)
    return cell
C = M

def pack(cell):
    '''Pack the input args of :class:`Cell` to a dict, which can be serialized
    with :mod:`pickle`
    '''
    cldic = mole.pack(cell)
    cldic['a'] = cell.a
    cldic['fractional'] = cell.fractional
    cldic['precision'] = cell.precision
    cldic['ke_cutoff'] = cell.ke_cutoff
    cldic['exp_to_discard'] = cell.exp_to_discard
    cldic['_mesh'] = cell._mesh
    cldic['_rcut'] = cell._rcut
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
    exclude_keys = {'output', 'stdout', '_keys', '_ctx_lock',
                    'symm_orb', 'irrep_id', 'irrep_name', 'lattice_symmetry'}

    celldic = dict(cell.__dict__)
    for k in exclude_keys:
        if k in celldic:
            del (celldic[k])
    for k in celldic:
        if isinstance(celldic[k], (np.ndarray, np.generic)):
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
                    isinstance(v, (str, bool, int, float))):
                    dic1[k] = v
                elif isinstance(v, (list, tuple)):
                    dic1[k] = v   # Should I recursively skip_vaule?
                elif isinstance(v, set):
                    dic1[k] = list(v)
                elif isinstance(v, dict):
                    dic1[k] = skip_value(v)
                elif isinstance(v, np.generic):
                    dic1[k] = v.tolist()
                else:
                    msg =('Function cell.dumps drops attribute %s because '
                          'it is not JSON-serializable' % k)
                    warnings.warn(msg)
            return dic1
        return json.dumps(skip_value(celldic), skipkeys=True)

def loads(cellstr):
    '''Deserialize a str containing a JSON document to a Cell object.
    '''
    from numpy import array  # noqa
    celldic = json.loads(cellstr)
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
    cell._mesh = np.array(cell._mesh)

    # Symmetry class cannot be serialized by dumps function.
    # Recreate it manually
    if cell.natm > 0 and cell.space_group_symmetry:
        cell.build_lattice_symmetry()

    return cell

def conc_cell(cell1, cell2):
    '''Concatenate two Cell objects.
    '''
    mol3 = mole.conc_mol(cell1, cell2)
    cell3 = Cell()
    cell3.__dict__.update(mol3.__dict__)

    # lattice_vectors needs to be consistent with cell3.unit (Bohr)
    cell3.a = cell1.lattice_vectors()
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
    cell3.rcut = max(cell1.rcut, cell2.rcut)
    return cell3

def intor_cross(intor, cell1, cell2, comp=None, hermi=0, kpts=None, kpt=None,
                shls_slice=None, **kwargs):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    intor, comp = moleintor._get_intor_and_comp(cell1._add_suffix(intor), comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    pcell = cell1.copy(deep=False)
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

    rcut = max(cell1.rcut, cell2.rcut)
    Ls = cell1.get_lattice_Ls(rcut=rcut)
    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCnr2c_drv

    kderiv = kwargs.get('kderiv', 0)
    if kderiv > 0:
        hermi = 0
        aosym = 's1'
        mat = np.empty((nkpts,(3**kderiv)*comp,ni,nj), dtype=np.complex128)
        if kderiv == 1:
            fac = 1j * lib.einsum('kl,lx->xkl', expkL, Ls)
        elif kderiv == 2:
            fac = -lib.einsum('kl,lx,ly->xykl', expkL, Ls, Ls).reshape(-1,nkpts,len(Ls))
        else:
            raise NotImplementedError

        for x in range(fac.shape[0]):
            facx = np.asarray(fac[x], order='C')
            drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
                Ls.ctypes.data_as(ctypes.c_void_p),
                facx.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*4)(i0, i1, j0, j1),
                ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
                env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

            for k, kpt in enumerate(kpts_lst):
                v = out[k]
                if hermi != 0:
                    for ic in range(comp):
                        lib.hermi_triu(v[ic], hermi=hermi, inplace=True)
                mat[k, x*comp:(x+1)*comp] = v.copy()
        return mat

    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
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

def _intor_cross_screened(
        intor, cell1, cell2, comp=None, hermi=0, kpts=None, kpt=None,
        shls_slice=None, **kwargs):
    '''`intor_cross` with prescreening.

    Notes:
         This function may be subject to change.
    '''
    from pyscf.pbc.gto.neighborlist import NeighborListOpt
    intor, comp = moleintor._get_intor_and_comp(cell1._add_suffix(intor), comp)

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    pcell = cell1.copy(deep=False)
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
    fill = getattr(libpbc, 'PBCnr2c_screened_fill_k'+aosym)
    fintor = getattr(moleintor.libcgto, intor)
    drv = libpbc.PBCnr2c_screened_drv

    rcut = max(cell1.rcut, cell2.rcut)
    Ls = cell1.get_lattice_Ls(rcut=rcut)
    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')

    neighbor_list = kwargs.get('neighbor_list', None)
    if neighbor_list is None:
        nlopt = NeighborListOpt(cell1)
        nlopt.build(cell1, cell2, Ls, set_optimizer=False)
        neighbor_list = nlopt.nl

    cintopt = lib.c_null_ptr()

    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
        ctypes.byref(neighbor_list))

    nlopt = None

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

    # nimgs determines the supercell size
    rcut = estimate_rcut(cell, precision)
    nimgs = cell.get_bounding_sphere(rcut)
    return nimgs

def _estimate_rcut(alpha, l, c, precision=INTEGRAL_PRECISION):
    '''rcut based on the overlap integrals. This estimation is too conservative
    in many cases. A possible replacement can be the value of the basis
    function at rcut ~ c*r^(l+2)*exp(-alpha*r^2) < precision'''
    theta = alpha * .5
    a1 = (alpha * 2)**-.5
    norm_ang = (2*l+1)/(4*np.pi)
    fac = 2*np.pi*c**2*norm_ang / theta / precision
    r0 = 20
    # The estimation is enough for overlap. Errors are slightly larger for
    # kinetic operator. The error in kinetic integrals may be dominant.
    # For kinetic operator, basis becomes 2*a*r*|orig-basis>.
    # A penalty term 4*a^2*r^2 is included in the estimation.
    fac *= 4*alpha**2
    r0 = (np.log(fac * r0 * (r0*.5+a1)**(2*l+2) + 1.) / theta)**.5
    r0 = (np.log(fac * r0 * (r0*.5+a1)**(2*l+2) + 1.) / theta)**.5
    return r0

def bas_rcut(cell, bas_id, precision=None):
    r'''Estimate the largest distance between the function and its image to
    reach the precision in overlap

    precision ~ \int g(r-0) g(r-Rcut)
    '''
    if precision is None:
        precision = cell.precision
    l = cell.bas_angular(bas_id)
    es = cell.bas_exp(bas_id)
    cs = abs(cell._libcint_ctr_coeff(bas_id)).max(axis=1)
    rcut = _estimate_rcut(es, l, cs, precision)
    return rcut.max()

def estimate_rcut(cell, precision=None):
    '''Lattice-sum cutoff for the entire system'''
    if cell.nbas == 0:
        return 0.01
    if precision is None:
        precision = cell.precision
    if cell.use_loose_rcut:
        return cell.rcut_by_shells(precision).max()

    exps, cs = _extract_pgto_params(cell, 'min')
    ls = cell._bas[:,mole.ANG_OF]
    rcut = _estimate_rcut(exps, ls, cs, precision)
    return rcut.max()

def _estimate_ke_cutoff(alpha, l, c, precision=INTEGRAL_PRECISION, omega=0):
    '''Energy cutoff estimation for nuclear attraction integrals'''
    norm_ang = (2*l+1)/(4*np.pi)
    fac = 32*np.pi**2*(2*np.pi)**1.5 * c**2*norm_ang / (2*alpha)**(2*l+.5) / precision
    Ecut = 20.
    if omega <= 0:
        Ecut = np.log(fac * (Ecut*2)**(l-.5) + 1.) * 4*alpha
        Ecut = np.log(fac * (Ecut*2)**(l-.5) + 1.) * 4*alpha
    else:
        theta = 1./(1./(4*alpha) + 1./(2*omega**2))
        Ecut = np.log(fac * (Ecut*2)**(l-.5) + 1.) * theta
        Ecut = np.log(fac * (Ecut*2)**(l-.5) + 1.) * theta
    return Ecut

def estimate_ke_cutoff(cell, precision=None):
    '''Energy cutoff estimation for nuclear attraction integrals'''
    if cell.nbas == 0:
        return 0.
    if precision is None:
        precision = cell.precision
    #precision /= cell.atom_charges().sum()
    exps, cs = _extract_pgto_params(cell, 'max')
    ls = cell._bas[:,mole.ANG_OF]
    Ecut = _estimate_ke_cutoff(exps, ls, cs, precision, cell.omega)
    return Ecut.max()

def _extract_pgto_params(cell, op='min'):
    '''A helper function for estimate_xxx function'''
    es = []
    cs = []
    if op == 'min':
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = cell._libcint_ctr_coeff(i)
            idx = e.argmin()
            es.append(e[idx])
            cs.append(abs(c[idx]).max())
    else:
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = cell._libcint_ctr_coeff(i)
            idx = e.argmax()
            es.append(e[idx])
            cs.append(abs(c[idx]).max())
    return np.array(es), np.array(cs)

def error_for_ke_cutoff(cell, ke_cutoff, omega=None):
    '''Error estimation based on nuclear attraction integrals'''
    if omega is None:
        omega = cell.omega
    exps, cs = _extract_pgto_params(cell, 'max')
    ls = cell._bas[:,mole.ANG_OF]
    norm_ang = (2*ls+1)/(4*np.pi)
    fac = 32*np.pi**2*(2*np.pi)**1.5 * cs**2*norm_ang / (2*exps)**(2*ls+.5)
    if omega <= 0:
        err = fac * (2*ke_cutoff)**(ls-.5) * np.exp(-ke_cutoff/(4*exps))
    else:
        theta = 1./(1./(4*exps) + 1./(2*omega**2))
        err = fac * (2*ke_cutoff)**(ls-.5) * np.exp(-ke_cutoff/theta)
    return err.max()

def get_bounding_sphere(cell, rcut):
    '''Finds all the lattice points within a sphere of radius rcut.

    Defines a parallelepiped given by -N_x <= n_x <= N_x, with x in [1,3]
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
        nimgs[i] = 0
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
    return get_Gv_weights(cell, mesh, **kwargs)[0]

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

    #:Gv = np.dot(lib.cartesian_prod(Gvbase), b)
    # NOTE mesh can be different from the input mesh
    mesh = np.asarray((len(rx),len(ry),len(rz)), dtype=np.int32)
    Gv = np.empty((*mesh,3), order='C', dtype=float)
    b = np.asarray(b, order='C')
    rx = np.asarray(rx, order='C')
    ry = np.asarray(ry, order='C')
    rz = np.asarray(rz, order='C')
    libpbc.get_Gv(
        Gv.ctypes.data_as(ctypes.c_void_p),
        rx.ctypes.data_as(ctypes.c_void_p),
        ry.ctypes.data_as(ctypes.c_void_p),
        rz.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
    )
    Gv = Gv.reshape(-1, 3)

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

def get_SI(cell, Gv=None, mesh=None, atmlst=None):
    '''Calculate the structure factor (0D, 1D, 2D, 3D) for all atoms; see MH (3.34).

    Args:
        cell : instance of :class:`Cell`

        Gv : (N,3) array
            G vectors

        atmlst : list of ints, optional
            Indices of atoms for which the structure factors are computed.

    Returns:
        SI : (natm, ngrids) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    '''
    coords = cell.atom_coords()
    if atmlst is not None:
        coords = coords[np.asarray(atmlst)]
    if Gv is None:
        if mesh is None:
            mesh = cell.mesh
        basex, basey, basez = cell.get_Gv_weights(mesh)[1]
        b = cell.reciprocal_vectors()
        rb = np.dot(coords, b.T)
        SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
        SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
        SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
        SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
        natm = coords.shape[0]
        SI = SI.reshape(natm, -1)
    else:
        SI = np.exp(-1j*np.dot(coords, Gv.T))
    return SI

def get_ewald_params(cell, precision=None, mesh=None):
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

    if precision is None:
        precision = cell.precision

    if (cell.dimension < 2 or
          (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        # Non-uniform PW grids are used for low-dimensional ewald summation.  The cutoff
        # estimation for long range part based on exp(G^2/(4*eta^2)) does not work for
        # non-uniform grids.  Smooth model density is preferred.
        ew_cut = cell.rcut
        ew_eta = np.sqrt(max(np.log(4*np.pi*ew_cut**2/precision)/ew_cut**2, .1))
    elif cell.dimension == 2:
        a = cell.lattice_vectors()
        ew_cut = a[2,2] / 2
        # ewovrl ~ erfc(eta*rcut) / rcut ~ e^{(-eta**2 rcut*2)} < precision
        log_precision = np.log(precision / (cell.atom_charges().sum()*16*np.pi**2))
        ew_eta = (-log_precision)**.5 / ew_cut
    else:  # dimension == 3
        ew_eta = 1./cell.vol**(1./6)
        ew_cut = _estimate_rcut(ew_eta**2, 0, 1., precision)
    return ew_eta, ew_cut

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

    if cell.dimension == 3 and cell.use_particle_mesh_ewald:
        from pyscf.pbc.gto import ewald_methods
        return ewald_methods.particle_mesh_ewald(cell, ew_eta, ew_cut)

    chargs = cell.atom_charges()

    if ew_eta is None or ew_cut is None:
        ew_eta, ew_cut = cell.get_ewald_params()
    log_precision = np.log(cell.precision / (chargs.sum()*16*np.pi**2))
    ke_cutoff = -2*ew_eta**2*log_precision
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    logger.debug1(cell, 'mesh for ewald %s', mesh)

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

    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        # TODO: truncated Coulomb for 0D. The non-uniform grids for inf-vacuum
        # have relatively large error
        coulG = 4*np.pi / absG2
        coulG *= weights

        #:ZSI = np.einsum('i,ij->j', chargs, cell.get_SI(Gv))
        ngrids = len(Gv)
        ZSI = np.empty((ngrids,), dtype=np.complex128)
        mem_avail = cell.max_memory - lib.current_memory()[0]
        blksize = int((mem_avail*1e6 - cell.natm*24)/((3+cell.natm*2)*8))
        blksize = min(ngrids, max(mesh[2], blksize))
        for ig0, ig1 in lib.prange(0, ngrids, blksize):
            np.einsum('i,ij->j', chargs, cell.get_SI(Gv[ig0:ig1]), out=ZSI[ig0:ig1])

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
              scaled_center=None,
              space_group_symmetry=False, time_reversal_symmetry=False,
              **kwargs):
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
        space_group_symmetry : bool
            Whether to consider space group symmetry
        time_reversal_symmetry : bool
            Whether to consider time reversal symmetry

    Returns:
        kpts in absolute value (unit 1/Bohr).  Gamma point is placed at the
        first place in the k-points list;
        instance of :class:`KPoints` if k-point symmetry is considered

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
    if space_group_symmetry or time_reversal_symmetry:
        from pyscf.pbc.lib import kpts as libkpts
        if space_group_symmetry and not cell.space_group_symmetry:
            raise RuntimeError('Using k-point symmetry now requires cell '
                               'to be built with space group symmetry info:\n'
                               'cell.space_group_symmetry = True\n'
                               'cell.symmorphic = False\n'
                               'cell.build()')
        kpts = libkpts.make_kpts(cell, kpts, space_group_symmetry,
                                 time_reversal_symmetry)
    return kpts

def get_uniform_grids(cell, mesh=None, wrap_around=True):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    if mesh is None: mesh = cell.mesh

    if wrap_around:
        # wrap the coordinates around the origin. If coordinates are generated
        # inside the primitive cell without wrap-around, an extra layer would be
        # needed in function get_lattice_Ls for 2D calculations.
        qv = lib.cartesian_prod([np.fft.fftfreq(x) for x in mesh])
        coords = np.dot(qv, cell.lattice_vectors())
    else:
        mesh = np.asarray(mesh, dtype=np.double)
        qv = lib.cartesian_prod([np.arange(x) for x in mesh])
        a_frac = np.einsum('i,ij->ij', 1./mesh, cell.lattice_vectors())
        coords = np.dot(qv, a_frac)
    return coords
gen_uniform_grids = get_uniform_grids

def _split_basis(cell, delimiter=EXP_DELIMITER):
    '''
    Split the contracted basis to small segmant.  The new basis has more
    shells.  Each shell has less primitive basis and thus is more local.
    '''
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

    pcell = cell.copy(deep=False)
    pcell._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pcell._env = _env
    return pcell, scipy.linalg.block_diag(*contr_coeff)

def tot_electrons(cell, nkpts=1):
    '''Total number of electrons
    '''
    if cell._nelectron is None:
        nelectron = cell.atom_charges().sum() * nkpts - cell.charge
    else: # Custom cell.nelectron stands for num. electrons per cell
        nelectron = cell._nelectron * nkpts
    # Round off to the nearest integer
    nelectron = int(nelectron+0.5)
    return nelectron

def _mesh_inf_vaccum(cell):
    #prec ~ exp(-0.436392335*mesh -2.99944305)*nelec
    meshz = (np.log(cell.nelectron/cell.precision)-2.99944305)/0.436392335
    # meshz has to be even number due to the symmetry on z+ and z-
    return int(meshz*.5 + .999) * 2

def pgf_rcut(l, alpha, coeff, precision=INTEGRAL_PRECISION,
             rcut=0, max_cycle=RCUT_MAX_CYCLE, eps=RCUT_EPS):
    '''Estimate the cutoff radii of primitive Gaussian functions
    based on their values in real space:
    `c*rcut^(l+2)*exp(-alpha*rcut^2) ~ precision`.
    '''
    c = np.log(coeff / precision)

    rmin = np.sqrt(.5 * (l+2) / alpha) * 2
    eps = np.minimum(rmin/10, eps)
    rcut = np.maximum(rcut, rmin+eps)
    for i in range(max_cycle):
        rcut_last = rcut
        rcut = np.sqrt(((l+2) * np.log(rcut) + c) / alpha)
        if np.all(abs(rcut - rcut_last) < eps):
            return rcut
    warnings.warn(f'cell.pgf_rcut failed to converge in {max_cycle} cycles.')
    return rcut

def rcut_by_shells(cell, precision=None, rcut=0,
                   return_pgf_radius=False):
    '''Compute shell and primitive gaussian function radii.
    '''
    # TODO the internal implementation loops over all shells,
    # which can be optimized to loop over atom types.
    if precision is None:
        precision = cell.precision

    bas = np.asarray(cell._bas, order='C')
    env = np.asarray(cell._env, order='C')
    nbas = len(bas)
    shell_radius = np.empty((nbas,), order='C', dtype=float)
    if return_pgf_radius:
        nprim = bas[:,mole.NPRIM_OF].max()
        # be careful that the unused memory blocks are not initialized
        pgf_radius = np.empty((nbas,nprim), order='C', dtype=np.double)
        ptr_pgf_radius = lib.ndarray_pointer_2d(pgf_radius).ctypes
    else:
        ptr_pgf_radius = lib.c_null_ptr()
    libpbc.rcut_by_shells(
        shell_radius.ctypes.data_as(ctypes.c_void_p),
        ptr_pgf_radius,
        bas.ctypes.data_as(ctypes.c_void_p),
        env.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        ctypes.c_double(rcut),
        ctypes.c_double(precision),
    )
    if return_pgf_radius:
        return shell_radius, pgf_radius
    return shell_radius

def tostring(cell, format='poscar'):
    '''Convert cell geometry to a string of the required format.

    Supported output formats:
        | poscar: VASP POSCAR
        | xyz: Extended XYZ with Lattice information
    '''
    format = format.lower()
    output = []
    atmfmt = '%17.8f %17.8f %17.8f'
    if format == 'poscar' or format == 'vasp' or format == 'xyz':
        lattice_vectors = cell.lattice_vectors() * param.BOHR
        coords = cell.atom_coords() * param.BOHR
        if format == 'poscar' or format == 'vasp':
            output.append('Written by PySCF, units are A')
            output.append('1.0')
            for lattice_vector in lattice_vectors:
                ax, ay, az = lattice_vector
                output.append(atmfmt % (ax, ay, az))
            unique_atoms = dict()
            for atom in cell.elements:
                if atom not in unique_atoms:
                    unique_atoms[atom] = 1
                else:
                    unique_atoms[atom] += 1
            output.append(' '.join(unique_atoms))
            output.append(' '.join(str(count) for count in unique_atoms.values()))
            output.append('Cartesian')
            for atom_type in unique_atoms:
                for atom, coord in zip(cell.elements, coords):
                    if atom == atom_type:
                        x, y, z = coord
                        output.append(atmfmt % (x, y, z))
            return '\n'.join(output)
        elif format == 'xyz':
            output.append('%d' % cell.natm)
            output.append('Lattice="'+' '.join(f'{ax:17.8f}' for ax in lattice_vectors.ravel())
                +'" Properties=species:S:1:pos:R:3')
            for i in range(cell.natm):
                symb = cell.atom_pure_symbol(i)
                x, y, z = coords[i]
                output.append(('%-4s ' + atmfmt) %
                              (symb, x, y, z))
            return '\n'.join(output)
    else:
        raise NotImplementedError(f'format={format}')

def tofile(cell, filename, format=None):
    if format is None:  # Guess format based on filename
        if filename.lower() == 'poscar':
            format = 'poscar'
        else:
            format = os.path.splitext(filename)[1][1:]
    string = tostring(cell, format)
    with open(filename,  'w', encoding='utf-8') as f:
        f.write(string)
        f.write('\n')
    return string

def fromfile(filename, format=None):
    '''Read cell geometry from a file
    (in testing)

    Supported formats:
        | poscar: VASP POSCAR file format
        | xyz: Extended XYZ with Lattice information
    '''
    if format is None:  # Guess format based on filename
        if filename.lower() == 'poscar':
            format = 'poscar'
        else:
            format = os.path.splitext(filename)[1][1:].lower()
        if format not in ('poscar', 'vasp', 'xyz'):
            format = 'raw'
    with open(filename, 'r') as f:
        return fromstring(f.read(), format)

def fromstring(string, format='poscar'):
    '''Convert the string of the specified format to internal format
    (in testing)

    Supported formats:
        | poscar: VASP POSCAR file format
        | xyz: Extended XYZ with Lattice information

    Returns:
        a: Lattice vectors
        atom: Atomic elements and xyz coordinates
    '''
    format = format.lower()
    if format == 'poscar' or format == 'vasp':
        lines = string.splitlines()
        scale = float(lines[1])
        a = lines[2:5]
        lattice_vectors = np.array([np.fromstring(ax, sep=' ') for ax in a])
        lattice_vectors *= scale
        a = []
        for i in range(3):
            a.append(' '.join(str(ax) for ax in lattice_vectors[i]))
        atom_position_type = lines[7].strip()
        unique_atoms = dict()
        natm = 0
        for atom, count in zip(lines[5].split(), lines[6].split()):
            unique_atoms[atom] = int(count)
            natm += int(count)
        atoms = []
        start = 8
        for atom_type in unique_atoms:
            end = start + unique_atoms[atom_type]
            for line in lines[start:end]:
                coords = np.fromstring(line, sep=' ')
                if atom_position_type.lower() == 'cartesian':
                    x, y, z = coords * scale
                elif atom_position_type.lower() == 'direct':
                    x, y, z = np.dot(coords, lattice_vectors)
                else:
                    raise RuntimeError('Error reading VASP geometry due to '
                        f'atom position type "{atom_position_type}". Atom '
                        'positions must be Direct or Cartesian.')
                atoms.append('%s %17.8f %17.8f %17.8f'
                    % (atom_type, x, y, z))
            start = end
        return '\n'.join(a), '\n'.join(atoms)
    elif format == 'xyz':
        lines = string.splitlines()
        natm = int(lines[0])
        lattice_vectors = lines[1].split('Lattice=')[1].split('"')[1].split()
        a = []
        for i in range(3):
            a.append(" ".join(lattice_vectors[3*i:3*i+3]))
        return '\n'.join(a), '\n'.join(lines[2:natm+2])
    elif format == 'raw':
        lines = string.splitlines()
        return '\n'.join(lines[:3]), '\n'.join(lines[4:])
    else:
        raise NotImplementedError


class Cell(mole.MoleBase):
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
        use_loose_rcut : bool
            If set to True, a loose `rcut` determined by shell radius is used,
            which is usually accurate enough for pure DFT calculations;
            otherwise, a tight `rcut` determined by overlap integral is used.
            Default value is False. Has no effect if `rcut` is set manually.
        use_particle_mesh_ewald : bool
            If set to True, use particle-mesh Ewald to compute the nuclear repulsion.
            Default value is False, meaning to use classical Ewald summation.
        space_group_symmetry : bool
            Whether to consider space group symmetry. Default is False.
        symmorphic : bool
            Whether the lattice is symmorphic. If set to True, even if the
            lattice is non-symmorphic, only symmorphic space group symmetry
            will be considered. Default is False, meaning the space group is
            determined by the lattice symmetry to be symmorphic or non-symmorphic.
        lattice_symmetry : None or :class:`pbc.symm.Symmetry` instance
            The object containing the lattice symmetry information. Default is None.

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

    _keys = {
        'precision', 'exp_to_discard',
        'a', 'ke_cutoff', 'pseudo', 'fractional', 'dimension', 'low_dim_ft_type',
        'space_group_symmetry', 'symmorphic', 'lattice_symmetry', 'mesh', 'rcut',
        'use_loose_rcut', 'use_particle_mesh_ewald',
    }

    tostring = tostring
    tofile = tofile

    def __init__(self, **kwargs):
        mole.MoleBase.__init__(self)
        self.a = None # lattice vectors, (a1,a2,a3)
        # if set, defines a spherical cutoff
        # of fourier components, with .5 * G**2 < ke_cutoff
        self.ke_cutoff = None

        self.fractional = False
        self.dimension = 3
        # TODO: Simple hack for now; the implementation of ewald depends on the
        #       density-fitting class.  This determines how the ewald produces
        #       its energy.
        self.low_dim_ft_type = None
        self.use_loose_rcut = False
        self.use_particle_mesh_ewald = False
        self.space_group_symmetry = False
        self.symmorphic = False
        self.lattice_symmetry = None

##################################################
# These attributes are initialized by build function if not specified
        self.mesh = None
        self.rcut = None
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fromstring(self, string, format='poscar'):
        '''Update the Cell object based on the input geometry string'''
        a, atom = fromstring(string, format)
        self.a = a
        self.set_geom_(atom, unit='Angstrom', inplace=True)
        return self

    def fromfile(self, filename, format=None):
        '''Update the Cell object based on the input geometry file'''
        a, atom = fromfile(filename, format)
        self.a = a
        self.set_geom_(atom, unit='Angstrom', inplace=True)
        return self

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
        warnings.warn("cell.ew_eta is deprecated. Use function get_ewald_params instead.")
        return self.get_ewald_params()[0]

    @property
    def ew_cut(self):
        warnings.warn("cell.ew_cut is deprecated. Use function get_ewald_params instead.")
        return self.get_ewald_params()[1]

    @ew_eta.setter
    def ew_eta(self, val):
        warnings.warn("ew_eta is no longer stored in the cell object. Setting it has no effect")

    @ew_cut.setter
    def ew_cut(self, val):
        warnings.warn("ew_cut is no longer stored in the cell object. Setting it has no effect")

    @property
    def nelec(self):
        ne = self.nelectron
        nalpha = (ne + self.spin) // 2
        nbeta = nalpha - self.spin
        # nelec method defined in Mole class raises error when the attributes .spin
        # and .nelectron are inconsistent.  In PBC, when the system has even number of
        # k-points, it is valid that .spin is odd while .nelectron is even.
        if nalpha + nbeta != ne:
            warnings.warn('Electron number %d and spin %d are not consistent '
                          'in cell\n' % (ne, self.spin))
        return nalpha, nbeta

    def __getattr__(self, key):
        '''To support accessing methods (cell.HF, cell.KKS, cell.KUCCSD, ...)
        from Cell object.
        '''
        if key[0] == '_':  # Skip private attributes and Python builtins
            # https://bugs.python.org/issue45985
            # https://github.com/python/cpython/issues/103936
            # @property and __getattr__ conflicts. As a temporary fix, call
            # object.__getattribute__ method to re-raise AttributeError
            return object.__getattribute__(self, key)

        # Import all available modules. Some methods are registered to other
        # classes/modules when importing modules in __all__.
        from pyscf.pbc import __all__  # noqa
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
            elif 'CI' in key or 'CC' in key or 'MP' in key:
                mf = scf.KHF(self)
            else:
                return object.__getattribute__(self, key)
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
            elif 'CI' in key or 'CC' in key or 'MP' in key:
                mf = scf.HF(self)
            else:
                return object.__getattribute__(self, key)

        method = getattr(mf, key)

        if self.nelectron != 0:
            mf.run()
        return method

    tot_electrons = tot_electrons

    def _build_symmetry(self, kpts=None, **kwargs):
        '''Construct symmetry adapted crystalline atomic orbitals
        '''
        from pyscf.pbc.lib.kpts import KPoints
        from pyscf.pbc.symm.basis import symm_adapted_basis
        if kpts is None:
            return mole.Mole._build_symmetry(self)
        elif isinstance(kpts, KPoints):
            self.symm_orb, self.irrep_id = symm_adapted_basis(self, kpts)
            return self
        else:
            raise RuntimeError('Symmetry information not found in kpts. '
                               'kpts needs to be initialized as a KPoints object.')

    def symmetrize_mesh(self, mesh=None):
        if mesh is None:
            mesh = self.mesh
        if not self.space_group_symmetry:
            return mesh

        _, mesh1 = self.lattice_symmetry.check_mesh_symmetry(mesh=mesh,
                                                             return_mesh=True)
        if np.prod(mesh1) != np.prod(mesh):
            logger.debug(self, 'mesh %s is symmetrized as %s', mesh, mesh1)
        m1size = np.prod(mesh1)
        msize = np.prod(mesh)
        if m1size > 8 * msize and m1size > 1000 + msize:
            wstr = ('''WARNING!
  Symmetrization significantly increased the mesh size,
  from {} to {}. This might indicate a nearly symmetric input
  structure, and it might cause memory issues. Consider symmetrizing your
  structure, increasing the symmetry tolerance `pbc_symm_space_group_symprec`,
  or turning off symmetry.\n\n''')
            wstr = wstr.format(mesh, mesh1)
            sys.stderr.write(wstr)
        return mesh1

    def build_lattice_symmetry(self, check_mesh_symmetry=True):
        '''Build cell.lattice_symmetry object.

        Kwargs:
            check_mesh_symmetry : bool
                For nonsymmorphic symmetry groups, `cell.mesh` may have
                lower symmetry than the lattice. In this case, if
                `check_mesh_symmetry` is `True`, the lower symmetry group will
                be used. Otherwise, if `check_mesh_symmetry` is `False`,
                the mesh grid will be modified to satisfy the higher symmetry.
                Default value is `True`.

        Note:
            This function modifies the attributes of `cell`.
        '''
        from pyscf.pbc.symm import Symmetry
        self.lattice_symmetry = Symmetry(self).build(
                                    space_group_symmetry=True,
                                    symmorphic=self.symmorphic,
                                    check_mesh_symmetry=check_mesh_symmetry)
        if not check_mesh_symmetry:
            _mesh_from_build = self._mesh_from_build
            self.mesh = self.symmetrize_mesh()
            self._mesh_from_build = _mesh_from_build
        return self

    @lib.with_doc(mole.format_atom.__doc__)
    def format_atom(self, atoms, origin=0, axes=None,
                    unit=getattr(__config__, 'UNIT', 'Ang')):
        if not self.fractional:
            return mole.format_atom(atoms, origin, axes, unit)
        _atoms = mole.format_atom(atoms, origin, axes, unit=1.)
        _a = self.lattice_vectors()
        c = np.array([a[1] for a in _atoms]).dot(_a)
        return [(a[0], r) for a, r in zip(_atoms, c.tolist())]

#Note: Exculde dump_input, parse_arg, basis from kwargs to avoid parsing twice
    def build(self, dump_input=True, parse_arg=mole.ARGPARSE,
              a=None, mesh=None, ke_cutoff=None, precision=None, nimgs=None,
              h=None, dimension=None, rcut= None, low_dim_ft_type=None,
              space_group_symmetry=None, symmorphic=None,
              use_loose_rcut=None, use_particle_mesh_ewald=None,
              fractional=None, *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.
        Whenever you change the value of the attributes of :class:`Cell`,
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            a : (3,3) ndarray
                The real-space cell lattice vectors. Each row represents
                a lattice vector.
            fractional : bool
                Whether the atom postions are specified in fractional coordinates.
                The default value is False, which means the coordinates are
                interpreted as Cartesian coordinate.
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
            space_group_symmetry : bool
                Whether to consider space group symmetry. Default is False.
            symmorphic : bool
                Whether the lattice is symmorphic. If set to True, even if the
                lattice is non-symmorphic, only symmorphic space group symmetry
                will be considered. Default is False, meaning the space group is
                determined by the lattice symmetry to be symmorphic or non-symmorphic.
        '''
        if h is not None: self.h = h
        if a is not None: self.a = a
        if mesh is not None: self.mesh = mesh
        if nimgs is not None: self.nimgs = nimgs
        if dimension is not None: self.dimension = dimension
        if fractional is not None: self.fractional = fractional
        if precision is not None: self.precision = precision
        if rcut is not None: self.rcut = rcut
        if ke_cutoff is not None: self.ke_cutoff = ke_cutoff
        if low_dim_ft_type is not None: self.low_dim_ft_type = low_dim_ft_type
        if use_loose_rcut is not None:
            self.use_loose_rcut = use_loose_rcut
        if use_particle_mesh_ewald is not None:
            self.use_particle_mesh_ewald = use_particle_mesh_ewald
        if space_group_symmetry is not None:
            self.space_group_symmetry = space_group_symmetry
        if symmorphic is not None:
            self.symmorphic = symmorphic

        if self.a is None:
            raise RuntimeError('Lattice parameters not specified')

        _built = self._built
        mole.MoleBase.build(self, False, parse_arg, *args, **kwargs)

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
            _env = self._env.copy()
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
                        _env[pe:pe+nprim] = es
                        cs = mole._nomalize_contracted_ao(l, es, cs)
                        _env[ptr:ptr+nprim*nc] = cs.T.reshape(-1)
                if nprim > 0:
                    steep_shls.append(ib)
            self._env = _env
            self._bas = np.asarray(self._bas[steep_shls], order='C')
            logger.info(self, 'Discarded %d diffused primitive functions, '
                        '%d contracted functions', nprim_drop, nctr_drop)
            #logger.debug1(self, 'Old shells %s', steep_shls)

        if self.rcut is None or self._rcut_from_build:
            self._rcut = estimate_rcut(self, self.precision)
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

        if self.space_group_symmetry:
            _check_mesh_symm = not self._mesh_from_build
            self.build_lattice_symmetry(check_mesh_symmetry=_check_mesh_symm)

        if dump_input and not _built and self.verbose > logger.NOTE:
            self.dump_input()
            logger.info(self, 'lattice vectors  a1 [%.9f, %.9f, %.9f]', *_a[0])
            logger.info(self, '                 a2 [%.9f, %.9f, %.9f]', *_a[1])
            logger.info(self, '                 a3 [%.9f, %.9f, %.9f]', *_a[2])
            logger.info(self, 'dimension = %s', self.dimension)
            logger.info(self, 'low_dim_ft_type = %s', self.low_dim_ft_type)
            logger.info(self, 'Cell volume = %g', self.vol)
            # Check atoms coordinates
            if self.dimension > 0 and self.natm > 0:
                scaled_atom_coords = self.get_scaled_atom_coords(_a)
                atom_boundary_max = scaled_atom_coords[:,:self.dimension].max(axis=0)
                atom_boundary_min = scaled_atom_coords[:,:self.dimension].min(axis=0)
                if (np.any(atom_boundary_max > 1) or np.any(atom_boundary_min < -1)):
                    logger.warn(self, 'Atoms found out of the primitive cell.')

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
            if self.space_group_symmetry:
                self.lattice_symmetry.dump_info()
        return self
    kernel = build

    @property
    def h(self):
        return np.asarray(self.a).T
    @h.setter
    def h(self, x):
        warnings.warn('cell.h is deprecated.  It is replaced by the '
                      '(row-based) lattice vectors cell.a:  cell.a = cell.h.T\n')
        if isinstance(x, str):
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
        rcut_guess = estimate_rcut(self, self.precision)
        if self.rcut > rcut_guess*1.5:
            msg = ('.nimgs is a deprecated attribute.  It is replaced by .rcut '
                   'attribute for lattice sum cutoff radius.  The given nimgs '
                   '%s is far over the estimated cutoff radius %s. ' %
                   (x, rcut_guess))
            warnings.warn(msg)

    def lattice_vectors(self):
        '''Convert the primitive lattice vectors.

        Return 3x3 array in which each row represents one direction of the
        lattice vectors (unit in Bohr)
        '''
        if isinstance(self.a, str):
            a = self.a.replace(';',' ').replace(',',' ').replace('\n',' ')
            a = np.asarray([float(x) for x in a.split()]).reshape(3,3)
        else:
            a = np.asarray(self.a, dtype=np.double).reshape(3,3)
        if isinstance(self.unit, str):
            if is_au(self.unit):
                return a
            else:
                return a/param.BOHR
        else:
            return a/self.unit

    def get_scaled_atom_coords(self, a=None):
        ''' Get scaled atomic coordinates.
        '''
        if a is None:
            a = self.lattice_vectors()
        return np.dot(self.atom_coords(), np.linalg.inv(a))

    def reciprocal_vectors(self, norm_to=2*np.pi):
        r'''
        .. math::

            \begin{align}
            \mathbf{b_1} &= 2\pi \frac{\mathbf{a_2} \times \mathbf{a_3}}{\mathbf{a_1} \cdot (\mathbf{a_2} \times \mathbf{a_3})} \\
            \mathbf{b_2} &= 2\pi \frac{\mathbf{a_3} \times \mathbf{a_1}}{\mathbf{a_2} \cdot (\mathbf{a_3} \times \mathbf{a_1})} \\
            \mathbf{b_3} &= 2\pi \frac{\mathbf{a_1} \times \mathbf{a_2}}{\mathbf{a_3} \cdot (\mathbf{a_1} \times \mathbf{a_2})}
            \end{align}

        '''  # noqa: E501
        a = self.lattice_vectors()
        if self.dimension == 1:
            assert (abs(np.dot(a[0], a[1])) < 1e-9 and
                   abs(np.dot(a[0], a[2])) < 1e-9 and
                   abs(np.dot(a[1], a[2])) < 1e-9)
        elif self.dimension == 2:
            assert (abs(np.dot(a[0], a[2])) < 1e-9 and
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

    def get_scaled_kpts(self, abs_kpts, kpts_in_ibz=True):
        '''Get scaled k-points, given absolute k-points in 1/Bohr.

        Args:
            abs_kpts : (nkpts, 3) ndarray of floats or :class:`KPoints` object

            kpts_in_ibz : bool
                If True, return k-points in IBZ; otherwise, return k-points in BZ.
                Default value is True. This has effects only if abs_kpts is a
                :class:`KPoints` object

        Returns:
            scaled_kpts : (nkpts, 3) ndarray of floats
        '''
        from pyscf.pbc.lib.kpts import KPoints
        if isinstance(abs_kpts, KPoints):
            if kpts_in_ibz:
                return abs_kpts.kpts_scaled_ibz
            else:
                return abs_kpts.kpts_scaled
        return 1./(2*np.pi)*np.dot(abs_kpts, self.lattice_vectors().T)

    def cutoff_to_mesh(self, ke_cutoff):
        '''Convert KE cutoff to FFT-mesh

        Args:
            ke_cutoff : float
                KE energy cutoff in a.u.

        Returns:
            mesh : (3,) array
        '''
        a = self.lattice_vectors()
        dim = self.dimension
        mesh = pbctools.cutoff_to_mesh(a, ke_cutoff)
        if dim < 2 or (dim == 2 and self.low_dim_ft_type == 'inf_vacuum'):
            mesh[dim:] = self.mesh[dim:]
        return mesh

    make_kpts = get_kpts = make_kpts

    pack = pack

    @classmethod
    @lib.with_doc(unpack.__doc__)
    def unpack(cls, moldic):
        return unpack(moldic)

    @lib.with_doc(unpack.__doc__)
    def unpack_(self, moldic):
        self.__dict__.update(moldic)
        return self

    dumps = dumps

    @classmethod
    @lib.with_doc(loads.__doc__)
    def loads(cls, molstr):
        return loads(molstr)

    @lib.with_doc(unpack.__doc__)
    def loads_(self, molstr):
        self.__dict__.update(loads(molstr).__dict__)
        return self

    bas_rcut = bas_rcut
    rcut_by_shells = rcut_by_shells

    get_lattice_Ls = pbctools.get_lattice_Ls

    get_nimgs = get_nimgs

    get_ewald_params = get_ewald_params

    get_bounding_sphere = get_bounding_sphere

    get_Gv = get_Gv
    get_Gv_weights = get_Gv_weights

    get_SI = get_SI

    ewald = ewald
    energy_nuc = get_enuc = ewald

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
        if self.use_loose_rcut:
            return _intor_cross_screened(
                            intor, self, self, comp, hermi, kpts, kpt,
                            shls_slice, **kwargs)
        return intor_cross(intor, self, self, comp, hermi, kpts, kpt,
                           shls_slice, **kwargs)

    pbc_eval_ao = pbc_eval_gto = pbc_eval_gto

    @lib.with_doc(pbc_eval_gto.__doc__)
    def eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None,
                 shls_slice=None, non0tab=None, ao_loc=None, cutoff=None,
                 out=None):
        if eval_name[:3] == 'PBC':
            return self.pbc_eval_gto(eval_name, coords, comp, kpts, kpt,
                                     shls_slice, non0tab, ao_loc, cutoff, out)
        else:
            return mole.eval_gto(self, eval_name, coords, comp,
                                 shls_slice, non0tab, ao_loc, cutoff, out)
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
        mol = self.view(mole.Mole)
        del mol.a
        mol.__dict__.pop('fractional', None)
        mol.__dict__.pop('ke_cutoff', None)
        mol.__dict__.pop('_mesh', None)
        mol.__dict__.pop('_rcut', None)
        mol.__dict__.pop('dimension', None)
        mol.__dict__.pop('low_dim_ft_type', None)
        mol.enuc = None #reset nuclear energy
        if mol.symmetry:
            mol._build_symmetry()
        return mol

del (INTEGRAL_PRECISION, WRAP_AROUND, WITH_GAMMA, EXP_DELIMITER)
