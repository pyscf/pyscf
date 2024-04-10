#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy
from pyscf import __config__
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import gamma_point

PP_WITH_RHO_CORE = getattr(__config__, 'pbc_dft_multigrid_pp_with_rho_core', True)

libpbc = lib.load_library('libpbc')
libdft = lib.load_library('libdft')

def make_rho_core(cell, mesh=None, precision=None, atm_id=None):
    if mesh is None:
        mesh = cell.mesh
    fakecell, max_radius = fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = numpy.asarray(cell.lattice_vectors(), order='C', dtype=float)
    if abs(a - numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
        raise NotImplementedError
    eval_fn = 'make_rho_lda' + lattice_type

    b = numpy.asarray(numpy.linalg.inv(a.T), order='C', dtype=float)
    mesh = numpy.asarray(mesh, order='C', dtype=numpy.int32)
    rho_core = numpy.zeros((numpy.prod(mesh),), order='C', dtype=float)
    drv = getattr(libdft, 'build_core_density', None)
    try:
        drv(getattr(libdft, eval_fn),
            rho_core.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p),
            mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.dimension),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(max_radius))
    except Exception as e:
        raise RuntimeError("Failed to compute rho_core. %s" % e)
    return rho_core


def _get_pp_without_erf(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    vpp = pp_int.get_pp_loc_part2(cell, kpts_lst)
    vppnl = pp_int.get_pp_nl(cell, kpts_lst)

    for k, kpt in enumerate(kpts_lst):
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl[k].real
        else:
            vpp[k] += vppnl[k]
    vppnl = None

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)


def get_pp_loc_part1_gs(cell, Gv):
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = numpy.einsum('ix,ix->i', Gv, Gv)
    G0idx = numpy.where(G2==0)[0]
    ngrid = len(G2)
    Gv = numpy.asarray(Gv, order='C', dtype=numpy.double)
    coulG = numpy.asarray(coulG, order='C', dtype=numpy.double)
    G2 = numpy.asarray(G2, order='C', dtype=numpy.double)

    coords = cell.atom_coords()
    coords = numpy.asarray(coords, order='C', dtype=numpy.double)
    Z = numpy.empty([cell.natm,], order='C', dtype=numpy.double)
    rloc = numpy.empty([cell.natm,], order='C', dtype=numpy.double)
    for ia in range(cell.natm):
        Z[ia] = cell.atom_charge(ia)
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            rloc[ia] = cell._pseudo[symb][1]
        else:
            rloc[ia] = -999

    out = numpy.empty((ngrid,), order='C', dtype=numpy.complex128)
    fn = getattr(libpbc, "pp_loc_part1_gs", None)
    try:
        fn(out.ctypes.data_as(ctypes.c_void_p),
           coulG.ctypes.data_as(ctypes.c_void_p),
           Gv.ctypes.data_as(ctypes.c_void_p),
           G2.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(G0idx), ctypes.c_int(ngrid),
           Z.ctypes.data_as(ctypes.c_void_p),
           coords.ctypes.data_as(ctypes.c_void_p),
           rloc.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(cell.natm))
    except Exception as e:
        raise RuntimeError("Failed to get vlocG part1. %s" % e)
    return out


def _get_vpplocG_part1(mydf, with_rho_core=PP_WITH_RHO_CORE):
    cell = mydf.cell
    mesh = mydf.mesh

    if not with_rho_core:
        # compute rho_core directly in G-space
        # this is much slower that the following
        Gv = cell.get_Gv(mesh)
        vpplocG_part1 = get_pp_loc_part1_gs(cell, Gv)
    else:
        # compute rho_core in real space then transform to G-space
        weight = cell.vol / numpy.prod(mesh)
        rho_core = make_rho_core(cell)
        rhoG_core = weight * tools.fft(rho_core, mesh)
        rho_core = None
        coulG = tools.get_coulG(cell, mesh=mesh)
        vpplocG_part1 = rhoG_core * coulG
        rhoG_core = coulG = None
        # G = 0 contribution
        chargs = cell.atom_charges()
        rloc = []
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            rloc.append(cell._pseudo[symb][1])
        rloc = numpy.asarray(rloc)
        vpplocG_part1[0] += 2. * numpy.pi * numpy.sum(rloc * rloc * chargs)
    return vpplocG_part1


def get_vpploc_part1_ip1(mydf, kpts=numpy.zeros((1,3))):
    from .multigrid_pair import _get_j_pass2_ip1
    if mydf.pp_with_erf:
        return 0

    mesh = mydf.mesh
    vG = mydf.vpplocG_part1
    vG.reshape(-1,*mesh)

    vpp_kpts = _get_j_pass2_ip1(mydf, vG, kpts, hermi=0, deriv=1)
    if gamma_point(kpts):
        vpp_kpts = vpp_kpts.real
    if len(kpts) == 1:
        vpp_kpts = vpp_kpts[0]
    return vpp_kpts


def vpploc_part1_nuc_grad(mydf, dm, kpts=numpy.zeros((1,3)), atm_id=None, precision=None):
    from .multigrid_pair import _eval_rhoG
    t0 = (logger.process_clock(), logger.perf_counter())
    cell = mydf.cell
    fakecell, max_radius = fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = numpy.asarray(cell.lattice_vectors(), order='C', dtype=float)
    if abs(a - numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
        raise NotImplementedError
    eval_fn = 'eval_mat_lda' + lattice_type + '_ip1'

    b = numpy.asarray(numpy.linalg.inv(a.T), order='C', dtype=float)
    mesh = numpy.asarray(mydf.mesh, order='C', dtype=numpy.int32)
    ngrids = numpy.prod(mesh)
    comp = 3
    grad = numpy.zeros((len(atm),comp), order="C", dtype=float)
    drv = getattr(libdft, 'int_gauss_charge_v_rs', None)

    if mydf.rhoG is None:
        rhoG = _eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=0)
    else:
        rhoG = mydf.rhoG
    rhoG = rhoG[...,0,:]
    rhoG = rhoG.reshape(-1,ngrids)
    if rhoG.shape[0] == 2: #unrestricted
        rhoG = rhoG[0] + rhoG[1]
    else:
        assert rhoG.shape[0] == 1
        rhoG = rhoG[0]

    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.multiply(rhoG, coulG)

    v_rs = numpy.asarray(tools.ifft(vG, mesh).real, order="C")
    try:
        drv(getattr(libdft, eval_fn),
            grad.ctypes.data_as(ctypes.c_void_p),
            v_rs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp),
            atm.ctypes.data_as(ctypes.c_void_p),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p),
            mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.dimension),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(max_radius))
    except Exception as e:
        raise RuntimeError("Failed to computed nuclear gradients of vpploc part1. %s" % e)
    grad *= -1
    t0 = logger.timer(mydf, 'vpploc_part1_nuc_grad', *t0)
    return grad


def fake_cell_vloc_part1(cell, atm_id=None, precision=None):
    '''
    Generate fakecell for the non-local term of the local part of
    the GTH pseudo-potential. Also stores the atomic radii.
    Differs from pp_int.fake_cell_vloc(cell, cn=0) in the normalization factors.
    '''
    from pyscf.pbc.gto.cell import pgf_rcut
    if atm_id is None:
        atm_id = numpy.arange(cell.natm)
    else:
        atm_id = numpy.asarray(atm_id)
    natm = len(atm_id)

    if precision is None:
        precision = cell.precision

    max_radius = 0
    kind = {}
    # FIXME prec may be too tight
    prec = precision ** 2
    for symb in cell._pseudo:
        charge = numpy.sum(cell._pseudo[symb][0])
        rloc = cell._pseudo[symb][1]
        zeta = .5 / rloc**2
        norm = (zeta / numpy.pi) ** 1.5
        radius = pgf_rcut(0, zeta, charge*norm, precision=prec)
        max_radius = max(radius, max_radius)
        kind[symb] = [zeta, norm, radius]

    fake_env = [cell.atom_coords()[atm_id].ravel()]
    fake_atm = cell._atm[atm_id].copy().reshape(natm,-1)
    fake_atm[:,gto.PTR_COORD] = numpy.arange(0, natm*3, 3)
    ptr = natm * 3
    fake_bas = []
    for ia, atm in enumerate(atm_id):
        if cell.atom_charge(atm) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(atm)
        if symb in kind:
            fake_env.append(kind[symb])
        else:
            alpha = 1e16
            norm = (alpha / numpy.pi) ** 1.5
            radius = 0.0
            fake_env.append([alpha, norm, radius])
        fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        fake_atm[ia,gto.PTR_RADIUS] = ptr+2
        ptr += 3

    fakecell = cell.copy(deep=False)
    fakecell._atm = numpy.asarray(fake_atm, order="C", dtype=numpy.int32)
    fakecell._bas = numpy.asarray(fake_bas, order="C", dtype=numpy.int32).reshape(-1, gto.BAS_SLOTS)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), order="C", dtype=float)
    return fakecell, max_radius
