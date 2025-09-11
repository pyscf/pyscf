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
import numpy as np
from pyscf import __config__
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import fft
from pyscf.pbc.dft.multigrid import _backend_c as backend

PP_WITH_RHO_CORE = getattr(__config__, 'pbc_dft_multigrid_pp_with_rho_core', True)


def make_rho_core(cell, mesh=None, precision=None, atm_id=None):
    if mesh is None:
        mesh = cell.mesh
    fakecell, max_radius = fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)

    a = cell.lattice_vectors()
    b = np.linalg.inv(a.T)

    rho_core = backend.build_core_density(
        fakecell._atm,
        fakecell._bas,
        fakecell._env,
        mesh,
        cell.dimension,
        a,
        b,
        max_radius,
    )
    logger.debug(cell, 'Number of core electrons: %.9f',
                 -np.sum(rho_core) * cell.vol / np.prod(mesh))
    return rho_core


def _get_pp_without_erf(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts = np.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts_ibz
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)

    vpp = pp_int.get_pp_loc_part2(cell, kpts)
    vppnl = pp_int.get_pp_nl(cell, kpts)

    for k, kpt in enumerate(kpts):
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl[k].real
        else:
            vpp[k] += vppnl[k]
    vppnl = None

    if is_single_kpt:
        vpp = vpp[0]
    return np.asarray(vpp)


def get_pp_loc_part1_gs(cell, Gv):
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = np.einsum('ix,ix->i', Gv, Gv)
    G0idx = np.where(G2==0)[0][0]

    coords = cell.atom_coords()

    natm = cell.natm
    Z = np.empty(natm)
    rloc = np.empty(natm)
    for ia in range(natm):
        Z[ia] = cell.atom_charge(ia)
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            rloc[ia] = cell._pseudo[symb][1]
        else:
            rloc[ia] = -999

    out = backend.pp_loc_part1_gs(
            coulG, Gv, G2, G0idx,
            Z, coords, rloc)
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
        weight = cell.vol / np.prod(mesh)
        rho_core = make_rho_core(cell)
        rhoG_core = weight * tools.fft(rho_core, mesh)
        rho_core = None
        coulG = tools.get_coulG(cell, mesh=mesh)
        vpplocG_part1 = rhoG_core * coulG
        rhoG_core = coulG = None
        # G = 0 contribution
        chargs = cell.atom_charges()
        symbs = map(cell.atom_symbol, range(cell.natm))
        rloc = [cell._pseudo[symb][1] if symb in cell._pseudo else 0.
                for symb in symbs]
        rloc = np.asarray(rloc)
        vpplocG_part1[0] += 2. * np.pi * np.sum(rloc * rloc * chargs)
    return vpplocG_part1


def get_vpploc_part1_ip1(mydf, kpts=np.zeros((1,3))):
    from .multigrid_pair import _get_j_pass2_ip1
    if isinstance(kpts, KPoints):
        raise NotImplementedError
    vG = mydf.vpplocG_part1
    if vG is None:
        vG = _get_vpplocG_part1(mydf)

    vpp_kpts = _get_j_pass2_ip1(mydf, vG, kpts, hermi=0, deriv=1)
    if gamma_point(kpts):
        vpp_kpts = vpp_kpts.real
    if len(kpts) == 1:
        vpp_kpts = vpp_kpts[0]
    return vpp_kpts


def vpploc_part1_nuc_grad(mydf, dm, kpts=np.zeros((1,3)), atm_id=None, precision=None):
    from .multigrid_pair import _eval_rhoG
    if isinstance(kpts, KPoints):
        raise NotImplementedError
    t0 = (logger.process_clock(), logger.perf_counter())
    cell = mydf.cell
    fakecell, max_radius = fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = cell.lattice_vectors()
    b = np.linalg.inv(a.T)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    comp = 3

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
    vG = np.multiply(rhoG, coulG)
    v_rs = tools.ifft(vG, mesh).real

    grad = backend.int_gauss_charge_v_rs(
        v_rs,
        comp,
        atm,
        bas,
        env,
        mesh,
        cell.dimension,
        a,
        b,
        max_radius,
    )
    grad *= -1
    t0 = logger.timer(mydf, 'vpploc_part1_nuc_grad', *t0)
    return grad


def fake_cell_vloc_part1(cell, atm_id=None, precision=None):
    '''
    Generate fakecell for the long-range term of the local part of
    the GTH pseudo-potential. Also stores the atomic radii.
    Differs from pp_int.fake_cell_vloc(cell, cn=0) in the normalization factors.
    '''
    from pyscf.pbc.gto.cell import pgf_rcut
    if atm_id is None:
        atm_id = np.arange(cell.natm)
    else:
        atm_id = np.asarray(atm_id)
    natm = len(atm_id)

    if precision is None:
        precision = cell.precision

    max_radius = 0
    kind = {}
    # FIXME prec may be too tight
    prec = precision ** 2
    for symb in cell._pseudo:
        charge = np.sum(cell._pseudo[symb][0])
        rloc = cell._pseudo[symb][1]
        zeta = .5 / rloc**2
        norm = (zeta / np.pi) ** 1.5
        radius = pgf_rcut(0, zeta, charge*norm, precision=prec)
        max_radius = max(radius, max_radius)
        kind[symb] = [zeta, norm, radius]

    fake_env = [cell.atom_coords()[atm_id].ravel()]
    fake_atm = cell._atm[atm_id].copy().reshape(natm,-1)
    fake_atm[:,gto.PTR_COORD] = np.arange(0, natm*3, 3)
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
            norm = (alpha / np.pi) ** 1.5
            radius = 0.0
            fake_env.append([alpha, norm, radius])
        fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        fake_atm[ia,gto.PTR_RADIUS] = ptr+2
        ptr += 3

    fakecell = cell.copy(deep=False)
    fakecell._atm = np.asarray(fake_atm, order="C", dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, order="C", dtype=np.int32).reshape(-1, gto.BAS_SLOTS)
    fakecell._env = np.asarray(np.hstack(fake_env), order="C", dtype=float)
    return fakecell, max_radius
