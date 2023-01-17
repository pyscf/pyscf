#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

'''
Self-consistente continuum solvation model

Reference:
1. J. Chem. Phys. 136, 064102 (2012); https://doi.org/10.1063/1.3676407
2. J. Chem. Phys. 144, 014103 (2016); https://doi.org/10.1063/1.4939125
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.tools import write_cube

KE_RATIO = 3.

libpbc = lib.load_library('libpbc')

def _get_eps(rho_elec, rho_aux, rho_min, rho_max, eps0):
    if rho_aux is not None:
        rho_elec = rho_elec + rho_aux
    ng = rho_elec.size
    eps = numpy.empty_like(rho_elec, order='C', dtype=float)
    deps_intermediate = numpy.empty_like(rho_elec, order='C', dtype=float)
    fun = getattr(libpbc, 'get_eps')
    fun(eps.ctypes.data_as(ctypes.c_void_p),
        deps_intermediate.ctypes.data_as(ctypes.c_void_p),
        rho_elec.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(rho_min), ctypes.c_double(rho_max),
        ctypes.c_double(eps0), ctypes.c_size_t(ng))
    return eps, deps_intermediate

def _get_log_eps_gradient(cell, eps, Gv=None, mesh=None, method='FFT'):
    log_eps = numpy.log(eps)
    if method.upper() == 'FFT':
        out = tools.gradient_by_fft(log_eps, Gv, mesh)
    elif method.upper() == 'FDIFF':
        out = tools.gradient_by_fdiff(cell, log_eps, mesh)
    else:
        raise KeyError
    return out

def _get_deps_drho(eps, deps_intermediate):
    return lib.multiply(eps, deps_intermediate)

def _pgd(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
         gradient_method=None, mixing_factor=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method
    if mixing_factor is None:
        mixing_factor = sccs.mixing_factor

    log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)
    if sccs.rho_pol is not None:
        # use the polarization density from previous scf step
        # as the initial guess
        rho_tot = sccs.rho_pol + rho_solute
    else:
        rho_tot = rho_solute

    phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
    else:
        raise NotImplementedError

    fac = 4 * numpy.pi

    tmp = lib.multiply(log_eps1[0], dphi_tot[0])
    for x in range(1,3):
        tmp = lib.add(tmp, lib.multiply(log_eps1[x], dphi_tot[x]), out=tmp)
    tmp = lib.multiply(tmp, eps, out=tmp)

    tmp1 = rho_solute / eps
    tmp1 = lib.subtract(rho_tot, tmp1, out=tmp1)
    tmp1 = lib.multiply(eps, tmp1, out=tmp1)
    tmp1 = lib.multiply(fac, tmp1, out=tmp1)
    r = lib.subtract(tmp1, tmp, out=tmp1)
    tmp = None

    fac1 = fac * numpy.sqrt(lib.vdot(rho_solute, rho_solute))
    invs_eps = lib.reciprocal(-fac * eps)
    for i in range(max_cycle):
        r_norm = numpy.sqrt(lib.vdot(r,r)) / fac1
        logger.info(sccs, 'cycle= %d  res= %4.3g', i, r_norm)
        if r_norm < conv_tol:
            break

        fake_rho = lib.multiply(r, invs_eps)
        v = tools.solve_poisson(cell, fake_rho, coulG=coulG, Gv=Gv, mesh=mesh)[0]
        fake_rho = None

        if gradient_method.upper() == "FFT":
            dv = tools.gradient_by_fft(v, Gv, mesh)
        elif gradient_method.upper() == "FDIFF":
            dv = tools.gradient_by_fdiff(cell, v, mesh)
        else:
            raise NotImplementedError

        Av = lib.multiply(log_eps1[0], dv[0])
        for x in range(1,3):
            Av = lib.add(Av, lib.multiply(log_eps1[x], dv[x]), out=Av)
        Av = lib.multiply(eps, Av, out=Av)
        Av = lib.add(r, Av, out=Av)

        #alpha = lib.vdot(v, r) / lib.vdot(v,Av)
        #alpha = lib.vdot(r,Av) / lib.vdot(Av,Av)
        alpha = mixing_factor
        r = lib.subtract(r, lib.multiply(alpha, Av), out=r)
        phi_tot = lib.add(phi_tot, lib.multiply(alpha, v), out=phi_tot)

    return phi_tot

def get_multiple_meshes(cell, mesh, ngrids=1, ke_ratio=KE_RATIO):
    a = cell.lattice_vectors()
    ke1 = tools.mesh_to_cutoff(a, mesh).max()
    cutoff = []
    for i in range(ngrids-1):
        ke1 /= ke_ratio
        cutoff.append(ke1)
    meshes = [mesh,]
    for ke in cutoff:
        meshes.append(tools.cutoff_to_mesh(a, ke))
    return meshes

def _mgpgd(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
           gradient_method=None, mixing_factor=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    if mesh is None or coulG is None or Gv is None:
        return _pgd(sccs, rho_solute, eps, coulG=coulG, Gv=Gv, mesh=mesh,
                    gradient_method=gradient_method, mixing_factor=mixing_factor,
                    conv_tol=conv_tol, max_cycle=max_cycle)
    if gradient_method is None:
        gradient_method = sccs.gradient_method
    if mixing_factor is None:
        mixing_factor = sccs.mixing_factor

    ngrids = len(eps)

    if sccs.rho_pol is not None:
        # use the polarization density from previous scf step
        # as the initial guess
        rho_tot = sccs.rho_pol + rho_solute
    else:
        rho_tot = rho_solute

    fac = 4 * numpy.pi

    phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG[0], Gv=Gv[0], mesh=mesh[0])[0]

    log_eps1 = []
    invs_eps = []
    for i in range(ngrids):
        log_eps1.append(_get_log_eps_gradient(cell, eps[i], Gv[i], mesh[i], gradient_method))
        invs_eps.append(lib.reciprocal(-fac * eps[i]))

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv[0], mesh[0])
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh[0])
    else:
        raise NotImplementedError

    tmp = lib.multiply(log_eps1[0][0], dphi_tot[0])
    for x in range(1,3):
        tmp = lib.add(tmp, lib.multiply(log_eps1[0][x], dphi_tot[x]), out=tmp)
    tmp = lib.multiply(tmp, eps[0], out=tmp)

    tmp1 = rho_solute / eps[0]
    tmp1 = lib.subtract(rho_tot, tmp1, out=tmp1)
    tmp1 = lib.multiply(eps[0], tmp1, out=tmp1)
    tmp1 = lib.multiply(fac, tmp1, out=tmp1)
    r = lib.subtract(tmp1, tmp, out=tmp1)
    tmp = None

    def residual(x, rhs, mesh, aux_data):
        Gv = aux_data["Gv"]
        eps = aux_data["eps"]
        log_eps1 = aux_data["log_eps1"]
        lap_x = tools.laplacian_by_fft(x, Gv, mesh)
        lap_x = lib.multiply(eps, lap_x, out=lap_x)
        dx = tools.gradient_by_fdiff(cell, x, mesh)
        tmp = lib.multiply(log_eps1[0], dx[0])
        for x in range(1,3):
            tmp = lib.add(tmp, lib.multiply(log_eps1[x], dx[x]), out=tmp)
        tmp = lib.multiply(tmp, eps, out=tmp)
        r = lib.subtract(rhs, lib.add(tmp, lap_x))
        return r

    def smoothing(phi_tot, r, mesh, aux_data):
        max_cycle = aux_data["max_cycle"]
        fac1 = aux_data["fac1"]
        invs_eps = aux_data["invs_eps"]
        coulG = aux_data["coulG"]
        Gv = aux_data["Gv"]
        log_eps1 = aux_data["log_eps1"]
        eps = aux_data["eps"]
        conv_tol = aux_data["conv_tol"]

        for i in range(max_cycle):
            r_norm = numpy.sqrt(lib.vdot(r,r)) / fac1
            #logger.info(sccs, 'micro cycle= %d  res= %4.3g', i, r_norm)
            if r_norm < conv_tol:
                break

            fake_rho = lib.multiply(r, invs_eps)
            v = tools.solve_poisson(cell, fake_rho, coulG=coulG, Gv=Gv, mesh=mesh)[0]
            fake_rho = None

            if gradient_method.upper() == "FFT":
                dv = tools.gradient_by_fft(v, Gv, mesh)
            elif gradient_method.upper() == "FDIFF":
                dv = tools.gradient_by_fdiff(cell, v, mesh)
            else:
                raise NotImplementedError

            Av = lib.multiply(log_eps1[0], dv[0])
            for x in range(1,3):
                Av = lib.add(Av, lib.multiply(log_eps1[x], dv[x]), out=Av)
            Av = lib.multiply(eps, Av, out=Av)
            Av = lib.add(r, Av, out=Av)

            alpha = mixing_factor
            r = lib.subtract(r, lib.multiply(alpha, Av), out=r)
            phi_tot = lib.add(phi_tot, lib.multiply(alpha, v), out=phi_tot)
        return phi_tot, r

    def v_cycle(phi_tot, rhs, r, meshes, ilevel):
        mesh = meshes[ilevel]

        aux_data = {}
        aux_data["max_cycle"] = max_cycle
        aux_data["fac1"] = numpy.sqrt(lib.vdot(rhs, rhs))
        aux_data["invs_eps"] = invs_eps[ilevel]
        aux_data["coulG"] = coulG[ilevel]
        aux_data["Gv"] = Gv[ilevel]
        aux_data["log_eps1"] = log_eps1[ilevel]
        aux_data["eps"] = eps[ilevel]
        aux_data["conv_tol"] = conv_tol

        # stop recursion at smallest grid size, otherwise continue recursion
        if ilevel == len(meshes) - 1:
            aux_data["max_cycle"] = max_cycle
            return smoothing(phi_tot, r, mesh, aux_data)
        else:
            # Pre-Smoothing
            r_norm = numpy.sqrt(lib.vdot(r,r)) / aux_data["fac1"]
            aux_data["conv_tol"] = max(r_norm / 10, conv_tol)
            phi_tot, r = smoothing(phi_tot, r, mesh, aux_data)
            if numpy.sqrt(lib.vdot(r,r)) / aux_data["fac1"] < conv_tol:
                return phi_tot, r

            # Restriction
            submesh = meshes[ilevel+1]
            r_sub = tools.restrict_by_fft(r, mesh, submesh)
            error = numpy.zeros_like(r_sub)
            error, _ = v_cycle(error, r_sub, r_sub, meshes, ilevel+1)

            # Prolongation and Correction
            tmp = lib.multiply(mixing_factor, tools.prolong_by_fft(error, mesh, submesh))
            phi_tot = lib.add(phi_tot, tmp, out=phi_tot)
            r = residual(phi_tot, rhs, mesh, aux_data)

        # Post-Smoothing
        phi_tot, r = smoothing(phi_tot, r, mesh, aux_data)
        return phi_tot, r

    rhs = lib.multiply(-fac, rho_solute)
    fac1 = numpy.sqrt(lib.vdot(rhs, rhs))
    for i in range(max_cycle):
        r_norm = numpy.sqrt(lib.vdot(r,r))/fac1
        logger.info(sccs, 'cycle= %d  res= %4.3g', i, r_norm)
        if r_norm < conv_tol:
            break

        phi_tot, r = v_cycle(phi_tot, rhs, r, mesh, 0)
    return phi_tot

def _pcg(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
         gradient_method=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method

    sqrt_eps = numpy.sqrt(eps)
    if gradient_method.upper() == "FFT":
        lap_sqrt_eps = tools.laplacian_by_fft(sqrt_eps, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        lap_sqrt_eps = tools.laplacian_by_fdiff(cell, sqrt_eps, mesh)
    else:
        raise NotImplementedError
    q = lib.multiply(sqrt_eps, lap_sqrt_eps)

    log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)

    if sccs.rho_pol is not None:
        # use the polarization density from previous scf step
        # as the initial guess
        rho_tot = sccs.rho_pol + rho_solute
    else:
        rho_tot = rho_solute

    '''
    if sccs.phi_tot is None:
        phi_tot = tools.solve_poisson(cell, rho_solute, coulG=coulG, Gv=Gv, mesh=mesh)[0]
    else:
        phi_tot = sccs.phi_tot
    '''
    phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv, mesh)
        #deps = tools.gradient_by_fft(eps, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
        #deps = tools.gradient_by_fdiff(cell, eps, mesh)
    else:
        raise NotImplementedError

    fac = 4 * numpy.pi
    tmp = lib.multiply(log_eps1[0], dphi_tot[0])
    for x in range(1,3):
        tmp = lib.add(tmp, lib.multiply(log_eps1[x], dphi_tot[x]), out=tmp)
    tmp = lib.multiply(tmp, eps, out=tmp)

    tmp1 = rho_solute / eps
    tmp1 = lib.subtract(rho_tot, tmp1, out=tmp1)
    tmp1 = lib.multiply(eps, tmp1, out=tmp1)
    tmp1 = lib.multiply(fac, tmp1, out=tmp1)
    r = lib.subtract(tmp1, tmp, out=tmp1)
    tmp = None

    fac1 = fac * numpy.sqrt(lib.vdot(rho_solute, rho_solute))
    invs_sqrt_eps = lib.reciprocal(sqrt_eps)
    vr = vr_old = None
    for i in range(max_cycle):
        r_norm = numpy.sqrt(lib.vdot(r,r)) / fac1
        logger.info(sccs, 'cycle= %d  res= %4.3g', i, r_norm)
        if r_norm < conv_tol:
            break

        fake_rho = lib.multiply(r, invs_sqrt_eps)
        v = lib.multiply(tools.solve_poisson(cell, fake_rho, coulG=coulG, Gv=Gv, mesh=mesh)[0], invs_sqrt_eps)
        fake_rho = None
        vr = lib.vdot(v, r)
        if i == 0:
            beta = 0.0
            p = v
        else:
            beta = vr / vr_old
            p = lib.add(v, lib.multiply(beta, p, out=p), out=p)
        Av = lib.multiply(v, q)
        Av = lib.subtract(lib.multiply(-fac, r), Av, out=Av)
        if i == 0:
            Ap = Av
        else:
            Ap = lib.add(Av, lib.multiply(beta, Ap, out=Ap), out=Ap)
        Av = None
        alpha = vr / lib.vdot(p, Ap)

        vr_old = vr

        r = lib.subtract(r, lib.multiply(alpha, Ap), out=r)
        phi_tot = lib.add(phi_tot, lib.multiply(alpha, p), out=phi_tot)

    if r_norm > conv_tol:
        logger.warn(sccs, 'SCCS did not converge.')

    return phi_tot

def _mixing(sccs, rho_solute, eps, rho_pol=None, coulG=None, Gv=None, mesh=None,
            gradient_method=None, mixing_factor=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method
    if mixing_factor is None:
        mixing_factor = sccs.mixing_factor

    fac = 4. * numpy.pi
    log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)
    log_eps1 = lib.multiply(1./fac, log_eps1, out=log_eps1)

    rho_solute_over_eps = numpy.divide(rho_solute, eps)
    if rho_pol is not None:
        # use the polarization density from previous scf step
        # as the initial guess
        rho_iter = lib.subtract(rho_pol, rho_solute_over_eps)
        rho_iter = lib.add(rho_iter, rho_solute, out=rho_iter)
    else:
        rho_iter = lib.subtract(rho_solute, rho_solute_over_eps)
    rho_iter_old = rho_iter

    r_norm = 0
    fac1 = fac * numpy.sqrt(lib.vdot(rho_solute, rho_solute))
    for i in range(max_cycle):
        rho_tot = lib.add(rho_solute_over_eps, rho_iter)
        if gradient_method.upper() == "FFT":
            _, dphi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh,
                                              compute_potential=False, compute_gradient=True)
        elif gradient_method.upper() == "FDIFF":
            phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]
            dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
            phi_tot = None
        else:
            raise NotImplementedError
        rho_tot = None

        rho_iter = lib.multiply(dphi_tot[0], log_eps1[0])
        for x in range(1,3):
            rho_iter = lib.add(rho_iter, lib.multiply(dphi_tot[x], log_eps1[x]), out=rho_iter)

        r = lib.subtract(rho_iter_old, rho_iter)
        r = lib.multiply(eps, r, out=r)
        r_norm = fac * numpy.sqrt(lib.vdot(r,r)) / fac1
        logger.info(sccs, 'cycle= %d  res= %4.3g', i+1, r_norm)
        if r_norm < conv_tol:
            break
        r = None

        rho_iter = lib.multiply(mixing_factor, rho_iter, out=rho_iter)
        rho_iter = lib.add(rho_iter, (1.-mixing_factor)*rho_iter_old, out=rho_iter)
        rho_iter_old = rho_iter

    if r_norm > conv_tol:
        logger.warn(sccs, 'SCCS did not converge.')

    rho_tot = lib.add(rho_solute_over_eps, rho_iter)
    return rho_tot

def kernel(sccs, rho_elec, rho_core=None, method="mixing",
           rho_min=1e-4, rho_max=1.5e-3, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    mesh = sccs.mesh
    eps0 = sccs.eps
    ngrids = sccs.ngrids
    meshes = get_multiple_meshes(cell, mesh, ngrids=ngrids, ke_ratio=sccs.ke_ratio)
    if hasattr(rho_elec, "ndim"):
        rho_elec = [rho_elec,]

    if len(rho_elec) < ngrids:
        # doing interpolation
        assert len(rho_elec) == 1
        rho_elec_sub = tools.restrict_by_fft(rho_elec[0], meshes[0], meshes[1:])
        if isinstance(rho_elec_sub, list):
            rho_elec.extend(rho_elec_sub)
        else:
            rho_elec.append(rho_elec_sub)

    if rho_core is None:
        rho_solute = rho_elec[0]
    else:
        rho_solute = lib.add(rho_elec[0], rho_core)

    eps = [None,]
    eps[0], deps_intermediate = _get_eps(rho_elec[0], None, rho_min, rho_max, eps0)
    for i in range(1, ngrids):
        eps.append(_get_eps(rho_elec[i], None, rho_min, rho_max, eps0)[0])

    Gv = []
    coulG = []
    log_eps1 = []
    for i, submesh in enumerate(meshes):
        Gv.append(cell.get_Gv(mesh=submesh))
        coulG.append(tools.get_coulG(cell, mesh=submesh))
        log_eps1.append(_get_log_eps_gradient(cell, eps[i], Gv[i], submesh, sccs.gradient_method))

    rho_tot = None
    phi_tot = None
    if method.upper() == "PCG":
        phi_tot = _pcg(sccs, rho_solute, eps[0], coulG=coulG[0], Gv=Gv[0], mesh=meshes[0],
                       conv_tol=conv_tol, max_cycle=max_cycle)
        sccs.phi_tot = phi_tot
    elif method.upper() == "MIXING":
        rho_tot = _mixing(sccs, rho_solute, eps[0], rho_pol=sccs.rho_pol,
                          coulG=coulG[0], Gv=Gv[0], mesh=meshes[0],
                          conv_tol=conv_tol, max_cycle=max_cycle)
    elif method.upper() == "PGD":
        phi_tot = _pgd(sccs, rho_solute, eps[0], coulG=coulG[0], Gv=Gv[0], mesh=meshes[0],
                       conv_tol=conv_tol, max_cycle=max_cycle)
        sccs.phi_tot = phi_tot
    elif method.upper() == 'MGPGD':
        phi_tot = _mgpgd(sccs, rho_solute, eps, coulG=coulG, Gv=Gv, mesh=meshes,
                         conv_tol=conv_tol, max_cycle=max_cycle)
        sccs.phi_tot = phi_tot
    else:
        raise KeyError(f"Unrecognized method: {method}.")

    deps_drho = _get_deps_drho(eps[0], deps_intermediate)
    deps_intermediate = None

    e_pol, phi_sccs = get_veff(sccs, rho_solute, eps[0], deps_drho, rho_tot, phi_tot,
                               coulG[0], Gv[0], meshes[0])
    return e_pol, phi_sccs

def get_veff(sccs, rho_solute, eps, deps_drho, rho_tot=None, phi_tot=None,
             coulG=None, Gv=None, mesh=None, gradient_method=None):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if Gv is None:
        Gv = cell.get_Gv(mesh)
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method
    if rho_tot is None and phi_tot is None:
        raise KeyError("Either rho_tot or phi_tot need to be specified.")

    rho_pol = None
    if rho_tot is not None:
        sccs.rho_pol = rho_pol = lib.subtract(rho_tot, rho_solute)
        phi_pol = tools.solve_poisson(cell, rho_pol, coulG=coulG, Gv=Gv, mesh=mesh)[0]
    if phi_tot is None:
        phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
    else:
        raise NotImplementedError

    if rho_pol is None:
        log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)
        log_eps1 = lib.multiply(.25/numpy.pi, log_eps1, out=log_eps1)

        rho_iter=None
        for x in range(3):
            if x == 0:
                rho_iter = lib.multiply(dphi_tot[x], log_eps1[x], out=rho_iter)
            else:
                tmp = lib.multiply(dphi_tot[x], log_eps1[x])
                rho_iter = lib.add(rho_iter, tmp, out=rho_iter)
        sccs.rho_pol = rho_pol = rho_iter - rho_solute + rho_solute / eps
        phi_pol = tools.solve_poisson(cell, rho_pol, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    weight = cell.vol / numpy.prod(mesh)
    e_pol = .5 * lib.sum(lib.multiply(phi_pol, rho_solute)) * weight
    logger.info(sccs, 'Polarization energy = %.8g', e_pol)

    dphi_tot_square = None
    for x in range(3):
        if x == 0:
            dphi_tot_square = lib.multiply(dphi_tot[x], dphi_tot[x])
        else:
            tmp = lib.multiply(dphi_tot[x], dphi_tot[x])
            dphi_tot_square = lib.add(dphi_tot_square, tmp, out=dphi_tot_square)
    dphi_tot = None

    deps_drho = lib.multiply(-0.125/numpy.pi, deps_drho, out=deps_drho)
    sccs.phi_eps = phi_eps = lib.multiply(deps_drho, dphi_tot_square)
    dphi_tot_square = None
    phi_sccs = lib.add(phi_pol, phi_eps)
    return e_pol, phi_sccs


class SCCS(lib.StreamObject):
    def __init__(self, cell, mesh, eps=78.3553, rho_min=1e-4, rho_max=1.5e-3):
        self.cell = cell
        self.mesh = mesh
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.eps = eps
        self.method = 'mixing'
        self.mixing_factor = 0.6
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.max_cycle = 100
        self.conv_tol = 1e-5
        self.rho_pol = None
        self.phi_eps = None
        self.phi_tot = None
        self.rho_core = None
        self.gradient_method = 'fft'
        self.ngrids = 1
        self.ke_ratio = 3.0

    def kernel(self, rho, rho_core=None):
        return kernel(self, rho, rho_core=rho_core, method=self.method,
                      rho_min=self.rho_min, rho_max=self.rho_max,
                      conv_tol=self.conv_tol, max_cycle=self.max_cycle)
