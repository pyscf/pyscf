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

import unittest
from functools import reduce
import numpy
import scipy.special
from pyscf import gto, scf, lib, dft
from pyscf.solvent import ddcosmo
from pyscf.symm import sph


def make_v_phi(mol, dm, r_vdw, lebedev_order):
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    natm = mol.natm
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)

    pmol = mol.copy()
    v_phi = []
    for ia in range(natm):
        for i,c in enumerate(coords_1sph):
            r = atom_coords[ia] + r_vdw[ia] * c
            dr = atom_coords - r
            v_nuc = (atom_charges / numpy.linalg.norm(dr, axis=1)).sum()
            pmol.set_rinv_orig(r)
            v_e = numpy.einsum('ij,ji', pmol.intor('int1e_rinv'), dm)
            v_phi.append(v_nuc - v_e)
    v_phi = numpy.array(v_phi).reshape(natm,-1)
    return v_phi

def make_L(pcmobj, r_vdw, lebedev_order, lmax, eta=0.1):
    mol = pcmobj.mol
    natm = mol.natm
    nlm = (lmax+1)**2

    leb_coords, leb_weights = ddcosmo.make_grids_one_sphere(lebedev_order)
    nleb_grid = leb_weights.size
    atom_coords = mol.atom_coords()
    Ylm_sphere = numpy.vstack(sph.real_sph_vec(leb_coords, lmax, True))
    fi = ddcosmo.make_fi(pcmobj, r_vdw)

    L_diag = numpy.zeros((natm,nlm))
    p1 = 0
    for l in range(lmax+1):
        p0, p1 = p1, p1 + (l*2+1)
        L_diag[:,p0:p1] = 4*numpy.pi/(l*2+1)
    L_diag /= r_vdw.reshape(-1,1)
    L = numpy.diag(L_diag.ravel()).reshape(natm,nlm,natm,nlm)
    for ja in range(natm):
        for ka in range(natm):
            if ja == ka:
                continue
            vjk = r_vdw[ja] * leb_coords + atom_coords[ja] - atom_coords[ka]
            v = lib.norm(vjk, axis=1)
            tjk = v / r_vdw[ka]
            sjk = vjk / v.reshape(-1,1)
            Ys = sph.real_sph_vec(sjk, lmax, True)
            # scale the weight, see JCTC 9, 3637, Eq (16)
            wjk = pcmobj.regularize_xt(tjk, eta, r_vdw[ka])
            wjk[fi[ja]>1] /= fi[ja,fi[ja]>1]
            tt = numpy.ones_like(wjk)
            p1 = 0
            for l in range(lmax+1):
                fac = 4*numpy.pi/(l*2+1) / r_vdw[ka]
                p0, p1 = p1, p1 + (l*2+1)
                val = numpy.einsum('n,xn,n,mn->xm', leb_weights, Ylm_sphere, wjk*tt, Ys[l])
                L[ja,:,ka,p0:p1] += -fac * val
                tt *= tjk
    return L.reshape(natm*nlm,natm*nlm)

def make_psi(mol, dm, r_vdw, lmax):
    grids = dft.gen_grid.Grids(mol)
    atom_grids_tab = grids.gen_atomic_grids(mol)
    grids.build()

    ao = dft.numint.eval_ao(mol, grids.coords)
    den = dft.numint.eval_rho(mol, ao, dm)
    den *= grids.weights
    natm = mol.natm
    nlm = (lmax+1)**2
    psi = numpy.empty((natm,nlm))
    i1 = 0
    for ia in range(natm):
        xnj, w = atom_grids_tab[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + w.size
        r = lib.norm(xnj, axis=1)
        snj = xnj/r.reshape(-1,1)
        Ys = sph.real_sph_vec(snj, lmax, True)
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            rr = numpy.zeros_like(r)
            rr[r<=r_vdw[ia]] = r[r<=r_vdw[ia]]**l / r_vdw[ia]**(l+1)
            rr[r> r_vdw[ia]] = r_vdw[ia]**l / r[r>r_vdw[ia]]**(l+1)
            psi[ia,p0:p1] = -fac * numpy.einsum('n,n,mn->m', den[i0:i1], rr, Ys[l])
        psi[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)
    return psi

def make_vmat(pcm, r_vdw, lebedev_order, lmax, LX, LS):
    mol = pcm.mol
    grids = dft.gen_grid.Grids(mol)
    atom_grids_tab = grids.gen_atomic_grids(mol)
    grids.build()
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
    ao = dft.numint.eval_ao(mol, grids.coords)
    nao = ao.shape[1]
    vmat = numpy.zeros((nao,nao))
    i1 = 0
    for ia in range(mol.natm):
        xnj, w = atom_grids_tab[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + w.size
        r = lib.norm(xnj, axis=1)
        Ys = sph.real_sph_vec(xnj/r.reshape(-1,1), lmax, True)
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            rr = numpy.zeros_like(r)
            rr[r<=r_vdw[ia]] = r[r<=r_vdw[ia]]**l / r_vdw[ia]**(l+1)
            rr[r> r_vdw[ia]] = r_vdw[ia]**l / r[r>r_vdw[ia]]**(l+1)
            eta_nj = fac * numpy.einsum('n,mn,m->n', rr, Ys[l], LX[ia,p0:p1])
            vmat -= numpy.einsum('n,np,nq->pq', grids.weights[i0:i1] * eta_nj,
                                 ao[i0:i1], ao[i0:i1])

    atom_coords = mol.atom_coords()
    Ylm_sphere = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))
    fi = ddcosmo.make_fi(pcm, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0
    xi_nj = numpy.einsum('n,jn,xn,jx->jn', weights_1sph, ui, Ylm_sphere, LS)

    pmol = mol.copy()
    for ia in range(mol.natm):
        for i,c in enumerate(coords_1sph):
            r = atom_coords[ia] + r_vdw[ia] * c
            pmol.set_rinv_orig(r)
            vmat += pmol.intor('int1e_rinv') * xi_nj[ia,i]
    return vmat


mol = gto.Mole()
mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
               H                 -0.00000000   -0.84695236    0.59109389
               H                 -0.00000000    0.89830571    0.52404783 '''
mol.basis = '3-21g'
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

class KnownValues(unittest.TestCase):
    def test_ddcosmo_scf(self):
        mol = gto.M(atom=''' H 0 0 0 ''', charge=1, basis='sto3g', verbose=7,
                    output='/dev/null')
        pcm = ddcosmo.DDCOSMO(mol)
        pcm.lmax = 10
        pcm.lebedev_order = 29
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol), pcm)
        mf.init_guess = '1e'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.1645636146393864, 9)

        mol = gto.M(atom='''
               6        0.000000    0.000000   -0.542500
               8        0.000000    0.000000    0.677500
               1        0.000000    0.935307   -1.082500
               1        0.000000   -0.935307   -1.082500
                    ''', basis='sto3g', verbose=7,
                    output='/dev/null')
        pcm = ddcosmo.DDCOSMO(mol)
        pcm.lmax = 6
        pcm.lebedev_order = 17
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol), pcm).run()
        self.assertAlmostEqual(mf.e_tot, -112.35450855007909, 9)

    def test_make_ylm(self):
        numpy.random.seed(1)
        lmax = 6
        r = numpy.random.random((100,3)) - numpy.ones(3)*.5
        r = r / lib.norm(r,axis=1).reshape(-1,1)

        ngrid = r.shape[0]
        cosphi = r[:,2]
        sinphi = (1-cosphi**2)**.5
        costheta = numpy.ones(ngrid)
        sintheta = numpy.zeros(ngrid)
        costheta[sinphi!=0] = r[sinphi!=0,0] / sinphi[sinphi!=0]
        sintheta[sinphi!=0] = r[sinphi!=0,1] / sinphi[sinphi!=0]
        costheta[costheta> 1] = 1
        costheta[costheta<-1] =-1
        sintheta[sintheta> 1] = 1
        sintheta[sintheta<-1] =-1
        varphi = numpy.arccos(cosphi)
        theta = numpy.arccos(costheta)
        theta[sintheta<0] = 2*numpy.pi - theta[sintheta<0]
        ylmref = []
        for l in range(lmax+1):
            ylm = numpy.empty((l*2+1,ngrid))
            ylm[l] = scipy.special.sph_harm(0, l, theta, varphi).real
            for m in range(1, l+1):
                f1 = scipy.special.sph_harm(-m, l, theta, varphi)
                f2 = scipy.special.sph_harm( m, l, theta, varphi)
                # complex to real spherical functions
                if m % 2 == 1:
                    ylm[l-m] = (-f1.imag - f2.imag) / numpy.sqrt(2)
                    ylm[l+m] = ( f1.real - f2.real) / numpy.sqrt(2)
                else:
                    ylm[l-m] = (-f1.imag + f2.imag) / numpy.sqrt(2)
                    ylm[l+m] = ( f1.real + f2.real) / numpy.sqrt(2)
            if l == 1:
                ylm = ylm[[2,0,1]]
            ylmref.append(ylm)
        ylmref = numpy.vstack(ylmref)
        ylm = numpy.vstack(sph.real_sph_vec(r, lmax, True))
        self.assertTrue(abs(ylmref - ylm).max() < 1e-14)

    def test_L_x(self):
        pcm = ddcosmo.DDCOSMO(mol)
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        n = mol.natm * (pcm.lmax+1)**2
        Lref = make_L(pcm, r_vdw, pcm.lebedev_order, pcm.lmax, pcm.eta).reshape(n,n)

        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        fi = ddcosmo.make_fi(pcm, r_vdw)
        L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi).reshape(n,n)

        numpy.random.seed(1)
        x = numpy.random.random(n)
        self.assertTrue(abs(Lref.dot(n)-L.dot(n)).max() < 1e-12)

    def test_phi(self):
        pcm = ddcosmo.DDCOSMO(mol)
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0

        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T

        v_phi = make_v_phi(mol, dm, r_vdw, pcm.lebedev_order)
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        phi = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)
        phi1 = ddcosmo.make_phi(pcm, dm, r_vdw, ui)
        self.assertTrue(abs(phi - phi1).max() < 1e-12)

    def test_psi_vmat(self):
        pcm = ddcosmo.DDCOSMO(mol)
        pcm.lmax = 2
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        grids = dft.gen_grid.Grids(mol).build()
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)

        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        natm = mol.natm
        nlm = (pcm.lmax+1)**2
        LX = numpy.random.random((natm,nlm))

        L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
        psi, vmat = ddcosmo.make_psi_vmat(pcm, dm, r_vdw, ui, grids,
                                          ylm_1sph, cached_pol, LX, L)[:2]
        psi_ref = make_psi(pcm.mol, dm, r_vdw, pcm.lmax)
        self.assertAlmostEqual(abs(psi_ref - psi).max(), 0, 12)

        LS = numpy.linalg.solve(L.T.reshape(natm*nlm,-1),
                                psi_ref.ravel()).reshape(natm,nlm)
        vmat_ref = make_vmat(pcm, r_vdw, pcm.lebedev_order, pcm.lmax, LX, LS)
        self.assertAlmostEqual(abs(vmat_ref - vmat).max(), 0, 12)

    def test_vmat(self):
        mol = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1', verbose=0)
        pcmobj = ddcosmo.DDCOSMO(mol)
        f = pcmobj.as_solver()
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm1 = numpy.random.random((nao,nao))
        dm1 = dm1 + dm1.T
        e0, vmat0 = f(dm1)
        dx = 0.0001
        vmat1 = numpy.zeros_like(dm1)
        for i in range(nao):
            for j in range(i):
                dm1[i,j] += dx
                dm1[j,i] += dx
                e1 = f(dm1)[0]
                vmat1[i,j] = vmat1[j,i] = (e1 - e0) / (dx*2)
                dm1[i,j] -= dx
                dm1[j,i] -= dx
            dm1[i,i] += dx
            e1 = f(dm1)[0]
            vmat1[i,i] = (e1 - e0) / dx
            dm1[i,i] -= dx
        self.assertAlmostEqual(abs(vmat0-vmat1).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for ddcosmo")
    unittest.main()

