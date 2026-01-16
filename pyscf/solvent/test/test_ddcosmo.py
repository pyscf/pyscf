#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto, scf, dft, ao2mo, df
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent
from pyscf.symm import sph
from pyscf.lib import Ylm


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
            wjk = pcmobj.regularize_xt(tjk, eta)
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
    grids = ddcosmo.Grids(mol)
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
    grids = ddcosmo.Grids(mol)
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


def make_B(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    mol = pcmobj.mol
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    mol = pcmobj.mol
    natm = mol.natm
    nao  = mol.nao
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids

    extern_point_idx = ui > 0
    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))

    cav_coords = cav_coords[extern_point_idx]
    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                         mol._env, int3c2e)
    fakemol = gto.fakemol_for_charges(cav_coords)
    v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s2ij', cintopt=cintopt)
    nao_pair = v_nj.shape[0]
    v_phi = numpy.zeros((nao_pair, natm, ngrid_1sph))
    v_phi[:,extern_point_idx] += v_nj

    phi = numpy.einsum('n,xn,jn,ijn->ijx', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi.reshape(-1,natm*nlm).T)
    Xvec = Xvec.reshape(natm,nlm,nao_pair)

    ao = mol.eval_gto('GTOval', grids.coords)
    aow = numpy.einsum('gi,g->gi', ao, grids.weights)
    aopair = lib.pack_tril(numpy.einsum('gi,gj->gij', ao, aow))

    psi = numpy.zeros((nao_pair, natm, nlm))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            psi[:,ia,p0:p1] = -fac * numpy.einsum('mn,ni->im', fak_pol[l], aopair[i0:i1])

    B = lib.einsum('pnl,nlq->pq', psi, Xvec)
    B = B + B.T
    B = ao2mo.restore(1, B, nao)
    return B


def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

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
        self.assertEqual(mf.undo_solvent().__class__.__name__, 'RHF')

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
        self.assertAlmostEqual(mf.e_tot, -112.35463433688, 9)

    def test_ddcosmo_scf_with_overwritten_attributes(self):
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol))
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -75.57006258287, 9)

        mf.with_solvent.lebedev_order = 15
        mf.with_solvent.lmax = 5
        mf.with_solvent.eps = .5
        mf.with_solvent.conv_tol = 1e-8
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -75.55351392557, 9)

        mf.with_solvent.grids.radi_method = dft.mura_knowles
        mf.with_solvent.grids.atom_grid = {"H": (8, 50), "O": (8, 50),}
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -75.55237426980, 9)

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
            ylm[l] = Ylm(l, 0, varphi, theta).real
            for m in range(1, l+1):
                f1 = Ylm(l, -m, varphi, theta)
                f2 = Ylm(l,  m, varphi, theta)
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
        phi1 = ddcosmo.make_phi(pcm, dm, r_vdw, ui, ylm_1sph)
        self.assertTrue(abs(phi - phi1).max() < 1e-12)

    def test_psi_vmat(self):
        pcm = ddcosmo.DDCOSMO(mol)
        pcm.lmax = 2
        pcm.eps = 0
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        grids = ddcosmo.Grids(mol).build()
        pcm.grids = grids
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
        psi, vmat = ddcosmo.make_psi_vmat(pcm, dm, r_vdw, ui,
                                          ylm_1sph, cached_pol, LX, L)[:2]
        psi_ref = make_psi(pcm.mol, dm, r_vdw, pcm.lmax)
        self.assertAlmostEqual(abs(psi_ref - psi).max(), 0, 12)

        LS = numpy.linalg.solve(L.reshape(natm*nlm,-1).T,
                                psi_ref.ravel()).reshape(natm,nlm)
        vmat_ref = make_vmat(pcm, r_vdw, pcm.lebedev_order, pcm.lmax, LX, LS)
        self.assertAlmostEqual(abs(vmat_ref - vmat).max(), 0, 12)

    def test_B_dot_x(self):
        pcm = ddcosmo.DDCOSMO(mol)
        pcm.lmax = 2
        pcm.eps = 0
        natm = mol.natm
        nao = mol.nao
        nlm = (pcm.lmax+1)**2
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        grids = ddcosmo.Grids(mol).run(level=0)
        pcm.grids = grids
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)
        L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
        B = make_B(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)

        numpy.random.seed(19)
        dm = numpy.random.random((2,nao,nao))
        Bx = numpy.einsum('ijkl,xkl->xij', B, dm)

        phi = ddcosmo.make_phi(pcm, dm, r_vdw, ui, ylm_1sph, with_nuc=False)
        Xvec = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi.reshape(-1,natm*nlm).T)
        Xvec = Xvec.reshape(natm,nlm,-1).transpose(2,0,1)
        psi, vref, LS = ddcosmo.make_psi_vmat(pcm, dm, r_vdw, ui, ylm_1sph,
                                              cached_pol, Xvec, L, with_nuc=False)
        self.assertAlmostEqual(abs(Bx - vref).max(), 0, 12)
        e1 = numpy.einsum('nij,nij->n', psi, Xvec)
        e2 = numpy.einsum('nij,nij->n', phi, LS)
        e3 = numpy.einsum('nij,nij->n', dm, vref) * .5
        self.assertAlmostEqual(abs(e1-e2).max(), 0, 12)
        self.assertAlmostEqual(abs(e1-e3).max(), 0, 12)

        vmat = pcm._B_dot_x(dm)
        self.assertEqual(vmat.shape, (2,nao,nao))
        self.assertAlmostEqual(abs(vmat-vref*.5).max(), 0, 12)

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

    def test_as_scanner(self):
        mol = gto.M(atom='''
               6        0.000000    0.000000   -0.542500
               8        0.000000    0.000000    0.677500
               1        0.000000    0.935307   -1.082500
               1        0.000000   -0.935307   -1.082500
                    ''', basis='sto3g', verbose=7,
                    output='/dev/null')
        mf_scanner = ddcosmo.ddcosmo_for_scf(scf.RHF(mol)).as_scanner()
        mf_scanner(mol)
        self.assertEqual(mf_scanner.with_solvent.grids.coords.shape, (48212, 3))
        mf_scanner('H  0. 0. 0.; H  0. 0. .9')
        self.assertEqual(mf_scanner.with_solvent.grids.coords.shape, (20048, 3))

        h2 = gto.M(atom='H  0. 0. 0.; H  0. 0. .9', basis='sto3g', verbose=7,
                   output='/dev/null')
        mf_h2 = ddcosmo.ddcosmo_for_scf(scf.RHF(h2)).run()
        self.assertAlmostEqual(mf_h2.e_tot, mf_scanner.e_tot, 9)

    def test_newton_rohf(self):
        mf = mol.ROHF(max_memory=0).ddCOSMO()
        mf = mf.newton()
        e = mf.kernel()
        self.assertAlmostEqual(e, -75.57006258287, 9)

        mf = mol.RHF().ddCOSMO()
        e = mf.kernel()
        self.assertAlmostEqual(e, -75.57006258287, 9)

    def test_convert_scf(self):
        mf = mol.RHF().ddCOSMO()
        mf = mf.to_uhf()
        self.assertTrue(isinstance(mf, scf.uhf.UHF))
        self.assertTrue(isinstance(mf, _attach_solvent._Solvation))

    def test_reset(self):
        mol1 = gto.M(atom='H 0 0 0; H 0 0 .9', basis='cc-pvdz')
        mf = scf.RHF(mol).density_fit().ddCOSMO().newton()
        mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertTrue(mf.with_solvent.mol is mol1)
        self.assertTrue(mf._scf.with_df.mol is mol1)
        self.assertTrue(mf._scf.with_solvent.mol is mol1)

    def test_df_cosmo(self):
        mol = gto.M(atom='H 0 0 0 ; H 0 0 1')
        auxbasis = [[0, [1, 1]]]
        mf1 = mol.RHF().density_fit(auxbasis=auxbasis).ddCOSMO().run()
        mf2 = mol.RHF().ddCOSMO().density_fit(auxbasis=auxbasis).run()
        assert abs(mf1.e_tot - mf2.e_tot) < 1e-12

class SolventWithDefaultGrids(unittest.TestCase):
    def test_rhf_tda(self):
        # TDA with equilibrium_solvation
        mf = mol.RHF().ddCOSMO().run(conv_tol=1e-12)
        td = mf.TDA(equilibrium_solvation=True).run(conv_tol=1e-10)
        ref = numpy.array([0.30125788456, 0.358731044210, 0.39502266389])
        self.assertAlmostEqual(abs(ref - td.e).max(), 0, 7)
        self.assertEqual(td.undo_solvent().__class__.__name__, 'TDA')

        # TDA without equilibrium_solvation
        mf = mol.RHF().ddCOSMO().run(conv_tol=1e-10)
        td = mf.TDA().run()
        ref = numpy.array([0.301354470812, 0.358764482083, 0.398123841665])
        self.assertAlmostEqual(abs(ref - td.e).max(), 0, 7)

# TODO: add tests for direct-scf, ROHF, ROKS, .newton(), and their mixes


if __name__ == "__main__":
    print("Full Tests for ddcosmo")
    unittest.main()
