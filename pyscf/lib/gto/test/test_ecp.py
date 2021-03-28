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

import ctypes
import unittest
import numpy
import scipy.misc
import scipy.special
try:
    from scipy.special import factorial2
except ImportError:
    from scipy.misc import factorial2
from pyscf import lib
from pyscf import gto
from pyscf.dft import radi
from pyscf.symm import sph

libecp = gto.moleintor.libcgto

mol = gto.M(atom='''
            Na 0.5 0.5 0.
            H  0.  1.  1.
            ''',
            basis={'Na':'lanl2dz',
                   'H':[[0,[1.21,1.],[.521,1.]],
                        [1,[3.12,1.],[.512,1.]],
                        [2,[2.54,1.],[.554,1.]],
                        [3,[0.98,1.],[.598,1.]],
                        [4,[0.79,1.],[.579,1.]]]},
            ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
0      2.0000000              6.0000000
1    175.5502590            -10.0000000
2      2.3365719             -6.0637782
2      0.7799867             -0.7299393
Na S
0    243.3605846              3.0000000
#1     41.5764759             36.2847626
#2     13.2649167             72.9304880
#2      0.9764209              6.0123861
#Na P
#0   1257.2650682              5.0000000
#1    189.6248810            117.4495683
#2     54.5247759            423.3986704
#2      0.9461106              7.1241813
''')})


CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_ZETA   = 3
ATM_SLOTS  = 6
# for _ecpbas
ATOM_OF    = 0
ANG_OF     = 1 # <0 means local function
NPRIM_OF   = 2
RADI_POWER = 3
SO_TYPE_OF = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8

def type1_by_shell(mol, shls, ecpatm_id, ecpbas):
    ish, jsh = shls

    li = mol.bas_angular(ish)
    npi = mol.bas_nprim(ish)
    nci = mol.bas_nctr(ish)
    ai = mol.bas_exp(ish)
    ci = mol._libcint_ctr_coeff(ish)
    icart = (li+1) * (li+2) // 2

    lj = mol.bas_angular(jsh)
    npj = mol.bas_nprim(jsh)
    ncj = mol.bas_nctr(jsh)
    aj = mol.bas_exp(jsh)
    cj = mol._libcint_ctr_coeff(jsh)
    jcart = (lj+1) * (lj+2) // 2

    rc = mol.atom_coord(ecpatm_id)
    rca = rc - mol.bas_coord(ish)
    r2ca = numpy.dot(rca, rca)
    rcb = rc - mol.bas_coord(jsh)
    r2cb = numpy.dot(rcb, rcb)
# Note the Mole._libcint_ctr_coeff are normalized to radial part
    cei = numpy.einsum('ij,i->ij', ci, numpy.exp(-ai * r2ca))
    cej = numpy.einsum('ij,i->ij', cj, numpy.exp(-aj * r2cb))
    #rs, ws = radi.treutler(99)
    rs, ws = radi.gauss_chebyshev(99)

    ur = rad_part(mol, ecpbas, rs) * ws
    rad_ang_all = numpy.zeros((nci,ncj,li+lj+1,li+lj+1,li+lj+1))
    for ip in range(npi):
        for jp in range(npj):
            rij = ai[ip] * rca + aj[jp] * rcb
            aij = ai[ip] + aj[jp]
            k = 2*numpy.linalg.norm(rij)
            rad_all = type1_rad_part(li+lj, k, aij, ur, rs)
            #ang_all = type1_ang_part(li+lj, -rij)
            #rad_ang = numpy.einsum('pl,lijk->pijk', rad_all, ang_all)
            rad_ang = type1_rad_ang(li+lj, rij, rad_all)
            for ic in range(nci):
                for jc in range(ncj):
                    rad_ang_all[ic,jc] += rad_ang * cei[ip,ic]*cej[jp,jc] * (4*numpy.pi)**2
    ifac = type1_cache_fac(li, rca)
    jfac = type1_cache_fac(lj, rcb)

    g1 = numpy.zeros((nci,ncj,icart,jcart))
    for ic in range(nci):
        for jc in range(ncj):
            for mi,(ix,iy,iz) in enumerate(loop_cart(li)):
                for mj,(jx,jy,jz) in enumerate(loop_cart(lj)):
                    tmp = 0
                    for i1, i2, i3 in loop_xyz(ix, iy, iz):
                        for j1, j2, j3 in loop_xyz(jx, jy, jz):
                            fac = ifac[mi,i1,i2,i3] * jfac[mj,j1,j2,j3]
                            tmp += fac * rad_ang_all[ic,jc,i1+j1,i2+j2,i3+j3]
                    g1[ic,jc,mi,mj] = tmp

    gsph = numpy.empty((nci,ncj,li*2+1,lj*2+1))
    for ic in range(nci):
        for jc in range(ncj):
            tmp = c2s_bra(lj, g1[ic,jc].T.copy())
            gsph[ic,jc] = c2s_bra(li, tmp.T.copy())
    return gsph.transpose(0,2,1,3).reshape(nci*(li*2+1),-1)

def type1_cache_fac(li, ri):
    facs = cache_fac(li, ri)
    facs4 = numpy.zeros(((li+1)*(li+2)//2,li+1,li+1,li+1))
    for mi,(ix,iy,iz) in enumerate(loop_cart(li)):
        for i1, i2, i3 in loop_xyz(ix, iy, iz):
            facs4[mi,i1,i2,i3] =(facs[0,ix,i1] * facs[1,iy,i2] * facs[2,iz,i3])
    return facs4

def type1_rad_part(lmax, k, aij, ur, rs):
    rad_all = numpy.empty((lmax+1,lmax+1))
    bessel_val = sph_ine(lmax, k*rs)
    ur_base = numpy.exp(k**2/(4*aij)) * ur * numpy.exp(-aij*(rs-k/(2*aij))**2)
    idx = abs(ur_base) > 1e-80
    for lab in range(lmax+1):
        val = ur_base[idx] * rs[idx]**lab
        for l in range(lmax+1):
            if (lab+l) % 2 == 0:
                val1 = val * bessel_val[l,idx]
                rad_all[lab,l] = val1.sum()
            else:
                rad_all[lab,l] = 0
    return rad_all

def type1_rad_ang(lmax, rij, rad_all):
    norm_rij = numpy.linalg.norm(rij)
    if norm_rij > 1e-18:
        unitr = -rij/norm_rij
    else:
        unitr = -rij
    omega_nuc = []
    for lmb in range(lmax+1):
        c2smat = c2s_bra(lmb, numpy.eye((lmb+1)*(lmb+2)//2))
        omega_nuc.append(numpy.dot(ang_nuc_part(lmb, unitr), c2smat))

    rad_ang = numpy.zeros((lmax+1,lmax+1,lmax+1))
    for i in range(lmax+1):
        for j in range(lmax+1-i):
            for k in range(lmax+1-i-j):
                for lmb in range(lmax+1):
                    if (i+j+k+lmb) % 2 == 0:
                        tmp = 0
                        for n, (i1, j1, k1) in enumerate(loop_cart(lmb)):
                            tmp += omega_nuc[lmb][n] * int_unit_xyz(i+i1, j+j1, k+k1)
                        rad_ang[i,j,k] += rad_all[i+j+k,lmb] * tmp
    return rad_ang

def type2_by_shell(mol, shls, ecpatm_id, ecpbas):
    ish, jsh = shls

    li = mol.bas_angular(ish)
    npi = mol.bas_nprim(ish)
    nci = mol.bas_nctr(ish)
    ai = mol.bas_exp(ish)
    ci = mol._libcint_ctr_coeff(ish)
    icart = (li+1) * (li+2) // 2

    lj = mol.bas_angular(jsh)
    npj = mol.bas_nprim(jsh)
    ncj = mol.bas_nctr(jsh)
    aj = mol.bas_exp(jsh)
    cj = mol._libcint_ctr_coeff(jsh)
    jcart = (lj+1) * (lj+2) // 2

    rc = mol.atom_coord(ecpatm_id)
    rcb = rc - mol.bas_coord(jsh)
    r_cb = numpy.linalg.norm(rcb)
    rca = rc - mol.bas_coord(ish)
    r_ca = numpy.linalg.norm(rca)
    #rs, ws = radi.treutler(99)
    rs, ws = radi.gauss_chebyshev(99)

    i_fac_cache = cache_fac(li, rca)
    j_fac_cache = cache_fac(lj, rcb)

    g1 = numpy.zeros((nci,ncj,icart,jcart))
    for lc in range(5): # up to g function
        ecpbasi = ecpbas[ecpbas[:,ANG_OF] == lc]
        if len(ecpbasi) == 0:
            continue
        ur = rad_part(mol, ecpbasi, rs) * ws
        idx = abs(ur) > 1e-80
        rur = numpy.array([ur[idx] * rs[idx]**lab for lab in range(li+lj+1)])

        fi = facs_rad(mol, ish, lc, r_ca, rs)[:,:,idx].copy()
        fj = facs_rad(mol, jsh, lc, r_cb, rs)[:,:,idx].copy()
        angi = facs_ang(type2_ang_part(li, lc, -rca), li, lc, i_fac_cache)
        angj = facs_ang(type2_ang_part(lj, lc, -rcb), lj, lc, j_fac_cache)

        for ic in range(nci):
            for jc in range(ncj):
                rad_all = numpy.einsum('pr,ir,jr->pij', rur, fi[ic], fj[jc])

                for i1 in range(li+1):
                    for j1 in range(lj+1):
                        g1[ic,jc] += numpy.einsum('pq,imp,jmq->ij', rad_all[i1+j1],
                                                  angi[i1], angj[j1])

    g1 *= (numpy.pi*4)**2
    gsph = numpy.empty((nci,ncj,li*2+1,lj*2+1))
    for ic in range(nci):
        for jc in range(ncj):
            tmp = c2s_bra(lj, g1[ic,jc].T.copy())
            gsph[ic,jc] = c2s_bra(li, tmp.T.copy())
    return gsph.transpose(0,2,1,3).reshape(nci*(li*2+1),-1)

def so_by_shell(mol, shls, ecpatm_id, ecpbas):
    '''SO-ECP
    i/2 <Pauli_matrix dot l U(r)>
    '''
    ish, jsh = shls

    li = mol.bas_angular(ish)
    npi = mol.bas_nprim(ish)
    nci = mol.bas_nctr(ish)
    ai = mol.bas_exp(ish)
    ci = mol._libcint_ctr_coeff(ish)
    icart = (li+1) * (li+2) // 2

    lj = mol.bas_angular(jsh)
    npj = mol.bas_nprim(jsh)
    ncj = mol.bas_nctr(jsh)
    aj = mol.bas_exp(jsh)
    cj = mol._libcint_ctr_coeff(jsh)
    jcart = (lj+1) * (lj+2) // 2

    rc = mol.atom_coord(ecpatm_id)
    rcb = rc - mol.bas_coord(jsh)
    r_cb = numpy.linalg.norm(rcb)
    rca = rc - mol.bas_coord(ish)
    r_ca = numpy.linalg.norm(rca)
    #rs, ws = radi.treutler(99)
    rs, ws = radi.gauss_chebyshev(99)

    i_fac_cache = cache_fac(li, rca)
    j_fac_cache = cache_fac(lj, rcb)

    g1 = numpy.zeros((nci,ncj,3,icart,jcart), dtype=numpy.complex128)
    for lc in range(5): # up to g function
        ecpbasi = ecpbas[ecpbas[:,ANG_OF] == lc]
        if len(ecpbasi) == 0:
            continue
        ur = rad_part(mol, ecpbasi, rs) * ws
        idx = abs(ur) > 1e-80
        rur = numpy.array([ur[idx] * rs[idx]**lab for lab in range(li+lj+1)])

        fi = facs_rad(mol, ish, lc, r_ca, rs)[:,:,idx].copy()
        fj = facs_rad(mol, jsh, lc, r_cb, rs)[:,:,idx].copy()
        angi = facs_ang(type2_ang_part(li, lc, -rca), li, lc, i_fac_cache)
        angj = facs_ang(type2_ang_part(lj, lc, -rcb), lj, lc, j_fac_cache)

        # Note the factor 2/(2l+1) in JCP 82, 2664 (1985); DOI:10.1063/1.448263 is not multiplied here
        # because the ECP parameter has been scaled by 2/(2l+1) in CRENBL
        jmm = angular_moment_matrix(lc)

        for ic in range(nci):
            for jc in range(ncj):
                rad_all = numpy.einsum('pr,ir,jr->pij', rur, fi[ic], fj[jc])

                for i1 in range(li+1):
                    for j1 in range(lj+1):
                        g1[ic,jc] += numpy.einsum('pq,imp,jnq,lmn->lij', rad_all[i1+j1],
                                                  angi[i1], angj[j1], jmm)

    g1 *= (numpy.pi*4)**2
    gspinor = numpy.empty((nci,ncj,li*4+2,lj*4+2), dtype=numpy.complex128)
    for ic in range(nci):
        for jc in range(ncj):
            ui = numpy.asarray(gto.cart2spinor_l(li))
            uj = numpy.asarray(gto.cart2spinor_l(lj))
            s = lib.PauliMatrices * .5j
            gspinor[ic,jc] = numpy.einsum('sxy,spq,xpi,yqj->ij', s,
                                          g1[ic,jc], ui.conj(), uj)
    return gspinor.transpose(0,2,1,3).reshape(nci*(li*4+2),-1)

def cache_fac(l, r):
    facs = numpy.empty((3,l+1,l+1))
    for i in range(l+1):
        for j in range(i+1):
            facs[0,i,j] = scipy.special.binom(i,j) * r[0]**(i-j)
            facs[1,i,j] = scipy.special.binom(i,j) * r[1]**(i-j)
            facs[2,i,j] = scipy.special.binom(i,j) * r[2]**(i-j)
    return facs

def sph_in(l, xs):
    '''Modified spherical Bessel function of the first kind'''
    return numpy.asarray([scipy.special.spherical_in(numpy.arange(l+1), x) for x in xs]).T

def sph_ine(l, xs):
    '''exponentially scaled modified spherical Bessel function'''
    bval = sph_in(l, xs)
    return numpy.einsum('ij,j->ij', bval, numpy.exp(-xs))

def loop_xyz(nx, ny, nz):
    for ix in range(nx+1):
        for iy in range(ny+1):
            for iz in range(nz+1):
                yield ix, iy, iz

def loop_cart(l):
    for ix in reversed(range(l+1)):
        for iy in reversed(range(l-ix+1)):
            iz = l - ix - iy
            yield ix, iy, iz

def rad_part(mol, ecpbas, rs):
    ur = numpy.zeros_like(rs)
    for ecpsh in ecpbas:
        npk = ecpsh[NPRIM_OF]
        r_order = ecpsh[RADI_POWER]
        ak = mol._env[ecpsh[PTR_EXP]:ecpsh[PTR_EXP]+npk]
        ck = mol._env[ecpsh[PTR_COEFF]:ecpsh[PTR_COEFF]+npk]
        u1 = numpy.zeros_like(ur)
        for kp, a1 in enumerate(ak):
            u1 += ck[kp] * numpy.exp(-a1*rs**2)
        u1 *= rs**r_order
        ur += u1
    return ur

def facs_rad(mol, ish, lc, r_ca, rs):
    facs = []
    li = mol.bas_angular(ish)
    ai = mol.bas_exp(ish)
    ci = mol._libcint_ctr_coeff(ish)
    npi = mol.bas_nprim(ish)
    for ip in range(npi):
        ka = 2*ai[ip]*r_ca
        facs.append(numpy.einsum('ij,j->ij', sph_ine(li+lc, ka*rs),
                                 numpy.exp(-ai[ip]*(rs-r_ca)**2)))
    facs = numpy.einsum('pk,pij->kij', ci, facs)
    return facs

# x**n*y**n*z**n * c2s * c2s.T, to project out 3s, 4p, ...
def type1_ang_part(lmax, rij):
    norm_rij = numpy.linalg.norm(rij)
    if norm_rij > 1e-18:
        unitr = rij/norm_rij
    else:
        unitr = rij
    omega_nuc = []
    for lmb in range(lmax+1):
        c2smat = c2s_bra(lmb, numpy.eye((lmb+1)*(lmb+2)//2))
        omega_nuc.append(4*numpy.pi * numpy.dot(ang_nuc_part(lmb, unitr), c2smat))

    omega = numpy.empty((lmax+1,lmax+1,lmax+1,lmax+1))
    for lmb in range(lmax+1):
        omega_elec = numpy.empty((lmb+1)*(lmb+2)//2)
        for i in range(lmax+1):
            for j in range(lmax+1-i):
                for k in range(lmax+1-i-j):
                    if (i+j+k+lmb) % 2 == 0:
                        for n, (i1, j1, k1) in enumerate(loop_cart(lmb)):
                            omega_elec[n] = int_unit_xyz(i+i1, j+j1, k+k1)
                        omega[lmb,i,j,k] = numpy.dot(omega_nuc[lmb], omega_elec)
                    else:
                        omega[lmb,i,j,k] = 0
    return omega

def type2_ang_part(li, lc, ri):
    # [lambda,m,a,b,c]
    norm_ri = numpy.linalg.norm(ri)
    if norm_ri > 1e-18:
        unitr = ri/norm_ri
    else:
        unitr = ri
    omega = numpy.empty((li+1,li+1,li+1,lc*2+1,li+lc+1))
    lcart = (lc+1)*(lc+2)//2
    omega_nuc = []
    for lmb in range(li+lc+1):
        c2smat = c2s_bra(lmb, numpy.eye((lmb+1)*(lmb+2)//2))
        omega_nuc.append(4*numpy.pi * numpy.dot(ang_nuc_part(lmb, unitr), c2smat))
    tmp = numpy.empty((lcart,li+lc+1))
    for a in range(li+1):
        for b in range(li+1-a):
            for c in range(li+1-a-b):
                for lmb in range(li+lc+1):
                    if (lc+a+b+c+lmb) % 2 == 0:
                        omega_xyz = numpy.empty((lcart, (lmb+1)*(lmb+2)//2))
                        for m,(u,v,w) in enumerate(loop_cart(lc)):
                            for n, (i1, j1, k1) in enumerate(loop_cart(lmb)):
                                omega_xyz[m,n] = int_unit_xyz(a+u+i1, b+v+j1, c+w+k1)
                        tmp[:,lmb] = numpy.dot(omega_xyz, omega_nuc[lmb])
                    else:
                        tmp[:,lmb] = 0
                omega[a,b,c,:,:] = c2s_bra(lc, tmp)
    return omega

def angular_moment_matrix(l):
    '''Matrix of angular moment operator l*1j on the real spherical harmonic
    basis'''
    lz = numpy.diag(numpy.arange(-l, l+1, dtype=numpy.complex128))
    lx = numpy.zeros_like(lz)
    ly = numpy.zeros_like(lz)
    for mi in range(-l, l+1):
        mj = mi + 1
        if mj <= l:
            lx[l+mi,l+mj] = .5  * ((l+mj)*(l-mj+1))**.5
            ly[l+mi,l+mj] = .5j * ((l+mj)*(l-mj+1))**.5

        mj = mi - 1
        if mj >= -l:
            lx[l+mi,l+mj] = .5  * ((l-mj)*(l+mj+1))**.5
            ly[l+mi,l+mj] =-.5j * ((l-mj)*(l+mj+1))**.5

    u = sph.sph_pure2real(l)
    lx = u.conj().T.dot(lx).dot(u)
    ly = u.conj().T.dot(ly).dot(u)
    lz = u.conj().T.dot(lz).dot(u)
    return numpy.array((lx, ly, lz))

def facs_ang(omega, l, lc, fac_cache):
    #                 (a+b+c,cart_nlm,        m,    lambda )
    facs = numpy.zeros((l+1,(l+1)*(l+2)//2,lc*2+1,l+lc+1))
    for mi,(ix,iy,iz) in enumerate(loop_cart(l)):
        for i1, i2, i3 in loop_xyz(ix, iy, iz):
            fac = fac_cache[0,ix,i1] * fac_cache[1,iy,i2] * fac_cache[2,iz,i3]
            facs[i1+i2+i3,mi,:,:] += fac * omega[i1,i2,i3]
    return facs

def ang_nuc_part(l, rij):
    omega_xyz = numpy.empty((l+1)*(l+2)//2)
    k = 0
    for i1 in reversed(range(l+1)):
        for j1 in reversed(range(l-i1+1)):
            k1 = l - i1 - j1
            omega_xyz[k] = rij[0]**i1 * rij[1]**j1 * rij[2]**k1
            k += 1
    if l == 0:
        return omega_xyz * 0.282094791773878143
    elif l == 1:
        return omega_xyz * 0.488602511902919921
    else:
        omega = numpy.empty((2*l+1))
        fc2s = libecp.CINTc2s_ket_sph
        fc2s(omega.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
             omega_xyz.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(l))
        return omega

def int_unit_xyz(i, j, k):
    if i % 2 or j % 2 or k % 2:
        return 0
    else:
        return (_fac2[i-1] * _fac2[j-1] * _fac2[k-1] / _fac2[i+j+k+1])

_fac2 = factorial2(numpy.arange(80))
_fac2[-1] = 1

def c2s_bra(l, gcart):
    if l == 0:
        return gcart * 0.282094791773878143
    elif l == 1:
        return gcart * 0.488602511902919921
    else:
        m = gcart.shape[1]
        gsph = numpy.empty((l*2+1,m))
        fc2s = libecp.CINTc2s_ket_sph
        fc2s(gsph.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(m),
             gcart.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(l))
        return gsph

class KnownValues(unittest.TestCase):
    def test_bessel(self):
        rs = radi.gauss_chebyshev(99)[0]
        bessel1 = numpy.empty(8)
        for i,x in enumerate(rs):
            bessel0 = scipy.special.spherical_in(numpy.arange(7+1), x) * numpy.exp(-x)
            libecp.ECPsph_ine(bessel1.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(7), ctypes.c_double(x))
            self.assertTrue(numpy.allclose(bessel0, bessel1))

    def test_gauss_chebyshev(self):
        rs0, ws0 = radi.gauss_chebyshev(99)
        rs = numpy.empty_like(rs0)
        ws = numpy.empty_like(ws0)
        libecp.ECPgauss_chebyshev(rs.ctypes.data_as(ctypes.c_void_p),
                                  ws.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(99))
        self.assertTrue(numpy.allclose(rs0, rs))
        self.assertTrue(numpy.allclose(ws0, ws))

    def test_rad_part(self):
        rs, ws = radi.gauss_chebyshev(99)
        ur0 = rad_part(mol, mol._ecpbas, rs)
        ur1 = numpy.empty_like(ur0)
        cache = numpy.empty(100000)
        libecp.ECPrad_part(ur1.ctypes.data_as(ctypes.c_void_p),
                           rs.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(0), ctypes.c_int(len(rs)), ctypes.c_int(1),
                           (ctypes.c_int*2)(0, len(mol._ecpbas)),
                           mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                           mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                           mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                           mol._env.ctypes.data_as(ctypes.c_void_p),
                           lib.c_null_ptr(), cache.ctypes.data_as(ctypes.c_void_p))
        self.assertTrue(numpy.allclose(ur0, ur1))

    def test_type2_ang_part(self):
        numpy.random.seed(3)
        rca = numpy.random.random(3)
        cache = numpy.empty(100000)
        def type2_facs_ang(li, lc):
            i_fac_cache = cache_fac(li, rca)
            facs0 = facs_ang(type2_ang_part(li, lc, -rca), li, lc, i_fac_cache)
            facs1 = numpy.empty_like(facs0)
            libecp.type2_facs_ang(facs1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(li), ctypes.c_int(lc),
                                  rca.ctypes.data_as(ctypes.c_void_p),
                                  cache.ctypes.data_as(ctypes.c_void_p))
            self.assertTrue(numpy.allclose(facs0, facs1))
        for li in range(6):
            for lc in range(5):
                type2_facs_ang(li, lc)

    def test_type2_rad_part(self):
        rc = .8712
        rs, ws = radi.gauss_chebyshev(99)
        cache = numpy.empty(100000)
        def type2_facs_rad(ish, lc):
            facs0 = facs_rad(mol, ish, lc, rc, rs).transpose(0,2,1).copy()
            facs1 = numpy.empty_like(facs0)
            libecp.type2_facs_rad(facs1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(ish), ctypes.c_int(lc),
                                  ctypes.c_double(rc),
                                  rs.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(len(rs)), ctypes.c_int(1),
                                  mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                                  mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                                  mol._env.ctypes.data_as(ctypes.c_void_p),
                                  cache.ctypes.data_as(ctypes.c_void_p))
            self.assertTrue(numpy.allclose(facs0, facs1))
        for ish in range(mol.nbas):
            for lc in range(5):
                type2_facs_rad(ish, lc)

    def test_type2(self):
        cache = numpy.empty(100000)
        def gen_type2(shls):
            di = (mol.bas_angular(shls[0])*2+1) * mol.bas_nctr(shls[0])
            dj = (mol.bas_angular(shls[1])*2+1) * mol.bas_nctr(shls[1])
            mat0 = numpy.zeros((di,dj))
            for ia in range(mol.natm):
                ecpbas = mol._ecpbas[mol._ecpbas[:,ATOM_OF] == ia]
                if len(ecpbas) == 0:
                    continue
                mat0 += type2_by_shell(mol, shls, ia, ecpbas)
            mat1 = numpy.empty(mat0.shape, order='F')
            libecp.ECPtype2_sph(mat1.ctypes.data_as(ctypes.c_void_p),
                                (ctypes.c_int*2)(*shls),
                                mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(len(mol._ecpbas)),
                                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                                mol._env.ctypes.data_as(ctypes.c_void_p),
                                lib.c_null_ptr(), cache.ctypes.data_as(ctypes.c_void_p))
            if not numpy.allclose(mat0, mat1, atol=1e-8):
                print(i, j, 'error = ', numpy.linalg.norm(mat0-mat1))
            self.assertTrue(numpy.allclose(mat0, mat1, atol=1e-6))
            mat2 = gto.ecp.type2_by_shell(mol, shls)
            self.assertTrue(numpy.allclose(mat0, mat2, atol=1e-6))
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                gen_type2((i,j))

    def test_type1_state_fac(self):
        numpy.random.seed(3)
        ri = numpy.random.random(3) - .5
        cache = numpy.empty(100000)
        def tfacs(li):
            facs0 = type1_cache_fac(li, ri)
            facs1 = numpy.zeros_like(facs0)
            libecp.type1_static_facs(facs1.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(li),
                                     ri.ctypes.data_as(ctypes.c_void_p),
                                     cache.ctypes.data_as(ctypes.c_void_p))
            self.assertTrue(numpy.allclose(facs0, facs1))
        for l in range(6):
            tfacs(l)

    def test_type1_rad_ang(self):
        numpy.random.seed(4)
        ri = numpy.random.random(3) - .5
        def tfacs(lmax):
            rad_all = numpy.random.random((lmax+1,lmax+1))
            rad_ang0 = type1_rad_ang(lmax, ri, rad_all)
            rad_ang1 = numpy.empty_like(rad_ang0)
            libecp.type1_rad_ang(rad_ang1.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(lmax),
                                 ri.ctypes.data_as(ctypes.c_void_p),
                                 rad_all.ctypes.data_as(ctypes.c_void_p))
            self.assertTrue(numpy.allclose(rad_ang0, rad_ang1))
        for l in range(13):
            tfacs(l)

    def test_type1_rad(self):
        k = 1.621
        aij = .792
        rs, ws = radi.gauss_chebyshev(99)
        ur = rad_part(mol, mol._ecpbas, rs) * ws
        cache = numpy.empty(100000)
        def gen_type1_rad(li):
            rad_all0 = type1_rad_part(li, k, aij, ur, rs)
            rad_all1 = numpy.zeros_like(rad_all0)
            libecp.type1_rad_part(rad_all1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(li),
                                  ctypes.c_double(k), ctypes.c_double(aij),
                                  ur.ctypes.data_as(ctypes.c_void_p),
                                  rs.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(len(rs)), ctypes.c_int(1),
                                  cache.ctypes.data_as(ctypes.c_void_p))
            self.assertTrue(numpy.allclose(rad_all0, rad_all1))
        for l in range(13):
            gen_type1_rad(l)

    def test_type1(self):
        def gen_type1(shls):
            di = (mol.bas_angular(shls[0])*2+1) * mol.bas_nctr(shls[0])
            dj = (mol.bas_angular(shls[1])*2+1) * mol.bas_nctr(shls[1])
            mat0 = numpy.zeros((di,dj))
            for ia in range(mol.natm):
                ecpbas = mol._ecpbas[mol._ecpbas[:,ATOM_OF] == ia]
                if len(ecpbas) == 0:
                    continue
                ecpbas0 = ecpbas[ecpbas[:,ANG_OF] < 0]
                if len(ecpbas0) == 0:
                    continue
                mat0 += type1_by_shell(mol, shls, ia, ecpbas0)
            mat1 = numpy.empty(mat0.shape, order='F')
            cache = numpy.empty(100000)
            libecp.ECPtype1_sph(mat1.ctypes.data_as(ctypes.c_void_p),
                                (ctypes.c_int*2)(*shls),
                                mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(len(mol._ecpbas)),
                                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                                mol._env.ctypes.data_as(ctypes.c_void_p),
                                lib.c_null_ptr(), cache.ctypes.data_as(ctypes.c_void_p))
            if not numpy.allclose(mat0, mat1, atol=1e-8):
                print(i, j, numpy.linalg.norm(mat0-mat1))
            self.assertTrue(numpy.allclose(mat0, mat1, atol=1e-6))
            mat2 = gto.ecp.type1_by_shell(mol, shls)
            self.assertTrue(numpy.allclose(mat0, mat2, atol=1e-6))
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                gen_type1((i,j))

    def test_so_1atom(self):
        mol = gto.M(atom='''
                    Na 0.5 0.5 0.
                    ''',
                    charge=1,
                    basis={'Na': [(0, (1, 1)), (1, (4, 1)), (1, (1, 1)), (2, (1, 1))]},
                    ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 8
Na ul
1      0.    -3.    -3.
Na S
1      0.    -3.    -3.
Na P
1      0.    -3.    -3.
Na D
1      0.    -3.    -3.
Na F
1      0.    -3.    -3.
''')})
        def gen_so(shls):
            mat0 = 0
            for ia in range(mol.natm):
                ecpbas = mol._ecpbas[(mol._ecpbas[:,ATOM_OF]==ia) &
                                     (mol._ecpbas[:,SO_TYPE_OF]==1)]
                if len(ecpbas) == 0:
                    continue
                mat0 += so_by_shell(mol, shls, ia, ecpbas)

            s = lib.PauliMatrices * .5
            ui = numpy.asarray(gto.sph2spinor_l(mol.bas_angular(shls[0])))
            uj = numpy.asarray(gto.sph2spinor_l(mol.bas_angular(shls[1])))
            ref = numpy.einsum('sxy,spq,xpi,yqj->ij', s,
                               mol.intor_by_shell('int1e_inuc_rxp', shls),
                               ui.conj(), uj)
            self.assertAlmostEqual(abs(ref-mat0).max(), 0, 12)

            mat2 = .5 * gto.ecp.so_by_shell(mol, shls)
            self.assertTrue(numpy.allclose(ref, mat2, atol=1e-6))
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                gen_so((i,j))


if __name__ == '__main__':
    print('Full Tests for ecp')
    unittest.main()
