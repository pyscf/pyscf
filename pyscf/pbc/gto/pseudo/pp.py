#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''PP with numeric integration.  See also pyscf/pbc/gto/pesudo/pp_int.py

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.lib import Ylm

def get_alphas(cell):
    '''alpha parameters from the non-divergent Hartree+Vloc G=0 term.

    See ewald.pdf

    Returns:
        alphas : (natm,) ndarray
    '''
    return get_alphas_gth(cell)

def get_alphas_gth(cell):
    '''alpha parameters for the local GTH pseudopotential.'''
    G0 = np.zeros((1,3))
    return -get_gth_vlocG(cell, G0)

def get_vlocG(cell, Gv=None):
    '''Local PP kernel in G space: Vloc(G)

    Returns:
        (natm, ngrids) ndarray
    '''
    if Gv is None: Gv = cell.Gv
    vlocG = get_gth_vlocG(cell, Gv)
    return vlocG

def get_gth_vlocG(cell, Gv):
    '''Local part of the GTH pseudopotential.

    See MH (4.79).

    Args:
        Gv : (ngrids,3) ndarray

    Returns:
         (natm, ngrids) ndarray
    '''
    vlocG = pp_int.get_gth_vlocG_part1(cell, Gv)

    # Add the C1, C2, C3, C4 contributions
    G2 = np.einsum('ix,ix->i', Gv, Gv)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        pp = cell._pseudo[symb]
        rloc, nexp, cexp = pp[1:3+1]

        G2_red = G2 * rloc**2
        cfacs = 0
        if nexp >= 1:
            cfacs += cexp[0]
        if nexp >= 2:
            cfacs += cexp[1] * (3 - G2_red)
        if nexp >= 3:
            cfacs += cexp[2] * (15 - 10*G2_red + G2_red**2)
        if nexp >= 4:
            cfacs += cexp[3] * (105 - 105*G2_red + 21*G2_red**2 - G2_red**3)

        vlocG[ia,:] -= (2*np.pi)**(3/2.)*rloc**3*np.exp(-0.5*G2_red) * cfacs

    return vlocG

def get_projG(cell, kpt=np.zeros(3)):
    '''PP weight and projector for the nonlocal PP in G space.

    Returns:
        hs : list( list( np.array( , ) ) )
         - hs[atm][l][i,j]
        projs : list( list( list( list( np.array(ngrids) ) ) ) )
         - projs[atm][l][m][i][ngrids]
    '''
    return get_gth_projG(cell, kpt+cell.Gv)

def get_gth_projG(cell, Gvs):
    r'''G space projectors from the FT of the real-space projectors.

    \int e^{iGr} p_j^l(r) Y_{lm}^*(theta,phi)
    = i^l p_j^l(G) Y_{lm}^*(thetaG, phiG)

    See MH Eq.(4.80)
    '''
    Gs,thetas,phis = cart2polar(Gvs)

    hs = []
    projs = []
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        pp = cell._pseudo[symb]
        # nproj_types = pp[4]
        h_ia = []
        proj_ia = []
        for l,proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            h_ia.append( np.array(hl) )
            proj_ia_l = []
            for m in range(-l,l+1):
                projG_ang = Ylm(l,m,thetas,phis).conj()
                proj_ia_lm = []
                for i in range(nl):
                    projG_radial = projG_li(Gs,l,i,rl)
                    proj_ia_lm.append( (1j)**l * projG_radial*projG_ang )
                proj_ia_l.append(proj_ia_lm)
            proj_ia.append(proj_ia_l)
        hs.append(h_ia)
        projs.append(proj_ia)

    return hs, projs

def projG_li(G, l, i, rl):
    G = np.array(G)
    G_red = G*rl

    # MH Eq. (4.81)
    return (_qli(G_red,l,i) * np.pi**(5/4.) * G**l * np.sqrt(rl**(2*l+3))
            / np.exp(0.5*G_red**2) )

def _qli(x,l,i):
    # MH Eqs. (4.82)-(4.93) :: beware typos!
    # Mathematica formulas:
    # p[l_, i_, r_] = Sqrt[2] r^(l + 2 (i - 1)) Exp[-r^2/(2 R^2)]/(R^(l + (4 i - 1)/2) Sqrt[Gamma[l + (4 i - 1)/2]])
    # pG[l_, i_, G_] = Integrate[p[l, i, r] 4 Pi r^2 SphericalBesselJ[l, G r], {r, 0, Infinity}]
    # qG[l_, i_, G_] := pG[l, i, G]/(Pi^(5/4) G^l Sqrt[R^(2 l + 3)]/Exp[(G R)^2/2])
    # FullSimplify[qG[4, 3, G], R > 0 && G > 0]
    sqrt = np.sqrt
    if l==0 and i==0:
        return 4*sqrt(2.)
    elif l==0 and i==1:
        return 8*sqrt(2/15.)*(3-x**2) # MH & GTH (right)
        #return sqrt(8*2/15.)*(3-x**2) # HGH (wrong)
    elif l==0 and i==2:
        #return 16/3.*sqrt(2/105.)*(15-20*x**2+4*x**4) # MH (wrong)
        return 16/3.*sqrt(2/105.)*(15-10*x**2+x**4) # HGH (right)
    elif l==1 and i==0:
        return 8*sqrt(1/3.)
    elif l==1 and i==1:
        return 16*sqrt(1/105.)*(5-x**2)
    elif l==1 and i==2:
        #return 32/3.*sqrt(1/1155.)*(35-28*x**2+4*x**4) # MH (wrong)
        return 32/3.*sqrt(1/1155.)*(35-14*x**2+x**4) # HGH (right)
    elif l==2 and i==0:
        return 8*sqrt(2/15.)
    elif l==2 and i==1:
        return 16/3.*sqrt(2/105.)*(7-x**2)
    elif l==2 and i==2:
        #return 32/3.*sqrt(2/15015.)*(63-36*x**2+4*x**4) # MH (wrong I think)
        return 32/3.*sqrt(2/15015.)*(63-18*x**2+x**4) # TCB
    elif l==3 and i==0:
        return 16*sqrt(1/105.)
    elif l==3 and i==1:
        return 32/3.*sqrt(1/1155.)*(9-x**2)
    elif l==3 and i==2:
        return 64/45.*sqrt(1/1001.)*(99-22*x**2+x**4)
    elif l==4 and i==0:
        return 16/3.*sqrt(2/105.)
    elif l==4 and i==1:
        return 32/3.*sqrt(2/15015.)*(11-x**2)
    elif l==4 and i==2:
        return 64/45.*sqrt(2/17017.)*(143-26*x**2+x**4)
    else:
        print("*** WARNING *** l =", l, ", i =", i, "not yet implemented for NL PP!")
        return 0.

def Ylm_real(l,m,theta,phi):
    '''Real spherical harmonics, if desired.'''
    Ylabsm = Ylm(l,np.abs(m),theta,phi)
    if m < 0:
        return np.sqrt(2.) * Ylabsm.imag
    elif m > 0:
        return np.sqrt(2.) * Ylabsm.real
    else: # m == 0
        return Ylabsm.real

def cart2polar(rvec):
    # The rows of rvec are the 3-component vectors
    # i.e. rvec is N x 3
    x,y,z = rvec.T
    r = lib.norm(rvec, axis=1)
    # theta is the polar angle, 0 < theta < pi
    # catch possible 0/0
    theta = np.arccos(z/(r+1e-200))
    # phi is the azimuthal angle, 0 < phi < 2pi (or -pi < phi < pi)
    phi = np.arctan2(y,x)
    return r, theta, phi


def get_pp(cell, kpt=np.zeros(3)):
    '''Get the periodic pseudopotential nuc-el AO matrix
    '''
    from pyscf.pbc import tools
    coords = cell.get_uniform_grids()
    aoR = cell.pbc_eval_gto('GTOval', coords, kpt=kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    vlocG = get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)
    vpplocG[0] = np.sum(get_alphas(cell)) # from get_jvloc_G0 function

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.mesh).real
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokG = tools.fftk(np.asarray(aoR.T, order='C'),
                      cell.mesh, np.exp(-1j*np.dot(coords, kpt))).T
    ngrids = len(aokG)

    fakemol = mole.Mole()
    fakemol._atm = np.zeros((1,mole.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,mole.BAS_SLOTS), dtype=np.int32)
    ptr = mole.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,mole.NPRIM_OF ] = 1
    fakemol._bas[0,mole.NCTR_OF  ] = 1
    fakemol._bas[0,mole.PTR_EXP  ] = ptr+3
    fakemol._bas[0,mole.PTR_COEFF] = ptr+4
    Gv = np.asarray(cell.Gv+kpt)
    G_rad = lib.norm(Gv, axis=1)

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue
        pp = cell._pseudo[symb]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl > 0:
                hl = np.asarray(hl)
                fakemol._bas[0,mole.ANG_OF] = l
                fakemol._env[ptr+3] = .5*rl**2
                fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                pYlm_part = fakemol.eval_gto('GTOval', Gv)

                pYlm = np.empty((nl,l*2+1,ngrids))
                for k in range(nl):
                    qkl = _qli(G_rad*rl, l, k)
                    pYlm[k] = pYlm_part.T * qkl
                # pYlm is real
                SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
    vppnl *= (1./ngrids**2)

    if aoR.dtype == np.double:
        return vpploc.real + vppnl.real
    else:
        return vpploc + vppnl


def get_jvloc_G0(cell, kpt=np.zeros(3)):
    '''Get the (separately divergent) Hartree + Vloc G=0 contribution.
    '''
    ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt)
    return 1./cell.vol * np.sum(get_alphas(cell)) * ovlp
