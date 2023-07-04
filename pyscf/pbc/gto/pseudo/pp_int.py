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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Analytic PP integrals.  See also pyscf/pbc/gto/pesudo/pp.py

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

import ctypes
import copy
import numpy
import scipy.special
from pyscf import lib
from pyscf import gto
from pyscf import __config__

libpbc = lib.load_library('libpbc')

def get_pp_loc_part1(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to erf
    '''
    raise NotImplementedError

def get_gth_vlocG_part1(cell, Gv):
    '''PRB, 58, 3641 Eq (5) first term
    '''
    from pyscf.pbc import tools
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = numpy.einsum('ix,ix->i', Gv, Gv)
    G0idx = numpy.where(G2==0)[0]

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        vlocG = numpy.zeros((cell.natm, len(G2)))
        for ia in range(cell.natm):
            Zia = cell.atom_charge(ia)
            symb = cell.atom_symbol(ia)
            # Note the signs -- potential here is positive
            vlocG[ia] = Zia * coulG
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                vlocG[ia] *= numpy.exp(-0.5*rloc**2 * G2)
                # alpha parameters from the non-divergent Hartree+Vloc G=0 term.
                vlocG[ia,G0idx] = -2*numpy.pi*Zia*rloc**2

    elif cell.dimension == 2:
        # The following 2D ewald summation is taken from:
        # Minary, Tuckerman, Pihakari, Martyna J. Chem. Phys. 116, 5351 (2002)
        vlocG = numpy.zeros((cell.natm,len(G2)))
        b = cell.reciprocal_vectors()
        inv_area = numpy.linalg.norm(numpy.cross(b[0], b[1]))/(2*numpy.pi)**2
        lzd2 = cell.vol * inv_area / 2
        lz = lzd2*2.

        G2[G0idx] = 1e200
        Gxy = numpy.linalg.norm(Gv[:,:2],axis=1)
        Gz = abs(Gv[:,2])

        for ia in range(cell.natm):
            Zia = cell.atom_charge(ia)
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                vlocG[ia] = Zia * coulG
                continue

            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]

            ew_eta = 1./numpy.sqrt(2)/rloc
            JexpG2 = 4*numpy.pi / G2 * numpy.exp(-G2/(4*ew_eta**2))
            fac = 4*numpy.pi / G2 * numpy.cos(Gz*lzd2)
            JexpG2 -= fac * numpy.exp(-Gxy*lzd2)
            eta_z1 = (ew_eta**2 * lz + Gxy) / (2.*ew_eta)
            eta_z2 = (ew_eta**2 * lz - Gxy) / (2.*ew_eta)
            JexpG2 += fac * 0.5*(numpy.exp(-eta_z1**2)*scipy.special.erfcx(eta_z2) +
                                 numpy.exp(-eta_z2**2)*scipy.special.erfcx(eta_z1))
            vlocG[ia,:] = Zia * JexpG2

            JexpG0 = ( - numpy.pi * lz**2 / 2. * scipy.special.erf( ew_eta * lzd2 )
                       + numpy.pi/ew_eta**2 * scipy.special.erfc(ew_eta*lzd2)
                       - numpy.sqrt(numpy.pi)*lz/ew_eta * numpy.exp( - (ew_eta*lzd2)**2 ) )
            vlocG[ia,G0idx] = -2*numpy.pi*Zia*rloc**2 + Zia*JexpG0
    else:
        raise NotImplementedError('Low dimension ft_type %s'
                                  ' not implemented for dimension %d' %
                                  (cell.low_dim_ft_type, cell.dimension))
    return vlocG

# part2 Vnuc - Vloc
def get_pp_loc_part2(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
    '''
    from pyscf.pbc.df.aft import _IntPPBuilder
    vpploc = _IntPPBuilder(cell, kpts).get_pp_loc_part2()
    if kpts is None or numpy.shape(kpts) == (3,):
        vpploc = vpploc[0]
    return vpploc

def get_pp_nl(cell, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    nao = cell.nao_nr()
    buf = numpy.empty((3*9*nao), dtype=numpy.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    ppnl = numpy.zeros((nkpts,nao,nao), dtype=numpy.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                offset[i] = p0 + nd
            ppnl[k] += numpy.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)

    if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
        ppnl = ppnl.real

    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl


def fake_cell_vloc(cell, cn=0):
    '''Generate fake cell for V_{loc}.

    Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
    function.  The integral over V_{loc} can be transfered to the 3-center
    integrals, in which the auxiliary basis is given by the fake cell.

    The kwarg cn indiciates which term to generate for the fake cell.
    If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = numpy.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    half_sph_norm = .5/numpy.pi**.5
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(ia)
        if cn == 0:
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                alpha = .5 / rloc**2
            else:
                alpha = 1e16
            norm = half_sph_norm / gto.gaussian_int(2, alpha)
            fake_env.append([alpha, norm])
            fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
            ptr += 2
        elif symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            if cn <= nexp:
                alpha = .5 / rloc**2
                norm = cexp[cn-1]/rloc**(cn*2-2) / half_sph_norm
                fake_env.append([alpha, norm])
                fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
                ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = numpy.asarray(fake_atm, dtype=numpy.int32).reshape(-1, gto.ATM_SLOTS)
    fakecell._bas = numpy.asarray(fake_bas, dtype=numpy.int32).reshape(-1, gto.BAS_SLOTS)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), dtype=numpy.double)
    return fakecell

# sqrt(Gamma(l+1.5)/Gamma(l+2i+1.5))
_PLI_FAC = 1/numpy.sqrt(numpy.array((
    (1, 3.75 , 59.0625  ),  # l = 0,
    (1, 8.75 , 216.5625 ),  # l = 1,
    (1, 15.75, 563.0625 ),  # l = 2,
    (1, 24.75, 1206.5625),  # l = 3,
    (1, 35.75, 2279.0625),  # l = 4,
    (1, 48.75, 3936.5625),  # l = 5,
    (1, 63.75, 6359.0625),  # l = 6,
    (1, 80.75, 9750.5625))))# l = 7,

def fake_cell_vnl(cell):
    '''Generate fake cell for V_{nl}.

    gaussian function p_i^l Y_{lm}
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = numpy.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    hl_blocks = []
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            pp = cell._pseudo[symb]
            # nproj_types = pp[4]
            for l, (rl, nl, hl) in enumerate(pp[5:]):
                if nl > 0:
                    alpha = .5 / rl**2
                    norm = gto.gto_norm(l, alpha)
                    fake_env.append([alpha, norm])
                    fake_bas.append([ia, l, 1, 1, 0, ptr, ptr+1, 0])

#
# Function p_i^l (PRB, 58, 3641 Eq 3) is (r^{2(i-1)})^2 square normalized to 1.
# But here the fake basis is square normalized to 1.  A factor ~ p_i^l / p_1^l
# is attached to h^l_ij (for i>1,j>1) so that (factor * fake-basis * r^{2(i-1)})
# is normalized to 1.  The factor is
#       r_l^{l+(4-1)/2} sqrt(Gamma(l+(4-1)/2))      sqrt(Gamma(l+3/2))
#     ------------------------------------------ = ----------------------------------
#      r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))     sqrt(Gamma(l+2i-1/2)) r_l^{2i-2}
#
                    fac = numpy.array([_PLI_FAC[l,i]/rl**(i*2) for i in range(nl)])
                    hl = numpy.einsum('i,ij,j->ij', fac, numpy.asarray(hl), fac)
                    hl_blocks.append(hl)
                    ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = numpy.asarray(fake_atm, dtype=numpy.int32)
    fakecell._bas = numpy.asarray(fake_bas, dtype=numpy.int32)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), dtype=numpy.double)
    return fakecell, hl_blocks

def _int_vnl(cell, fakecell, hl_blocks, kpts):
    '''Vnuc - Vloc'''
    rcut = max(cell.rcut, fakecell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    nimgs = len(Ls)
    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order='C')
    nkpts = len(kpts)

    fill = getattr(libpbc, 'PBCnr2c_fill_ks1')
    cintopt = lib.c_null_ptr()

    def int_ket(_bas, intor):
        if len(_bas) == 0:
            return []
        intor = cell._add_suffix(intor)
        atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                     fakecell._atm, _bas, fakecell._env)
        atm = numpy.asarray(atm, dtype=numpy.int32)
        bas = numpy.asarray(bas, dtype=numpy.int32)
        env = numpy.asarray(env, dtype=numpy.double)
        natm = len(atm)
        nbas = len(bas)
        shls_slice = (cell.nbas, nbas, 0, cell.nbas)
        ao_loc = gto.moleintor.make_loc(bas, intor)
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        out = numpy.empty((nkpts,ni,nj), dtype=numpy.complex128)
        comp = 1

        fintor = getattr(gto.moleintor.libcgto, intor)

        drv = libpbc.PBCnr2c_drv
        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*(shls_slice[:4])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        return out

    hl_dims = numpy.asarray([len(hl) for hl in hl_blocks])
    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
    return out

