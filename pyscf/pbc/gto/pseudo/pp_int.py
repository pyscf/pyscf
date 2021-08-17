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
from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf.pbc.lib.kpts_helper import gamma_point

EPS_PPL = getattr(__config__, "pbc_gto_pseudo_eps_ppl", 1e-2)

libpbc = lib.load_library('libpbc')

def get_pp_loc_part1(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to erf
    '''
    raise NotImplementedError


def get_pp_loc_part1_gs(cell, Gv):
    from pyscf.pbc import tools
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
    except:
        raise RuntimeError("Failed to get vlocG part1.")
    return out


def get_gth_vlocG_part1(cell, Gv):
    '''PRB, 58, 3641 Eq (5) first term
    '''
    from pyscf.pbc import tools
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = numpy.einsum('ix,ix->i', Gv, Gv)
    #G2 = lib.multiply_sum(Gv, Gv, axis=1)
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
                vlocG[ia] *= lib.exp(-0.5*rloc**2 * G2)
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
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    buf = 0
    if gamma_point(kpts_lst):
        buf = get_pp_loc_part2_gamma_smallmem(cell, intors, kptij_lst)
    else:
        for cn in range(1, 5):
            fakecell = fake_cell_vloc(cell, cn)
            if fakecell.nbas > 0:
                v = incore.aux_e2(cell, fakecell, intors[cn], aosym='s2', comp=1,
                                  kptij_lst=kptij_lst)
                buf += numpy.einsum('...i->...', v)

    if isinstance(buf, int):
        if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
            pass
        else:
            lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                             'were not found in the system.', cell._pseudo.keys())
        vpploc = [0] * nkpts
    else:
        buf = buf.reshape(nkpts,-1)
        vpploc = []
        for k, kpt in enumerate(kpts_lst):
            v = lib.unpack_tril(buf[k])
            if abs(kpt).sum() < 1e-9:  # gamma_point:
                v = v.real
            vpploc.append(v)
    if kpts is None or numpy.shape(kpts) == (3,):
        vpploc = vpploc[0]
    return vpploc


def get_pp_loc_part2_gamma(cell, intors, kptij_lst):
    from pyscf.pbc.df import incore
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list
    Ls = cell.get_lattice_Ls()
    max_memory = cell.max_memory - lib.current_memory()[0]
    nao_pair = cell.nao * (cell.nao + 1) // 2
    blksize = max(1, int(max_memory*.95*1e6 / (nao_pair*8))-2)
    count = 0
    buf = None
    for cn in range(1, 5):
        fakecell = fake_cell_vloc(cell, cn)
        if fakecell.nbas > 0:
            neighbor_list = build_neighbor_list_for_shlpairs(fakecell, cell, Ls)
            for ib0 in range(0, fakecell.nbas, blksize):
                ib1 = min(ib0 + blksize, fakecell.nbas)
                v = incore.aux_e2(cell, fakecell, intors[cn], aosym='s2', comp=1,
                                  kptij_lst=kptij_lst,
                                  shls_slice=(0, cell.nbas, 0, cell.nbas, ib0, ib1),
                                  neighbor_list=neighbor_list)
                if count == 0:
                    buf = lib.sum(v, axis=-1)
                else:
                    buf = lib.add(buf, lib.sum(v, axis=-1), out=buf)
                v = None
                count += 1
            free_neighbor_list(neighbor_list)
    return buf


def get_pp_loc_part2_gamma_smallmem(cell, intors, kptij_lst):
    from pyscf.pbc.df import incore
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list
    Ls = cell.get_lattice_Ls()
    count = 0
    buf = None
    for cn in range(1, 5):
        fakecell = fake_cell_vloc(cell, cn)
        if fakecell.nbas > 0:
            neighbor_list = build_neighbor_list_for_shlpairs(fakecell, cell, Ls)
            v = incore.aux_e2_sum_auxbas(cell, fakecell, intors[cn], aosym='s2', comp=1,
                                         kptij_lst=kptij_lst, neighbor_list=neighbor_list)
            if count == 0:
                buf = v
            else:
                buf = lib.add(buf, v, out=buf)
            v = None
            count += 1
            free_neighbor_list(neighbor_list)
    return buf


def _contract_ppnl_gamma(cell, fakecell, hl_blocks, ppnl_half, kpts=None):
    # XXX only works for Gamma point
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    offset = [0] * 3
    hl_table = numpy.empty((len(hl_blocks),6), order='C', dtype=numpy.int32)
    hl_data = []
    ptr = 0
    for ib, hl in enumerate(hl_blocks):
        hl_dim = hl.shape[0]
        hl_table[ib,0], hl_table[ib,1] = hl_dim, ptr
        ptr += hl_dim**2
        hl_data.extend(list(hl.ravel()))
        l = fakecell.bas_angular(ib)
        nd = 2 * l + 1
        hl_table[ib,2] = nd
        for i in range(hl_dim):
            p0 = offset[i]
            hl_table[ib, i+3] = offset[i]
            offset[i] += nd
    hl_data = numpy.asarray(hl_data, order='C', dtype=numpy.double)
    ppnl = []
    nao = cell.nao_nr()
    for k, kpt in enumerate(kpts_lst):
        ptr_ppnl_half0 = lib.c_null_ptr()
        ptr_ppnl_half1 = lib.c_null_ptr()
        ptr_ppnl_half2 = lib.c_null_ptr()
        if len(ppnl_half[0]) > 0:
            ppnl_half0 = ppnl_half[0][k].real
            ppnl_half0 = numpy.asarray(ppnl_half0, order='C', dtype=numpy.double)
            ptr_ppnl_half0 = ppnl_half0.ctypes.data_as(ctypes.c_void_p)
        if len(ppnl_half[1]) > 0:
            ppnl_half1 = ppnl_half[1][k].real
            ppnl_half1 = numpy.asarray(ppnl_half1, order='C', dtype=numpy.double)
            ptr_ppnl_half1 = ppnl_half1.ctypes.data_as(ctypes.c_void_p)
        if len(ppnl_half[2]) > 0:
            ppnl_half2 = ppnl_half[2][k].real
            ppnl_half2 = numpy.asarray(ppnl_half2, order='C', dtype=numpy.double)
            ptr_ppnl_half2 = ppnl_half2.ctypes.data_as(ctypes.c_void_p)

        ppnl_k = numpy.empty((nao,nao), order='C', dtype=numpy.double)
        fn = getattr(libpbc, "contract_ppnl", None)
        try:
            fn(ppnl_k.ctypes.data_as(ctypes.c_void_p), 
               ptr_ppnl_half0, ptr_ppnl_half1, ptr_ppnl_half2, 
               hl_table.ctypes.data_as(ctypes.c_void_p),
               hl_data.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(len(hl_blocks)),
               ctypes.c_int(nao))
        except:
            raise RuntimeError("Failed to contract ppnl.")
        ppnl.append(ppnl_k)

    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl


def get_pp_nl(cell, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    nao = cell.nao_nr()

    if gamma_point(kpts_lst):
        return _contract_ppnl_gamma(cell, fakecell, hl_blocks, ppnl_half, kpts)

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
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
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
    fakecell._atm = numpy.asarray(fake_atm, dtype=numpy.int32)
    fakecell._bas = numpy.asarray(fake_bas, dtype=numpy.int32)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), dtype=numpy.double)
    fakecell.precision = EPS_PPL
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
    intopt = lib.c_null_ptr()

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
            ao_loc.ctypes.data_as(ctypes.c_void_p), intopt, lib.c_null_ptr(),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        return out

    hl_dims = numpy.asarray([len(hl) for hl in hl_blocks])
    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
    return out

