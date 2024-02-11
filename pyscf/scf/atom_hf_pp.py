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

import copy
import numpy
from scipy.special import erf

from pyscf import lib
from pyscf import gto, scf
from pyscf.dft import gen_grid, numint
from pyscf.pbc import gto as pbcgto
from pyscf.scf import atom_hf, rohf

def get_pp_loc_part1_rs(mol, coords):
    atm_coords = mol.atom_coords()
    out = 0
    for ia in range(mol.natm):
        r0 = atm_coords[ia]
        r2 = numpy.sum((coords - r0)**2, axis=1)
        r = numpy.sqrt(r2)
        Zia = mol.atom_charge(ia)
        symb = mol.atom_symbol(ia)
        if symb in mol._pseudo:
            pp = mol._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
        else:
            rloc = 1e16
        alpha = 1.0 / (numpy.sqrt(2) * rloc)
        out += - Zia / r * erf(alpha * r)
    return out

def _aux_e2(cell, auxcell, intor, aosym='s1', comp=1):
    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                         cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = numpy.asarray(numpy.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                           dtype=numpy.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    nbas = cell.nbas
    shls_slice = (0, nbas, nbas, nbas*2, nbas*2, nbas*2+auxcell.nbas)
    comp = 1
    out = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice=shls_slice,
                                  comp=comp, aosym=aosym, ao_loc=ao_loc)
    return out

def get_pp_loc_part2(mol):
    buf = 0
    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    for cn in range(1, 5):
        fakecell = pbcgto.pseudo.pp_int.fake_cell_vloc(mol, cn)
        if fakecell.nbas > 0:
            v = _aux_e2(mol, fakecell, intors[cn], aosym='s2', comp=1)
            buf += numpy.einsum('...i->...', v)
    if numpy.isscalar(buf):
        vpp_loc =  buf
    else:
        vpp_loc = lib.unpack_tril(buf)
    return vpp_loc

def get_pp_loc(mol):
    # TODO use analytic integral
    grids = gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    _numint = numint.NumInt()

    vpp = 0
    for ao, mask, weight, coords in _numint.block_loop(mol, grids):
        vloc = get_pp_loc_part1_rs(mol, coords)
        vpp += numpy.einsum("g,g,gi,gj->ij", weight, vloc, ao, ao)
    vpp += get_pp_loc_part2(mol)
    return vpp

def get_pp_nl(mol):
    nao = mol.nao
    fakecell, hl_blocks = pbcgto.pseudo.pp_int.fake_cell_vnl(mol)
    ppnl_half = _int_vnl(mol, fakecell, hl_blocks)

    ppnl = numpy.zeros((nao,nao), dtype=numpy.double)
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakecell.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.double)
        for i in range(hl_dim):
            p0 = offset[i]
            ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        ppnl += numpy.einsum('ilp,ij,jlq->pq', ilp, hl, ilp)
    return ppnl

def _int_vnl(cell, fakecell, hl_blocks):
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
        nbas = len(bas)
        shls_slice = (cell.nbas, nbas, 0, cell.nbas)
        ao_loc = gto.moleintor.make_loc(bas, intor)
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        out = numpy.empty((ni,nj), dtype=numpy.double)
        comp = 1
        out = gto.moleintor.getints2c(intor, atm, bas, env, shls_slice=shls_slice, comp=comp, hermi=0,
                                      ao_loc=ao_loc, cintopt=intopt, out=out)
        return out

    hl_dims = numpy.asarray([len(hl) for hl in hl_blocks])
    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
    return out

class AtomSCFPP(atom_hf.AtomSphAverageRHF):
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h = mol.intor('int1e_kin', hermi=1)
        h += get_pp_nl(mol)
        h += get_pp_loc(mol)
        return h

class AtomHF1ePP(rohf.HF1e, AtomSCFPP):
    eig = AtomSCFPP.eig
    get_hcore = AtomSCFPP.get_hcore
