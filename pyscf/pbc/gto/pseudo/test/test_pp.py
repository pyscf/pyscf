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
import numpy as np
import pyscf.dft
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.data.nist import BOHR


def get_pp_loc_part2(cell, kpt=np.zeros(3)):
    coords = gen_grid.gen_uniform_grids(cell)
    aoR = numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    G = lib.norm(cell.Gv, axis=1)
    vlocG = np.zeros((cell.natm,len(G)))
    for ia in range(cell.natm):
        Zia = cell.atom_charge(ia)
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            vlocG[ia] = 0
            continue
        pp = cell._pseudo[symb]
        rloc, nexp, cexp = pp[1:3+1]

        G_red = G*rloc
        cfacs = np.array(
                [1*G_red**0,
                 3 - G_red**2,
                 15 - 10*G_red**2 + G_red**4,
                 105 - 105*G_red**2 + 21*G_red**4 - G_red**6])

        with np.errstate(divide='ignore'):
            # Note the signs -- potential here is positive
            vlocG[ia,:] = (# 4*np.pi * Zia * np.exp(-0.5*G_red**2)/G**2
                           - (2*np.pi)**(3/2.)*rloc**3*np.exp(-0.5*G_red**2)*(
                                np.dot(cexp, cfacs[:nexp])) )

    vpplocG = -np.sum(SI * vlocG, axis=0)
    vpplocR = tools.ifft(vpplocG, cell.mesh).real
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)
    if aoR.dtype == np.double:
        return vpploc.real
    else:
        return vpploc

def get_pp_nl(cell, kpt=np.zeros(3)):
    coords = gen_grid.gen_uniform_grids(cell)
    aoR = numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()
    SI = cell.get_SI()
    aokG = tools.fftk(np.asarray(aoR.T, order='C'),
                      cell.mesh, np.exp(-1j*np.dot(coords, kpt))).T
    ngrids = len(aokG)

    fakemol = pyscf.gto.Mole()
    fakemol._atm = np.zeros((1,pyscf.gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,pyscf.gto.BAS_SLOTS), dtype=np.int32)
    ptr = pyscf.gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,pyscf.gto.NPRIM_OF ] = 1
    fakemol._bas[0,pyscf.gto.NCTR_OF  ] = 1
    fakemol._bas[0,pyscf.gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,pyscf.gto.PTR_COEFF] = ptr+4
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
                fakemol._bas[0,pyscf.gto.ANG_OF] = l
                fakemol._env[ptr+3] = .5*rl**2
                fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                pYlm_part = pyscf.dft.numint.eval_ao(fakemol, Gv, deriv=0)

                pYlm = np.empty((nl,l*2+1,ngrids))
                for k in range(nl):
                    qkl = pseudo.pp._qli(G_rad*rl, l, k)
                    pYlm[k] = pYlm_part.T * qkl
                # pYlm is real
                SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
    vppnl *= (1./ngrids**2)

    if aoR.dtype == np.double:
        return vppnl.real
    else:
        return vppnl


def get_pp(cell, kpt=np.zeros(3)):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    aoR = numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    vlocG = pseudo.get_vlocG(cell)
    vlocG[:,0] = 0
    vpplocG = -np.sum(SI * vlocG, axis=0)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.mesh)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokG = np.empty(aoR.shape, np.complex128)
    expmikr = np.exp(-1j*np.dot(coords,kpt))
    for i in range(nao):
        aokG[:,i] = tools.fftk(aoR[:,i], cell.mesh, expmikr)
    ngrids = len(aokG)

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    hs, projGs = pseudo.get_projG(cell, kpt)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                SPG_lm_aoG = np.zeros((nl,nao), dtype=np.complex128)
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    SPG_lm_aoG[i,:] = np.einsum('g,gp->p', SPG_lmi.conj(), aokG)
                for i in range(nl):
                    for j in range(nl):
                        # Note: There is no (-1)^l here.
                        vppnl += h[i,j]*np.einsum('p,q->pq',
                                                   SPG_lm_aoG[i,:].conj(),
                                                   SPG_lm_aoG[j,:])
    vppnl *= (1./ngrids**2)

    ovlp = cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpt)
    vpploc += 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * ovlp
    return vpploc + vppnl


class KnowValues(unittest.TestCase):
    def test_pp_int(self):
        L = 2.
        n = 20
        cell = pbcgto.Cell()
        cell.atom = 'He  1.  .1  .3; He  .0  .8  1.1'
        cell.a = np.eye(3) * L
        cell.mesh = [n] * 3
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [1, (1.2, 1.0)]
                             ]}
        cell.pseudo = {'He': pbcgto.pseudo.parse('''
He
    2
     0.40000000    3    -1.98934751    -0.75604821    0.95604821
    2
     0.29482550    3     1.23870466    .855         .3
                                       .71         -1.1
                                                    .9
     0.32235865    2     2.25670239    -0.39677748
                                        0.93894690
                                                 ''')}
        cell.precision = 1e-11
        cell.build()
        np.random.seed(9)
        kpt = np.random.random(3)

        ref = get_pp_nl(cell)
        dat = pp_int.get_pp_nl(cell)
        self.assertTrue(dat.dtype == np.double)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 11)

        ref = get_pp_nl(cell, kpt)
        dat = pp_int.get_pp_nl(cell, (kpt,kpt))
        self.assertTrue(dat.dtype == np.complex128)
        self.assertAlmostEqual(abs(ref-dat[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(ref-dat[1]).max(), 0, 11)

        ref = get_pp_loc_part2(cell)
        dat = pp_int.get_pp_loc_part2(cell)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 8)

        ref = get_pp_loc_part2(cell, kpt)
        dat = pp_int.get_pp_loc_part2(cell, kpt)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 8)

    def test_pp_loc_part2(self):
        cell = pbcgto.Cell()
        cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                      'C' :'gth-szv',}
        cell.pseudo = {'C':'gth-pade'}
        cell.a = np.eye(3) * 2.5
        cell.mesh = [30] * 3
        cell.precision = 1e-9
        cell.build()
        np.random.seed(1)
        kpt = np.random.random(3)

        ref = get_pp_loc_part2(cell)
        dat = pp_int.get_pp_loc_part2(cell)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        ref = get_pp_loc_part2(cell, kpt)
        dat = pp_int.get_pp_loc_part2(cell, kpt)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_pp(self):
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
        cell.a = np.diag([4, 4, 4])
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.mesh = [20]*3
        cell.build()

        np.random.seed(1)
        k = np.random.random(3)
        v0 = get_pp(cell, k)
        v1 = pseudo.get_pp(cell, k)
        self.assertAlmostEqual(abs(v0-v1).max(), 0, 6)

    def test_pp_nuc_grad(self):
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0; Na 0 0 0.8'
        cell.a = np.diag([6,6,6])
        cell.basis='gth-szv'
        cell.pseudo='gth-pade'
        cell.ke_cutoff=200
        cell.build()

        cellp = cell.copy()
        cellp.atom = 'H 0 0 0; Na 0 0 0.8001'
        cellp.build()

        cellm = cell.copy()
        cellm.atom = 'H 0 0 0; Na 0 0 0.7999'
        cellm.build()

        np.random.seed(1)
        dm = np.random.rand(cell.nao, cell.nao)
        dm = (dm + dm.T) / 2

        # local_part2
        vp = pp_int.get_pp_loc_part2(cellp)
        vm = pp_int.get_pp_loc_part2(cellm)
        v_fd = (vp - vm) / (0.0002 / BOHR)
        grad = pp_int.vpploc_part2_nuc_grad(cell, dm)[1,2]
        grad_fd = np.einsum("ij,ij->", v_fd, dm)
        self.assertAlmostEqual(abs(grad - grad_fd), 0, 7)

        # non-local
        vp = pp_int.get_pp_nl(cellp)
        vm = pp_int.get_pp_nl(cellm)
        v_fd = (vp - vm) / (0.0002 / BOHR)
        grad = pp_int.vppnl_nuc_grad(cell, dm)[1,2]
        grad_fd = np.einsum("ij,ij->", v_fd, dm)
        self.assertAlmostEqual(abs(grad - grad_fd), 0, 7)

if __name__ == '__main__':
    print("Full Tests for pbc.gto.pseudo")
    unittest.main()
