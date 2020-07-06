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
#
# Author: Yang Gao <younggao1994@gmail.com>

#
'''
Non-relativistic analytical nuclear gradients for restricted Hartree Fock with kpoints sampling
'''
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as molgrad
from pyscf.pbc.gto.pseudo.pp import get_vlocG, get_alphas, get_projG, projG_li, _qli
from pyscf.pbc.dft.numint import eval_ao_kpts
from pyscf.pbc import gto, tools
from pyscf.gto import mole
import scipy
import time

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of KRHF/KRKS gradients
    Args:
        mf_grad : pbc.grad.krhf.Gradients or pbc.grad.krks.Gradients object
    '''
    mf = mf_grad.base
    cell = mf_grad.cell
    kpts = mf.kpts
    nkpts = len(kpts)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(cell.natm)

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    hcore_deriv = mf_grad.hcore_generator(cell, kpts)
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(dm0, kpts)
    log.timer('gradients of 2e part', *t0)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    aoslices = cell.aoslice_by_atom()
    de = np.zeros([len(atmlst),3])
    for x, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        h1ao = hcore_deriv(ia)
        de[x] += np.einsum('xkij,kji->x', h1ao, dm0).real
        # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[x] += np.einsum('xkij,kji->x', vhf[:,:,p0:p1], dm0[:,:,p0:p1]).real * 2
        de[x] -= np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2
        de[x] /= nkpts
        de[x] += mf_grad.extra_force(ia, locals())

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(log, cell, de, atmlst)
    return de

def _make_fakemol():
    fakemol = mole.Mole()
    fakemol._atm = np.zeros((1,mole.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,mole.BAS_SLOTS), dtype=np.int32)
    ptr = mole.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,mole.NPRIM_OF ] = 1
    fakemol._bas[0,mole.NCTR_OF  ] = 1
    fakemol._bas[0,mole.PTR_EXP  ] = ptr+3
    fakemol._bas[0,mole.PTR_COEFF] = ptr+4
    return fakemol

def get_hcore(cell, kpts):
    '''Part of the nuclear gradients of core Hamiltonian'''
    h1 = np.asarray(cell.pbc_intor('int1e_ipkin', kpts=kpts))
    dtype = h1.dtype
    if cell._pseudo:
        SI=cell.get_SI()
        nao = cell.nao_nr()
        Gv = cell.Gv
        natom = cell.natm
        coords = cell.get_uniform_grids()
        ngrids, nkpts = len(coords), len(kpts)
        vlocG = get_vlocG(cell)
        vpplocG = -np.einsum('ij,ij->j', SI, vlocG)
        vpplocG[0] = np.sum(get_alphas(cell))
        vpplocR = tools.ifft(vpplocG, cell.mesh).real
        fakemol = _make_fakemol()
        ptr = mole.PTR_ENV_START
        for kn, kpt in enumerate(kpts):
            aos = eval_ao_kpts(cell, coords, kpt, deriv=1)[0]
            vloc = np.einsum('agi,g,gj->aij', aos[1:].conj(), vpplocR, aos[0])
            expir = np.exp(-1j*np.dot(coords, kpt))
            aokG = np.asarray([tools.fftk(np.asarray(ao.T, order='C'),
                              cell.mesh, expir).T for ao in aos])
            Gk = Gv + kpt
            G_rad = lib.norm(Gk, axis=1)
            vnl = np.zeros(vloc.shape, dtype=np.complex128)
            for ia in range(natom):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl >0:
                        hl = np.asarray(hl)
                        fakemol._bas[0,mole.ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)
                        pYlm = np.empty((nl,l*2+1,ngrids))
                        for k in range(nl):
                            qkl = _qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                        SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                        SPG_lm_aoG = np.einsum('nmg,agp->anmp', SPG_lmi, aokG)
                        tmp = np.einsum('ij,ajmp->aimp', hl, SPG_lm_aoG[1:])
                        vnl += np.einsum('aimp,imq->apq', tmp.conj(), SPG_lm_aoG[0])
            vnl  *= (1./ngrids**2)
            if dtype == np.float64:
                h1[kn,:] += vloc.real + vnl.real
            else:
                h1[kn,:] += vloc + vnl
    else:
        raise NotImplementedError
    return h1

def get_ovlp(cell, kpts):
    return -np.asarray(cell.pbc_intor('int1e_ipovlp', kpts=kpts))

def hcore_generator(mf, cell=None, kpts=None):
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    h1 = get_hcore(cell, kpts)
    dtype = h1.dtype

    aoslices = cell.aoslice_by_atom()
    SI=cell.get_SI()  ##[natom ,grid]
    mesh = cell.mesh
    Gv = cell.Gv    ##[grid, 3]
    ngrids = len(Gv)
    coords = cell.get_uniform_grids()
    vlocG = get_vlocG(cell)  ###[natom, grid]
    ptr = mole.PTR_ENV_START
    def hcore_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        symb = cell.atom_symbol(atm_id)
        fakemol = _make_fakemol()
        vloc_g = 1j * np.einsum('ga,g->ag', Gv, SI[atm_id]*vlocG[atm_id])
        nkpts, nao = h1.shape[0], h1.shape[2]
        hcore = np.zeros([3,nkpts,nao,nao], dtype=h1.dtype)
        for kn, kpt in enumerate(kpts):

            ao = eval_ao_kpts(cell, coords, kpt)[0]
            rho = np.einsum('gi,gj->gij',ao.conj(),ao)
            for ax in range(3):
                vloc_R = tools.ifft(vloc_g[ax], mesh).real
                vloc = np.einsum('gij,g->ij', rho, vloc_R)
                hcore[ax,kn] += vloc
            rho = None
            aokG= tools.fftk(np.asarray(ao.T, order='C'),
                              mesh, np.exp(-1j*np.dot(coords, kpt))).T
            ao = None
            Gk = Gv + kpt
            G_rad = lib.norm(Gk, axis=1)
            if symb not in cell._pseudo: continue
            pp = cell._pseudo[symb]
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl >0:
                    hl = np.asarray(hl)
                    fakemol._bas[0,mole.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)
                    pYlm = np.empty((nl,l*2+1,ngrids))
                    for k in range(nl):
                        qkl = _qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    SPG_lmi = np.einsum('g,nmg->nmg', SI[atm_id].conj(), pYlm)
                    SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    SPG_lmi_G = 1j * np.einsum('nmg, ga->anmg', SPG_lmi, Gv)
                    SPG_lm_G_aoG = np.einsum('anmg, gp->anmp', SPG_lmi_G, aokG)
                    tmp_1 = np.einsum('ij,ajmp->aimp', hl, SPG_lm_G_aoG)
                    tmp_2 = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl = np.einsum('imp,aimq->apq', SPG_lm_aoG.conj(), tmp_1) + np.einsum('aimp,imq->apq',   SPG_lm_G_aoG.conj(), tmp_2)
                    vppnl *=(1./ngrids**2)
                    if dtype==np.float64:
                        hcore[:,kn] += vppnl.real
                    else:
                        hcore[:,kn] += vppnl
            hcore[:,kn,p0:p1] -= h1[kn,:,p0:p1]
            hcore[:,kn,:,p0:p1] -= h1[kn,:,p0:p1].transpose(0,2,1).conj()
        return hcore
    return hcore_deriv

def grad_nuc(cell, atmlst):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    ew_eta = cell.ew_eta
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls()
    natom = len(chargs)
    ewovrl_grad = np.zeros([natom,3])

    for i, qi in enumerate(chargs):
        ri = coords[i]
        for j in range(natom):
            if j == i:
                continue
            qj = chargs[j]
            rj = coords[j]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            r = r.reshape(len(r),1)
            ewovrl_grad[i] += np.sum(- (qi * qj / r ** 3 * r1 * \
                                    scipy.special.erfc(ew_eta * r).reshape(len(r),1)), axis = 0)
            ewovrl_grad[i] += np.sum(- qi * qj / r ** 2 * r1 * 2 * ew_eta / np.sqrt(np.pi) * \
                                    np.exp(-ew_eta**2 * r ** 2).reshape(len(r),1), axis = 0)

    mesh = gto.cell._cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2==0] = 1e200
    ewg_grad = np.zeros([natom,3])
    SI = cell.get_SI(Gv)
    if cell.low_dim_ft_type is None or cell.dimension == 3:
        coulG = 4*np.pi / absG2
        coulG *= weights
        ZSI = np.einsum("i,ij->j", chargs, SI)
        ZexpG2 = coulG * np.exp(-absG2/(4*ew_eta**2))
        ZexpG2_mod = ZexpG2.reshape(len(ZexpG2),1) * Gv
    for i, qi in enumerate(chargs):
        Zfac = np.imag(ZSI * SI[i].conj()) * qi
        ewg_grad[i] = - np.sum(Zfac.reshape((len(Zfac),1)) * ZexpG2_mod, axis = 0)

    ew_grad = ewg_grad + ewovrl_grad
    if atmlst is not None:
        ew_grad = ew_grad[atmlst]
    return ew_grad

def get_jk(mf_grad, dm, kpts):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    vj, vk = mf_grad.get_jk(dm, kpts)
    return vj, vk

def get_j(mf_grad, dm, kpts):
    vk= mf_grad.get_j(dm, kpts)
    return vj

def get_k(mf_grad, dm, kpts):
    vk= mf_grad.get_k(dm, kpts)
    return vk

def get_veff(mf_grad, dm, kpts):
    '''NR Hartree-Fock Coulomb repulsion'''
    vj, vk = mf_grad.get_jk(dm, kpts)
    return vj - vk * .5

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    nkpts = len(mo_occ)
    dm1e = [molgrad.make_rdm1e(mo_energy[k], mo_coeff[k], mo_occ[k]) for k in range(nkpts)]
    return np.asarray(dm1e)

class GradientsBasics(molgrad.GradientsBasics):
    '''
    Basic nuclear gradient functions for non-relativistic methods
    '''
    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.cell = method.cell
        self.base = method
        self.kpts = method.kpts
        self.max_memory = self.cell.max_memory
        self.atmlst = None
        self.de = None
        self._keys = set(self.__dict__.keys())

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_hcore(cell, kpts)

    hcore_generator = hcore_generator

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_ovlp(cell, kpts)

    def get_jk(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        exxdiv = self.base.exxdiv
        cpu0 = (time.clock(), time.time())
        vj, vk = self.base.with_df.get_jk_e1(dm, kpts, exxdiv=exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj = self.base.with_df.get_j_e1(dm, kpts)
        logger.timer(self, 'vj', *cpu0)
        return vj

    def get_k(self, dm=None, kpts=None, kpts_band=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        exxdiv = self.base.exxdiv
        cpu0 = (time.clock(), time.time())
        vk = self.base.with_df.get_k_e1(dm, kpts, kpts_band, exxdiv)
        logger.timer(self, 'vk', *cpu0)
        return vk

    def grad_nuc(self, cell=None, atmlst=None):
        if cell is None: cell = self.cell
        return grad_nuc(cell, atmlst)

def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "cell" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)

    class SCF_GradScanner(mf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, cell_or_geom, **kwargs):
            if isinstance(cell_or_geom, gto.Cell):
                cell = cell_or_geom
            else:
                cell = self.cell.set_geom_(cell_or_geom, inplace=False)

            mf_scanner = self.base
            e_tot = mf_scanner(cell)
            self.cell = cell

            # If second integration grids are created for RKS and UKS
            # gradients
            if getattr(self, 'grids', None):
                self.grids.reset(cell)

            de = self.kernel(**kwargs)
            return e_tot, de
    return SCF_GradScanner(mf_grad)


class Gradients(GradientsBasics):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, dm, kpts)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        #1 force from exxdiv corrections when madelung constant has non-zero derivative
        #2 DFT grid response
        return 0

    grad_elec = grad_elec
    as_scanner = as_scanner

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            self._write(self.cell, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (time.clock(), time.time())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

if __name__=='__main__':
    from pyscf.pbc import scf
    cell = gto.Cell()
    cell.atom = '''
    He 0.0 0.0 0.0
    He 1.0 1.1 1.2
    '''
    cell.basis = 'gth-dzv'
    cell.a = np.eye(3) * 3
    cell.unit='bohr'
    cell.pseudo='gth-pade'
    cell.verbose=4
    cell.build()

    nmp = [1,1,3]
    kpts = cell.make_kpts(nmp)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    kmf.kernel()
    mygrad = Gradients(kmf)
    grad = mygrad.kernel()
