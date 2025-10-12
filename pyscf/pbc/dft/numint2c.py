#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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

'''
Numerical integration functions for (2-component) GKS and KGKS

Ref:
    Phys. Rev. Research 5, 013036
'''

import numpy as np
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft import numint2c
from pyscf.pbc.dft import numint as pnumint
from pyscf.pbc.lib.kpts import KPoints


class NumInt2C(lib.StreamObject, numint.LibXCMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''
    collinear         = numint2c.NumInt2C.collinear
    spin_samples      = numint2c.NumInt2C.spin_samples
    collinear_thrd    = numint2c.NumInt2C.collinear_thrd
    collinear_samples = numint2c.NumInt2C.collinear_samples

    make_mask = lib.invalid_method('make_mask')
    eval_ao = staticmethod(pnumint.eval_ao)
    eval_rho = staticmethod(numint2c.eval_rho)
    eval_rho2 = numint2c.NumInt2C.eval_rho2

    def eval_rho1(self, cell, ao, dm, screen_index=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=None, ao_cutoff=None, pair_mask=None,
                  verbose=None):
        return self.eval_rho(cell, ao, dm, screen_index, xctype, hermi,
                             with_lapl, verbose=verbose)

    def cache_xc_kernel(self, cell, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        kpt=None, max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        if kpt is None:
            kpt = np.zeros(3)
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = mo_coeff.shape[0]
        nao = n2c // 2

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            with_lapl = False
            rho = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpt, None,
                                       max_memory):
                rho.append(self.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask,
                                          xctype, with_lapl))
            rho = np.concatenate(rho,axis=-1)
            assert rho.dtype == np.double
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            # rhoa and rhob must be real
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            dm_a = dm[:nao,:nao].copy('C')
            dm_b = dm[nao:,nao:].copy('C')
            ni = self._to_numint1c()
            with_lapl = True
            hermi = 1
            rhoa = []
            rhob = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpt, None,
                                     max_memory):
                # rhoa and rhob must be real
                rhoa.append(ni.eval_rho(cell, ao_k1, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(cell, ao_k1, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def cache_xc_kernel1(self, cell, grids, xc_code, dm, spin=0,
                         kpt=None, max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        if kpt is None:
            kpt = np.zeros(3)
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = dm.shape[0]
        nao = n2c // 2

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            hermi = 1 # rho must be real. We need to assume dm hermitian
            with_lapl = False
            rho = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpt, None,
                                       max_memory):
                rho.append(self.eval_rho1(cell, ao_k1, dm, mask, xctype, hermi,
                                          with_lapl))
            rho = np.concatenate(rho,axis=-1)
            assert rho.dtype == np.double
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            hermi = 1
            dm_a = dm[:nao,:nao].copy('C')
            dm_b = dm[nao:,nao:].copy('C')
            ni = self._to_numint1c()
            with_lapl = True
            rhoa = []
            rhob = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpt, None,
                                     max_memory):
                rhoa.append(ni.eval_rho(cell, ao_k1, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(cell, ao_k1, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def get_rho(self, cell, dm, grids, kpt=np.zeros((1,3)), max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao]
        dm_b = dm[nao:,nao:]
        ni = self._to_numint1c()
        return ni.get_rho(cell, dm_a+dm_b, grids, kpt, max_memory)

    def _gks_mcol_vxc(self, cell, grids, xc_code, dms, relativity=0, hermi=0,
                      kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpt is None:
            kpt = np.zeros(3)
        xctype = self._xc_type(xc_code)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()

        make_rho, nset, n2c = self._gen_rho_evaluator(cell, dms, hermi, False)
        nao = n2c // 2

        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

        if xctype in ('LDA', 'GGA', 'MGGA'):
            f_eval_mat = {
                ('LDA' , 'n'): (numint2c._ncol_lda_vxc_mat , 0),
                ('LDA' , 'm'): (numint2c._mcol_lda_vxc_mat , 0),
                ('GGA' , 'm'): (numint2c._mcol_gga_vxc_mat , 1),
                ('MGGA', 'm'): (numint2c._mcol_mgga_vxc_mat, 1),
            }
            fmat, ao_deriv = f_eval_mat[(xctype, self.collinear[0])]

            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff

            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpt, kpts_band,
                                       max_memory):
                for i in range(nset):
                    rho = make_rho(i, ao_k2, mask, xctype)
                    exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
                    if xctype == 'LDA':
                        den = rho[0] * weight
                    else:
                        den = rho[0,0] * weight
                    nelec[i] += den.sum()
                    excsum[i] += np.dot(den, exc)
                    vmat[i] += fmat(cell, ao_k1, weight, rho, vxc, mask, shls_slice,
                                    ao_loc, hermi)

        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f'numint2c.get_vxc for functional {xc_code}')

        if hermi:
            vmat = vmat + vmat.conj().transpose(0,2,1)
        if isinstance(dms, np.ndarray) and dms.ndim == 2:
            vmat = vmat[0]
            nelec = nelec[0]
            excsum = excsum[0]
        return nelec, excsum, vmat

    def _gks_mcol_fxc(self, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      rho0=None, vxc=None, fxc=None, kpt=None, max_memory=2000,
                      verbose=None):
        assert self.collinear[0] == 'm'  # mcol
        if kpt is None:
            kpt = np.zeros(3)
        xctype = self._xc_type(xc_code)
        if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0,
                                        kpt=kpt, max_memory=max_memory)[2]

        if xctype == 'MGGA':
            fmat, ao_deriv = (numint2c._mcol_mgga_fxc_mat , 1)
        elif xctype == 'GGA':
            fmat, ao_deriv = (numint2c._mcol_gga_fxc_mat  , 1)
        else:
            fmat, ao_deriv = (numint2c._mcol_lda_fxc_mat  , 0)

        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        make_rho1, nset, n2c = self._gen_rho_evaluator(cell, dms, hermi, False)
        nao = n2c // 2
        vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

        if xctype in ('LDA', 'GGA', 'MGGA'):
            _rho0 = None
            p1 = 0
            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpt, None,
                                       max_memory):
                p0, p1 = p1, p1 + weight.size
                _fxc = fxc[:,:,:,:,p0:p1]
                for i in range(nset):
                    rho1 = make_rho1(i, ao_k1, mask, xctype)
                    vmat[i] += fmat(cell, ao_k1, weight, _rho0, rho1, _fxc,
                                    mask, shls_slice, ao_loc, hermi)
        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

        if hermi:
            vmat = vmat + vmat.conj().transpose(0,2,1)
        if isinstance(dms, np.ndarray) and dms.ndim == 2:
            vmat = vmat[0]
        return vmat

    @lib.with_doc(pnumint.NumInt.nr_vxc.__doc__)
    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            n, exc, vmat = self._gks_mcol_vxc(
                cell, grids, xc_code, dms, relativity, hermi,
                kpt, kpts_band, max_memory, verbose)
        else:
            nao = dms.shape[-1] // 2
            # ground state density is always real
            dm_a = dms[...,:nao,:nao].copy('C')
            dm_b = dms[...,nao:,nao:].copy('C')
            dm1 = (dm_a, dm_b)
            ni = self._to_numint1c()
            n, exc, v = ni.nr_uks(cell, grids, xc_code, dm1, relativity, hermi,
                                  kpt, kpts_band, max_memory, verbose)
            vmat = np.zeros(dms.shape, dtype=np.result_type(*v))
            vmat[...,:nao,:nao] = v[0]
            vmat[...,nao:,nao:] = v[1]
        return n.sum(), exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    @lib.with_doc(pnumint.NumInt.nr_fxc.__doc__)
    def nr_fxc(self, cell, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpt=None, max_memory=2000,
               verbose=None):
        if self.collinear[0] not in ('c', 'm'):  # col or mcol
            raise NotImplementedError('non-collinear fxc')

        if self.collinear[0] == 'm':  # mcol
            fxcmat = self._gks_mcol_fxc(cell, grids, xc_code, dm0, dms,
                                        relativity, hermi, rho0, vxc, fxc,
                                        kpt, max_memory, verbose)
        else:
            dms = np.asarray(dms)
            nao = dms.shape[-1] // 2
            if dm0 is not None:
                dm0 = np.asarray(dm0)
                dm0a = dm0[...,:nao,:nao].copy('C')
                dm0b = dm0[...,nao:,nao:].copy('C')
                dm0 = (dm0a, dm0b)
            dms_a = dms[...,:nao,:nao].copy('C')
            dms_b = dms[...,nao:,nao:].copy('C')
            dm1 = (dms_a, dms_b)
            ni = self._to_numint1c()
            vmat = ni.nr_uks_fxc(cell, grids, xc_code, dm0, dm1, relativity,
                                 hermi, rho0, vxc, fxc, kpt, max_memory, verbose)
            fxcmat = np.zeros(dms.shape, dtype=np.result_type(*vmat))
            fxcmat[...,:nao,:nao] = vmat[0]
            fxcmat[...,nao:,nao:] = vmat[1]
        return fxcmat
    get_fxc = nr_gks_fxc = nr_fxc

    eval_xc_eff = numint2c._eval_xc_eff
    mcfun_eval_xc_adapter = numint2c.mcfun_eval_xc_adapter
    block_loop = pnumint.NumInt.block_loop
    _gen_rho_evaluator = pnumint.NumInt._gen_rho_evaluator

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(pnumint.NumInt)


class KNumInt2C(lib.StreamObject, numint.LibXCMixin):
    def __init__(self, kpts=np.zeros((1,3))):
        self.kpts = np.reshape(kpts, (-1,3))

    collinear         = numint2c.NumInt2C.collinear
    spin_samples      = numint2c.NumInt2C.spin_samples
    collinear_thrd    = numint2c.NumInt2C.collinear_thrd
    collinear_samples = numint2c.NumInt2C.collinear_samples

    make_mask = lib.invalid_method('make_mask')
    eval_ao = staticmethod(pnumint.eval_ao_kpts)

    def reset(self, cell=None):
        return self

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, with_lapl=True, verbose=None):
        '''Collocate the density (opt. gradients) on the real-space grid.

        Args:
            cell : Mole or Cell object
            ao_kpts : (nkpts, ngrids, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngrids,) ndarray
        '''
        eval_rho = numint2c.eval_rho
        nkpts = len(ao_kpts)
        rho_ks = [eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                           hermi, with_lapl, verbose)
                  for k in range(nkpts)]
        dtype = np.result_type(*rho_ks)
        rho = np.zeros(rho_ks[0].shape, dtype=dtype)
        for k in range(nkpts):
            rho += rho_ks[k]
        rho *= 1./nkpts
        return rho

    def eval_rho1(self, cell, ao_kpts, dm_kpts, screen_index=None, xctype='LDA',
                  hermi=0, with_lapl=True, cutoff=None, ao_cutoff=None,
                  pair_mask=None, verbose=None):
        return self.eval_rho(cell, ao_kpts, dm_kpts, screen_index, xctype,
                             hermi, with_lapl, verbose=verbose)

    def eval_rho2(self, cell, ao_kpts, mo_coeff_kpts, mo_occ_kpts,
                  non0tab=None, xctype='LDA', with_lapl=True, verbose=None):
        if self.collinear[0] not in ('n', 'm'):
            raise NotImplementedError(self.collinear)

        dm = [(mo*occ).dot(mo.conj().T)
              for mo, occ in zip(mo_coeff_kpts, mo_occ_kpts)]
        hermi = 1
        return self.eval_rho(cell, ao_kpts, dm, non0tab, xctype, hermi,
                             with_lapl, verbose)

    def cache_xc_kernel(self, cell, grids, xc_code, mo_coeff_kpts, mo_occ_kpts,
                        spin=0, kpts=None, max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts
            mo_coeff = kpts.transform_mo_coeff(mo_coeff_kpts)
            mo_occ = kpts.transform_mo_occ(mo_occ_kpts)
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = mo_coeff[0].shape[0]
        nao = n2c // 2

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            with_lapl = False
            rho = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpts, None,
                                       max_memory):
                rho.append(self.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask,
                                          xctype, with_lapl))
            rho = np.concatenate(rho,axis=-1)
            assert rho.dtype == np.double
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            # rhoa and rhob must be real
            dm_a = [(mo[:nao]*occ).dot(mo[:nao].conj().T)
                    for mo, occ in zip(mo_coeff_kpts, mo_occ_kpts)]
            dm_b = [(mo[nao:]*occ).dot(mo[nao:].conj().T)
                    for mo, occ in zip(mo_coeff_kpts, mo_occ_kpts)]
            hermi = 1
            ni = self._to_numint1c()
            with_lapl = True
            hermi = 1
            rhoa = []
            rhob = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None,
                                     max_memory):
                rhoa.append(ni.eval_rho(cell, ao_k1, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(cell, ao_k1, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def cache_xc_kernel1(self, cell, grids, xc_code, dm_kpts, spin=0, kpts=None,
                         max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = dm_kpts.shape[0]
        nao = n2c // 2

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            hermi = 1 # rho must be real. We need to assume dm hermitian
            with_lapl = False
            rho = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in self.block_loop(cell, grids, nao, ao_deriv, kpts, None,
                                       max_memory):
                rho.append(self.eval_rho1(cell, ao_k1, dm_kpts, mask, xctype,
                                          hermi, with_lapl))
            rho = np.concatenate(rho,axis=-1)
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            hermi = 1
            dm_a = dm_kpts[:,:nao,:nao].copy('C')
            dm_b = dm_kpts[:,nao:,nao:].copy('C')
            ni = self._to_numint1c()
            with_lapl = True
            rhoa = []
            rhob = []
            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None,
                                     max_memory):
                rhoa.append(ni.eval_rho(cell, ao_k1, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(cell, ao_k1, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def get_rho(self, cell, dm, grids, kpts=np.zeros((1,3)), max_memory=2000):
        '''Density in real space
        '''
        if dm.ndim != 3:
            raise RuntimeError(f'dm dimension error {dm.ndim}')
        nao = dm.shape[-1] // 2
        dm = [x[:nao,:nao] + x[nao:,nao:] for x in dm]
        ni = self._to_numint1c()
        return ni.get_rho(cell, dm, grids, kpts, max_memory)

    def _gks_mcol_vxc(ni, cell, grids, xc_code, dms, relativity=0, hermi=0,
                      kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts

        xctype = ni._xc_type(xc_code)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        assert dms.ndim >= 3
        make_rho, nset, n2c = ni._gen_rho_evaluator(cell, dms, hermi, False)
        nao = n2c // 2
        nkpts = len(kpts)

        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        vmat = np.zeros((nset,nkpts,n2c,n2c), dtype=np.complex128)

        if xctype in ('LDA', 'GGA', 'MGGA'):
            f_eval_mat = {
                ('LDA' , 'n'): (numint2c._ncol_lda_vxc_mat , 0),
                ('LDA' , 'm'): (numint2c._mcol_lda_vxc_mat , 0),
                ('GGA' , 'm'): (numint2c._mcol_gga_vxc_mat , 1),
                ('MGGA', 'm'): (numint2c._mcol_mgga_vxc_mat, 1),
            }
            fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

            if ni.collinear[0] == 'm':  # mcol
                eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = ni.eval_xc_eff

            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                     max_memory):
                for i in range(nset):
                    rho = make_rho(i, ao_k2, mask, xctype)
                    exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
                    if xctype == 'LDA':
                        den = rho[0] * weight
                    else:
                        den = rho[0,0] * weight
                    nelec[i] += den.sum()
                    excsum[i] += np.dot(den, exc)
                    for k in range(nkpts):
                        vmat[i,k] += fmat(cell, ao_k1[k], weight, rho, vxc,
                                          mask, shls_slice, ao_loc, hermi)
        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f'KNUMINT2C.get_vxc for functional {xc_code}')

        if dms.ndim == 3:
            vmat = vmat[0]
            nelec = nelec[0]
            excsum = excsum[0]
        return nelec, excsum, vmat

    def _gks_mcol_fxc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
                      verbose=None):
        assert ni.collinear[0] == 'm'  # mcol
        xctype = ni._xc_type(xc_code)
        if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
            fxc = ni.cache_xc_kernel1(cell, grids, xc_code, dm0,
                                      kpts=kpts, max_memory=max_memory)[2]
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts

        if xctype == 'MGGA':
            fmat, ao_deriv = (numint2c._mcol_mgga_fxc_mat , 1)
        elif xctype == 'GGA':
            fmat, ao_deriv = (numint2c._mcol_gga_fxc_mat  , 1)
        else:
            fmat, ao_deriv = (numint2c._mcol_lda_fxc_mat  , 0)

        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        assert dms.ndim >= 3
        make_rho1, nset, n2c = ni._gen_rho_evaluator(cell, dms, hermi, False)
        nao = n2c // 2
        nkpts = len(kpts)
        vmat = np.zeros((nset,nkpts,n2c,n2c), dtype=np.complex128)

        if xctype in ('LDA', 'GGA', 'MGGA'):
            _rho0 = None
            p1 = 0
            for ao_k1, ao_k2, mask, weight, coords \
                    in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
                p0, p1 = p1, p1 + weight.size
                _fxc = fxc[:,:,:,:,p0:p1]
                for i in range(nset):
                    rho1 = make_rho1(i, ao_k1, mask, xctype)
                    for k in range(nkpts):
                        vmat[i,k] += fmat(cell, ao_k1[k], weight, _rho0, rho1,
                                          _fxc, mask, shls_slice, ao_loc, hermi)
        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

        if dms.ndim == 3:
            vmat = vmat[0]
        return vmat

    @lib.with_doc(pnumint.KNumInt.nr_vxc.__doc__)
    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            n, exc, vmat = self._gks_mcol_vxc(
                cell, grids, xc_code, dms, relativity, hermi,
                kpts, kpts_band, max_memory, verbose)
        else:
            dms = np.asarray(dms)
            nao = dms.shape[-1] // 2
            # ground state density is always real
            dm_a = dms[...,:nao,:nao].copy('C')
            dm_b = dms[...,nao:,nao:].copy('C')
            dm1 = (dm_a, dm_b)
            ni = self._to_numint1c()
            n, exc, v = ni.nr_uks(cell, grids, xc_code, dm1, relativity, hermi,
                                  kpts, kpts_band, max_memory, verbose)
            vmat = np.zeros(dms.shape, dtype=np.result_type(*v))
            vmat[...,:nao,:nao] = v[0]
            vmat[...,nao:,nao:] = v[1]
        return n.sum(), exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    get_fxc = nr_gks_fxc = nr_fxc = NumInt2C.nr_fxc

    eval_xc_eff = numint2c._eval_xc_eff
    mcfun_eval_xc_adapter = numint2c.mcfun_eval_xc_adapter
    block_loop = pnumint.KNumInt.block_loop
    _gen_rho_evaluator = pnumint.KNumInt._gen_rho_evaluator

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(pnumint.KNumInt)
