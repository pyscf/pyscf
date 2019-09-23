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


import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import dhf
from pyscf.scf import _vhf
from pyscf import __config__

class X2C(lib.StreamObject):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the j-adapted spinor basis.
    '''
    exp_drop = getattr(__config__, 'x2c_X2C_exp_drop', 0.2)
    approx = getattr(__config__, 'x2c_X2C_approx', '1e')  # 'atom1e'
    xuncontract = getattr(__config__, 'x2c_X2C_xuncontract', True)
    basis = getattr(__config__, 'x2c_X2C_basis', None)
    def __init__(self, mol=None):
        self.mol = mol

    def dump_flags(self):
        log = logger.Logger(self.mol.stdout, self.mol.verbose)
        log.info('\n')
        log.info('******** %s flags ********', self.__class__)
        log.info('exp_drop = %g', self.exp_drop)
        log.info('approx = %s',    self.approx)
        log.info('xuncontract = %d', self.xuncontract)
        if self.basis is not None:
            log.info('basis for X matrix = %s', self.basis)
        return self

    def get_xmol(self, mol=None):
        if mol is None:
            mol = self.mol

        if self.basis is not None:
            xmol = copy.copy(mol)
            xmol.build(False, False, basis=self.basis)
            return xmol, None
        elif self.xuncontract:
            xmol, contr_coeff = _uncontract_mol(mol, self.xuncontract,
                                                self.exp_drop)
            return xmol, contr_coeff
        else:
            return mol, None

    def get_hcore(self, mol=None):
        '''2-component X2c Foldy-Wouthuysen (FW) Hamiltonian (including
        spin-free and spin-dependent terms) in the j-adapted spinor basis.
        '''
        if mol is None: mol = self.mol
        xmol, contr_coeff_nr = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex)
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
                t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
                with xmol.with_rinv_as_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = _get_hcore_fw(t, v, w, s, x, c)
        else:
            h1 = _x2c1e_get_hcore(t, v, w, s, c)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
            s21 = mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
            c = lib.cho_solve(s22, s21)
            h1 = reduce(numpy.dot, (c.T.conj(), h1, c))
        elif self.xuncontract:
            np, nc = contr_coeff_nr.shape
            contr_coeff = numpy.zeros((np*2,nc*2))
            contr_coeff[0::2,0::2] = contr_coeff_nr
            contr_coeff[1::2,1::2] = contr_coeff_nr
            h1 = reduce(numpy.dot, (contr_coeff.T.conj(), h1, contr_coeff))
        return h1


def get_hcore(mol):
    '''2-component X2c hcore Hamiltonian (including spin-free and
    spin-dependent terms) in the j-adapted spinor basis.
    '''
    x2c = X2C(mol)
    return x2c.get_hcore(mol)

def get_jk(mol, dm, hermi=1, mf_opt=None):
    '''non-relativistic J/K matrices (without SSO,SOO etc) in the j-adapted
    spinor basis.
    '''
    n2c = dm.shape[0]
    dd = numpy.zeros((n2c*2,)*2, dtype=numpy.complex)
    dd[:n2c,:n2c] = dm
    dhf._call_veff_llll(mol, dd, hermi, None)
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dm, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    return dhf._jk_triu_(vj, vk, hermi)

def make_rdm1(mo_coeff, mo_occ):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    dm = hf.init_guess_by_minao(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    mf = UHF(mol)
    return mf.init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    dm = dhf.init_guess_by_chkfile(mol, chkfile_name, project)
    n2c = dm.shape[0] // 2
    return dm[:n2c,:n2c].copy()

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    elif key.lower() == '1e':
        return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.hf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)


class X2C_UHF(hf.SCF):
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.with_x2c = X2C(mol)
        #self.with_x2c.xuncontract = False
        self._keys = self._keys.union(['with_x2c'])

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(self.mol)

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        if self.with_x2c:
            self.with_x2c.dump_flags()
        return self

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def _eigh(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,range(len(e))].real<0] *= -1
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return self.with_x2c.get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('int1e_ovlp_spinor')

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:mol.nelectron] = 1
        if mol.nelectron < len(mo_energy):
            logger.info(self, 'nocc = %d  HOMO = %.12g  LUMO = %.12g', \
                        mol.nelectron, mo_energy[mol.nelectron-1],
                        mo_energy[mol.nelectron])
        else:
            logger.info(self, 'nocc = %d  HOMO = %.12g  no LUMO', \
                        mol.nelectron, mo_energy[mol.nelectron-1])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        def set_vkscreen(opt, name):
            opt._this.contents.r_vkscreen = _vhf._fpointer(name)
        opt = _vhf.VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                          'CVHFrkbllll_direct_scf',
                          'CVHFrkbllll_direct_scf_dm')
        opt.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt, 'CVHFrkbllll_vkscreen')
        return opt

    def get_jk(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        vj, vk = get_jk(mol, dm, hermi, self.opt)
        logger.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Dirac-Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return dhf.analyze(self, verbose)
UHF = X2C_UHF

try:
    from pyscf.dft import rks, dks
    class X2C_UKS(X2C_UHF):
        def dump_flags(self):
            hf.SCF.dump_flags(self)
            logger.info(self, 'XC functionals = %s', self.xc)
            logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
            self.grids.dump_flags()
            if self.with_x2c:
                self.with_x2c.dump_flags()
            return self

        get_veff = dks.get_veff
        energy_elec = rks.energy_elec
        define_xc_ = rks.define_xc_

    UKS = X2C_UKS
except ImportError:
    pass


def _uncontract_mol(mol, xuncontract=False, exp_drop=0.2):
    '''mol._basis + uncontracted steep functions'''
    pmol = copy.copy(mol)
    _bas = []
    _env = []
    ptr = len(pmol._env)
    contr_coeff = []
    for ib in range(mol.nbas):
        if isinstance(xuncontract, str):
            ia = mol.bas_atom(ib)
            uncontract_me = ((xuncontract == mol.atom_pure_symbol(ia)) or
                             (xuncontract == mol.atom_symbol(ia)))
        elif isinstance(xuncontract, (tuple, list)):
            ia = mol.bas_atom(ib)
            uncontract_me = ((mol.atom_pure_symbol(ia) in xuncontract) or
                             (mol.atom_symbol(ia) in xuncontract) or
                             (ia in xuncontract))
        else:
            uncontract_me = xuncontract

        nc = mol._bas[ib,mole.NCTR_OF]
        l = mol._bas[ib,mole.ANG_OF]
        if mol.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = l * 2 + 1
        if uncontract_me:
            np = mol._bas[ib,mole.NPRIM_OF]
            pexp = mol._bas[ib,mole.PTR_EXP]
# Modfied partially uncontraction to avoid potentially lindep in the
# segment-contracted basis
            nkept = (pmol._env[pexp:pexp+np] > exp_drop).sum()
            if nkept > nc:
                b_coeff = mol.bas_ctr_coeff(ib)
                importance = numpy.einsum('ij->i', abs(b_coeff))
                idx = numpy.argsort(importance[:nkept])
                contracted = numpy.sort(idx[nkept-nc:])
                primitive  = numpy.sort(idx[:nkept-nc])

# part1: pGTOs that are associated with small coefficients
                bs = numpy.empty((nkept-nc,mol._bas.shape[1]), dtype=numpy.int32)
                bs[:] = mol._bas[ib]
                bs[:,mole.NCTR_OF] = bs[:,mole.NPRIM_OF] = 1
                for k, i in enumerate(primitive):
                    norm = mole.gto_norm(l, mol._env[pexp+i])
                    _env.append(mol._env[pexp+i])
                    _env.append(norm)
                    bs[k,mole.PTR_EXP] = ptr
                    bs[k,mole.PTR_COEFF] = ptr + 1
                    ptr += 2
                _bas.append(bs)
                part1 = numpy.zeros((degen*(nkept-nc),degen*nc))
                c = b_coeff[primitive]
                for i in range(degen):
                    part1[i::degen,i::degen] = c

# part2: binding the pGTOs of small exps to the pGTOs of large coefficients
                bs = mol._bas[ib].copy()
                bs[mole.NPRIM_OF] = np - nkept + nc
                idx = numpy.hstack((contracted, numpy.arange(nkept,np)))
                exps = mol._env[pexp:pexp+np][idx]
                cs = mol._libcint_ctr_coeff(ib)[idx]
                ee = mole.gaussian_int(l*2+2, exps[:,None] + exps)
                s1 = numpy.einsum('pi,pq,qi->i', cs, ee, cs)
                s1 = numpy.sqrt(s1)
                cs = numpy.einsum('pi,i->pi', cs, 1/s1)
                _env.extend(exps)
                _env.extend(cs.T.reshape(-1))
                bs[mole.PTR_EXP] = ptr
                bs[mole.PTR_COEFF] = ptr + exps.size
                ptr += exps.size + cs.size
                _bas.append(bs)

                part2 = numpy.eye(degen*nc)
                for i in range(nc):
                    part2[i*degen:(i+1)*degen,i*degen:(i+1)*degen] *= s1[i]
                contr_coeff.append(numpy.vstack((part1, part2)))
            else:
                _bas.append(mol._bas[ib])
                contr_coeff.append(numpy.eye(degen*nc))
        else:
            _bas.append(mol._bas[ib])
            contr_coeff.append(numpy.eye(degen*nc))
    pmol._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pmol._env = numpy.hstack((mol._env, _env))
    return pmol, scipy.linalg.block_diag(*contr_coeff)

def _sqrt(a, tol=1e-14):
    e, v = numpy.linalg.eigh(a)
    idx = e > tol
    return numpy.dot(v[:,idx]*numpy.sqrt(e[idx]), v[:,idx].T.conj())

def _invsqrt(a, tol=1e-14):
    e, v = numpy.linalg.eigh(a)
    idx = e > tol
    return numpy.dot(v[:,idx]/numpy.sqrt(e[idx]), v[:,idx].T.conj())

def _get_hcore_fw(t, v, w, s, x, c):
    s1 = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5/c**2)
    tx = numpy.dot(t, x)
    h1 =(v + tx + tx.T.conj() - numpy.dot(x.T.conj(), tx) +
         reduce(numpy.dot, (x.T.conj(), w, x)) * (.25/c**2))

    r = _get_r(s, s1)
    h1 = reduce(numpy.dot, (r.T.conj(), h1, r))
    return h1

def _get_r(s, snesc):
    # R^dag \tilde{S} R = S
    # R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
    w, v = numpy.linalg.eigh(s)
    idx = w > 1e-14
    v = v[:,idx]
    w_sqrt = numpy.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt

    # eigenvectors of S as the new basis
    snesc = reduce(numpy.dot, (v.conj().T, snesc, v))
    r_mid = numpy.einsum('i,ij,j->ij', w_invsqrt, snesc, w_invsqrt)
    w1, v1 = numpy.linalg.eigh(r_mid)
    idx1 = w1 > 1e-14
    v1 = v1[:,idx1]
    r_mid = numpy.dot(v1/numpy.sqrt(w1[idx1]), v1.conj().T)
    r = numpy.einsum('i,ij,j->ij', w_invsqrt, r_mid, w_sqrt)
    # Back transform to AO basis
    r = reduce(numpy.dot, (v, r, v.conj().T))
    return r

def _x2c1e_xmatrix(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao,nao:]
    cs = a[nao:,nao:]
    x = numpy.linalg.solve(cl.T, cs.T).T  # B = XA
    return x

def _x2c1e_get_hcore(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao,nao:]
    cs = a[nao:,nao:]
# The so obtaied X seems not numerically stable.  We changed to the
# transformed matrix
# [1 1] [ V T ] [1 0]
# [0 1] [ T W ] [1 1]
#            h[:nao,:nao] = h[:nao,nao:] = h[nao:,:nao] = h[nao:,nao:] = w * (.25/c**2)
#            m[:nao,:nao] = m[:nao,nao:] = m[nao:,:nao] = m[nao:,nao:] = t * (.5/c**2)
#            h[:nao,:nao]+= v + t
#            h[nao:,nao:]-= t
#            m[:nao,:nao]+= s
#            e, a = scipy.linalg.eigh(h, m)
#            cl = a[:nao,nao:]
#            cs = a[nao:,nao:]
#            x = numpy.eye(nao) + numpy.linalg.solve(cl.T, cs.T).T  # B = XA
#            h1 = _get_hcore_fw(t, v, w, s, x, c)

# Taking A matrix as basis and rewrite the FW Hcore formula, to avoid inversing matrix
#   R^dag \tilde{S} R = S
#   R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
# Using A matrix as basis, the representation of R is
#   R[A] = (A^+ S A)^{1/2} = (A^+ S A)^{-1/2} A^+ S A
# Construct h = R^+ h1 R in two steps, first in basis A matrix, then back
# transformed to AO basis
#   h  = (A^+)^{-1} R[A]^+ (A^+ h1 A) R[A] A^{-1}         (0)
# Using (A^+)^{-1} = \tilde{S} A, h can be transformed to
#   h  = \tilde{S} A R[A]^+ A^+ h1 A R[A] A^+ \tilde{S}   (1)
# Using R[A] = R[A]^{-1} A^+ S A,  Eq (0) turns to
#      = S A R[A]^{-1}^+ A^+ h1 A R[A]^{-1} A^+ S
#      = S A R[A]^{-1}^+ e R[A]^{-1} A^+ S                (2)
    w, u = numpy.linalg.eigh(reduce(numpy.dot, (cl.T.conj(), s, cl)))
    idx = w > 1e-14
# Adopt (2) here becuase X is not appeared in Eq (2).
# R[A] = u w^{1/2} u^+,  so R[A]^{-1} A^+ S in Eq (2) is
    r = reduce(numpy.dot, (u[:,idx]/numpy.sqrt(w[idx]), u[:,idx].T.conj(),
                           cl.T.conj(), s))
    h1 = reduce(numpy.dot, (r.T.conj()*e[nao:], r))
    return h1


def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, numpy.eye(mol_nr.nao_nr()), mol)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm_ll = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
    return dm_ll


# A tag to label the derived SCF class
class _X2C_SCF:
    pass


if __name__ == '__main__':
    mol = mole.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz-dk',
    )

    method = hf.RHF(mol)
    enr = method.kernel()
    print('E(NR) = %.12g' % enr)

    method = UHF(mol)
    ex2c = method.kernel()
    print('E(X2C1E) = %.12g' % ex2c)
    method.with_x2c.basis = {'O': 'unc-ccpvqz', 'H':'unc-ccpvdz'}
    print('E(X2C1E) = %.12g' % method.kernel())
    method.with_x2c.approx = 'atom1e'
    print('E(X2C1E) = %.12g' % method.kernel())

