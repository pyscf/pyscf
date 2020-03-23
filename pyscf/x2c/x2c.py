#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
from pyscf.data import nist
from pyscf import __config__

LINEAR_DEP_THRESHOLD = 1e-9

class X2C(lib.StreamObject):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the j-adapted spinor basis.
    '''
    approx = getattr(__config__, 'x2c_X2C_approx', '1e')  # 'atom1e'
    xuncontract = getattr(__config__, 'x2c_X2C_xuncontract', True)
    basis = getattr(__config__, 'x2c_X2C_basis', None)
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
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
            xmol, contr_coeff = _uncontract_mol(mol, self.xuncontract)
            return xmol, contr_coeff
        else:
            return mol, None

    def get_hcore(self, mol=None):
        '''2-component X2c Foldy-Wouthuysen (FW) Hamiltonian (including
        spin-free and spin-dependent terms) in the j-adapted spinor basis.
        '''
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        if 'get_xmat' in self.__dict__:
            # If the get_xmat method is overwritten by user, build the X
            # matrix with the external get_xmat method
            x = self.get_xmat(xmol)
            h1 = _get_hcore_fw(t, v, w, s, x, c)

        elif 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex)
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
                t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
                with xmol.with_rinv_at_nucleus(ia):
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

    def _picture_change(self, xmol, even_operator=(None, None), odd_operator=None):
        '''Picture change for even_operator + odd_operator

        even_operator has two terms at diagonal blocks
        [ v  0 ]
        [ 0  w ]

        odd_operator has the term at off-diagonal blocks
        [ 0    p ]
        [ p^T  0 ]

        v, w, and p can be strings (integral name) or matrices.
        '''
        c = lib.param.LIGHT_SPEED
        v_op, w_op = even_operator
        if isinstance(v_op, str):
            v_op = xmol.intor(v_op)
        if isinstance(w_op, str):
            w_op = xmol.intor(w_op)
            w_op *= (.5/c)**2
        if isinstance(odd_operator, str):
            odd_operator = xmol.intor(odd_operator) * (.5/c)

        if v_op is not None:
            shape = v_op.shape
        elif w_op is not None:
            shape = w_op.shape
        elif odd_operator is not None:
            shape = odd_operator.shape
        else:
            raise RuntimeError('No operators provided')

        x = self.get_xmat()
        r = self._get_rmat(x)
        def transform(mat):
            nao = mat.shape[-1] // 2
            xv = mat[:nao] + x.conj().T.dot(mat[nao:])
            h = xv[:,:nao] + xv[:,nao:].dot(x)
            return reduce(numpy.dot, (r.T.conj(), h, r))

        nao = shape[-1]
        dtype = numpy.result_type(v_op, w_op, odd_operator)

        if len(shape) == 2:
            mat = numpy.zeros((nao*2,nao*2), dtype)
            if v_op is not None:
                mat[:nao,:nao] = v_op
            if w_op is not None:
                mat[nao:,nao:] = w_op
            if odd_operator is not None:
                mat[:nao,nao:] = odd_operator
                mat[nao:,:nao] = odd_operator.conj().T
            pc_mat = transform(mat)

        else:
            assert len(shape) == 3
            mat = numpy.zeros((shape[0],nao*2,nao*2), dtype)
            if v_op is not None:
                mat[:,:nao,:nao] = v_op
            if w_op is not None:
                mat[:,nao:,nao:] = w_op
            if odd_operator is not None:
                mat[:,:nao,nao:] = odd_operator
                mat[:,nao:,:nao] = odd_operator.conj().transpose(0,2,1)
            pc_mat = numpy.asarray([transform(m) for m in mat])

        return pc_mat

    def picture_change(self, even_operator=(None, None), odd_operator=None):
        '''Picture change for even_operator + odd_operator

        even_operator has two terms at diagonal blocks
        [ v  0 ]
        [ 0  w ]

        odd_operator has the term at off-diagonal blocks
        [ 0    p ]
        [ p^T  0 ]

        v, w, and p can be strings (integral name) or matrices.
        '''
        mol = self.mol
        xmol, contr_coeff_nr = self.get_xmol(mol)
        pc_mat = self._picture_change(xmol, even_operator, odd_operator)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
            s21 = mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
            c = lib.cho_solve(s22, s21)

        elif self.xuncontract:
            np, nc = contr_coeff_nr.shape
            c = numpy.zeros((np*2,nc*2))
            c[0::2,0::2] = contr_coeff_nr
            c[1::2,1::2] = contr_coeff_nr

        else:
            return pc_mat

        if pc_mat.ndim == 2:
            return lib.einsum('pi,pq,qj->ij', c.conj(), pc_mat, c)
        else:
            return lib.einsum('pi,xpq,qj->xij', c.conj(), pc_mat, c)


    def get_xmat(self, mol=None):
        if mol is None:
            xmol = self.get_xmol(mol)[0]
        else:
            xmol = mol
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex)
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
                t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            s = xmol.intor_symmetric('int1e_ovlp_spinor')
            t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
            v = xmol.intor_symmetric('int1e_nuc_spinor')
            w = xmol.intor_symmetric('int1e_spnucsp_spinor')
            x = _x2c1e_xmatrix(t, v, w, s, c)
        return x

    def _get_rmat(self, x=None):
        '''The matrix (in AO basis) that changes metric from NESC metric to NR metric'''
        xmol = self.get_xmol()[0]
        if x is None:
            x = self.get_xmat(xmol)
        c = lib.param.LIGHT_SPEED
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        s1 = s + reduce(numpy.dot, (x.conj().T, t, x)) * (.5/c**2)
        return _get_r(s, s1)

    def reset(self, mol):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        self.mol = mol
        return self


def get_hcore(mol):
    '''2-component X2c hcore Hamiltonian (including spin-free and
    spin-dependent terms) in the j-adapted spinor basis.
    '''
    x2c = X2C(mol)
    return x2c.get_hcore(mol)

def get_jk(mol, dm, hermi=1, mf_opt=None, with_j=True, with_k=True, omega=None):
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

def make_rdm1(mo_coeff, mo_occ, **kwargs):
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
        self.opt = None
        return self

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
        if self.with_x2c:
            self.with_x2c.dump_flags(verbose)
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

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

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

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        vj, vk = get_jk(mol, dm, hermi, self.opt, with_j, with_k)
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

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, **kwargs):
        r''' Dipole moment calculation with picture change correction

        Args:
             mol: an instance of :class:`Mole`
             dm : a 2D ndarrays density matrices

        Kwarg:
            picture_change (bool) : Whether to compute the dipole moment with
            picture change correction.

        Return:
            A list: the dipole moment on x, y and z component
        '''
        if mol is None: mol = self.mol
        if dm is None: dm =self.make_rdm1()
        log = logger.new_logger(mol, verbose)

        if 'unit_symbol' in kwargs:  # pragma: no cover
            log.warn('Kwarg "unit_symbol" was deprecated. It was replaced by kwarg '
                     'unit since PySCF-1.5.')
            unit = kwargs['unit_symbol']

        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            # UHF denisty matrices
            dm = dm[0] + dm[1]

        with mol.with_common_orig((0,0,0)):
            if picture_change:
                ao_dip = self.with_x2c.picture_change(('int1e_r_spinor',
                                                       'int1e_sprsp_spinor'))
            else:
                ao_dip = mol.intor_symmetric('int1e_r_spinor')

        el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real

        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nucl_dip = numpy.einsum('i,ix->x', charges, coords)
        mol_dip = nucl_dip - el_dip

        if unit.upper() == 'DEBYE':
            mol_dip *= nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

UHF = X2C_UHF

try:
    from pyscf.dft import rks, dks, r_numint
    class X2C_UKS(X2C_UHF, rks.KohnShamDFT):
        def __init__(self, mol):
            X2C_UHF.__init__(self, mol)
            rks.KohnShamDFT.__init__(self)
            self._numint = r_numint.RNumInt()

        def dump_flags(self, verbose=None):
            hf.SCF.dump_flags(self, verbose)
            rks.KohnShamDFT.dump_flags(self, verbose)
            if self.with_x2c:
                self.with_x2c.dump_flags(verbose)
            return self

        get_veff = dks.get_veff
        energy_elec = rks.energy_elec

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

    def _to_full_contraction(mol, bas_idx):
        es = numpy.hstack([mol.bas_exp(ib) for ib in bas_idx])
        cs = scipy.linalg.block_diag(*[mol._libcint_ctr_coeff(ib) for ib in bas_idx])

        es, e_idx, rev_idx = numpy.unique(es.round(9), True, True)
        cs_new = numpy.zeros((es.size, cs.shape[1]))
        for i, j in enumerate(rev_idx):
            cs_new[j] += cs[i]
        return es[::-1], cs_new[::-1]

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        if isinstance(xuncontract, str):
            uncontract_me = ((xuncontract == mol.atom_pure_symbol(ia)) or
                             (xuncontract == mol.atom_symbol(ia)))
        elif isinstance(xuncontract, (tuple, list)):
            uncontract_me = ((mol.atom_pure_symbol(ia) in xuncontract) or
                             (mol.atom_symbol(ia) in xuncontract) or
                             (ia in xuncontract))
        else:
            uncontract_me = xuncontract

        if not uncontract_me:
            p0, p1 = aoslices[ia]
            _bas.append(mol._bas[ib0:ib1])
            contr_coeff.append(numpy.eye(p1-p0))
            continue

        lmax = mol._bas[ib0:ib1,mole.ANG_OF].max()
        assert(all(mol._bas[ib0:ib1, mole.KAPPA_OF] == 0))
        # TODO: loop based on kappa
        for l in range(lmax+1):
            bas_idx = ib0 + numpy.where(mol._bas[ib0:ib1,mole.ANG_OF] == l)[0]
            if len(bas_idx) == 0:
                continue

# Some input basis may be the segmented basis from a general contracted set.
# This may lead to duplicated pGTOs. First contract all basis to remove
# duplicated primitive functions.
            mol_exps, b_coeff = _to_full_contraction(mol, bas_idx)

            if mol.cart:
                degen = (l + 1) * (l + 2) // 2
            else:
                degen = l * 2 + 1
            np, nc = b_coeff.shape

            # Uncontract all basis. Use pGTO basis for X
            bs = numpy.zeros((np, mole.BAS_SLOTS), dtype=numpy.int32)
            bs[:,mole.ATOM_OF] = ia
            bs[:,mole.ANG_OF ] = l
            bs[:,mole.NCTR_OF] = bs[:,mole.NPRIM_OF] = 1
            norm = mole.gto_norm(l, mol_exps)
            _env.append(mol_exps)
            _env.append(norm)
            bs[:,mole.PTR_EXP] = numpy.arange(ptr, ptr+np)
            bs[:,mole.PTR_COEFF] = numpy.arange(ptr+np, ptr+np*2)
            _bas.append(bs)
            ptr += np * 2

            c = b_coeff * 1/norm[:,None]
            c = scipy.linalg.block_diag(*([c,] * degen))
            c = c.reshape((degen, np, degen, nc))
            c = c.transpose(1,0,3,2).reshape(np*degen, nc*degen)
            contr_coeff.append(c)

    pmol._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pmol._env = numpy.hstack([mol._env,] + _env)
    contr_coeff = scipy.linalg.block_diag(*contr_coeff)

    return pmol, contr_coeff


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

    try:
        e, a = scipy.linalg.eigh(h, m)
        cl = a[:nao,nao:]
        cs = a[nao:,nao:]
        x = numpy.linalg.solve(cl.T, cs.T).T  # B = XA
    except scipy.linalg.LinAlgError:
        d, t = numpy.linalg.eigh(m)
        idx = d>LINEAR_DEP_THRESHOLD
        t = t[:,idx] / numpy.sqrt(d[idx])
        tht = reduce(numpy.dot, (t.T.conj(), h, t))
        e, a = numpy.linalg.eigh(tht)
        a = numpy.dot(t, a)
        idx = e > -c**2
        cl = a[:nao,idx]
        cs = a[nao:,idx]
        # X = B A^{-1} = B A^T S
        x = cs.dot(cl.conj().T).dot(m)
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

    try:
        e, a = scipy.linalg.eigh(h, m)
        cl = a[:nao,nao:]
        cs = a[nao:,nao:]
        e = e[nao:]
    except scipy.linalg.LinAlgError:
        d, t = numpy.linalg.eigh(m)
        idx = d>LINEAR_DEP_THRESHOLD
        t = t[:,idx] / numpy.sqrt(d[idx])
        tht = reduce(numpy.dot, (t.T.conj(), h, t))
        e, a = numpy.linalg.eigh(tht)
        a = numpy.dot(t, a)
        idx = e > -c**2
        cl = a[:nao,idx]
        cs = a[nao:,idx]
        e = e[idx]

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
    h1 = reduce(numpy.dot, (r.T.conj()*e, r))
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

    method = UKS(mol)
    ex2c = method.kernel()
    print('E(X2C1E) = %.12g' % ex2c)
