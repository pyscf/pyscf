#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

'''
X2C 2-component HF methods
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf, ghf, dhf
from pyscf.scf import _vhf
from pyscf.data import nist
from pyscf import __config__

LINEAR_DEP_THRESHOLD = 1e-9

class X2CHelperBase(lib.StreamObject):
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
            xmol = mol.copy(deep=False)
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
        assert ('1E' in self.approx.upper())
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
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex128)
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
        assert ('1E' in self.approx.upper())

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex128)
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

class SpinorX2CHelper(X2CHelperBase):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the j-adapted spinor basis.
    '''
    pass

X2C = SpinorX2CHelper

class SpinOrbitalX2CHelper(X2CHelperBase):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the Gaussian type spin-orbital basis (as the spin-orbital basis in GHF)
    '''
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())

        t = _block_diag(xmol.intor_symmetric('int1e_kin'))
        v = _block_diag(xmol.intor_symmetric('int1e_nuc'))
        s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
        w = _sigma_dot(xmol.intor('int1e_spnucsp'))
        if 'get_xmat' in self.__dict__:
            # If the get_xmat method is overwritten by user, build the X
            # matrix with the external get_xmat method
            x = self.get_xmat(xmol)
            h1 = _get_hcore_fw(t, v, w, s, x, c)

        elif 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            # spin-orbital basis is twice the size of NR basis
            atom_slices[:,2:] *= 2
            nao = xmol.nao_nr() * 2
            x = numpy.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = _block_diag(xmol.intor('int1e_kin', shls_slice=shls_slice))
                s1 = _block_diag(xmol.intor('int1e_ovlp', shls_slice=shls_slice))
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = _block_diag(z * xmol.intor('int1e_rinv', shls_slice=shls_slice))
                    w1 = _sigma_dot(z * xmol.intor('int1e_sprinvsp', shls_slice=shls_slice))
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = _get_hcore_fw(t, v, w, s, x, c)

        else:
            h1 = _x2c1e_get_hcore(t, v, w, s, c)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp')
            s21 = mole.intor_cross('int1e_ovlp', xmol, mol)
            c = _block_diag(lib.cho_solve(s22, s21))
            h1 = reduce(lib.dot, (c.T, h1, c))
        if self.xuncontract and contr_coeff is not None:
            contr_coeff = _block_diag(contr_coeff)
            h1 = reduce(lib.dot, (contr_coeff.T, h1, contr_coeff))
        return h1

    @lib.with_doc(X2CHelperBase.picture_change.__doc__)
    def picture_change(self, even_operator=(None, None), odd_operator=None):
        mol = self.mol
        xmol, c = self.get_xmol(mol)
        pc_mat = self._picture_change(xmol, even_operator, odd_operator)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp')
            s21 = mole.intor_cross('int1e_ovlp', xmol, mol)
            c = lib.cho_solve(s22, s21)

        elif self.xuncontract:
            pass

        else:
            return pc_mat

        c = _block_diag(c)
        if pc_mat.ndim == 2:
            return lib.einsum('pi,pq,qj->ij', c, pc_mat, c)
        else:
            return lib.einsum('pi,xpq,qj->xij', c, pc_mat, c)

    def get_xmat(self, mol=None):
        if mol is None:
            xmol = self.get_xmol(mol)[0]
        else:
            xmol = mol
        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            # spin-orbital basis is twice the size of NR basis
            atom_slices[:,2:] *= 2
            nao = xmol.nao_nr() * 2
            x = numpy.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = _block_diag(xmol.intor('int1e_kin', shls_slice=shls_slice))
                s1 = _block_diag(xmol.intor('int1e_ovlp', shls_slice=shls_slice))
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = _block_diag(z * xmol.intor('int1e_rinv', shls_slice=shls_slice))
                    w1 = _sigma_dot(z * xmol.intor('int1e_sprinvsp', shls_slice=shls_slice))
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            t = _block_diag(xmol.intor_symmetric('int1e_kin'))
            v = _block_diag(xmol.intor_symmetric('int1e_nuc'))
            s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
            w = _sigma_dot(xmol.intor('int1e_spnucsp'))
            x = _x2c1e_xmatrix(t, v, w, s, c)
        return x

    def _get_rmat(self, x=None):
        '''The matrix (in AO basis) that changes metric from NESC metric to NR metric'''
        xmol = self.get_xmol()[0]
        if x is None:
            x = self.get_xmat(xmol)
        c = lib.param.LIGHT_SPEED
        s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
        t = _block_diag(xmol.intor_symmetric('int1e_kin'))
        s1 = s + reduce(numpy.dot, (x.conj().T, t, x)) * (.5/c**2)
        return _get_r(s, s1)


def get_hcore(mol):
    '''2-component X2c hcore Hamiltonian (including spin-free and
    spin-dependent terms) in the j-adapted spinor basis.
    '''
    return SpinorX2CHelper(mol).get_hcore(mol)

def get_jk(mol, dm, hermi=1, mf_opt=None, with_j=True, with_k=True, omega=None):
    '''non-relativistic J/K matrices (without SSO,SOO etc) in the j-adapted
    spinor basis.
    '''
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dm, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    vj = vj.reshape(dm.shape)
    vk = vk.reshape(dm.shape)
    return dhf._jk_triu_(mol, vj, vk, hermi)

make_rdm1 = hf.make_rdm1

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


class SCF(hf.SCF):
    '''The full X2C problem (scaler + soc terms) in j-adapted spinor basis'''

    _keys = set(['with_x2c'])

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.with_x2c = SpinorX2CHelper(mol)
        #self.with_x2c.xuncontract = False

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(mol)
        return self

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
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
        nocc = mol.nelectron
        mo_occ[:nocc] = 1
        if nocc < len(mo_energy):
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(self, 'nocc = %d  HOMO = %.12g  LUMO = %.12g',
                            nocc, mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'nocc = %d  HOMO = %.12g  no LUMO',
                        nocc, mo_energy[nocc-1])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    make_rdm1 = lib.module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        opt = dhf._VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                          'CVHFrkb_q_cond', 'CVHFrkb_dm_cond',
                          direct_scf_tol=self.direct_scf_tol)
        opt._this.r_vkscreen = _vhf._fpointer('CVHFrkbllll_vkscreen')
        return opt

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        if self.direct_scf and self._opt.get(omega) is None:
            with mol.with_range_coulomb(omega):
                self._opt[omega] = self.init_direct_scf(mol)
        vhfopt = self._opt.get(omega)
        vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k)
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

    def x2c1e(self):
        return self
    x2c = x2c1e

    def sfx2c1e(self):
        raise NotImplementedError

    def newton(self):
        from pyscf.x2c.newton_ah import newton
        return newton(self)

    def stability(self, internal=None, external=None, verbose=None, return_status=False):
        '''
        X2C-HF/X2C-KS stability analysis.

        See also pyscf.scf.stability.rhf_stability function.

        Kwargs:
            return_status: bool
                Whether to return `stable_i` and `stable_e`

        Returns:
            If return_status is False (default), the return value includes
            two set of orbitals, which are more close to the stable condition.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.

            Else, another two boolean variables (indicating current status:
            stable or unstable) are returned.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.
        '''
        from pyscf.x2c.stability import x2chf_stability
        return x2chf_stability(self, verbose, return_status)

    def nuc_grad_method(self):
        raise NotImplementedError

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.x2c()
        return hf.SCF._transfer_attrs_(self, dst)

X2C_SCF = SCF

class UHF(SCF):
    def to_ks(self, xc='HF'):
        '''Convert the input mean-field object to an X2C-KS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.x2c import dft
        return self._transfer_attrs_(dft.UKS(self.mol, xc=xc))

X2C_UHF = UHF

class RHF(SCF):
    def __init__(self, mol):
        super().__init__(mol)
        if dhf.zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted X2C-RHF')

    def _eigh(self, h, s):
        return dhf.zquatev.solve_KR_FCSCE(self.mol, h, s)

    def to_ks(self, xc='HF'):
        '''Convert the input mean-field object to an X2C-KS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.x2c import dft
        return self._transfer_attrs_(dft.RKS(self.mol, xc=xc))

X2C_RHF = RHF

def x2c1e_ghf(mf):
    '''
    For the given *GHF* object, generate X2C-GSCF object in GHF spin-orbital
    basis. Note the orbital basis of X2C_GSCF is different to the X2C_RHF and
    X2C_UHF objects. X2C_RHF and X2C_UHF use spinor basis.

    Args:
        mf : an GHF/GKS object

    Returns:
        An GHF/GKS object

    Examples:

    >>> mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.GHF(mol).x2c1e().run()
    '''
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, _X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2CHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinOrbitalX2CHelper):
            # An object associated to sfx2c1e.SpinFreeX2CHelper
            raise NotImplementedError
        else:
            return mf

    return lib.set_class(X2C1E_GSCF(mf), (X2C1E_GSCF, mf.__class__))

# A tag to label the derived SCF class
class _X2C_SCF:
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.with_x2c:
            self.with_x2c.dump_flags(verbose)
        return self

    def reset(self, mol):
        self.with_x2c.reset(mol)
        return super().reset(mol)

class X2C1E_GSCF(_X2C_SCF):
    '''
    Attributes for spin-orbital X2C:
        with_x2c : X2C object
    '''

    __name_mixin__ = 'X2C1e'

    _keys = set(['with_x2c'])

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinOrbitalX2CHelper(mf.mol)

    def undo_x2c(self):
        '''Remove the X2C Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, X2C1E_GSCF))
        del obj.with_x2c
        return obj

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return self.with_x2c.get_hcore(mol)

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
        if dm is None: dm = self.make_rdm1()
        log = logger.new_logger(mol, verbose)
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nucl_dip = numpy.einsum('i,ix->x', charges, coords)
        with mol.with_common_orig(nucl_dip):
            r = mol.intor_symmetric('int1e_r')
            ao_dip = numpy.array([_block_diag(x) for x in r])
            if picture_change:
                xmol = self.with_x2c.get_xmol()[0]
                nao = xmol.nao
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,0]
                prp = numpy.array([_block_diag(x) for x in prp])
                ao_dip = self.with_x2c.picture_change((ao_dip, prp))

        mol_dip = -numpy.einsum('xij,ji->x', ao_dip, dm).real

        if unit.upper() == 'DEBYE':
            mol_dip *= nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.x2c()
        return hf.SCF._transfer_attrs_(self, dst)

    def to_ks(self, xc='HF'):
        raise NotImplementedError


def _uncontract_mol(mol, xuncontract=None, exp_drop=0.2):
    '''mol._basis + uncontracted steep functions'''
    pmol, contr_coeff = mol.decontract_basis(atoms=xuncontract)
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
    # s1 = s + (1/2c^2)(X^{\dag}*T*X)
    s1 = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5/c**2)
    # tx = T * X
    tx = numpy.dot(t, x)
    # h1 = (v + T*X + V^{\dag}*T^{\dag} - (X^{\dag} * T * X) + (X^{\dag} * W * X)*(1/4c^2)
    h1 =(v + tx + tx.T.conj() - numpy.dot(x.T.conj(), tx) +
         reduce(numpy.dot, (x.T.conj(), w, x)) * (.25/c**2))
    # R = S^{-1/2} * (S^{-1/2}\tilde{S}S^{-1/2})^{-1/2} * S^{1/2}
    r = _get_r(s, s1)
    # H1 = R^{\dag} * H1 * R
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
    dtype = numpy.result_type(t, v, w, s)
    h = numpy.zeros((n2,n2), dtype=dtype)
    m = numpy.zeros((n2,n2), dtype=dtype)
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
    dtype = numpy.result_type(t, v, w, s)
    h = numpy.zeros((n2,n2), dtype=dtype)
    m = numpy.zeros((n2,n2), dtype=dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)

    try:
        e, a = scipy.linalg.eigh(h, m)
        cl = a[:nao,nao:]
        # cs = a[nao:,nao:]
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
        # cs = a[nao:,idx]
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

def _block_diag(mat):
    '''
    [A 0]
    [0 A]
    '''
    return scipy.linalg.block_diag(mat, mat)

def _sigma_dot(mat):
    '''sigma dot A x B + A dot B'''
    quaternion = numpy.vstack([1j * lib.PauliMatrices, numpy.eye(2)[None,:,:]])
    nao = mat.shape[-1] * 2
    return lib.einsum('sxy,spq->xpyq', quaternion, mat).reshape(nao, nao)


if __name__ == '__main__':
    from pyscf.x2c import dft
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

    method = dft.UKS(mol)
    ex2c = method.kernel()
    print('E(X2C1E) = %.12g' % ex2c)
