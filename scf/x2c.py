#!/usr/bin/env python


import time
import ctypes
import _ctypes
from functools import reduce
import numpy
import scipy.linalg
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import dhf
from pyscf.scf import _vhf


EXP_DROP = .2

def sfx2c1e(mf):
    '''Spin-free X2C.
    For the given SCF object, update the hcore constructor.

    Args:
        mf : an SCF object

    Attributes:
        xuncontract : bool or str or list of str/int
            Use uncontracted basis to expand X matrix.
            When atom symbol (str type) is assigned to this attribute, the
            uncontracted basis will be used for the specified atoms.  If a
            list is given, the uncontracted basis will be applied for the
            atoms or atom-ID specified by the given list.

    Returns:
        An SCF object

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.sfx2c1e(scf.RHF(mol))
    >>> mf.scf()

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = scf.sfx2c1e(scf.UHF(mol))
    >>> mf.scf()
    '''
    mf_class = mf.__class__
    class HF(mf_class):
        def __init__(self):
            self.xuncontract = True
            self.xequation = '1e'
            self._tag_x2c = True
            self.__dict__.update(mf.__dict__)
            self._keys = self._keys.union(['xequation', 'xuncontract'])

        def get_hcore(self, mol=None):
            if not self._tag_x2c:
                return mf_class.get_hcore(self, mol)

            mol = mf.mol
            if self.xuncontract:
                xmol, contr_coeff = _uncontract_mol(mol, self.xuncontract)
            else:
                xmol = mol
            c = mol.light_speed
            t = xmol.intor_symmetric('cint1e_kin_sph')
            v = xmol.intor_symmetric('cint1e_nuc_sph')
            s = xmol.intor_symmetric('cint1e_ovlp_sph')
            w = xmol.intor_symmetric('cint1e_pnucp_sph')

            nao = t.shape[0]
            n2 = nao * 2
            h = numpy.zeros((n2,n2))
            m = numpy.zeros((n2,n2))
            h[:nao,:nao] = v
            h[:nao,nao:] = t
            h[nao:,:nao] = t
            h[nao:,nao:] = w * (.25/c**2) - t
            m[:nao,:nao] = s
            m[nao:,nao:] = t * (.5/c**2)

            e, a = scipy.linalg.eigh(h, m)
            cl = a[:nao,nao:]
            cs = a[nao:,nao:]

# Taking A matrix as basis and rewrite the FW Hcore formula, to avoid inversing matrix
#   R^dag \tilde{S} R = S
#   R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
#   R = (A^+ S A)^{1/2} = (A^+ S A)^{-1/2} A^+ S A
#   h1 = (A^+)^{-1} R^+ (A^+ h1 A) R A^{-1}
#      = \tilde{S} A R^+ A^+ h1 A R A^+ \tilde{S}   (1)
#      = S A R^{-1}^+ A^+ h1 A R{-1} A^+ S          (2)
            w, u = numpy.linalg.eigh(reduce(numpy.dot, (cl.T, s, cl)))
            idx = w > 1e-14
# Adopt (2) here becuase X is not appeared in Eq (2).
            r = reduce(numpy.dot, (u[:,idx]/numpy.sqrt(w[idx]), u[:,idx].T, cl.T, s))
            h1 = reduce(numpy.dot, (r.T.conj()*e[nao:], r))

            if self.xuncontract:
                h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))
            return h1

    return HF()

sfx2c = sfx2c1e


def get_hcore(mol, xuncontract=False):
    '''Foldy-Wouthuysen hcore'''
    if xuncontract:
        xmol, contr_coeff_nr = _uncontract_mol(mol, xuncontract)
        np, nc = contr_coeff_nr.shape
        contr_coeff = numpy.zeros((np*2,nc*2))
        contr_coeff[0::2,0::2] = contr_coeff_nr
        contr_coeff[1::2,1::2] = contr_coeff_nr
    else:
        xmol = mol

    c = mol.light_speed
    s = xmol.intor_symmetric('cint1e_ovlp')
    t = xmol.intor_symmetric('cint1e_spsp') * .5
    v = xmol.intor_symmetric('cint1e_nuc')
    w = xmol.intor_symmetric('cint1e_spnucsp')

    n2c = t.shape[0]
    n4c = n2c * 2
    h = numpy.zeros((n4c,n4c), dtype=numpy.complex)
    m = numpy.zeros((n4c,n4c), dtype=numpy.complex)
    h[:n2c,:n2c] = v
    h[:n2c,n2c:] = t
    h[n2c:,:n2c] = t
    h[n2c:,n2c:] = w * (.25/c**2) - t
    m[:n2c,:n2c] = s
    m[n2c:,n2c:] = t * (.5/c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:n2c,n2c:]
    cs = a[n2c:,n2c:]
    x = numpy.linalg.solve(cl.T, cs.T).T  # B = XA
    h1 = _get_hcore_fw(t, v, w, s, x, c)
    if xuncontract:
        h1 = reduce(numpy.dot, (contr_coeff.T.conj(), h1, contr_coeff))
    return h1

def get_jk(mol, dm, hermi=1, mf_opt=None):
    n2c = dm.shape[0]
    dd = numpy.zeros((n2c*2,)*2, dtype=numpy.complex)
    dd[:n2c,:n2c] = dm
    dhf._call_veff_llll(mol, dd, hermi, None)
    vj, vk = _vhf.rdirect_mapdm('cint2e', 's8',
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

def init_guess_by_chkfile(mol, chkfile_name, project=True):
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


class UHF(hf.SCF):
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.xuncontract = False
        self.xequation = '1e'
        self._keys = self._keys.union(['xequation', 'xuncontract'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        logger.info(self, 'X equation %s', self.xequation)
        return self

    def build_(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(self.mol)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_chkfile(self, chkfile=None, project=True):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def eig(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,range(len(e))].real<0] *= -1
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol, self.xuncontract)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('cint1e_ovlp')

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
            opt._this.contents.r_vkscreen = \
                ctypes.c_void_p(_ctypes.dlsym(_vhf.libcvhf._handle, name))
        opt = _vhf.VHFOpt(mol, 'cint2e', 'CVHFrkbllll_prescreen',
                          'CVHFrkbllll_direct_scf',
                          'CVHFrkbllll_direct_scf_dm')
        opt.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt, 'CVHFrkbllll_vkscreen')
        return opt

    def get_jk_(self, mol=None, dm=None, hermi=1):
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


def _uncontract_mol(mol, xuncontract=False):
    '''mol._basis + uncontracted steep functions'''
    pmol = mol.copy()
    _bas = []
    _env = []
    ptr = len(pmol._env)
    contr_coeff = []
    for ib in range(mol.nbas):
        if isinstance(xuncontract, bool):
            uncontract_me = xuncontract
        elif isinstance(xuncontract, str):
            ia = mol.bas_atom(ib)
            uncontract_me = ((xuncontract == mol.atom_pure_symbol(ia)) or
                             (xuncontract == mol.atom_symbol(ia)))
        elif isinstance(xuncontract, (tuple, list)):
            ia = mol.bas_atom(ib)
            uncontract_me = ((mol.atom_pure_symbol(ia) in xuncontract) or
                             (mol.atom_symbol(ia) in xuncontract) or
                             (ia in xuncontract))
        else:
            raise RuntimeError('xuncontract needs to be bool, str, tuple or list type')

        if uncontract_me:
            np = mol._bas[ib,mole.NPRIM_OF]
            nc = mol._bas[ib,mole.NCTR_OF]
            l = mol.bas_angular(ib)
            pexp = mol._bas[ib,mole.PTR_EXP]
# Modfied partially uncontraction to avoid potentially lindep in the
# segment-contracted basis
            nkept = (pmol._env[pexp:pexp+np] > EXP_DROP).sum()
            if nkept > nc:
                b_coeff = mol.bas_ctr_coeff(ib)
                norms = mole.gto_norm(l, mol._env[pexp:pexp+np])
                importance = numpy.einsum('i,ij->i', 1/norms, abs(b_coeff))
                idx = numpy.argsort(importance[:nkept])
                contracted = numpy.sort(idx[nkept-nc:])
                primitive  = numpy.sort(idx[:nkept-nc])

# part1: pGTOs that are associated with small coefficients
                bs = numpy.empty((nkept-nc,mol._bas.shape[1]), dtype=numpy.int32)
                bs[:] = mol._bas[ib]
                bs[:,mole.NCTR_OF] = bs[:,mole.NPRIM_OF] = 1
                norms = numpy.empty(nkept-nc)
                for k, i in enumerate(primitive):
                    norms[k] = mole.gto_norm(l, mol._env[pexp+i])
                    _env.append(mol._env[pexp+i])
                    _env.append(norms[k])
                    bs[k,mole.PTR_EXP] = ptr
                    bs[k,mole.PTR_COEFF] = ptr + 1
                    ptr += 2
                _bas.append(bs)
                d = l * 2 + 1
                part1 = numpy.zeros((d*(nkept-nc),d*nc))
                c = numpy.einsum('i,ij->ij', 1/norms, b_coeff[primitive])
                for i in range(d):
                    part1[i::d,i::d] = c

# part2: binding the pGTOs of small exps to the pGTOs of large coefficients
                bs = mol._bas[ib].copy()
                bs[mole.NPRIM_OF] = np - nkept + nc
                idx = numpy.hstack((contracted, numpy.arange(nkept,np)))
                exps = mol._env[pexp:pexp+np][idx]
                cs = b_coeff[idx]
                ee = mole._gaussian_int(l*2+2, exps[:,None] + exps)
                s1 = numpy.einsum('pi,pq,qi->i', cs, ee, cs)
                s1 = numpy.sqrt(s1)
                cs = numpy.einsum('pi,i->pi', cs, 1/s1)
                _env.extend(exps)
                _env.extend(cs.T.reshape(-1))
                bs[mole.PTR_EXP] = ptr
                bs[mole.PTR_COEFF] = ptr + exps.size
                ptr += exps.size + cs.size
                _bas.append(bs)

                part2 = numpy.eye(d*nc)
                for i in range(nc):
                    part2[i*d:(i+1)*d,i*d:(i+1)*d] *= s1[i]
                contr_coeff.append(numpy.vstack((part1, part2)))
            else:
                _bas.append(mol._bas[ib])
                contr_coeff.append(numpy.eye((l*2+1)*nc))
        else:
            _bas.append(mol._bas[ib])
            l = mol.bas_angular(ib)
            d = l * 2 + 1
            contr_coeff.append(numpy.eye(d*mol.bas_nctr(ib)))
    pmol._bas = numpy.vstack(_bas)
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

    # R^dag \tilde{S} R = S
    # R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
    sa = _invsqrt(s)
    sb = _invsqrt(reduce(numpy.dot, (sa, s1, sa)))
    r = reduce(numpy.dot, (sa, sb, sa, s))
    h1 = reduce(numpy.dot, (r.T.conj(), h1, r))
    return h1


def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, 1, mol)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm_ll = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
    return dm_ll


if __name__ == '__main__':
    import pyscf.gto
    import pyscf.scf
    mol = pyscf.gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    method = pyscf.scf.RHF(mol)
    enr = method.kernel()
    print('E(NR) = %.12g' % enr)

    method = sfx2c1e(pyscf.scf.RHF(mol))
    esfx2c = method.kernel()
    print('E(SFX2C1E) = %.12g' % esfx2c)

    method = UHF(mol)
    ex2c = method.kernel()
    print('E(X2C1E) = %.12g' % ex2c)

