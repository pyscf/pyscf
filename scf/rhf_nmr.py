#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic NMR shielding tensor
'''


import sys
import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf

# flatten([[XX, XY, XZ],
#          [YX, YY, YZ],
#          [ZX, ZY, ZZ]])
TENSOR_IDX = numpy.arange(9)
# flatten([[XX, YX, ZX],
#          [XY, YY, ZY],
#          [XZ, YZ, ZZ]])
TENSOR_TRANSPOSE = TENSOR_IDX.reshape(3,3).T.flatten()
def dia(mol, dm0, gauge_orig=None, shielding_nuc=None):
    if shielding_nuc is None:
        shielding_nuc = range(1, mol.natm+1)
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)

    msc_dia = []
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id-1))
        if gauge_orig is None:
            h11 = mol.intor('int1e_giao_a11part_sph', 9)
        else:
            h11 = mol.intor('int1e_cg_a11part_sph', 9)
        trh11 = -(h11[0] + h11[4] + h11[8])
        h11[0] += trh11
        h11[4] += trh11
        h11[8] += trh11
        if gauge_orig is None:
            g11 = mol.intor('int1e_a01gp_sph', 9)
            h11 = h11 + g11[TENSOR_TRANSPOSE] # (mu,B) => (B,mu)
        a11 = [numpy.einsum('ij,ji', dm0, (x+x.T)*.5) for x in h11]
        msc_dia.append(a11)
        #     XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
        #  => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
    return numpy.array(msc_dia).reshape(-1, 3, 3)

def para(mol, mo10, mo_coeff, mo_occ, shielding_nuc=None):
    if shielding_nuc is None:
        shielding_nuc = range(1, mol.natm+1)
    msc_para = numpy.zeros((len(shielding_nuc),3,3))
    para_vir = numpy.zeros((len(shielding_nuc),3,3))
    para_occ = numpy.zeros((len(shielding_nuc),3,3))
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id-1))
        # 1/2(A01 dot p + p dot A01) => (ia01p - c.c.)/2 => <ia01p>
        h01 = mol.intor_asymmetric('int1e_ia01p_sph', 3)
        # *2 for doubly occupied orbitals
        h01_mo = _mat_ao2mo(h01, mo_coeff, mo_occ) * 2
        for b in range(3):
            for m in range(3):
                # c10^T * h01 + c.c.
                p = numpy.einsum('ij,ji->i',h01_mo[m], mo10[b].T) * 2
                msc_para[n,b,m] = p.sum()
                para_occ[n,b,m] = p[mo_occ>0].sum()
                para_vir[n,b,m] = msc_para[n,b,m] - para_occ[n,b,m]
    return msc_para, para_vir, para_occ

def make_h10(mol, dm0, gauge_orig=None, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if gauge_orig is None:
        # A10_i dot p + p dot A10_i consistents with <p^2 g>
        # A10_j dot p + p dot A10_j consistents with <g p^2>
        # A10_j dot p + p dot A10_j => i/2 (rjxp - pxrj) = irjxp
        h1 = .5 * mol.intor('int1e_giao_irjxp_sph', 3)
        log.debug('First-order GIAO Fock matrix')
        h1 += make_h10giao(mol, dm0)
    else:
        mol.set_common_origin(gauge_orig)
        h1 = .5 * mol.intor('int1e_cg_irxp_sph', 3)
    return h1

def make_h10giao(mol, dm0):
    vj, vk = _vhf.direct_mapdm('int2e_ig1_sph',  # (g i,j|k,l)
                               'a4ij', ('lk->s1ij', 'jk->s1il'),
                               dm0, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
# J = i[(i i|\mu g\nu) + (i gi|\mu \nu)] = i (i i|\mu g\nu)
# K = i[(\mu gi|i \nu) + (\mu i|i g\nu)]
#   = (\mu g i|i \nu) - h.c.   anti-symm because of the factor i
    vk = vk - vk.transpose(0,2,1)
    h1 = vj - .5 * vk
    h1 += mol.intor_asymmetric('int1e_ignuc_sph', 3)
    h1 += mol.intor('int1e_igkin_sph', 3)
    return h1

def make_s10(mol, gauge_orig=None):
    if gauge_orig is None:
        s1 = mol.intor_asymmetric('int1e_igovlp_sph', 3)
    else:
        nao = mol.nao_nr()
        s1 = numpy.zeros((3,nao,nao))
    return s1

def make_rdm1_1(mo1occ, mo0, occ):
    ''' DM^1 = (i * C_occ^1 C_occ^{0,dagger}) + c.c.  on AO'''
    mocc = mo0[:,occ>0] * occ[occ>0]
    dm1 = []
    for i in range(3):
        tmp = reduce(numpy.dot, (mo0, mo1occ[i], mocc.T.conj()))
        # note the minus sign due to the phase i
        dm1.append(tmp - tmp.T)
    return numpy.array(dm1)

def solve_mo1(mo_energy, mo_occ, h1, s1):
    '''uncoupled equation'''
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

    hs = h1 - s1 * e_i

    mo10 = numpy.empty_like(hs)
    mo10[:,mo_occ==0,:] = -hs[:,mo_occ==0,:] * e_ai
    mo10[:,mo_occ>0,:] = -s1[:,mo_occ>0,:] * .5

    e_ji = e_i.reshape(-1,1) - e_i
    mo_e10 = hs[:,mo_occ>0,:] + mo10[:,mo_occ>0,:] * e_ji
    return mo10, mo_e10


class NMR(lib.StreamObject):
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        self.shielding_nuc = [i+1 for i in range(self.mol.natm)]
# gauge_orig=None will call GIAO. Specify coordinate for common gauge
        self.gauge_orig = None
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys())

## ** default method **
#        # RHF: exchange parts
#        if not isinstance(scf_method, pyscf.scf.hf.RHF):
#            raise AttributeError('TODO: UHF')

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s shielding for %s ********',
                 self.__class__, self._scf.__class__)
        if self.gauge_orig is None:
            log.info('gauge = GIAO')
        else:
            log.info('Common gauge = %s', str(self.gauge_orig))
        log.info('shielding for atoms %s', str(self.shielding_nuc))
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')
        return self

    def kernel(self, mo1=None):
        return self.shielding(mo1)
    def shielding(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.dump_flags()
        if self.verbose >= logger.WARN:
            self.check_sanity()

        facppm = 1e6/lib.param.LIGHT_SPEED**2
        msc_para, para_vir, para_occ = [x*facppm for x in self.para(mo10=mo1)]
        msc_dia = self.dia() * facppm
        e11 = msc_para + msc_dia

        logger.timer(self, 'NMR shielding', *cput0)
        if self.verbose > logger.QUIET:
            for i, atm_id in enumerate(self.shielding_nuc):
                _write(self.stdout, e11[i], \
                       '\ntotal shielding of atom %d %s' \
                       % (atm_id, self.mol.atom_symbol(atm_id-1)))
                _write(self.stdout, msc_dia[i], 'dia-magnetism')
                _write(self.stdout, msc_para[i], 'para-magnetism')
                if self.verbose >= logger.INFO:
                    _write(self.stdout, para_occ[i], 'occ part of para-magnetism')
                    _write(self.stdout, para_vir[i], 'vir part of para-magnetism')
        self.stdout.flush()
        return e11

    def dia(self, mol=None, dm0=None, gauge_orig=None, shielding_nuc=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        if shielding_nuc is None: shielding_nuc = self.shielding_nuc
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return dia(mol, dm0, gauge_orig, shielding_nuc)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             shielding_nuc=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if shielding_nuc is None: shielding_nuc = self.shielding_nuc
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(mol, mo10, mo_coeff, mo_occ, shielding_nuc)

    def make_rdm1_1(self, mo1occ, mo0=None, occ=None):
        if mo0 is None: mo0 = self._scf.mo_coeff
        if occ is None: occ = self._scf.mo_occ
        return make_rdm1_1(mo1occ, mo0, occ)

    def make_h10(self, mol=None, dm0=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        if gauge_orig is None: gauge_orig = self.gauge_orig
        log = logger.Logger(self.stdout, self.verbose)
        h1 = make_h10(mol, dm0, gauge_orig, log)
        lib.chkfile.dump(self.chkfile, 'nmr/h1', h1)
        return h1

    def make_s10(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return make_s10(mol, gauge_orig)

    def solve_mo1(self, mo_energy=None, mo_occ=None, h1=None, s1=None):
        cput1 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_occ    is None: mo_occ = self._scf.mo_occ

        mol = self.mol
        if h1 is None:
            mo_coeff = self._scf.mo_coeff
            dm0 = self._scf.make_rdm1(mo_coeff, mo_occ)
            h1 = _mat_ao2mo(self.make_h10(mol, dm0), mo_coeff, mo_occ)
        if s1 is None:
            s1 = _mat_ao2mo(self.make_s10(mol), mo_coeff, mo_occ)

        cput1 = log.timer('first order Fock matrix', *cput1)
        if self.cphf:
            mo10, mo_e10 = cphf.solve(self.get_vind, mo_energy, mo_occ, h1, s1,
                                      self.max_cycle_cphf, self.conv_tol,
                                      verbose=log)
        else:
            mo10, mo_e10 = solve_mo1(mo_energy, mo_occ, h1, s1)
        logger.timer(self, 'solving mo1 eqn', *cput1)
        return mo10, mo_e10

    def get_vind(self, mo1):
        '''Induced potential'''
        mo_coeff = self._scf.mo_coeff
        mo_occ = self._scf.mo_occ
        dm1 = self.make_rdm1_1(mo1, mo_coeff, mo_occ)
        #direct_scf_bak, self._scf.direct_scf = self._scf.direct_scf, False
        v_ao = self._scf.get_veff(self.mol, dm1, hermi=2)
        #self._scf.direct_scf = direct_scf_bak
        return _mat_ao2mo(v_ao, mo_coeff, mo_occ)


def _mat_ao2mo(mat, mo_coeff, occ):
    '''transform an AO-based matrix to a MO-based matrix. The MO-based
    matrix only has the occupied columns M[:,:nocc]'''
    mo0 = mo_coeff[:,occ>0]
    mat_mo = [reduce(numpy.dot, (mo_coeff.T.conj(),i,mo0)) for i in mat]
    return numpy.array(mat_mo)

def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('B_x %s\n' % str(msc3x3[0]))
    stdout.write('B_y %s\n' % str(msc3x3[1]))
    stdout.write('B_z %s\n' % str(msc3x3[2]))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.nucmod = {'F': 2} # gaussian nuclear model
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.RHF(mol)
    rhf.max_memory = 0
    rhf.scf()
    nmr = NMR(rhf)
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    msc = nmr.shielding() # _xx,_yy = 375.232839, _zz = 483.002139
    print(msc)

