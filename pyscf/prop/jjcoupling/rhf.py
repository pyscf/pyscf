#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic J-J coupling tensor
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf
from pyscf.scf.newton_ah import _gen_rhf_response
from pyscf.prop.nmr import rhf as rhf_nmr

UNIT_PPM = lib.param.ALPHA**2 * 1e6
NUMINT_GRIDS = 30

def dia(mol, dm0, nuc_pair):
    jj_dia = []
    for ia, ja in nuc_pair:
        h11 = jj_integral(mol, mol.atom_coord(ia), mol.atom_coord(ja))
        a11 = -numpy.einsum('xyij,ji->xy', h11, dm0)
        a11 = a11 - a11.trace() * numpy.eye(3)
        jj_dia.append(a11)
    return numpy.asarray(jj_dia)

def jj_integral(mol, orig1, orig2):
    '''Integral of vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
    Ref. JCP, 73, 5718'''
    t, w = numpy.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1+t)/(1-t) * .8
    w *= 2/(1-t)**2 * .8
    fakemol = gto.Mole()
    fakemol._atm = numpy.asarray([[0, 0, 0, 0, 0, 0]], dtype=numpy.int32)
    fakemol._bas = numpy.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                 dtype=numpy.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = numpy.hstack((orig2, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
    fakemol._built = True

    pmol = mol + fakemol
    pmol.set_rinv_origin(orig1)
    # <nabla i, j | k>  k is a fictitious basis for numerical integraion
    mat1 = pmol.intor('int3c1e_iprinv_sph', comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
    # <i, j | nabla k>
    mat  = pmol.intor('int3c1e_iprinv_sph', comp=3,
                      shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
    mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
    return mat


# Note mo10 is the imaginary part of MO^1
def para(mol, mo1, mo_coeff, mo_occ, nuc_pair):
    para = []
    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    # *2 for doubly occupied orbitals
    dm10 = numpy.asarray([reduce(numpy.dot, (orbv, x*2, orbo.T.conj())) for x in mo1])
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
    h1 = make_h1_mo(mol, mo_coeff, mo_occ, atm1lst)
    h1 = numpy.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)
    for i,j in nuc_pair:
        # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
        e = numpy.einsum('xij,yij->xy', h1[atm1dic[i]], mo1[atm2dic[j]]) * 2
        para.append(e)
    return numpy.asarray(para)

def make_h1_mo(mol, mo_coeff, mo_occ, atmlst):
    # Imaginary part of H01 operator
    # 1/2(A01 dot p + p dot A01) => (a01p + c.c.)/2 ~ <a01p> 
    # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]

    h1 = []
    for ia in atmlst:
        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.intor_asymmetric('int1e_prinvxp', 3)
        h1 += [reduce(numpy.dot, (orbv.T.conj(), x, orbo)) for x in h1ao]
    return h1

def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('mu_x %s\n' % str(msc3x3[0]))
    stdout.write('mu_y %s\n' % str(msc3x3[1]))
    stdout.write('mu_z %s\n' % str(msc3x3[2]))
    stdout.flush()


class SpinSpinCoupling(rhf_nmr.NMR):
    def __init__(self, scf_method):
        mol = scf_method.mol
        self.nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        rhf_nmr.NMR.__init__(self, scf_method)

    def dump_flags(self):
        rhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'nuc_pair %s', self.nuc_pair)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        jj_dia = self.dia() * UNIT_PPM
        jj_para = self.para(mo10=mo1)
        jj_para *= UNIT_PPM
        e11 = jj_para + jj_dia

        logger.timer(self, 'JJ coupling', *cput0)
        if self.verbose > logger.QUIET:
            for k, (i, j) in enumerate(self.nuc_pair):
                _write(self.stdout, e11[k],
                       '\nJJ coupling between %d %s and %d %s' \
                       % (i, self.mol.atom_symbol(i),
                          j, self.mol.atom_symbol(j)))
                if self.verbose >= logger.INFO:
                    _write(self.stdout, jj_dia [k], 'dia-magnetism')
                    _write(self.stdout, jj_para[k], 'para-magnetism')
        return e11

    def dia(self, mol=None, dm0=None, nuc_pair=None):
        if mol is None: mol = self.mol
        if nuc_pair is None: nuc_pair = self.nuc_pair
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return dia(mol, dm0, nuc_pair)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             nuc_pair=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if nuc_pair is None: nuc_pair = self.nuc_pair
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo1 = self.mo10
        return para(mol, mo1, mo_coeff, mo_occ, nuc_pair)

    def solve_mo1(self, mo_energy=None, mo_occ=None, nuc_pair=None,
                  with_cphf=None):
        cput1 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_occ    is None: mo_occ = self._scf.mo_occ
        if nuc_pair  is None: nuc_pair = self.nuc_pair
        if with_cphf is None: with_cphf = self.cphf

        atmlst = sorted(set([j for i,j in nuc_pair]))
        mol = self.mol
        h1 = numpy.asarray(make_h1_mo(mol, self._scf.mo_coeff, mo_occ, atmlst))

        if with_cphf:
            vind = self.gen_vind(self._scf)
            mo1, mo_e1 = cphf.solve(vind, mo_energy, mo_occ, h1, None,
                                    self.max_cycle_cphf, self.conv_tol,
                                    verbose=log)
        else:
            e_ai = -1 / (mo_energy[mo_occ==0] - mo_energy[mo_oc>0])
            mo1 = h1 * e_ai
            mo_e1 = None
        logger.timer(self, 'solving mo1 eqn', *cput1)
        return mo1, mo_e1

    def gen_vind(self, mf):
        '''Induced potential'''
        vresp = _gen_rhf_response(mf, hermi=2)
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        orbv = mo_coeff[:,~occidx]
        nocc = orbo.shape[1]
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        def vind(mo1):
            #direct_scf_bak, mf.direct_scf = mf.direct_scf, False
            dm1 = [reduce(numpy.dot, (orbv, x*2, orbo.T.conj()))
                   for x in mo1.reshape(-1,nvir,nocc)]
            dm1 = numpy.asarray([d1-d1.conj().T for d1 in dm1])
            v1mo = numpy.asarray([reduce(numpy.dot, (orbv.T.conj(), x, orbo))
                                  for x in vresp(dm1)])
            #mf.direct_scf = direct_scf_bak
            return v1mo.ravel()
        return vind
JJ = JJCoupling = SpinSpinCoupling


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

    rhf = scf.RHF(mol).run()
    nmr = JJ(rhf)
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    jj = nmr.kernel() # _xx,_yy = , _zz =
    print(jj)
    print(lib.finger(jj))
