#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic RHF spin-spin coupling (SSC) constants
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf
from pyscf.scf.newton_ah import _gen_rhf_response
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.hfc.parameters import get_nuc_g_factor

NUMINT_GRIDS = 30


def make_dso(sscobj, mol, dm0, nuc_pair=None):
    '''orbital diamagnetic term'''
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    ssc_dia = []
    for ia, ja in nuc_pair:
        h11 = dso_integral(mol, mol.atom_coord(ia), mol.atom_coord(ja))
        a11 = -numpy.einsum('xyij,ji->xy', h11, dm0)
        a11 = a11 - a11.trace() * numpy.eye(3)
        ssc_dia.append(a11)
    return numpy.asarray(ssc_dia) * lib.param.ALPHA**4

def dso_integral(mol, orig1, orig2):
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
    mat1 = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
    # <i, j | nabla k>
    mat  = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
    mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
    return mat


# Note mo10 is the imaginary part of MO^1
def make_pso(sscobj, mol, mo1, mo_coeff, mo_occ, nuc_pair=None):
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
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
    h1 = make_h1_pso(mol, mo_coeff, mo_occ, atm1lst)
    h1 = numpy.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)
    for i,j in nuc_pair:
        # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
        e = numpy.einsum('xij,yij->xy', h1[atm1dic[i]], mo1[atm2dic[j]]) * 2
        para.append(e)
    return numpy.asarray(para) * lib.param.ALPHA**4

def make_h1_pso(mol, mo_coeff, mo_occ, atmlst):
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

def make_h1_fc(mol, mo_coeff, mo_occ, atmlst):
    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)
    mo = ao.dot(mo_coeff)
    mo_o = mo[:,mo_occ>0]
    h1 = []
    for ia in atmlst:
        h1.append(8*numpy.pi/3 * numpy.einsum('p,i->pi', mo[ia], mo_o[ia]))
    return h1

def make_h1_sd(mol, mo_coeff, mo_occ, atmlst):
    orbo = mo_coeff[:,mo_occ> 0]
    nao, nmo = mo_coeff.shape

    h1 = []
    for ia in atmlst:
        mol.set_rinv_origin(mol.atom_coord(ia))
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip
        h1ao = h1ao + h1ao.transpose(1,0,3,2)
        for i in range(3):
            for j in range(3):
                h1.append(orbv.T.conj().dot(h1ao[i,j]).dot(orbo))
    return h1

def _uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

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
        self.with_sd = False

    def dump_flags(self):
        rhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'nuc_pair %s', self.nuc_pair)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        mol = self.mol

        dm0 = self._scf.make_rdm1()
        mo_coeff = self._scf.mo_coeff
        mo_occ = self._scf.mo_occ

        ssc_dia = self.make_dso(mol, dm0)

        if mo1 is None:
            mo1 = self.mo10 = self.solve_mo1()[0]
        ssc_para = self.make_pso(mol, mo1, mo_coeff, mo_occ)
        e11 = ssc_para + ssc_dia
        logger.timer(self, 'spin-spin coupling', *cput0)

        if self.verbose > logger.QUIET:
            nuc_mag = .5 * (lib.param.E_MASS/lib.param.PROTON_MASS)  # e*hbar/2m
            au2Hz = lib.param.HARTREE2J / lib.param.PLANCK
            #logger.debug('Unit AU -> Hz %s', au2Hz*nuc_mag**2)
            iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
            natm = mol.natm
            ktensor = numpy.zeros((natm,natm))
            for k, (i, j) in enumerate(self.nuc_pair):
                ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
                if self.verbose >= logger.DEBUG:
                    _write(self.stdout, ssc_dia[k]+ssc_para[k],
                           '\nSSC E11 between %d %s and %d %s' \
                           % (i, self.mol.atom_symbol(i),
                              j, self.mol.atom_symbol(j)))
                    _write(self.stdout, ssc_dia [k], 'dia-magnetism')
                    _write(self.stdout, ssc_para[k], 'para-magnetism')

            gyro = [get_nuc_g_factor(mol.atom_symbol(ia)) for ia in range(natm)]
            jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
            label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
            logger.note(self, 'Reduced spin-spin coupling constant K (Hz)')
            tools.dump_mat.dump_tri(self.stdout, ktensor, label)
            logger.info(self, '\nNuclear g factor %s', gyro)
            logger.note(self, 'Spin-spin coupling constant J (Hz)')
            tools.dump_mat.dump_tri(self.stdout, jtensor, label)
        return e11

    dia = make_dso = make_dso
    para = make_pso = make_pso

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
        h1 = numpy.asarray(make_h1_pso(mol, self._scf.mo_coeff, mo_occ, atmlst))

        if with_cphf:
            vind = self.gen_vind(self._scf)
            mo1, mo_e1 = cphf.solve(vind, mo_energy, mo_occ, h1, None,
                                    self.max_cycle_cphf, self.conv_tol,
                                    verbose=log)
        else:
            e_ai = lib.direct_sum('i-a->ai', mo_energy[mo_occ>0], mo_energy[mo_occ==0])
            mo1 = h1 * (1 / e_ai)
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

SSC = SpinSpinCoupling


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.nucmod = {'F': 2} # gaussian nuclear model
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.RHF(mol).run()
    nmr = SSC(rhf)
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    jj = nmr.kernel() # _xx,_yy = , _zz =
    print(jj)
    print(lib.finger(jj))
