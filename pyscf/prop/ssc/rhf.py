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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic RHF spin-spin coupling (SSC) constants

Ref.
Chem. Rev. 99, 293 (1999); DOI:10.1021/cr960017t
JCP 113, 3530 (2000); DOI:10.1063/1.1286806
JCP 113, 9402 (2000); DOI:10.1063/1.1321296
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # noqa
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

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

    mf = sscobj._scf
    if getattr(mf, 'with_x2c', None):
        raise NotImplementedError('X2C for SSC')

    if getattr(mf, 'with_qmmm', None):
        raise NotImplementedError('SSC with QM/MM')

    if getattr(mf, 'with_solvent', None):
        raise NotImplementedError('SSC with Solvent')
    return numpy.asarray(ssc_dia) * nist.ALPHA**4

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
    nocc = numpy.count_nonzero(mo_occ> 0)
    nvir = numpy.count_nonzero(mo_occ==0)
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
    h1 = make_h1_pso(mol, mo_coeff, mo_occ, atm1lst)
    h1 = numpy.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)
    for i,j in nuc_pair:
        # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
        e = numpy.einsum('xij,yij->xy', h1[atm1dic[i]], mo1[atm2dic[j]])
        para.append(e*4)  # *4 for +c.c. and double occupnacy
    return numpy.asarray(para) * nist.ALPHA**4

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

def make_fc(sscobj, nuc_pair=None):
    '''Only Fermi-contact'''
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    mol = sscobj.mol
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    atm1dic, atm2dic = _uniq_atoms(nuc_pair)

    h1 = make_h1_fc(mol, mo_coeff, mo_occ, sorted(atm2dic.keys()))
    mo1 = solve_mo1_fc(sscobj, h1)
    h1 = make_h1_fc(mol, mo_coeff, mo_occ, sorted(atm1dic.keys()))
    para = []
    for i,j in nuc_pair:
        at1 = atm1dic[i]
        at2 = atm2dic[j]
        e = numpy.einsum('ij,ij', h1[at1], mo1[at2])
        para.append(e*4)  # *4 for +c.c. and for double occupancy
    return numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))

def solve_mo1_fc(sscobj, h1):
    cput1 = (time.clock(), time.time())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)
    mo_energy = sscobj._scf.mo_energy
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    nset = len(h1)
    eai = 1. / lib.direct_sum('a-i->ai', mo_energy[mo_occ==0], mo_energy[mo_occ>0])
    mo1 = numpy.asarray(h1) * -eai
    if not sscobj.cphf:
        return mo1

    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    nmo = nocc + nvir

    vresp = sscobj._scf.gen_response(singlet=False, hermi=1)
    mo_v_o = numpy.asarray(numpy.hstack((orbv,orbo)), order='F')
    def vind(mo1):
        dm1 = _dm1_mo2ao(mo1.reshape(nset,nvir,nocc), orbv, orbo*2)  # *2 for double occupancy
        dm1 = dm1 + dm1.transpose(0,2,1)
        v1 = vresp(dm1)
        v1 = _ao2mo.nr_e2(v1, mo_v_o, (0,nvir,nvir,nmo)).reshape(nset,nvir,nocc)
        v1 *= eai
        return v1.ravel()

    mo1 = lib.krylov(vind, mo1.ravel(), tol=sscobj.conv_tol,
                     max_cycle=sscobj.max_cycle_cphf, verbose=log)
    log.timer('solving FC CPHF eqn', *cput1)
    return mo1.reshape(nset,nvir,nocc)

def make_fcsd(sscobj, nuc_pair=None):
    '''FC + SD contributions to 2nd order energy'''
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    mol = sscobj.mol
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    atm1dic, atm2dic = _uniq_atoms(nuc_pair)

    h1 = make_h1_fcsd(mol, mo_coeff, mo_occ, sorted(atm2dic.keys()))
    mo1 = solve_mo1_fc(sscobj, h1)
    h1 = make_h1_fcsd(mol, mo_coeff, mo_occ, sorted(atm1dic.keys()))
    nocc = numpy.count_nonzero(mo_occ> 0)
    nvir = numpy.count_nonzero(mo_occ==0)
    mo1 = numpy.asarray(mo1).reshape(-1,3,3,nvir,nocc)
    h1  = numpy.asarray(h1).reshape(-1,3,3,nvir,nocc)
    para = []
    for i,j in nuc_pair:
        at1 = atm1dic[i]
        at2 = atm2dic[j]
        e = numpy.einsum('xwij,ywij->xy', h1[at1], mo1[at2])
        para.append(e*4)  # *4 for +c.c. and double occupancy
    return numpy.asarray(para) * nist.ALPHA**4


def make_h1_fc(mol, mo_coeff, mo_occ, atmlst):
    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)
    mo = ao.dot(mo_coeff)
    orbo = mo[:,mo_occ> 0]
    orbv = mo[:,mo_occ==0]
    fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
    h1 = []
    for ia in atmlst:
        h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))
    return h1

def make_h1_fcsd(mol, mo_coeff, mo_occ, atmlst):
    '''MO integrals for FC + SD'''
    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]

    h1 = []
    for ia in atmlst:
        h1ao = _get_integrals_fcsd(mol, ia)
        for i in range(3):
            for j in range(3):
                h1.append(orbv.T.conj().dot(h1ao[i,j]).dot(orbo) * .5)
    return h1

def _get_integrals_fcsd(mol, atm_id):
    '''AO integrals for FC + SD'''
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Note the fermi-contact part is different to the fermi-contact
        # operator in HFC, as well as the FC operator in EFG.
        # FC here is associated to the the integrals of
        # (-\nabla \nabla 1/r + I_3x3 \nabla\dot\nabla 1/r), which includes the
        # contribution of Poisson equation twice, i.e. 8\pi rho.
        # Therefore, -1./3 * (8\pi rho) is used as the contact contribution in
        # function _get_integrals_fc to remove the FC part.
        # In HFC or EFG, the factor of FC part is 4\pi/3.
        a01p = mol.intor('int1e_sa01sp', 12).reshape(3,4,nao,nao)
        h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
    return h1ao

def _get_integrals_fc(mol, atm_id):
    '''AO integrals for Fermi contact term'''
    # The factor -8\pi/3 is used because FC part is assoicated to the integrals
    # of (-\nabla \nabla 1/r + I_3x3 \nabla\dot\nabla 1/r). See also the
    # function _get_integrals_fcsd above.
    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    return -8*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)

def _uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

def _dm1_mo2ao(dm1, ket, bra):
    nao, nket = ket.shape
    nbra = bra.shape[1]
    nset = len(dm1)
    dm1 = lib.ddot(ket, dm1.transpose(1,0,2).reshape(nket,nset*nbra))
    dm1 = dm1.reshape(nao,nset,nbra).transpose(1,0,2).reshape(nset*nao,nbra)
    return lib.ddot(dm1, bra.T).reshape(nset,nao,nao)


def solve_mo1(sscobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              h1=None, s1=None, with_cphf=None):
    cput1 = (time.clock(), time.time())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)
    if mo_energy is None: mo_energy = sscobj._scf.mo_energy
    if mo_coeff  is None: mo_coeff = sscobj._scf.mo_coeff
    if mo_occ    is None: mo_occ = sscobj._scf.mo_occ
    if with_cphf is None: with_cphf = sscobj.cphf

    mol = sscobj.mol
    if h1 is None:
        atmlst = sorted(set([j for i,j in sscobj.nuc_pair]))
        h1 = numpy.asarray(make_h1_pso(mol, mo_coeff, mo_occ, atmlst))

    if with_cphf:
        if callable(with_cphf):
            vind = with_cphf
        else:
            vind = gen_vind(sscobj._scf, mo_coeff, mo_occ)
        mo1, mo_e1 = cphf.solve(vind, mo_energy, mo_occ, h1, None,
                                sscobj.max_cycle_cphf, sscobj.conv_tol,
                                verbose=log)
    else:
        e_ai = lib.direct_sum('i-a->ai', mo_energy[mo_occ>0], mo_energy[mo_occ==0])
        mo1 = h1 * (1 / e_ai)
        mo_e1 = None
    logger.timer(sscobj, 'solving mo1 eqn', *cput1)
    return mo1, mo_e1

def gen_vind(mf, mo_coeff, mo_occ):
    '''Induced potential associated with h1_PSO'''
    vresp = mf.gen_response(singlet=True, hermi=2)
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    nocc = orbo.shape[1]
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    def vind(mo1):
        dm1 = [reduce(numpy.dot, (orbv, x*2, orbo.T.conj()))
               for x in mo1.reshape(-1,nvir,nocc)]
        dm1 = numpy.asarray([d1-d1.conj().T for d1 in dm1])
        v1mo = numpy.asarray([reduce(numpy.dot, (orbv.T.conj(), x, orbo))
                              for x in vresp(dm1)])
        return v1mo.ravel()
    return vind

def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('mu_x %s\n' % str(msc3x3[0]))
    stdout.write('mu_y %s\n' % str(msc3x3[1]))
    stdout.write('mu_z %s\n' % str(msc3x3[2]))
    stdout.flush()

def _atom_gyro_list(mol):
    gyro = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in mol.nucprop:
            prop = mol.nucprop[symb]
            mass = prop.get('mass', None)
            gyro.append(get_nuc_g_factor(symb, mass))
        else:
            # Get default isotope
            gyro.append(get_nuc_g_factor(symb))
    return numpy.array(gyro)


class SpinSpinCoupling(lib.StreamObject):
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        mol = scf_method.mol
        self.nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        self.with_fc = True
        self.with_fcsd = False

        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        log.info('nuc_pair %s', self.nuc_pair)
        log.info('with Fermi-contact  %s', self.with_fc)
        log.info('with Fermi-contact + spin-dipole  %s', self.with_fcsd)
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
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
        ssc_pso = self.make_pso(mol, mo1, mo_coeff, mo_occ)
        e11 = ssc_dia + ssc_pso
        if self.with_fcsd:
            ssc_fcsd = self.make_fcsd(self.nuc_pair)
            e11 += ssc_fcsd
        elif self.with_fc:
            ssc_fc = self.make_fc(self.nuc_pair)
            e11 += ssc_fc
        logger.timer(self, 'spin-spin coupling', *cput0)

        if self.verbose >= logger.NOTE:
            nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
            au2Hz = nist.HARTREE2J / nist.PLANCK
            unit = au2Hz * nuc_magneton ** 2
            logger.debug(self, 'Unit AU -> Hz %s', unit)

            iso_ssc = unit * numpy.einsum('kii->k', e11) / 3
            natm = mol.natm
            ktensor = numpy.zeros((natm,natm))
            for k, (i, j) in enumerate(self.nuc_pair):
                ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
                if self.verbose >= logger.DEBUG:
                    _write(self.stdout, e11[k],
                           '\nSSC E11 between %d %s and %d %s' \
                           % (i, self.mol.atom_symbol(i),
                              j, self.mol.atom_symbol(j)))
#                    _write(self.stdout, ssc_dia [k], 'dia-magnetism')
#                    _write(self.stdout, ssc_para[k], 'para-magnetism')

            gyro = _atom_gyro_list(mol)
            jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
            label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
            logger.note(self, 'Reduced spin-spin coupling constant K (Hz)')
            tools.dump_mat.dump_tri(self.stdout, ktensor, label)
            logger.info(self, '\nNuclear g factor %s', gyro)
            logger.note(self, 'Spin-spin coupling constant J (Hz)')
            tools.dump_mat.dump_tri(self.stdout, jtensor, label)
        return e11

    dia = make_dso = make_dso
    make_pso = make_pso
    make_fc = make_fc
    make_fcsd = make_fcsd

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             nuc_pair=None):
        ssc_para = self.make_pso(mol, mo10, mo_coeff, mo_occ)
        if self.with_fcsd:
            ssc_para += self.make_fcsd(mol, mo10, mo_coeff, mo_occ)
        elif self.with_fc:
            ssc_para += self.make_fc(mol, mo10, mo_coeff, mo_occ)
        return ssc_para

    solve_mo1 = solve_mo1

SSC = SpinSpinCoupling

from pyscf import scf
scf.hf.RHF.SSC = scf.hf.RHF.SpinSpinCoupling = lib.class_as_method(SSC)


if __name__ == '__main__':
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

    mf = scf.RHF(mol).run()
    ssc = mf.SSC()
    ssc.verbose = 4
    ssc.cphf = True
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()
    print(jj)
    print(lib.finger(jj)*1e8 - 0.12374812977885304)

    mol = gto.M(atom='''
                O 0 0      0
                H 0 -0.757 0.587
                H 0  0.757 0.587''',
                basis='ccpvdz')

    mf = scf.RHF(mol).run()
    ssc = SSC(mf)
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()
    print(lib.finger(jj)*1e8 - -0.11191697931377538)

    ssc.with_fc = True
    ssc.with_fcsd = False
    jj = ssc.kernel()
    print(lib.finger(jj)*1e8 - 0.82442034395656116)
