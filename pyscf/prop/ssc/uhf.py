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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic UHF spin-spin coupling (SSC) constants
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.soscf.newton_ah import _gen_uhf_response
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.ssc import rhf as rhf_ssc
from pyscf.prop.ssc.rhf import _uniq_atoms, _dm1_mo2ao, _write
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

NUMINT_GRIDS = 30


def make_dso(sscobj, mol, dm0, nuc_pair=None):
    '''orbital diamagnetic term'''
    if not (isinstance(dm0, numpy.ndarray) and dm0.ndim == 2):
        dm0 = dm0[0] + dm0[1]
    return rhf_ssc.make_dso(sscobj, mol, dm0, nuc_pair)

# Note mo10 is the imaginary part of MO^1
def make_pso(sscobj, mol, mo1, mo_coeff, mo_occ, nuc_pair=None):
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    atm1dic, atm2dic = _uniq_atoms(nuc_pair)
    h1a, h1b = make_h1_pso(mol, mo_coeff, mo_occ, sorted(atm1dic.keys()))
    mo1a, mo1b = mo1
    nvira, nocca = h1a[0].shape
    nvirb, noccb = h1b[0].shape
    mo1a = mo1a.reshape(len(atm2dic),3,nvira,nocca)
    mo1b = mo1b.reshape(len(atm2dic),3,nvirb,noccb)
    h1a = numpy.asarray(h1a).reshape(len(atm1dic),3,nvira,nocca)
    h1b = numpy.asarray(h1b).reshape(len(atm1dic),3,nvirb,noccb)
    para = []
    for i,j in nuc_pair:
        # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
        e = numpy.einsum('xij,yij->xy', h1a[atm1dic[i]], mo1a[atm2dic[j]]) * 2
        e+= numpy.einsum('xij,yij->xy', h1b[atm1dic[i]], mo1b[atm2dic[j]]) * 2
        para.append(e)
    return numpy.asarray(para) * nist.ALPHA**4

def make_h1_pso(mol, mo_coeff, mo_occ, atmlst):
    # Imaginary part of H01 operator
    # 1/2(A01 dot p + p dot A01) => (a01p + c.c.)/2 ~ <a01p> 
    # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
    orboa = mo_coeff[0][:,mo_occ[0]> 0]
    orbva = mo_coeff[0][:,mo_occ[0]==0]
    orbob = mo_coeff[1][:,mo_occ[1]> 0]
    orbvb = mo_coeff[1][:,mo_occ[1]==0]
    h1a = []
    h1b = []
    for ia in atmlst:
        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = mol.intor_asymmetric('int1e_prinvxp', 3)
        h1a += [reduce(numpy.dot, (orbva.T.conj(), -x, orboa)) for x in h1ao]
        h1b += [reduce(numpy.dot, (orbvb.T.conj(), -x, orbob)) for x in h1ao]
    return h1a, h1b

def make_fc(sscobj, nuc_pair=None):
    '''Only Fermi-contact'''
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    mol = sscobj.mol
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    atm1dic, atm2dic = _uniq_atoms(nuc_pair)

    h1 = make_h1_fc(mol, mo_coeff, mo_occ, sorted(atm2dic.keys()))
    mo1aa, mo1ab, mo1ba, mo1bb = solve_mo1_fc(sscobj, h1)
    h1 = None
    h1aa, h1ab, h1ba, h1bb = make_h1_fc(mol, mo_coeff, mo_occ, sorted(atm1dic.keys()))
    para = []
    for i,j in nuc_pair:
        at1 = atm1dic[i]
        at2 = atm2dic[j]
        ez  = numpy.einsum('ij,ij', h1aa[at1], mo1aa[at2]) * 2  # *2 for +c.c.
        ez += numpy.einsum('ij,ij', h1bb[at1], mo1bb[at2]) * 2
        ex  = numpy.einsum('ij,ij', h1ab[at1], mo1ab[at2]) * 2
        ex += numpy.einsum('ij,ij', h1ba[at1], mo1ba[at2]) * 2
        ey = ex
        para.append(numpy.diag([ex,ey,ez]))
    return numpy.asarray(para) * nist.ALPHA**4

# See also the UHF to GHF stability analysis
def solve_mo1_fc(sscobj, h1):
    cput1 = (time.clock(), time.time())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)
    mol = sscobj.mol
    mo_energy = sscobj._scf.mo_energy
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    h1aa, h1ab, h1ba, h1bb = h1
    nset = len(h1aa)
    eai_aa = 1. / lib.direct_sum('a-i->ai', mo_energy[0][mo_occ[0]==0], mo_energy[0][mo_occ[0]>0])
    eai_ab = 1. / lib.direct_sum('a-i->ai', mo_energy[0][mo_occ[0]==0], mo_energy[1][mo_occ[1]>0])
    eai_ba = 1. / lib.direct_sum('a-i->ai', mo_energy[1][mo_occ[1]==0], mo_energy[0][mo_occ[0]>0])
    eai_bb = 1. / lib.direct_sum('a-i->ai', mo_energy[1][mo_occ[1]==0], mo_energy[1][mo_occ[1]>0])

    mo1_fc = (numpy.asarray(h1aa) * -eai_aa,
              numpy.asarray(h1ab) * -eai_ab,
              numpy.asarray(h1ba) * -eai_ba,
              numpy.asarray(h1bb) * -eai_bb)
    h1aa = h1ab = h1ba = h1bb = None
    if not sscobj.cphf:
        return mo1_fc

    orboa = mo_coeff[0][:,mo_occ[0]> 0]
    orbva = mo_coeff[0][:,mo_occ[0]==0]
    orbob = mo_coeff[1][:,mo_occ[1]> 0]
    orbvb = mo_coeff[1][:,mo_occ[1]==0]
    nocca = orboa.shape[1]
    nvira = orbva.shape[1]
    noccb = orbob.shape[1]
    nvirb = orbvb.shape[1]
    p1 = nvira * nocca
    p2 = p1 + nvira * noccb
    p3 = p2 + nvirb * nocca
    def _split_mo1(mo1):
        mo1 = mo1.reshape(nset,-1)
        mo1aa = mo1[:,  :p1].reshape(nset,nvira,nocca)
        mo1ab = mo1[:,p1:p2].reshape(nset,nvira,noccb)
        mo1ba = mo1[:,p2:p3].reshape(nset,nvirb,nocca)
        mo1bb = mo1[:,p3:  ].reshape(nset,nvirb,noccb)
        return mo1aa, mo1ab, mo1ba, mo1bb

    mo1_fc = numpy.hstack((mo1_fc[0].reshape(nset,-1),
                           mo1_fc[1].reshape(nset,-1),
                           mo1_fc[2].reshape(nset,-1),
                           mo1_fc[3].reshape(nset,-1)))

    vresp = _gen_uhf_response(sscobj._scf, with_j=False, hermi=1)
    mo_va_oa = numpy.asarray(numpy.hstack((orbva,orboa)), order='F')
    mo_va_ob = numpy.asarray(numpy.hstack((orbva,orbob)), order='F')
    mo_vb_oa = numpy.asarray(numpy.hstack((orbvb,orboa)), order='F')
    mo_vb_ob = numpy.asarray(numpy.hstack((orbvb,orbob)), order='F')
    def vind(mo1):
        mo1aa, mo1ab, mo1ba, mo1bb = _split_mo1(mo1)
        dm1aa = _dm1_mo2ao(mo1aa, orbva, orboa)
        dm1ab = _dm1_mo2ao(mo1ab, orbva, orbob)
        dm1ba = _dm1_mo2ao(mo1ba, orbvb, orboa)
        dm1bb = _dm1_mo2ao(mo1bb, orbvb, orbob)
        dm1 = numpy.vstack([dm1aa+dm1aa.transpose(0,2,1),
                            dm1ab+dm1ba.transpose(0,2,1),
                            dm1ba+dm1ab.transpose(0,2,1),
                            dm1bb+dm1bb.transpose(0,2,1)])
        v1 = vresp(dm1)
        v1aa = _ao2mo.nr_e2(v1[      :nset  ], mo_va_oa, (0,nvira,nvira,nvira+nocca))
        v1ab = _ao2mo.nr_e2(v1[nset*1:nset*2], mo_va_ob, (0,nvira,nvira,nvira+noccb))
        v1ba = _ao2mo.nr_e2(v1[nset*2:nset*3], mo_vb_oa, (0,nvirb,nvirb,nvirb+nocca))
        v1bb = _ao2mo.nr_e2(v1[nset*3:      ], mo_vb_ob, (0,nvirb,nvirb,nvirb+noccb))
        v1aa = v1aa.reshape(nset,nvira,nocca)
        v1ab = v1ab.reshape(nset,nvira,noccb)
        v1ba = v1ba.reshape(nset,nvirb,nocca)
        v1bb = v1bb.reshape(nset,nvirb,noccb)
        v1aa *= eai_aa
        v1ab *= eai_ab
        v1ba *= eai_ba
        v1bb *= eai_bb
        v1mo = numpy.hstack((v1aa.reshape(nset,-1), v1ab.reshape(nset,-1),
                             v1ba.reshape(nset,-1), v1bb.reshape(nset,-1)))
        return v1mo.ravel()

    mo1 = lib.krylov(vind, mo1_fc.ravel(), tol=1e-9, max_cycle=20, verbose=log)
    log.timer('solving FC CPHF eqn', *cput1)
    mo1_fc = _split_mo1(mo1)
    return mo1_fc

def make_fcsd(sscobj, nuc_pair=None):
    '''FC + SD contributions to 2nd order energy'''
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    mol = sscobj.mol
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    atm1dic, atm2dic = _uniq_atoms(nuc_pair)

    h1 = make_h1_fcsd(mol, mo_coeff, mo_occ, sorted(atm2dic.keys()))
    mo1aa, mo1ab, mo1ba, mo1bb = solve_mo1_fc(sscobj, h1)
    h1 = None
    h1aa, h1ab, h1ba, h1bb = make_h1_fcsd(mol, mo_coeff, mo_occ, sorted(atm1dic.keys()))
    nocca = numpy.count_nonzero(mo_occ[0]> 0)
    nvira = numpy.count_nonzero(mo_occ[0]==0)
    noccb = numpy.count_nonzero(mo_occ[1]> 0)
    nvirb = numpy.count_nonzero(mo_occ[1]==0)
    mo1aa = numpy.asarray(mo1aa).reshape(-1,3,3,nvira,nocca)
    mo1ab = numpy.asarray(mo1ab).reshape(-1,3,3,nvira,noccb)
    mo1ba = numpy.asarray(mo1ba).reshape(-1,3,3,nvirb,nocca)
    mo1bb = numpy.asarray(mo1bb).reshape(-1,3,3,nvirb,noccb)
    h1aa = numpy.asarray(h1aa).reshape(-1,3,3,nvira,nocca)
    h1ab = numpy.asarray(h1ab).reshape(-1,3,3,nvira,noccb)
    h1ba = numpy.asarray(h1ba).reshape(-1,3,3,nvirb,nocca)
    h1bb = numpy.asarray(h1bb).reshape(-1,3,3,nvirb,noccb)
    para = []
    for i,j in nuc_pair:
        at1 = atm1dic[i]
        at2 = atm2dic[j]
        # x contributions
        e = numpy.einsum('xij,yij->xy', h1ab[at1,0], mo1ab[at2,0])
        e+= numpy.einsum('xij,yij->xy', h1ba[at1,0], mo1ba[at2,0])
        # y contributions
        e+= numpy.einsum('xij,yij->xy', h1ab[at1,1], mo1ab[at2,1])
        e+= numpy.einsum('xij,yij->xy', h1ba[at1,1], mo1ba[at2,1])
        # z contribution
        e+= numpy.einsum('xij,yij->xy', h1aa[at1,2], mo1aa[at2,2])
        e+= numpy.einsum('xij,yij->xy', h1bb[at1,2], mo1bb[at2,2])
        para.append(e*2)  # *2 for +c.c.
    return numpy.asarray(para) * nist.ALPHA**4

def make_h1_fc(mol, mo_coeff, mo_occ, atmlst):
    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)
    moa = ao.dot(mo_coeff[0])
    mob = ao.dot(mo_coeff[1])
    orboa = moa[:,mo_occ[0]> 0]
    orbva = moa[:,mo_occ[0]==0]
    orbob = mob[:,mo_occ[1]> 0]
    orbvb = mob[:,mo_occ[1]==0]
    h1aa = []
    h1ab = []
    h1ba = []
    h1bb = []
    fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
    for ia in atmlst:
        h1aa.append(fac * numpy.einsum('p,i->pi', orbva[ia], orboa[ia]))
        h1ab.append(fac * numpy.einsum('p,i->pi', orbva[ia], orbob[ia]))
        h1ba.append(fac * numpy.einsum('p,i->pi', orbvb[ia], orboa[ia]))
        h1bb.append(fac * numpy.einsum('p,i->pi', orbvb[ia], orbob[ia]))
    return h1aa, h1ab, h1ba, h1bb

def make_h1_fcsd(mol, mo_coeff, mo_occ, atmlst):
    '''FC + SD'''
    orboa = mo_coeff[0][:,mo_occ[0]> 0]
    orbva = mo_coeff[0][:,mo_occ[0]==0]
    orbob = mo_coeff[1][:,mo_occ[1]> 0]
    orbvb = mo_coeff[1][:,mo_occ[1]==0]
    nao = mo_coeff[0].shape[0]
    h1aa = []
    h1ab = []
    h1ba = []
    h1bb = []
    for ia in atmlst:
        mol.set_rinv_origin(mol.atom_coord(ia))
        a01p = mol.intor('int1e_sa01sp', 12).reshape(3,4,nao,nao)
        h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
        # *.5 due to s = 1/2 * pauli-matrix
        for i in range(3):
            for j in range(3):
                h1aa.append(orbva.T.conj().dot(h1ao[i,j]).dot(orboa) * .5)
                h1ab.append(orbva.T.conj().dot(h1ao[i,j]).dot(orbob) * .5)
                h1ba.append(orbvb.T.conj().dot(h1ao[i,j]).dot(orboa) * .5)
                h1bb.append(orbvb.T.conj().dot(h1ao[i,j]).dot(orbob) * .5)
    return h1aa, h1ab, h1ba, h1bb


class SpinSpinCoupling(uhf_nmr.NMR):
    def __init__(self, scf_method):
        mol = scf_method.mol
        self.nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        self.with_fc = True
        self.with_fcsd = False
        uhf_nmr.NMR.__init__(self, scf_method)

    def dump_flags(self):
        uhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'nuc_pair %s', self.nuc_pair)
        logger.info(self, 'with Fermi-contact  %s', self.with_fc)
        logger.info(self, 'with Fermi-contact + spin-dipole  %s', self.with_fcsd)
        return self

    def kernel(self, mo1=None):
        if len(self.nuc_pair) == 0:
            return

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

        if self.verbose > logger.QUIET:
            nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
            au2Hz = nist.HARTREE2J / nist.PLANCK
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
#                    _write(self.stdout, ssc_dia [k], 'dia-magnetism')
#                    _write(self.stdout, ssc_para[k], 'para-magnetism')

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
    make_pso = make_pso
    make_fc = make_fc
    make_fcsd = make_fcsd

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             nuc_pair=None):
        ssc_para = self.make_pso(mol, mo1, mo_coeff, mo_occ)
        if self.with_fcsd:
            ssc_para += self.make_fcsd(mol, mo1, mo_coeff, mo_occ)
        elif self.with_fc:
            ssc_para += self.make_fc(mol, mo1, mo_coeff, mo_occ)
        return ssc_para

    def solve_mo1(self, mo_energy=None, mo_occ=None, h1=None, with_cphf=None):
        cput1 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_occ    is None: mo_occ = self._scf.mo_occ
        if with_cphf is None: with_cphf = self.cphf

        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        if h1 is None:
            atmlst = sorted(set([j for i,j in self.nuc_pair]))
            h1a, h1b = make_h1_pso(mol, self._scf.mo_coeff, mo_occ, atmlst)
        else:
            h1a, h1b = h1
        h1a = numpy.asarray(h1a)
        h1b = numpy.asarray(h1b)

        if with_cphf:
            vind = self.gen_vind(self._scf, mo_coeff, mo_occ)
            mo1, mo_e1 = ucphf.solve(vind, mo_energy, mo_occ, (h1a,h1b), None,
                                     self.max_cycle_cphf, self.conv_tol,
                                     verbose=log)
        else:
            eai_aa = lib.direct_sum('i-a->ai', mo_energy[0][mo_occ[0]>0], mo_energy[0][mo_occ[0]==0])
            eai_bb = lib.direct_sum('i-a->ai', mo_energy[1][mo_occ[1]>0], mo_energy[1][mo_occ[1]==0])
            mo1 = (h1a * (1/eai_aa), h1b * (1/eai_bb))
            mo_e1 = None

        logger.timer(self, 'solving mo1 eqn', *cput1)
        return mo1, mo_e1

    def gen_vind(self, mf, mo_coeff, mo_occ):
        '''Induced potential associated with h1_PSO'''
        vresp = _gen_uhf_response(mf, with_j=False, hermi=0)
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        orboa = mo_coeff[0][:, occidxa]
        orbva = mo_coeff[0][:,~occidxa]
        orbob = mo_coeff[1][:, occidxb]
        orbvb = mo_coeff[1][:,~occidxb]
        nocca = orboa.shape[1]
        noccb = orbob.shape[1]
        nvira = orbva.shape[1]
        nvirb = orbvb.shape[1]
        nova = nocca * nvira
        novb = noccb * nvirb
        mo_va_oa = numpy.asarray(numpy.hstack((orbva,orboa)), order='F')
        mo_vb_ob = numpy.asarray(numpy.hstack((orbvb,orbob)), order='F')
        def vind(mo1):
            mo1a = mo1.reshape(-1,nova+novb)[:,:nova].reshape(-1,nvira,nocca)
            mo1b = mo1.reshape(-1,nova+novb)[:,nova:].reshape(-1,nvirb,noccb)
            nset = mo1a.shape[0]
            dm1a = _dm1_mo2ao(mo1a, orbva, orboa)
            dm1b = _dm1_mo2ao(mo1b, orbvb, orbob)
            dm1 = numpy.asarray([dm1a-dm1a.transpose(0,2,1),
                                 dm1b-dm1b.transpose(0,2,1)])
            v1 = vresp(dm1)
            v1a = _ao2mo.nr_e2(v1[0], mo_va_oa, (0,nvira,nvira,nvira+nocca))
            v1b = _ao2mo.nr_e2(v1[1], mo_vb_ob, (0,nvirb,nvirb,nvirb+noccb))
            v1mo = numpy.hstack((v1a.reshape(nset,-1), v1b.reshape(nset,-1)))
            return v1mo.ravel()
        return vind

SSC = SpinSpinCoupling


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

    mf = scf.UHF(mol).run()
    ssc = SSC(mf)
    ssc.verbose = 4
    ssc.cphf = True
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()
    print(jj)
    print(lib.finger(jj)*1e8 - 0.12374695912503765)
