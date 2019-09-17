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
Non-relativistic unrestricted Hartree-Fock hyperfine coupling tensor

Refs:
    JCP, 120, 2127
    JCP, 118, 3939
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.ssc import uhf as uhf_ssc
from pyscf.prop.ssc.rhf import _dm1_mo2ao
from pyscf.prop.zfs.uhf import koseki_charge
from pyscf.prop.gtensor.uhf import align, get_jk
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

def make_fcdip(hfcobj, dm0, hfc_nuc=None, verbose=None):
    '''The contribution of Fermi-contact term and dipole-dipole interactions'''
    log = logger.new_logger(hfcobj, verbose)
    mol = hfcobj.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5

    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        h1 = _get_integrals_fcdip(mol, atm_id)
        fcsd = numpy.einsum('xyij,ji->xy', h1, spindm)

        h1fc = _get_integrals_fc(mol, atm_id)
        fc = numpy.einsum('ij,ji', h1fc, spindm)

        sd = fcsd + numpy.eye(3) * fc

        log.info('FC of atom %d  %s (in MHz)', atm_id, fac * nuc_gyro * fc)
        if hfcobj.verbose >= logger.INFO:
            _write(hfcobj, align(fac*nuc_gyro*sd)[0], 'SD of atom %d (in MHz)' % atm_id)
        hfc.append(fac * nuc_gyro * fcsd)
    return numpy.asarray(hfc)

def _get_integrals_fcdip(mol, atm_id):
    '''AO integrals for FC + Dipole-dipole'''
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Note the fermi-contact part is different to the fermi-contact
        # operator in SSC.  FC here is associated to the the integrals of
        # (\nabla \nabla 1/r), which includes the contribution of Poisson
        # equation, 4\pi rho.  Factor 4.\pi/3 is used in the Fermi contact
        # contribution.  In SSC, the factor of FC part is -8\pi/3.
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)
        trace = h1ao[0,0] + h1ao[1,1] + h1ao[2,2]
        idx = numpy.arange(3)
        h1ao[idx,idx] -= trace
    return h1ao

def _get_integrals_fc(mol, atm_id):
    '''AO integrals for Fermi contact term'''
    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    return 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)

# Note mo1 is the imaginary part of MO^1
def make_pso_soc(hfcobj, hfc_nuc=None):
    '''Spin-orbit coupling correction'''
    mol = hfcobj.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)

    mf = hfcobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    effspin = mol.spin * .5
    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**4 / 4 / effspin * e_gyro * au2MHz

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:, occidxa]
    orbva = mo_coeff[0][:,~occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbvb = mo_coeff[1][:,~occidxb]
    # Note sigma_z is considered in h1_soc integral.
    # mo1b has the associated sign (-)
    mo1a, mo1b = hfcobj.solve_mo1()[0]
    dm1a = _dm1_mo2ao(mo1a, orbva, orboa)
    dm1b = _dm1_mo2ao(mo1b, orbvb, orbob)
    dm1 = dm1a + dm1b
    dm1 = dm1 - dm1.transpose(0,2,1)

    para = []
    for n, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        # Imaginary part of H01 operator
        # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        h1ao = -mol.intor_asymmetric('int1e_prinvxp', 3)
        de = numpy.einsum('xij,yij->xy', h1ao, dm1)
        de *= fac * nuc_gyro
        if hfcobj.verbose >= logger.INFO:
            _write(hfcobj, align(de)[0], 'PSO of atom %d (in MHz)' % atm_id)
        para.append(de)
    return numpy.asarray(para)

def solve_mo1_soc(hfcobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                  h1=None, s1=None, with_cphf=None):
    if h1 is None:
        if mo_occ is None:
            mo_occ = hfcobj._scf.mo_occ
        mo_coeff = hfcobj._scf.mo_coeff
        h1a, h1b = make_h1_soc(hfcobj, hfcobj._scf.make_rdm1())
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        orboa = mo_coeff[0][:, occidxa]
        orbva = mo_coeff[0][:,~occidxa]
        orbob = mo_coeff[1][:, occidxb]
        orbvb = mo_coeff[1][:,~occidxb]
        h1a = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orboa)) for x in h1a])
        h1b = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orbob)) for x in h1b])
        h1 = (h1a, h1b)
    mo1, mo_e1 = uhf_ssc.solve_mo1(hfcobj, mo_energy, mo_coeff, mo_occ,
                                   h1, s1, with_cphf)
    return mo1, mo_e1

def make_h1_soc(hfcobj, dm0):
    '''1-electron and 2-electron spin-orbit coupling integrals.

    1-electron SOC integral is the imaginary part of [i sigma dot pV x p],
    ie [sigma dot pV x p].

    Note sigma_z is considered in the SOC integrals (the (-) sign for beta-beta
    block is included in the integral).  The factor 1/2 in the spin operator
    s=sigma/2 is not included.
    '''
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    mol = hfcobj.mol
    assert(not mol.has_ecp())
    if hfcobj.so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            Z = koseki_charge(mol.atom_charge(ia))
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor('int1e_pnucxp', 3)
    hso = [hso1e, -hso1e]

    if hfcobj.para_soc2e:
        hso2e = hfcobj.make_h1_soc2e(dm0)
        hso[0] += hso2e[0]
        hso[1] += hso2e[1]

    return hso

# Note the (-) sign of beta-beta block is included in the integral
def make_h1_soc2e(hfcobj, dm0):
    mol = hfcobj.mol
    vj, vk = get_jk(mol, dm0)
    vjaa = 0
    vjbb = 0
    vkaa = 0
    vkbb = 0
    if 'SSO' in hfcobj.para_soc2e.upper():
        vj1 = vj[0] + vj[1]
        vjaa += vj1
        vjbb -= vj1
        vkaa += vk[0]
        vkbb -= vk[1]
    if 'SOO' in hfcobj.para_soc2e.upper():
        vj1 = vj[0] - vj[1]
        vjaa += vj1 * 2
        vjbb += vj1 * 2
        vkaa += vk[0] * 2
        vkbb -= vk[1] * 2
    hso2e = (vjaa-vkaa, vjbb-vkbb)
    return hso2e

def _write(hfcobj, tensor, title):
    hfcobj.stdout.write('%s %s\n' % (title, tensor.diagonal()))
    if hfcobj.verbose >= logger.DEBUG:
        hfcobj.stdout.write('%s\n' % title)
        hfcobj.stdout.write('I_x %s\n' % str(tensor[0]))
        hfcobj.stdout.write('I_y %s\n' % str(tensor[1]))
        hfcobj.stdout.write('I_z %s\n' % str(tensor[2]))
        hfcobj.stdout.flush()


class HyperfineCoupling(lib.StreamObject):
    '''dE = I dot gtensor dot s'''
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        # para_soc2e can be 'SSO', 'SOO', 'SSO+SOO', None/False, True (='SSO+SOO')
        # 'SOMF', 'AMFI' (='AMFI+SSO+SOO'), 'SOMF+AMFI', 'AMFI+SSO',
        # 'AMFI+SOO', 'AMFI+SSO+SOO'
        self.para_soc2e = 'SSO+SOO'
        self.so_eff_charge = False  # Koseki effective SOC charge
        self.hfc_nuc = range(scf_method.mol.natm)

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
        log.info('HFC for atoms %s', str(self.hfc_nuc))
        if self.cphf:
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        log.info('para_soc2e = %s', self.para_soc2e)
        log.info('so_eff_charge = %s (1e SO effective charge)',
                 self.so_eff_charge)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        mol = self.mol

        dm0 = self._scf.make_rdm1()
        hfc_tensor = self.make_fcdip(dm0, self.hfc_nuc)
        hfc_tensor += self.make_pso_soc(self.hfc_nuc)

        logger.timer(self, 'HFC tensor', *cput0)
        if self.verbose > logger.QUIET:
            for i, atm_id in enumerate(self.hfc_nuc):
                _write(self, align(hfc_tensor[i])[0],
                       '\nHyperfine coupling tensor of atom %d %s (in MHz)'
                       % (atm_id, mol.atom_symbol(atm_id)))
        return hfc_tensor

    make_fcdip = make_fcdip
    make_pso_soc = make_pso_soc
    solve_mo1 = solve_mo1_soc
    make_h1_soc2e = make_h1_soc2e

HFC = HyperfineCoupling


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol).run()
    hfc = HFC(mf)
    #hfc.verbose = 4
    hfc.para_soc2e = 'SSO+SOO'
    hfc.so_eff_charge = False
    print(lib.finger(hfc.kernel()) - -2050.5830130636241)

    mol = gto.M(atom='C 0 0 0; O 0 0 1.12',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol).run()
    hfc = HFC(mf)
    #hfc.verbose = 4
    hfc.para_soc2e = 'SSO+SOO'
    hfc.so_eff_charge = False
    print(lib.finger(hfc.kernel()) - 323.96293610190781)

    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol).run()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()) - -23.376864877326888)

    mol = gto.M(atom='''
                Li 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UHF(mol).run()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()) - 55.653597133253925)

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol).run()
    hfc = HFC(mf)
    print(lib.finger(hfc.kernel()) - 220.64088515064489)

    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(1)
    dm0 = numpy.random.random((2,nao,nao))
    dm0 = dm0 + dm0.transpose(0,2,1)
    hfc.para_soc2e = 'SSO+SOO'
    hfc.so_eff_charge = False
    h1a, h1b = make_h1_soc(hfc, dm0)
    print(lib.finger(h1a) - -13.574938902220254)
    print(lib.finger(h1b) - 30.266331488847413)

