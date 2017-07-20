#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic unrestricted Hartree-Fock hyperfine coupling tensor
(In testing)

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
from pyscf.prop.ssc.parameters import get_nuc_g_factor
from pyscf.prop.ssc.rhf import _dm1_mo2ao
from pyscf.prop.gtensor.uhf import koseki_charge

def make_fcsd(hfcobj, dm0, hfc_nuc=None, verbose=None):
    log = logger.new_logger(hfcobj, verbose)
    mol = hfcobj.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5

    e_gyro = .5 * lib.param.G_ELECTRON
    nuc_mag = .5 * (lib.param.E_MASS/lib.param.PROTON_MASS)  # e*hbar/2m
    au2MHz = lib.param.HARTREE2J / lib.param.PLANCK * 1e-6
    fac = lib.param.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)

    nao = dma.shape[0]
    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        mol.set_rinv_origin(mol.atom_coord(atm_id))
# a01p[mu,sigma] the imaginary part of integral <vec{r}/r^3 cross p>
        a01p = mol.intor('int1e_sa01sp', 12).reshape(3,4,nao,nao)
        h1 = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
        fcsd = numpy.einsum('xyij,ji->xy', h1, spindm)
        fc = 8*numpy.pi/3 * numpy.einsum('i,j,ji', ao[atm_id], ao[atm_id], spindm)
        sd = fcsd - numpy.eye(3) * fc

        log.info('FC %s', fac * nuc_gyro * fc)
        if hfcobj.verbose >= logger.INFO:
            _write(hfcobj, fac * nuc_gyro * sd, 'SD')
        hfc.append(fac * nuc_gyro * fcsd)
    return numpy.asarray(hfc)

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
    e_gyro = .5 * lib.param.G_ELECTRON
    nuc_mag = .5 * (lib.param.E_MASS/lib.param.PROTON_MASS)  # e*hbar/2m
    au2MHz = lib.param.HARTREE2J / lib.param.PLANCK * 1e-6
    fac = lib.param.ALPHA**4 / 4 / effspin * e_gyro * au2MHz

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
        e = numpy.einsum('xij,yij->xy', h1ao, dm1)
        para.append(fac * nuc_gyro * e)
    return numpy.asarray(para)

def solve_mo1_soc(hfcobj, mo_energy=None, mo_occ=None, h1=None, with_cphf=None):
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
    mo1, mo_e1 = uhf_ssc.SSC.solve_mo1(hfcobj, mo_energy, mo_occ, h1, with_cphf)
    return mo1, mo_e1

def make_h1_soc(gobj, dm0):
    '''1-electron and 2-electron spin-orbit coupling integrals.

    1-electron SOC integral is the imaginary part of [i sigma dot pV x p],
    ie [sigma dot pV x p].

    Note sigma_z is considered in the SOC integrals (the (-) sign for beta-beta
    block is included in the integral).  The factor 1/2 in the spin operator
    s=sigma/2 is not included.
    '''
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    mol = gobj.mol
    if gobj.with_so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            #FIXME: when ECP is enabled
            Z = koseki_charge(mol.atom_charge(ia))
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor('int1e_pnucxp', 3)
    hso = numpy.asarray((hso1e,-hso1e))

# TODO: test SOMF and the treatments in JCP, 122, 034107
    if gobj.with_sso or gobj.with_soo:
        hso2e = make_h1_soc2e(gobj, dm0)
        hso += hso2e

    return hso

# Note the (-) sign of beta-beta block is included in the integral
def make_h1_soc2e(gobj, dm0):
    dma, dmb = dm0
    nao = dma.shape[0]
# hso2e is the imaginary part of SSO
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3,nao,nao,nao,nao)
    vj = numpy.zeros((2,3,nao,nao))
    vk = numpy.zeros((2,3,nao,nao))
    if gobj.with_sso:
        vj[:] += numpy.einsum('yijkl,ji->ykl', hso2e, dma-dmb)
        vj[0] += numpy.einsum('yijkl,lk->yij', hso2e, dma+dmb)
        vj[1] -= numpy.einsum('yijkl,lk->yij', hso2e, dma+dmb)
        vk[0] += numpy.einsum('yijkl,jk->yil', hso2e, dma)
        vk[1] -= numpy.einsum('yijkl,jk->yil', hso2e, dmb)
        vk[0] += numpy.einsum('yijkl,li->ykj', hso2e, dma)
        vk[1] -= numpy.einsum('yijkl,li->ykj', hso2e, dmb)
    if gobj.with_soo:
        vj[0] += 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma+dmb)
        vj[1] -= 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma+dmb)
        vj[:] += 2 * numpy.einsum('yijkl,lk->yij', hso2e, dma-dmb)
        vk[0] += 2 * numpy.einsum('yijkl,jk->yil', hso2e, dma)
        vk[1] -= 2 * numpy.einsum('yijkl,jk->yil', hso2e, dmb)
        vk[0] += 2 * numpy.einsum('yijkl,li->ykj', hso2e, dma)
        vk[1] -= 2 * numpy.einsum('yijkl,li->ykj', hso2e, dmb)
    hso2e = vj - vk
    return hso2e

def _write(rec, msc3x3, title):
    rec.stdout.write('%s\n' % title)
    rec.stdout.write('I_x %s\n' % str(msc3x3[0]))
    rec.stdout.write('I_y %s\n' % str(msc3x3[1]))
    rec.stdout.write('I_z %s\n' % str(msc3x3[2]))
    rec.stdout.flush()


class HyperfineCoupling(uhf_ssc.SSC):
    '''dE = I dot gtensor dot s'''
    def __init__(self, scf_method):
        self.with_sso = False  # Two-electron spin-same-orbit coupling
        self.with_soo = False  # Two-electron spin-other-orbit coupling
        self.with_so_eff_charge = True
        self.hfc_nuc = range(scf_method.mol.natm)
        uhf_nmr.NMR.__init__(self, scf_method)

    def dump_flags(self):
        uhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'with_sso = %s (2e spin-same-orbit coupling)', self.with_sso)
        logger.info(self, 'with_soo = %s (2e spin-other-orbit coupling)', self.with_soo)
        logger.info(self, 'with_so_eff_charge = %s (1e SO effective charge)',
                    self.with_so_eff_charge)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        mol = self.mol

        dm0 = mf.make_rdm1()
        hfc_tensor = self.make_fcsd(dm0, self.hfc_nuc)
        hfc_tensor += self.make_pso_soc(self.hfc_nuc)

        logger.timer(self, 'HFC tensor', *cput0)
        if self.verbose > logger.QUIET:
            for i, atm_id in enumerate(self.hfc_nuc):
                _write(self, hfc_tensor[i],
                       '\nHyperfine coupling tensor of atom %d %s'
                       % (atm_id, mol.atom_symbol(atm_id)))
        return hfc_tensor

    make_fcsd = make_fcsd
    make_pso_soc = make_pso_soc
    solve_mo1 = solve_mo1_soc

HFC = HyperfineCoupling


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    hfc = HFC(mf)
    hfc.with_sso = True
    hfc.with_soo = True
    hfc.with_so_eff_charge = False
    print(lib.finger(hfc.kernel()))

    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()))

    mol = gto.M(atom='''
                Li 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()))

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    hfc = HFC(mf)
    print(lib.finger(hfc.kernel()))

