#!/usr/bin/env python
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
from pyscf.gto import mole
from pyscf.dft import numint
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.hfc import parameters
from pyscf.prop.gtensor.uhf import koseki_charge

# Due to the value of lib.param.NUC_MAGNETON, SI unit is used in this module

def dia(gobj, mol, dm0, hfc_nuc=None, verbose=None):
    log = logger.new_logger(gobj, verbose)
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5

    mu_B = 1  # lib.param.BOHR_MAGNETON
    mu_N = lib.param.PROTON_MASS * mu_B
    fac = lib.param.ALPHA / 2 / effspin
    fac*= lib.param.G_ELECTRON * mu_B * mu_N

    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)

    nao = dma.shape[0]
    dia = []
    for i, atm_id in enumerate(hfc_nuc):
        Z = mole._charge(mol.atom_symbol(atm_id))
        nuc_spin, g_nuc = parameters.ISOTOPE[Z][1:3]
# g factor of other isotopes can be found in file nuclear_g_factor.dat
        gyromag = 1e-6/(2*numpy.pi) * parameters.g_factor_to_gyromagnetic_ratio(g_nuc)
        log.info('Atom %d %s  nuc-spin %g  nuc-g-factor %g  gyromagnetic ratio %g (in MHz)',
                 atm_id, mol.atom_symbol(atm_id), nuc_spin, g_nuc, gyromag)
        mol.set_rinv_origin(mol.atom_coord(atm_id))
# a01p[mu,sigma] the imaginary part of integral <vec{r}/r^3 cross p>
# mu = gN * I * mu_N
        a01p = mol.intor('int1e_sa01sp', 12).reshape(3,4,nao,nao)
        h11 = a01p[:,1:] - a01p[:,1:].transpose(0,1,3,2)
        e11 = numpy.einsum('xyij,ji->xy', h11, spindm)
        e11 *= fac * gyromag
# e11 includes fermi-contact and spin-dipolar contriutions and a rank-2 contact
# term.  We ignore the contribution of rank-2 contact term, view it as part of
# SD contribution.  See also TCA, 73, 173
        fermi_contact = (4*numpy.pi/3 * fac * gyromag *
                         numpy.einsum('i,j,ji', ao[atm_id], ao[atm_id], spindm))
        dip = e11 - numpy.eye(3) * fermi_contact
        log.info('FC %s', fermi_contact)
        if gobj.verbose >= logger.INFO:
            _write(gobj, dip, 'SD')
        dia.append(e11)
    return numpy.asarray(dia)

# Note mo1 is the imaginary part of MO^1
def para(mol, mo1, mo_coeff, mo_occ, hfc_nuc=None):
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)

    effspin = mol.spin * .5
    mu_B = 1  # lib.param.BOHR_MAGNETON
    mu_N = lib.param.PROTON_MASS * mu_B
    fac = lib.param.ALPHA / 2 / effspin
    fac*= lib.param.G_ELECTRON * mu_B * mu_N

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    nao = mo_coeff[0].shape[0]
    dm10 = numpy.empty((3,nao,nao))
    for i in range(3):
        dm10[i] = reduce(numpy.dot, (mo_coeff[0], mo1[0][i], orboa.conj().T))
        dm10[i]+= reduce(numpy.dot, (mo_coeff[1], mo1[1][i], orbob.conj().T))
    para = numpy.empty((len(hfc_nuc),3,3))
    for n, atm_id in enumerate(hfc_nuc):
        Z = mole._charge(mol.atom_symbol(atm_id))
        nuc_spin, g_nuc = parameters.ISOTOPE[Z][1:3]
        gyromag = 1e-6/(2*numpy.pi) * parameters.g_factor_to_gyromagnetic_ratio(g_nuc)

        mol.set_rinv_origin(mol.atom_coord(atm_id))
        h01 = mol.intor_asymmetric('int1e_prinvxp', 3)
        para[n] = numpy.einsum('xji,yij->yx', dm10, h01) * 2
        para[n] *= fac * gyromag
    return para

def make_h10(mol, dm0):
# hso1e is the imaginary part of [i sigma dot pV x p]
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    if gobj.with_so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            #FIXME: when ECP is enabled
            Z = koseki_charge(mol.atom_charge(ia))
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor('int1e_pnucxp', 3)
    hso1e = numpy.asarray((hso1e,hso1e))

    if gobj.with_sso or gobj.with_soo:
        hso2e = make_soc2e(gobj, dm0)
    else:
        hso2e = 0

    hso = hso1e + hso2e
    hso *= lib.param.ALPHA**2/4
    return hso

def make_soc2e(gobj, dm0):
    dma, dmb = dm0
    nao = dma.shape[0]
    # FIXME: see JPC, 101, 3388 Eq (11c), why?
    g_so = (lib.param.G_ELECTRON - 1) * 2

# hso2e is the imaginary part of SSO
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3,nao,nao,nao,nao)
    vj = numpy.zeros((2,3,nao,nao))
    vk = numpy.zeros((2,3,nao,nao))
    if gobj.with_sso:
        vj[:] += g_so/2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma-dmb)
        vj[0] += g_so/2 * numpy.einsum('yijkl,lk->yij', hso2e, dma+dmb)
        vj[1] -= g_so/2 * numpy.einsum('yijkl,lk->yij', hso2e, dma+dmb)
        vk[0] += g_so/2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma)
        vk[1] -= g_so/2 * numpy.einsum('yijkl,ji->ykl', hso2e, dmb)
        vk[0] += g_so/2 * numpy.einsum('yijkl,lk->yij', hso2e, dma)
        vk[0] -= g_so/2 * numpy.einsum('yijkl,lk->yij', hso2e, dmb)
    if gobj.with_soo:
        vj[0] += 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma+dmb)
        vj[1] -= 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma+dmb)
        vj[:] += 2 * numpy.einsum('yijkl,lk->yij', hso2e, dma-dmb)
        vk[0] += 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dma)
        vk[1] -= 2 * numpy.einsum('yijkl,ji->ykl', hso2e, dmb)
        vk[0] += 2 * numpy.einsum('yijkl,lk->yij', hso2e, dma)
        vk[1] -= 2 * numpy.einsum('yijkl,lk->yij', hso2e, dmb)
    hso2e = vj - vk
    return hso2e

def _write(rec, msc3x3, title):
    rec.stdout.write('%s\n' % title)
    rec.stdout.write('I_x %s\n' % str(msc3x3[0]))
    rec.stdout.write('I_y %s\n' % str(msc3x3[1]))
    rec.stdout.write('I_z %s\n' % str(msc3x3[2]))


class HyperfineCoupling(uhf_nmr.NMR):
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

        hfc_dia = self.dia()
        hfc_para = self.para(mo10=mo1)
        hfc_tensor = hfc_para + hfc_dia

        logger.timer(self, 'HFC tensor', *cput0)
        if self.verbose > logger.QUIET:
            for i, atm_id in enumerate(self.hfc_nuc):
                _write(gobj, hfc_tensor[i],
                       '\nHyperfine coupling tensor of atom %d %s'
                       % (atm_id, self.mol.atom_symbol(atm_id)))
                if self.verbose >= logger.INFO:
                    _write(gobj, hfc_dia[n], 'HFC diamagnetic terms')
                    _write(gobj, hfc_para[n], 'HFC paramagnetic terms')
        self.stdout.flush()
        return hfc_tensor

    def dia(self, mol=None, dm0=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return dia(self, mol, dm0)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(mol, mo10, mo_coeff, mo_occ)

    def make_h10(self, mol=None, dm0=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return make_h10(mol, dm0)

    def make_s10(self, mol=None):
        nao = mol.nao_nr()
        return numpy.zeros((3,nao,nao))

HFC = HyperfineCoupling


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    gobj = HFC(mf)
    gobj.with_sso = True
    gobj.with_soo = True
    gobj.with_so_eff_charge = False
    print(gobj.kernel())

    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    gobj = HFC(mf)
    print(gobj.kernel())

    mol = gto.M(atom='''
                H 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    print(HFC(mf).kernel())

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    print(HFC(mf).kernel())

