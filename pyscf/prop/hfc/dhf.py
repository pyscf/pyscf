#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Unrestricted Dirac Hartree-Fock hyperfine coupling tensor
(In testing)

Refs: JCP, 134, 044111
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.prop.ssc import dhf as dhf_ssc
from pyscf.prop.ssc.parameters import get_nuc_g_factor

# TODO: 3 SCF for sx, sy, sz

def make_h01(mol, atm_id):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    t1 = mol.intor('int1e_sa01sp_spinor', 3)
    n2c = t1.shape[2]
    n4c = n2c * 2
    h1 = numpy.zeros((3, n4c, n4c), complex)
    for i in range(3):
        h1[i,:n2c,n2c:] += .5 * t1[i]
        h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
    return h1

def kernel(hfcobj, with_gaunt=False, verbose=None):
    log = lib.logger.new_logger(hfcobj, verbose)
    mf = hfcobj._scf
    mol = mf.mol
# Add finite field to remove degeneracy
    nuc_spin = numpy.ones(3) * 1e-6
    sc = numpy.dot(mf.get_ovlp(), mf.mo_coeff)
    h0 = reduce(numpy.dot, (sc*mf.mo_energy, sc.conj().T))
    c = lib.param.LIGHT_SPEED
    n4c = h0.shape[0]
    n2c = n4c // 2
    Sigma = numpy.zeros((3,n4c,n4c), dtype=h0.dtype)
    Sigma[:,:n2c,:n2c] = mol.intor('int1e_sigma_spinor', comp=3)
    Sigma[:,n2c:,n2c:] = .25/c**2 * mol.intor('int1e_spsigmasp_spinor', comp=3)

    hfc = []
    for atm_id in range(mol.natm):
        symb = mol.atom_symbol(atm_id)
        nuc_mag = .5 * (lib.param.E_MASS/lib.param.PROTON_MASS)  # e*hbar/2m
        nuc_gyro = get_nuc_g_factor(symb) * nuc_mag
        e_gyro = .5 * lib.param.G_ELECTRON
        au2MHz = lib.param.HARTREE2J / lib.param.PLANCK * 1e-6
        fac = lib.param.ALPHA**2 * nuc_gyro * e_gyro * au2MHz
        #logger.debug('factor (MHz) %s', fac)

        h01 = make_h01(mol, 0)
        mo_occ = mf.mo_occ
        mo_coeff = mf.mo_coeff
        if 0:
            h01b = h0 + numpy.einsum('xij,x->ij', h01, nuc_spin)
            h01b = reduce(numpy.dot, (mf.mo_coeff.conj().T, h01b, mf.mo_coeff))
            mo_energy, v = numpy.linalg.eigh(h01b)
            mo_coeff = numpy.dot(mf.mo_coeff, v)
            mo_occ = mf.get_occ(mo_energy, mo_coeff)

        occidx = mo_occ > 0
        orbo = mo_coeff[:,occidx]
        dm0 = numpy.dot(orbo, orbo.T.conj())
        e01 = numpy.einsum('xij,ji->x', h01, dm0) * fac

        effspin = numpy.einsum('xij,ji->x', Sigma, dm0) * .5
        log.debug('atom %d Eff-spin %s', atm_id, effspin.real)

        e01 = (e01 / effspin).real
        hfc.append(e01)
    return numpy.asarray(hfc)

class HyperfineCoupling(dhf_ssc.SSC):
    kernel = kernel

HFC = HyperfineCoupling

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.M(
        atom = [['Li', (0.,0.,0.)],
                #['He', (.4,.7,0.)],
               ],
        basis = 'ccpvdz', spin=1)
    mf = scf.DHF(mol).run()
    hfc = HFC(mf)
    print(hfc.kernel())
