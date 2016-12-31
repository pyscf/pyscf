#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
import scipy.special
import scipy.optimize
from pyscf.pbc import gto as pbcgto
from pyscf.lib import logger


def project_mo_nr2nr(cell1, mo1, cell2, kpt=None):
    r''' Project orbital coefficients

    .. math::

        |\psi1> = |AO1> C1

        |\psi2> = P |\psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2

        C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = cell2.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpt)
    s21 = pbcgto.intor_cross('cint1e_ovlp_sph', cell2, cell1, kpts=kpt)
    mo2 = numpy.dot(s21, mo1)
    return scipy.linalg.cho_solve(scipy.linalg.cho_factor(s22), mo2)


def smearing_(mf, sigma=None, method='fermi'):
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.pbc.scf import khf
    from pyscf.pbc.scf import kuhf
    mf_class = mf.__class__

    is_uhf = isinstance(mf, kuhf.KUHF)
    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if mf.sigma is None or mf.sigma == 0:
            return mo_occ_kpts

        if is_uhf:
            nkpts = len(mo_energy_kpts[0])
            nocc = mf.cell.nelectron * nkpts
        else:
            nkpts = len(mo_energy_kpts)
            nocc = (mf.cell.nelectron * nkpts) // 2
        mo_energy = numpy.sort(mo_energy_kpts.ravel())
        fermi = mo_energy[nocc-1]

        if method.lower() == 'fermi':  # Fermi-Dirac smearing
            # Optimize mu to give correct electron number
            def nelec_cost_fn(m):
                mo_occ_kpts = 2./(numpy.exp((mo_energy_kpts-m)/mf.sigma)+1.)
                if is_uhf:
                    mo_occ_kpts *= .5
                return ( mo_occ_kpts.sum()/nkpts - mf.cell.nelectron )**2
            res = scipy.optimize.minimize(nelec_cost_fn, fermi, method='Powell')
            mu = res.x
            mo_occ_kpts = f = 1./(numpy.exp((mo_energy_kpts-mu)/mf.sigma)+1.)
            f = f[(f>0) & (f<1)]
            mf.entropy = -(f*numpy.log(f) + (1-f)*numpy.log(1-f)).sum()
            if not is_uhf:
                mo_occ_kpts *= 2
                mf.entropy *= 2

        else:  # Gaussian smearing
            def nelec_cost_fn(m):
                mo_occ_kpts = 1 - scipy.special.erf((mo_energy_kpts-m)/mf.sigma)
                if is_uhf:
                    mo_occ_kpts *= .5
                return ( mo_occ_kpts.sum()/nkpts - mf.cell.nelectron )**2
            res = scipy.optimize.minimize(nelec_cost_fn, fermi, method='Powell')
            mu = res.x
            mo_occ_kpts = 1 - scipy.special.erf((mo_energy_kpts-mu)/mf.sigma)
            if is_uhf:
                mo_occ_kpts *= .5
# Is the entropy correct for spin unrestricted case?
            mf.entropy = numpy.exp(-((mo_energy_kpts-mu)/mf.sigma)**2).sum() / numpy.sqrt(numpy.pi)

        logger.debug(mf, '    Sum mo_occ_kpts = %s  should equal nelec = %s',
                     mo_occ_kpts.sum()/nkpts, mf.cell.nelectron)
        logger.info(mf, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                    mf.sigma, mu, mf.entropy)

        return mo_occ_kpts

    def get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
        nkpts = len(mf.kpts)
        grad_kpts = []
        for k in range(nkpts):
            f_mo = reduce(numpy.dot, (mo_coeff_kpts[k].T.conj(), fock[k],
                                      mo_coeff_kpts[k]))
            nmo = f_mo.shape[0]
            grad_kpts.append(f_mo[numpy.tril_indices(nmo, -1)])
        return numpy.hstack(grad_kpts)

    if isinstance(mf, kuhf.KUHF):
        def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
            if fock is None:
                dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
                fock = mf.get_hcore(mf.cell, mf.kpts) + mf.get_veff(mf.cell, dm1)
            ga = get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return numpy.hstack((ga,gb))

    elif isinstance(mf, khf.KRHF):
        def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
            if fock is None:
                dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
                fock = mf.get_hcore(mf.cell, mf.kpts) + mf.get_veff(mf.cell, dm1)
            return get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

    else:
        raise RuntimeError("Input mf %s is not k-point SCF object" % mf)

    def energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        e_tot = mf.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)[0] + mf.energy_nuc()
        if mf.entropy is not None and mf.verbose >= logger.INFO:
            e_free = e_tot - mf.sigma * mf.entropy
            e_zero = e_tot - mf.sigma * mf.entropy * .5
            logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, e_free, e_zero)
        return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf._keys.union(['sigma', 'smearing_method', 'entropy'])

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf

if __name__ == '__main__':
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = 'ccpvdz'
    cell.a = numpy.eye(3) * 4
    cell.gs = [8] * 3
    cell.verbose = 4
    cell.build()
    nks = [2,1,1]
    mf = pscf.KUHF(cell, cell.make_kpts(nks))
    mf = smearing_(mf, .1)
    mf.kernel()

