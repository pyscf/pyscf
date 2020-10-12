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
Non-relativistic magnetizability tensor for RHF
(In testing)

Refs:
[1] R. Cammi, J. Chem. Phys., 109, 3185 (1998)
[2] Todd A. Keith, Chem. Phys., 213, 123 (1996)
[3] S. Sauer et al., Mol. Phys., 76, 445 (1991)
'''


import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import jk
from pyscf.scf import _response_functions  # noqa
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.data import nist


#TODO: Eq (102) of TCA, 90, 421 to partition the dia- and para-magnetic terms
def dia(magobj, gauge_orig=None):
    '''Diamagnetic term of magnetizability.

    See also J. Olsen et al., Theor. Chem. Acc., 90, 421 (1995)
    '''
    mol = magobj.mol
    mf = magobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    orbo = mo_coeff[:,mo_occ > 0]
    dm0 = numpy.dot(orbo, orbo.T) * 2
    # Energy weighted density matrix
    dme0 = numpy.dot(orbo * mo_energy[mo_occ > 0], orbo.T) * 2

    e2 = _get_dia_1e(magobj, gauge_orig, dm0, dme0)

    # Add the contributions from two-electron interactions
    if gauge_orig is None:
        e2 = e2.ravel()
        # + 1/2 Tr[(J^{[20]} - K^{[20]}), DM]
        # Symmetry between 'ijkl,ji->s2kl' and 'ijkl,lk->s2ij' can be used.
        #vs = jk.get_jk(mol, [dm0]*4, ['ijkl,ji->s2kl',
        #                              'ijkl,lk->s2ij',
        #                              'ijkl,jk->s1il',
        #                              'ijkl,li->s1kj'],
        #               'int2e_gg1', 's4', 9, hermi=1)
        #e2 += numpy.einsum('xpq,qp->x', vs[0], dm0) * .5
        #e2 += numpy.einsum('xpq,qp->x', vs[1], dm0) * .5
        #e2 -= numpy.einsum('xpq,qp->x', vs[2], dm0) * .25
        #e2 -= numpy.einsum('xpq,qp->x', vs[3], dm0) * .25
        vs = jk.get_jk(mol, [dm0]*3, ['ijkl,ji->s2kl',
                                      'ijkl,jk->s1il',
                                      'ijkl,li->s1kj'],
                       'int2e_gg1', 's4', 9, hermi=1)
        e2 += numpy.einsum('xpq,qp->x', vs[0], dm0)
        e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0) * .25
        e2 -= numpy.einsum('xpq,qp->x', vs[2], dm0) * .25

        # J does not have contribution because integrals are anti-symmetric
        #vs = jk.get_jk(mol, [dm0]*2, ['ijkl,ji->s2kl',
        #                              'ijkl,jk->s1il'],
        #               'int2e_g1g2', 'aa4', 9, hermi=0)
        #e2 += numpy.einsum('xpq,qp->x', vs[0], dm0)
        #e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0) * .5
        vk = jk.get_jk(mol, dm0, 'ijkl,jk->s1il',
                       'int2e_g1g2', 'aa4', 9, hermi=0)
        e2 -= numpy.einsum('xpq,qp->x', vk, dm0) * .5

    # Note the sign of magnetizability (ksi) in the Talyor expansion
    #   E(B) = E0 - m * B - 1/2 B ksi B + ...
    # Magnetic susceptibility chi = mu0 * ksi
    return -e2.reshape(3, 3)

def _get_dia_1e(magobj, gauge_orig, dm0, dme0):
    '''The dia-magnetic magnetizability from one-electron operators.

    Args:
        dm0 : Density matrix

        dme0 : Energy-weighted density matrix
    '''
    mol = magobj.mol
    mf = magobj._scf

    if gauge_orig is None:
        h2 = mol.intor('int1e_rr_origj', comp=9)
    else:
        mol.set_common_origin(gauge_orig)
        h2 = mol.intor('int1e_rr', comp=9)

    if getattr(mf, 'with_x2c', None):
        raise NotImplementedError('X2C for magnetizability')

    if getattr(mf, 'with_qmmm', None):
        raise NotImplementedError('Magnetizability with QM/MM')

    if getattr(mf, 'with_solvent', None):
        raise NotImplementedError('Magnetizability with Solvent')

    e2 = numpy.einsum('xpq,qp->x', h2, dm0).reshape(3,3)
    e2 = numpy.eye(3) * e2.trace() - e2
    e2 *= .25

    # If gauge_orig is None, computing the GIAO contributions
    if gauge_orig is None:
        gl = numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_grjxp', comp=9))
        gl = gl.reshape(3,3)
        e2 += (gl + gl.T) * .5
        e2 = e2.ravel()
        e2 += numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_ggkin', comp=9))
        e2 += numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_ggnuc', comp=9))
        if mol.has_ecp():
            raise NotImplementedError
            e2+= numpy.einsum('qp,xpq->x', dm0, mol.intor('ECPscalar_ggnuc', comp=9))

        e2 -= numpy.einsum('qp,xpq->x', dme0, mol.intor('int1e_ggovlp', comp=9))

    # Note the sign of magnetizability (ksi) in the Talyor expansion
    #   E(B) = E0 - m * B - 1/2 B ksi B + ...
    # Magnetic susceptibility chi = mu0 * ksi
    return e2.reshape(3, 3)


# Note mo10 is the imaginary part of MO^1
def para(magobj, gauge_orig=None, h1=None, s1=None, with_cphf=None):
    '''Paramagnetic susceptibility tensor

    Kwargs:
        h1: (3,nmo,nocc) array
            First order Fock matrix in MO basis.
        s1: (3,nmo,nocc) array
            First order overlap matrix in MO basis.
        with_cphf : boolean or  function(dm_mo) => v1_mo
            If a boolean value is given, the value determines whether CPHF
            equation will be solved or not. The induced potential will be
            generated by the function gen_vind.
            If a function is given, CPHF equation will be solved, and the
            given function is used to compute induced potential
    '''
    log = logger.Logger(magobj.stdout, magobj.verbose)
    cput1 = (time.clock(), time.time())

    mol = magobj.mol
    mf = magobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:,occidx]

    if h1 is None:
        # Imaginary part of F10
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        h1 = lib.einsum('xpq,pi,qj->xij', magobj.get_fock(dm0, gauge_orig),
                        mo_coeff.conj(), orbo)
    if s1 is None:
        # Imaginary part of S10
        s1 = lib.einsum('xpq,pi,qj->xij', magobj.get_ovlp(mol, gauge_orig),
                        mo_coeff.conj(), orbo)
    cput1 = log.timer('first order Fock matrix', *cput1)

    with_cphf = magobj.cphf
    mo1, mo_e1 = rhf_nmr.solve_mo1(magobj, mo_energy, mo_coeff, mo_occ,
                                   h1, s1, with_cphf)
    cput1 = logger.timer(magobj, 'solving mo1 eqn', *cput1)

    mag_para = numpy.einsum('yji,xji->xy', mo1, h1)
    mag_para-= numpy.einsum('yji,xji,i->xy', mo1, s1, mo_energy[occidx])
    # + c.c.
    mag_para = mag_para + mag_para.conj()

    mag_para-= numpy.einsum('xij,yij->xy', s1[:,occidx], mo_e1)

    # *2 for double occupancy.
    mag_para *= 2
    return -mag_para


def _get_ao_coords(mol):
    atom_coords = mol.atom_coords()
    nao = mol.nao_nr()
    ao_coords = numpy.empty((nao, 3))
    aoslices = mol.aoslice_by_atom()
    for atm_id, (ish0, ish1, i0, i1) in enumerate(aoslices):
        ao_coords[i0:i1] = atom_coords[atm_id]
    return ao_coords


class Magnetizability(lib.StreamObject):
    def __init__(self, mf):
        mol = mf.mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.chkfile = mf.chkfile
        self._scf = mf

        self.gauge_orig = None
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
        if self.gauge_orig is None:
            log.info('gauge = GIAO')
        else:
            log.info('Common gauge = %s', str(self.gauge_orig))
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    def kernel(self):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        mag_dia = self.dia(self.gauge_orig)
        mag_para = self.para(self.gauge_orig)
        ksi = mag_para + mag_dia

        logger.timer(self, 'Magnetizability', *cput0)
        if self.verbose >= logger.NOTE:
            _write = rhf_nmr._write
            _write(self.stdout, ksi, '\nMagnetizability (au)')
            _write(self.stdout, mag_dia, 'dia-magnetic contribution (au)')
            _write(self.stdout, mag_para, 'para-magnetic contribution (au)')
            #if self.verbose >= logger.INFO:
            #    _write(self.stdout, para_occ, 'occ part of para-magnetic term')
            #    _write(self.stdout, para_vir, 'vir part of para-magnetic term')

            unit = nist.HARTREE2J / nist.AU2TESLA**2 * 1e30
            _write(self.stdout, ksi*unit, '\nMagnetizability (10^{-30} J/T^2)')
        return ksi

    dia = dia
    para = para
    get_fock = rhf_nmr.get_fock

    def get_ovlp(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return rhf_nmr.get_ovlp(mol, gauge_orig)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   0.
                  F  ,  0.   0.   .917'''
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run()
    mag = Magnetizability(mf)
    mag.cphf = True
    m = mag.kernel()
    print(lib.finger(m) - -0.43596639996758657)

    mag.gauge_orig = (0,0,1)
    m = mag.kernel()
    print(lib.finger(m) - -0.76996086788058238)

    mag.gauge_orig = (0,0,1)
    mag.cphf = False
    m = mag.kernel()
    print(lib.finger(m) - -0.7973915717274408)


    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='ccpvdz')
    mf = scf.RHF(mol).run()
    mag = Magnetizability(mf)
    mag.cphf = True
    m = mag.kernel()
    print(lib.finger(m) - -0.62173669377370366)
