#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
domain decomposition PCM (In testing)

See also
JCP, 144, 054101
JCP, 144, 160901
'''

import warnings
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf.solvent import _attach_solvent

warnings.warn('Module ddPCM is under testing')


@lib.with_doc(_attach_solvent._for_scf.__doc__)
def ddpcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = DDPCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def ddpcm_for_casscf(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), DDPCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = DDPCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def ddpcm_for_casci(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), DDPCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = DDPCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def ddpcm_for_post_scf(method, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(method._scf, 'with_solvent', None), DDPCM):
            solvent_obj = method._scf.with_solvent
        else:
            solvent_obj = DDPCM(method.mol)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_tdscf.__doc__)
def ddpcm_for_tdscf(method, solvent_obj=None, dm=None):
    scf_solvent = getattr(method._scf, 'with_solvent', None)
    assert scf_solvent is None or isinstance(scf_solvent, DDPCM)

    if solvent_obj is None:
        solvent_obj = DDPCM(method.mol)
    return _attach_solvent._for_tdscf(method, solvent_obj, dm)


# Inject ddPCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.ddPCM    = scf.hf.SCF.DDPCM    = ddpcm_for_scf
mp.mp2.MP2.ddPCM    = mp.mp2.MP2.DDPCM    = ddpcm_for_post_scf
ci.cisd.CISD.ddPCM  = ci.cisd.CISD.DDPCM  = ddpcm_for_post_scf
cc.ccsd.CCSD.ddPCM  = cc.ccsd.CCSD.DDPCM  = ddpcm_for_post_scf
tdscf.rhf.TDMixin.ddPCM = tdscf.rhf.TDMixin.DDPCM = ddpcm_for_tdscf
mcscf.casci.CASCI.ddPCM = mcscf.casci.CASCI.DDPCM = ddpcm_for_casci
mcscf.mc1step.CASSCF.ddPCM = mcscf.mc1step.CASSCF.DDPCM = ddpcm_for_casscf

def gen_ddpcm_solver(pcmobj, verbose=None):
    mol = pcmobj.mol
    if pcmobj.grids.coords is None:
        pcmobj.grids.build(with_non0tab=True)

    natm = mol.natm
    lmax = pcmobj.lmax

    r_vdw = ddcosmo.get_atomic_radii(pcmobj)
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0
    nexposed = numpy.count_nonzero(ui==1)
    nbury = numpy.count_nonzero(ui==0)
    on_shell = numpy.count_nonzero(ui>0) - nexposed
    logger.debug(pcmobj, 'Num points exposed %d', nexposed)
    logger.debug(pcmobj, 'Num points buried %d', nbury)
    logger.debug(pcmobj, 'Num points on shell %d', on_shell)

    nlm = (lmax+1)**2
    Lmat = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
    Lmat = Lmat.reshape(natm*nlm,-1)

    Amat = make_A(pcmobj, r_vdw, ylm_1sph, ui).reshape(natm*nlm,-1)
    fac = 2*numpy.pi * (pcmobj.eps+1) / (pcmobj.eps-1)
    A_diele = Amat + fac * numpy.eye(natm*nlm)
    A_inf = Amat + 2*numpy.pi * numpy.eye(natm*nlm)

    cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

    def gen_vind(dm):
        phi = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui)
        phi = numpy.linalg.solve(A_diele, A_inf.dot(phi.ravel()))

        Xvec = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
        psi, vmat = ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, pcmobj.grids,
                                          ylm_1sph, cached_pol, Xvec, Lmat)[:2]
        dielectric = pcmobj.eps
        f_epsilon = (dielectric-1.)/dielectric
        epcm = .5 * f_epsilon * numpy.einsum('jx,jx', psi, Xvec)
        vpcm = .5 * f_epsilon * vmat
        return epcm, vpcm
    return gen_vind

def energy(pcmobj, dm):
    r'''
    ddPCM energy
    Es = 1/2 f(eps) \int rho(r) W(r) dr
    '''
    epcm = gen_ddpcm_solver(pcmobj, pcmobj.verbose)(dm)[0]
    return epcm

def regularize_xt(t, eta):
    xt = numpy.zeros_like(t)
    inner = t <= 1-eta
    on_shell = (1-eta < t) & (t < 1)
    xt[inner] = 1
    ti = t[on_shell] - eta*.5
    # JCP, 144, 054101
    xt[on_shell] = 1./eta**4 * (1-ti)**2 * (ti-1+2*eta)**2
    return xt

def make_A(pcmobj, r_vdw, ylm_1sph, ui):
    # Part of A matrix defined in JCP, 144, 054101, Eq (43), (44)
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    # eta = pcmobj.eta
    nlm = (lmax+1)**2

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = weights_1sph.size
    atom_coords = mol.atom_coords()
    ylm_1sph = ylm_1sph.reshape(nlm,ngrid_1sph)
    Amat = numpy.zeros((natm,nlm,natm,nlm))

    for ja in range(natm):
        # w_u = precontract w_n U_j
        w_u = weights_1sph * ui[ja]
        p1 = 0
        for l in range(lmax+1):
            fac = 2*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            a = numpy.einsum('xn,n,mn->xm', ylm_1sph, w_u, ylm_1sph[p0:p1])
            Amat[ja,:,ja,p0:p1] += -fac * a

        for ka in ddcosmo.atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            vjk = r_vdw[ja] * coords_1sph + atom_coords[ja] - atom_coords[ka]
            rjk = lib.norm(vjk, axis=1)
            pol = sph.multipoles(vjk, lmax)
            p1 = 0
            weights = w_u / rjk**(l*2+1)
            for l in range(lmax+1):
                fac = 4*numpy.pi*l/(l*2+1) * r_vdw[ka]**(l+1)
                p0, p1 = p1, p1 + (l*2+1)
                a = numpy.einsum('xn,n,mn->xm', ylm_1sph, weights, pol[l])
                Amat[ja,:,ka,p0:p1] += -fac * a
    return Amat

class DDPCM(ddcosmo.DDCOSMO):
    def __init__(self, mol):
        ddcosmo.DDCOSMO.__init__(self, mol)

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'ddPCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'lmax = %s'         , self.lmax)
        logger.info(self, 'eta = %s'          , self.eta)
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.debug2(self, 'radii_table %s', self.radii_table)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        self.grids.dump_flags(verbose)
        return self

    def build(self):
        if self.grids.coords is None:
            self.grids.build(with_non0tab=True)

        mol = self.mol
        natm = mol.natm
        lmax = self.lmax

        r_vdw = ddcosmo.get_atomic_radii(self)
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(self.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

        fi = ddcosmo.make_fi(self, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        nexposed = numpy.count_nonzero(ui==1)
        nbury = numpy.count_nonzero(ui==0)
        on_shell = numpy.count_nonzero(ui>0) - nexposed
        logger.debug(self, 'Num points exposed %d', nexposed)
        logger.debug(self, 'Num points buried %d', nbury)
        logger.debug(self, 'Num points on shell %d', on_shell)

        nlm = (lmax+1)**2
        Lmat = ddcosmo.make_L(self, r_vdw, ylm_1sph, fi)
        Lmat = Lmat.reshape(natm*nlm,-1)

        Amat = make_A(self, r_vdw, ylm_1sph, ui).reshape(natm*nlm,-1)
        fac = 2*numpy.pi * (self.eps+1) / (self.eps-1)
        A_diele = Amat + fac * numpy.eye(natm*nlm)
        A_inf = Amat + 2*numpy.pi * numpy.eye(natm*nlm)

        cached_pol = ddcosmo.cache_fake_multipoles(self.grids, r_vdw, lmax)

        self._intermediates = {
            'r_vdw': r_vdw,
            'ylm_1sph': ylm_1sph,
            'ui': ui,
            'Lmat': Lmat,
            'A_diele': A_diele,
            'A_inf': A_inf,
            'cached_pol': cached_pol,
        }

    def _get_vind(self, dm):
        '''A single shot solvent effects for given density matrix.
        '''
        if not self._intermediates or self.grids.coords is None:
            self.build()

        mol = self.mol
        r_vdw      = self._intermediates['r_vdw'     ]
        ylm_1sph   = self._intermediates['ylm_1sph'  ]
        ui         = self._intermediates['ui'        ]
        Lmat       = self._intermediates['Lmat'      ]
        A_diele    = self._intermediates['A_diele'   ]
        A_inf      = self._intermediates['A_inf'     ]
        cached_pol = self._intermediates['cached_pol']

        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]

        phi = ddcosmo.make_phi(self, dm, r_vdw, ui, ylm_1sph)
        phi = numpy.linalg.solve(A_diele, A_inf.dot(phi.ravel()))

        Xvec = numpy.linalg.solve(Lmat, phi.ravel()).reshape(mol.natm,-1)
        psi, vmat = ddcosmo.make_psi_vmat(self, dm, r_vdw, ui, ylm_1sph,
                                          cached_pol, Xvec, Lmat)[:2]
        dielectric = self.eps
        f_epsilon = (dielectric-1.)/dielectric
        epcm = .5 * f_epsilon * numpy.einsum('jx,jx', psi, Xvec)
        vpcm = .5 * f_epsilon * vmat
        return epcm, vpcm

    def _B_dot_x(self, dm):
        '''
        Compute the matrix-vector product B * x. The B matrix, as defined in
        the paper R. Cammi, JPCA, 104, 5631 (2000), is the second order
        derivatives of E_solvation wrt density matrices.

        Note: In ddCOSMO, strictly, B is not symmetric. To make it compatible
        with the CIS framework, it is symmetrized in current implementation.
        '''
        if not self._intermediates or self.grids.coords is None:
            self.build()

        mol = self.mol
        r_vdw      = self._intermediates['r_vdw'     ]
        ylm_1sph   = self._intermediates['ylm_1sph'  ]
        ui         = self._intermediates['ui'        ]
        Lmat       = self._intermediates['Lmat'      ]
        A_diele    = self._intermediates['A_diele'   ]
        A_inf      = self._intermediates['A_inf'     ]
        cached_pol = self._intermediates['cached_pol']
        natm = mol.natm
        nlm = (self.lmax+1)**2

        phi = ddcosmo.make_phi(self, dm, r_vdw, ui, ylm_1sph, with_nuc=False)
        phi = numpy.linalg.solve(A_diele, A_inf.dot(phi.reshape(-1,natm*nlm).T))

        Xvec = numpy.linalg.solve(Lmat, phi)
        Xvec = Xvec.reshape(natm,nlm,-1).transpose(2,0,1)
        vmat = ddcosmo.make_psi_vmat(self, dm, r_vdw, ui, ylm_1sph,
                                     cached_pol, Xvec, Lmat, with_nuc=False)[1]
        dielectric = self.eps
        f_epsilon = (dielectric-1.)/dielectric
        return .5 * f_epsilon * vmat

    gen_solver = as_solver = gen_ddpcm_solver

    def regularize_xt(self, t, eta, scale=1):
        return regularize_xt(t, eta)

    def nuc_grad_method(self, grad_method):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1')
    numpy.random.seed(1)

    nao = mol.nao_nr()
    dm = numpy.random.random((nao,nao))
    dm = dm + dm.T
    #dm = scf.RHF(mol).run().make_rdm1()
    e, vmat = DDPCM(mol).kernel(dm)
    print(e + 1.2446306643473923)
    print(lib.fp(vmat) - 0.77873361914445294)

    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g' #cc-pvdz'
    mol.build()
    cm = DDPCM(mol)
    cm.verbose = 4
    mf = ddpcm_for_scf(scf.RHF(mol), cm)#.newton()
    mf.verbose = 4
    mf.kernel()  # -75.5697645601958
