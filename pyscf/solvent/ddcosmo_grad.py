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
Analytical nuclear gradients for domain decomposition COSMO

See also

[1] Fast Domain Decomposition Algorithm for Continuum Solvation Models: Energy and First Derivatives.
F. Lipparini, B. Stamm, E. Cances, Y. Maday, B. Mennucci
J. Chem. Theory Comput., 9, 3637-3648 (2013)
http://dx.doi.org/10.1021/ct400280b

[2] Quantum, classical, and hybrid QM/MM calculations in solution: General implementation of the ddCOSMO linear scaling strategy.
F. Lipparini, G. Scalmani, L. Lagardere, B. Stamm, E. Cances, Y. Maday, J.-P.Piquemal, M. J. Frisch, B. Mennucci
J. Chem. Phys., 141, 184108 (2014)
http://dx.doi.org/10.1063/1.4901304
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.dft import gen_grid, numint
from pyscf.symm import sph
from pyscf.solvent import ddcosmo
from pyscf.solvent._attach_solvent import _Solvation
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf as tdrhf_grad  # noqa


# TODO: Define attribute grad_method.base to point to the class of the 0th
# order calculation for all gradients class. Then this function can be
# extended and used as the general interface to initialize solvent gradients.
def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_solvent = kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                rhf_grad._write(self, self.mol, self.de, self.atmlst)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)


def kernel(pcmobj, dm, verbose=None):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    if pcmobj.grids.coords is None:
        pcmobj.grids.build(with_non0tab=True)

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF density matrix
        dm = dm[0] + dm[1]

    r_vdw = ddcosmo.get_atomic_radii(pcmobj)
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0

    cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

    nlm = (lmax+1)**2
    L0 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
    L0 = L0.reshape(natm*nlm,-1)
    L1 = make_L1(pcmobj, r_vdw, ylm_1sph, fi)

    phi0 = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui, ylm_1sph)
    phi1 = make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph)
    L0_X = numpy.linalg.solve(L0, phi0.ravel()).reshape(natm,-1)
    psi0, vmat, L0_S = \
            ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph,
                                  cached_pol, L0_X, L0)
    e_psi1 = make_e_psi1(pcmobj, dm, r_vdw, ui, ylm_1sph,
                         cached_pol, L0_X, L0)
    dielectric = pcmobj.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1
    de = .5 * f_epsilon * e_psi1
    de+= .5 * f_epsilon * numpy.einsum('jx,azjx->az', L0_S, phi1)
    de-= .5 * f_epsilon * numpy.einsum('aziljm,il,jm->az', L1, L0_S, L0_X)
    return de

def make_L1(pcmobj, r_vdw, ylm_1sph, fi):
    # See JCTC, 9, 3637, Eq (18)
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    eta = pcmobj.eta
    nlm = (lmax+1)**2

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = weights_1sph.size
    atom_coords = mol.atom_coords()
    ylm_1sph = ylm_1sph.reshape(nlm,ngrid_1sph)

    Lmat = numpy.zeros((natm,3,natm,nlm,natm,nlm))
    fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())

    for ja in range(natm):
        part_weights = weights_1sph.copy()
        part_weights[fi[ja]>1] /= fi[ja,fi[ja]>1]

        part_weights1 = numpy.zeros((natm,3,ngrid_1sph))
        tmp = part_weights[fi[ja]>1] / fi[ja,fi[ja]>1]
        part_weights1[:,:,fi[ja]>1] = -tmp * fi1[:,:,ja,fi[ja]>1]

        for ka in ddcosmo.atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            vjk = r_vdw[ja] * coords_1sph + atom_coords[ja] - atom_coords[ka]
            rv = lib.norm(vjk, axis=1)
            tjk = rv / r_vdw[ka]
            wjk0 = pcmobj.regularize_xt(tjk, eta, r_vdw[ka])
            wjk1 = regularize_xt1(tjk, eta*r_vdw[ka])
            sjk = vjk.T / rv
            wjk1 = 1./r_vdw[ka] * wjk1 * sjk

            wjk01 = wjk0 * part_weights1
            wjk0 *= part_weights
            wjk1 *= part_weights

            pol0 = sph.multipoles(vjk, lmax)
            pol1 = multipoles1(vjk, lmax)
            p1 = 0
            for l in range(lmax+1):
                fac = 4*numpy.pi/(l*2+1) / r_vdw[ka]**(l+1)
                p0, p1 = p1, p1 + (l*2+1)
                a = numpy.einsum('xn,zn,mn->zxm', ylm_1sph, wjk1, pol0[l])
                a+= numpy.einsum('xn,n,zmn->zxm', ylm_1sph, wjk0, pol1[l])
                Lmat[ja,:,ja,:,ka,p0:p1] += -fac * a
                Lmat[ka,:,ja,:,ka,p0:p1] -= -fac * a
                a = numpy.einsum('xn,azn,mn->azxm', ylm_1sph, wjk01, pol0[l])
                Lmat[:,:,ja,:,ka,p0:p1] += -fac * a
    return Lmat


def multipoles1(r, lmax, reorder_dipole=True):
    ngrid = r.shape[0]
    xs = numpy.ones((lmax+1,ngrid))
    ys = numpy.ones((lmax+1,ngrid))
    zs = numpy.ones((lmax+1,ngrid))
    for i in range(1,lmax+1):
        xs[i] = xs[i-1] * r[:,0]
        ys[i] = ys[i-1] * r[:,1]
        zs[i] = zs[i-1] * r[:,2]
    ylms = []
    for l in range(lmax+1):
        nd = (l+1)*(l+2)//2
        c = numpy.empty((nd,3,ngrid))
        k = 0
        for lx in reversed(range(0, l+1)):
            for ly in reversed(range(0, l-lx+1)):
                lz = l - lx - ly
                c[k,0] = lx * xs[lx-1] * ys[ly] * zs[lz]
                c[k,1] = ly * xs[lx] * ys[ly-1] * zs[lz]
                c[k,2] = lz * xs[lx] * ys[ly] * zs[lz-1]
                k += 1
        ylm = gto.cart2sph(l, c.reshape(nd,3*ngrid).T)
        ylm = ylm.reshape(3,ngrid,l*2+1).transpose(0,2,1)
        ylms.append(ylm)

# when call libcint, p functions are ordered as px,py,pz
# reorder px,py,pz to p(-1),p(0),p(1)
    if (not reorder_dipole) and lmax >= 1:
        ylms[1] = ylms[1][:,[1,2,0]]
    return ylms


def regularize_xt1(t, eta):
    xt = numpy.zeros_like(t)
    # no response if grids are inside the cavity
    # inner = t <= 1-eta
    # xt[inner] = 0
    on_shell = (1-eta < t) & (t < 1)
    ti = t[on_shell]
    xt[on_shell] = -30./eta**5 * (1-ti)**2 * (1-eta-ti)**2
    return xt

def make_fi1(pcmobj, r_vdw):
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    mol = pcmobj.mol
    eta = pcmobj.eta
    natm = mol.natm
    atom_coords = mol.atom_coords()
    ngrid_1sph = coords_1sph.shape[0]
    fi1 = numpy.zeros((natm,3,natm,ngrid_1sph))
    for ia in range(natm):
        for ja in ddcosmo.atoms_with_vdw_overlap(ia, atom_coords, r_vdw):
            v = r_vdw[ia]*coords_1sph + atom_coords[ia] - atom_coords[ja]
            rv = lib.norm(v, axis=1)
            t = rv / r_vdw[ja]
            xt1 = regularize_xt1(t, eta*r_vdw[ja])
            s_ij = v.T / rv
            xt1 = 1./r_vdw[ja] * xt1 * s_ij
            fi1[ia,:,ia] += xt1
            fi1[ja,:,ia] -= xt1

    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1[:,:,fi<1e-20] = 0
    return fi1

def make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph):
    mol = pcmobj.mol
    natm = mol.natm

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    tril_dm = lib.pack_tril(dm+dm.T)
    nao = dm.shape[0]
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    #extern_point_idx = ui > 0

    fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    ngrid_1sph = weights_1sph.size
    v_phi0 = numpy.empty((natm,ngrid_1sph))
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords
        v_phi0[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi1 = -numpy.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, v_phi0)

    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        for ja in range(natm):
            rs = atom_coords[ja] - cav_coords
            d_rs = lib.norm(rs, axis=1)
            v_phi = atom_charges[ja] * numpy.einsum('px,p->px', rs, 1./d_rs**3)
            tmp = numpy.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
            phi1[ja,:,ia] += tmp  # response of the other atoms
            phi1[ia,:,ia] -= tmp  # response of cavity grids

    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1')
        v_phi = numpy.einsum('ij,ijk->k', dm, v_nj)
        phi1[:,:,ia] += numpy.einsum('n,ln,azn,n->azl', weights_1sph, ylm_1sph, ui1[:,:,ia], v_phi)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3, aosym='s1')
        phi1_e2_nj  = numpy.einsum('ij,xijr->xr', dm, v_e1_nj)
        phi1_e2_nj += numpy.einsum('ji,xijr->xr', dm, v_e1_nj)
        phi1[ia,:,ia] += numpy.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_e2_nj)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            phi1_nj  = numpy.einsum('ij,xijr->xr', dm[p0:p1  ], v_e1_nj[:,p0:p1])
            phi1_nj += numpy.einsum('ji,xijr->xr', dm[:,p0:p1], v_e1_nj[:,p0:p1])
            phi1[ja,:,ia] -= numpy.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_nj)
    return phi1

def make_e_psi1(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, Xvec, L):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    grids = pcmobj.grids

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    ni = numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    den = numpy.empty((4,grids.weights.size))

    ao_loc = mol.ao_loc_nr()
    vmat = numpy.zeros((3,nao,nao))
    psi1 = numpy.zeros((natm,3))
    i1 = 0
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        i0, i1 = i1, i1 + weight.size
        ao = ni.eval_ao(mol, coords, deriv=1)
        mask = gen_grid.make_mask(mol, coords)
        den[:,i0:i1] = make_rho(0, ao, mask, 'GGA')

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        eta_nj = 0
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            eta_nj += fac * numpy.einsum('mn,m->n', fak_pol[l], Xvec[ia,p0:p1])
        psi1 -= numpy.einsum('n,n,zxn->zx', den[0,i0:i1], eta_nj, weight1)
        psi1[ia] -= numpy.einsum('xn,n,n->x', den[1:4,i0:i1], eta_nj, weight)

        vtmp = numpy.zeros((3,nao,nao))
        aow = numpy.einsum('pi,p->pi', ao[0], weight*eta_nj)
        rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
        vmat += vtmp

    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        shl0, shl1, p0, p1 = aoslices[ia]
        psi1[ia] += numpy.einsum('xij,ij->x', vmat[:,p0:p1], dm[p0:p1]) * 2
    return psi1


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B')
    mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol))
    mf.kernel()
    de = mf.nuc_grad_method().kernel()
    de_cosmo = kernel(mf.with_solvent, mf.make_rdm1())
    dm1 = mf.make_rdm1()

    mol = gto.M(atom='H 0 0 -0.001; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B')
    mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol))
    e1 = mf.kernel()
    e1_cosmo = mf.with_solvent.energy(dm1)

    mol = gto.M(atom='H 0 0 0.001; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B')
    mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol))
    e2 = mf.kernel()
    e2_cosmo = mf.with_solvent.energy(dm1)
    print(abs((e2-e1)/0.002 - de[0,2]).max())
    print(abs((e2_cosmo-e1_cosmo)/0.002 - de_cosmo[0,2]).max())
