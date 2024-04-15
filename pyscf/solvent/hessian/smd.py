# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Hessian of SMD solvent model, copied from GPU4PySCF with modification for CPU
'''
# pylint: disable=C0103

import numpy as np
from pyscf import scf, lib
from pyscf.solvent import smd
from pyscf.solvent.grad import smd as smd_grad
from pyscf.solvent.hessian import pcm as pcm_hess

from pyscf.solvent.smd import (
    sigma_water, sigma_n, sigma_alpha, sigma_beta, r_zz, swtich_function)
from pyscf.lib import logger

def hess_swtich_function(R, r, dr):
    if R < r + dr:
        dist = (R[0]**2 + R[1]**2 + R[2]**2)**.5
        hess = [
            [R[1]**2+R[2]**2, -R[0]*R[1], -R[0]*R[2]],
            [-R[0]*R[1], R[0]**2+R[2]**2, -R[1]*R[2]],
            [-R[0]*R[2], -R[1]*R[2], R[0]**2+R[1]**2]]
        return np.asarray(hess)/dist
    else:
        return np.zeros([3,3])

def atomic_surface_tension(symbols, coords, n, alpha, beta, water=True):
    '''
    TODO: debug later
    - list of atomic symbols
    - atomic coordinates in Anstrong
    - solvent descriptors: n, alpha, beta
    '''

    def get_bond_tension(bond):
        if water:
            return sigma_water.get(bond, 0.0)
        t = 0.0
        t += sigma_n.get(bond, 0.0) * n
        t += sigma_alpha.get(bond, 0.0) * alpha
        t += sigma_beta.get(bond, 0.0) * beta
        return t

    def get_atom_tension(sym_i):
        if water:
            return sigma_water.get(sym_i, 0.0)
        t = 0.0
        t += sigma_n.get(sym_i, 0.0) * n
        t += sigma_alpha.get(sym_i, 0.0) * alpha
        t += sigma_beta.get(sym_i, 0.0) * beta
        return t
    natm = coords.shape[0]
    ri_rj = coords[:,None,:] - coords[None,:,:]
    rij = np.sum(ri_rj**2, axis=2)**0.5
    np.fill_diagonal(rij, 1)
    #drij = ri_rj / np.expand_dims(rij, axis=-1)
    tensions = []
    for i, sym_i in enumerate(symbols):
        if sym_i not in ['H', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(np.zeros([natm,3]))
            continue

        tension = np.zeros([natm,3])
        if sym_i in ['F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(tension)
            continue

        if sym_i == 'H':
            dt_HC = np.zeros([natm,natm,3,3])
            dt_HO = np.zeros([natm,natm,3,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('H','C'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_HC[i,i] += dt_drij
                    dt_HC[i,j] -= dt_drij
                    dt_HC[j,i] -= dt_drij
                    dt_HC[j,j] += dt_drij
                if sym_j == 'O':
                    r, dr = r_zz.get(('H','O'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_HO[i,i] += dt_drij
                    dt_HO[i,j] -= dt_drij
                    dt_HO[j,i] -= dt_drij
                    dt_HO[j,j] += dt_drij
            sig_HC = get_bond_tension(('H','C'))
            sig_HO = get_bond_tension(('H','O'))
            tension += sig_HC * dt_HC + sig_HO * dt_HO
            tensions.append(tension)

        if sym_i == 'C':
            dt_CN = np.zeros([natm,3])
            d2t_CC = np.zeros([natm,natm,3,3])
            d2t_CN = np.zeros([natm,natm,3,3])
            t_CN = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C' and i != j:
                    r, dr = r_zz.get(('C', 'C'), (0.0, 0.0))
                    d2t_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    d2t_CC[i,i] += d2t_drij
                    d2t_CC[i,j] -= d2t_drij
                    d2t_CC[j,i] -= d2t_drij
                    d2t_CC[j,j] += d2t_drij
                if sym_j == 'N':
                    r, dr = r_zz.get(('C', 'N'), (0.0, 0.0))
                    t_CN += swtich_function(rij[i,j], r, dr)
                    dt_drij = smd_grad.grad_switch_function(coords[i]-coords[j], r, dr)
                    dt_CN[i] += dt_drij
                    dt_CN[j] -= dt_drij
                    d2t_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    d2t_CN[i,i] += d2t_drij
                    d2t_CN[i,j] -= d2t_drij
                    d2t_CN[j,i] -= d2t_drij
                    d2t_CN[j,j] += d2t_drij
            sig_CC = get_bond_tension(('C','C'))
            sig_CN = get_bond_tension(('C','N'))
            tension += sig_CC * d2t_CC + sig_CN * (2 * t_CN * d2t_CN + 2 * np.einsum('i,j->ij', dt_drij[i], dt_drij[j]))
            tensions.append(tension)

        if sym_i == 'N':
            t_NC = 0.0
            dt_NC = np.zeros([natm,natm,3,3])
            dt_NC3 = np.zeros([natm,natm,3,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('N','C'), (0.0, 0.0))
                    tk = 0.0
                    dtk = np.zeros([natm,natm,3,3])
                    for k, sym_k in enumerate(symbols):
                        if k != i and k != j:
                            rjk, drjk = r_zz.get(('C', sym_k), (0.0, 0.0))
                            tk += swtich_function(rij[j,k], rjk, drjk)
                            dtk_rjk = hess_swtich_function(coords[j]-coords[k], rjk, drjk)
                            dtk[j,j] += dtk_rjk
                            dtk[j,k] -= dtk_rjk
                            dtk[k,j] -= dtk_rjk
                            dtk[k,k] += dtk_rjk
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr) * tk**2
                    dt_NC[i,i] += dt_drij
                    dt_NC[i,j] -= dt_drij
                    dt_NC[j,i] -= dt_drij
                    dt_NC[j,j] += dt_drij
                    t = swtich_function(coords[i]-coords[j], r, dr)
                    dt_NC += t * (2 * tk * dtk)
                    t_NC += t * tk**2

                    r, dr = r_zz.get(('N','C3'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_NC3[i,i] += dt_drij
                    dt_NC3[i,j] -= dt_drij
                    dt_NC3[j,i] -= dt_drij
                    dt_NC3[j,j] += dt_drij
            sig_NC = get_bond_tension(('N','C'))
            sig_NC3= get_bond_tension(('N','C3'))
            tension += sig_NC * (1.3 * t_NC**0.3 * dt_NC) + sig_NC3 * dt_NC3
            tensions.append(tension)

        if sym_i == 'O':
            dt_OC = np.zeros([natm,natm,3,3])
            dt_ON = np.zeros([natm,natm,3,3])
            dt_OO = np.zeros([natm,natm,3,3])
            dt_OP = np.zeros([natm,natm,3,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('O','C'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_OC[i,i] += dt_drij
                    dt_OC[i,j] -= dt_drij
                    dt_OC[j,i] -= dt_drij
                    dt_OC[j,j] += dt_drij
                if sym_j == 'N':
                    r, dr = r_zz.get(('O','N'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_ON[i,i] += dt_drij
                    dt_ON[i,j] -= dt_drij
                    dt_ON[j,i] -= dt_drij
                    dt_ON[j,j] += dt_drij
                if sym_j == 'O' and j != i:
                    r, dr = r_zz.get(('O','O'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_OO[i,i] += dt_drij
                    dt_OO[i,j] -= dt_drij
                    dt_OO[j,i] -= dt_drij
                    dt_OO[j,j] += dt_drij
                if sym_j == 'P':
                    r, dr = r_zz.get(('O','P'), (0.0, 0.0))
                    dt_drij = hess_swtich_function(coords[i]-coords[j], r, dr)
                    dt_OP[i,i] += dt_drij
                    dt_OP[i,j] -= dt_drij
                    dt_OP[j,i] -= dt_drij
                    dt_OP[j,j] += dt_drij
            sig_OC = get_bond_tension(('O','C'))
            sig_ON = get_bond_tension(('O','N'))
            sig_OO = get_bond_tension(('O','O'))
            sig_OP = get_bond_tension(('O','P'))
            tension += sig_OC * dt_OC + sig_ON * dt_ON + sig_OO * dt_OO + sig_OP * dt_OP
            tensions.append(tension)
    return np.asarray(tensions)

def get_cds(smdobj):
    mol = smdobj.mol
    solvent = smdobj.solvent
    def smd_grad_scanner(mol):
        smdobj_tmp = smd.SMD(mol)
        smdobj_tmp.solvent = solvent
        return smd_grad.get_cds(smdobj_tmp)

    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    eps = 1e-4
    natm = mol.natm
    hess_cds = np.zeros([natm,natm,3,3])
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad0_cds = smd_grad_scanner(mol)

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad1_cds = smd_grad_scanner(mol)

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            hess_cds[ia,:,j] = (grad0_cds - grad1_cds) / (2.0 * eps)
    t1 = log.timer_debug1('solvent energy', *t1)
    return hess_cds # hartree

def make_hess_object(hess_method):
    '''For hess_method in vacuum, add nuclear Hessian of solvent smdobj'''
    if hess_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    name = (hess_method.base.with_solvent.__class__.__name__
            + hess_method.__class__.__name__)
    return lib.set_class(WithSolventHess(hess_method),
                         (WithSolventHess, hess_method.__class__), name)

class WithSolventHess:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        is_equilibrium = self.base.with_solvent.equilibrium_solvation
        self.base.with_solvent.equilibrium_solvation = True
        self.de_solvent = pcm_hess.hess_elec(self.base.with_solvent, dm, verbose=self.verbose)
        #self.de_solvent+= hess_nuc(self.base.with_solvent)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de_cds = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        self.base.with_solvent.equilibrium_solvation = is_equilibrium
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1(ao_repr=True)
            dv = pcm_hess.fd_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1(ao_repr=True)
            dm = dm[0] + dm[1]
            dva = pcm_hess.fd_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = pcm_hess.fd_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')
    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
