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
Hessian of SMD solvent model (for experiment and education)
copied from GPU4PySCF with modification for CPU
'''

import numpy as np
from pyscf.solvent.grad import smd as smd_grad
from pyscf.solvent.smd_experiment import (
    sigma_water, sigma_n, sigma_alpha, sigma_beta, r_zz, swtich_function)

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
