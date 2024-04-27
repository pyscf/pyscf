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
SMD solvent model (for experiment and education)
copied from GPU4PySCF with modification for CPU
'''

import numpy as np
import scipy
from pyscf.data import radii
from pyscf.lib import logger
from pyscf.solvent import pcm
from pyscf.solvent.smd import hartree2kcal

# see https://pubs.acs.org/doi/epdf/10.1021/jp810292n

sigma_water = {
    ('H'): 48.69,
    ('C'): 129.74,
    ('H','C'): -60.77,
    ('C','C'): -72.95,
    ('O','C'): 68.69,
    ('N','C'): -48.22,
    ('N','C3'): 84.10,
    ('O','N'): 121.98,
    ('F'): 38.18,
    ('Cl'): 9.82,
    ('Br'): -8.72,
    ('S'): -9.10,
    ('O','P'): 68.85}

sigma_n = {
    ('C'): 58.10,
    ('H','C'): -36.37,
    ('C','C'): -62.05,
    ('O'): -17.56,
    ('H','O'): -19.39,
    ('O','C'): -15.70,
    ('N'): 32.62,
    ('C','N'): -99.76,
    ('Cl'): -24.31,
    ('Br'): -35.42,
    ('S'): -33.17,
    ('Si'): -18.04
}

sigma_alpha = {
    ('C'): 48.10,
    ('O'): 193.06,
    ('O','C'): 95.99,
    ('C','N'): 152.20,
    ('N','C'): -41.00
}

sigma_beta = {
    ('C'): 32.87,
    ('O'): -43.79,
    ('O','O'):-128.16,
    ('O','N'):79.13
}

# Molecular surface tension (cal/mol*AA^-2)
sigma_gamma = 0.35
sigma_phi2 = -4.19
sigma_psi2 = -6.68
sigma_beta2 = 0.0
gamma0 = 1.0

# rzz and delta r_zz in AA
r_zz = {
    ('H','C'): [1.55, 0.3],
    ('H','O'): [1.55, 0.3],
    ('C','H'): [1.55, 0.3],
    ('C','C'): [1.84, 0.3],
    ('C','N'): [1.84, 0.3],
    ('C','O'): [1.84, 0.3],
    ('C','F'): [1.84, 0.3],
    ('C','P'): [2.2, 0.3],
    ('C','S'): [2.2, 0.3],
    ('C','Cl'): [2.1, 0.3],
    ('C','Br'): [2.3, 0.3],
    ('C','I'): [2.6, 0.3],
    ('N','C'): [1.84, 0.3],
    ('N','C3'): [1.225, 0.065],
    ('O','C'): [1.33, 0.1],
    ('O','N'): [1.5, 0.3],
    ('O','O'): [1.8, 0.3],
    ('O','P'): [2.1, 0.3]
}


def swtich_function(R, r, dr):
    return np.exp(dr/(R-dr-r)) if R<r+dr else 0

def atomic_surface_tension(symbols, coords, n, alpha, beta, water=True):
    '''
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

    rij = scipy.spatial.distance.cdist(coords, coords)
    tensions = []
    for i, sym_i in enumerate(symbols):
        if sym_i not in ['H', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(0)
            continue

        tension = get_atom_tension(sym_i)
        if sym_i in ['F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(tension)
            continue

        if sym_i == 'H':
            t_HC = 0.0
            t_HO = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('H','C'), (0.0, 0.0))
                    t_HC += swtich_function(rij[i,j], r, dr)
                if sym_j == 'O':
                    r, dr = r_zz.get(('H','O'), (0.0, 0.0))
                    t_HO += swtich_function(rij[i,j], r, dr)
            sig_HC = get_bond_tension(('H','C'))
            sig_HO = get_bond_tension(('H','O'))
            tension += sig_HC * t_HC + sig_HO * t_HO
            tensions.append(tension)
            continue

        if sym_i == 'C':
            t_CC = 0.0
            t_CN = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C' and i != j:
                    r, dr = r_zz.get(('C', 'C'), (0.0, 0.0))
                    t_CC += swtich_function(rij[i,j], r, dr)
                if sym_j == 'N':
                    r, dr = r_zz.get(('C', 'N'), (0.0, 0.0))
                    t_CN += swtich_function(rij[i,j], r, dr)
            sig_CC = get_bond_tension(('C','C'))
            sig_CN = get_bond_tension(('C','N'))
            tension += sig_CC * t_CC + sig_CN * t_CN**2
            tensions.append(tension)
            continue

        if sym_i == 'N':
            t_NC = 0.0
            t_NC3 = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('N','C'), (0.0,0.0))
                    tk = 0.0
                    for k, sym_k in enumerate(symbols):
                        if k != i and k != j:
                            rjk, drjk = r_zz.get(('C', sym_k), (0.0,0.0))
                            tk += swtich_function(rij[j,k], rjk, drjk)
                    t_NC += swtich_function(rij[i,j], r, dr) * tk**2

                    r, dr = r_zz.get(('N','C3'), (0.0, 0.0))
                    t_NC3 += swtich_function(rij[i,j], r, dr)
            sig_NC = get_bond_tension(('N','C'))
            sig_NC3= get_bond_tension(('N','C3'))
            tension += sig_NC * t_NC**1.3 + sig_NC3 * t_NC3
            tensions.append(tension)
            continue

        if sym_i == 'O':
            t_OC = 0.0
            t_ON = 0.0
            t_OO = 0.0
            t_OP = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('O','C'), (0.0, 0.0))
                    t_OC += swtich_function(rij[i,j], r, dr)
                if sym_j == 'N':
                    r, dr = r_zz.get(('O','N'), (0.0, 0.0))
                    t_ON += swtich_function(rij[i,j], r, dr)
                if sym_j == 'O' and j != i:
                    r, dr = r_zz.get(('O','O'), (0.0, 0.0))
                    t_OO += swtich_function(rij[i,j], r, dr)
                if sym_j == 'P':
                    r, dr = r_zz.get(('O','P'), (0.0, 0.0))
                    t_OP += swtich_function(rij[i,j], r, dr)
            sig_OC = get_bond_tension(('O','C'))
            sig_ON = get_bond_tension(('O','N'))
            sig_OO = get_bond_tension(('O','O'))
            sig_OP = get_bond_tension(('O','P'))
            tension += sig_OC * t_OC + sig_ON * t_ON + sig_OO * t_OO + sig_OP * t_OP
            tensions.append(tension)
            continue
    return np.asarray(tensions)

def molecular_surface_tension(beta, gamma, phi, psi):
    sig_gamma = sigma_gamma * gamma / gamma0
    sig_phi = sigma_phi2 * phi**2
    sig_psi = sigma_psi2 * psi**2
    sig_beta= sigma_beta2 * beta**2
    return sig_gamma + sig_phi + sig_psi + sig_beta

def naive_sasa(mol, rad):
    coords = mol.atom_coords(unit='A')
    charges = mol.atom_charges()
    radius = [rad[ch] for ch in charges]
    sasa = []
    for i in range(mol.natm):
        area = 4 * np.pi * radius[i]
        for j in range(mol.natm):
            if i != j:
                dr = coords[i] - coords[j]
                r = (dr[0]**2 + dr[1]**2 + dr[2]**2)**0.5
                r1 = radius[i]
                r2 = radius[j]
                if r < r1 + r2:
                    overlap = (r1 + r2 - r) / (r1 + r2)
                    area -= overlap * area
        sasa.append(area)
    return np.asarray(sasa)

def get_cds(smdobj):
    mol = smdobj.mol
    n, _, alpha, beta, gamma, _, phi, psi = smdobj.solvent_descriptors
    symbols = [mol.atom_symbol(ia) for ia in range(mol.natm)]
    coords = mol.atom_coords(unit='A')
    if smdobj._solvent.lower() != 'water':
        atm_tension = atomic_surface_tension(symbols, coords, n, alpha, beta, water=False)
        mol_tension = molecular_surface_tension(beta, gamma, phi, psi)
    else:
        logger.info(mol, 'no solvent descriptor is needed for water')
        atm_tension = atomic_surface_tension(symbols, coords, n, alpha, beta, water=True)
        mol_tension = 0.0

    # generate surface for calculating SASA
    rad = radii.VDW + 0.4/radii.BOHR
    surface = pcm.gen_surface(mol, ng=smdobj.sasa_ng, rad=rad)
    area = surface['area']
    gridslice = surface['gslice_by_atom']
    SASA = np.asarray([np.sum(area[p0:p1], axis=0) for p0,p1, in gridslice])
    SASA *= radii.BOHR**2
    mol_cds = mol_tension * np.sum(SASA) / 1000 # in kcal/mol
    atm_cds = np.sum(SASA * atm_tension) / 1000 # in kcal/mol
    return (mol_cds + atm_cds)/hartree2kcal # hartree
