#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
1D periodic HF method based on real-space lattice summation.

In this implementation, nuclear attraction and Coulomb integrals are computed
based on real-space lattice summation.  Because of the long-range character of
Coulomb interaction, large number of images are required to converge the total
energy.

This implementation is different to the algorithm in PBC module.  In PBC module,
Coulomb interactions are evaluated for density without background charge
(Coulomb kernel without G=0).  Coulomb potential of zero-charge density decays
fast in space and therefore small number of repeated images are required for
Coulomb integrals.
'''

import numpy
from pyscf import gto, scf, ao2mo, lib

def pbc1d_mf(unit_cell, T0, nimgs):
    T0 = numpy.asarray(T0)
    atom = []
    for n in range(-nimgs, nimgs+1):
        for i in range(unit_cell.natm):
            atom.append((unit_cell.atom_symbol(i), unit_cell.atom_coord(i)+n*T0))
    molN = unit_cell.copy()
    molN.atom = atom
    molN.build(0, 0)

    nao1 = unit_cell.nao_nr()
    naoN = molN.nao_nr()
    c = numpy.eye(nao1)
    c = numpy.vstack([c]*(nimgs*2+1))

    def energy_nuc(mol):
        Ts = numpy.array([T0*i for i in range(-nimgs, nimgs+1)])
        chargs = mol.atom_charges()
        coords = mol.atom_coords()
        enuc = 0
        for i, qi in enumerate(chargs):
            ri = coords[i]
            for j, qj in enumerate(chargs):
                rj = coords[j]
                r1 = ri-rj + Ts
                r = numpy.sqrt(numpy.einsum('ji,ji->j', r1, r1))
                r[r<1e-7] = 1e200
                enuc += (qi * qj / r).sum()
        return enuc * 0.5

    def madelung(nimgs):
        e = 0
        t = numpy.linalg.norm(T0)
        for j in range(nimgs*2+1):
            if j != nimgs:
                e += 1. / abs(t*(nimgs-j))
        return -e

    enuc = energy_nuc(unit_cell)
    mad = madelung(nimgs)
    lib.logger.debug(unit_cell, 'Enuc = %s  Madelung const = %s', enuc, mad)

# lattice sum for eri = \sum_{abc} (i^0 j^a | k^b l^c)
    nbas1 = unit_cell.nbas
    nbasN = molN.nbas
    nao_pair = naoN*(naoN+1)//2
    blk = int(max(4, 1e9/nao_pair/nao1**2))*nbas1
    eri = 0
    cintopt = gto.moleintor.make_cintopt(molN._atm, molN._bas, molN._env, 'int2e_sph')
    for p0, p1 in lib.prange(0, nbasN, blk):
        eri1 = gto.moleintor.getints('int2e_sph', molN._atm, molN._bas, molN._env,
                                     shls_slice=[p0,p1, nimgs*nbas1,(nimgs+1)*nbas1, 0,nbasN, 0,nbasN],
                                     aosym='s2kl', cintopt=cintopt)
        n_img = (p1-p0) // nbas1
        eri1 = eri1.reshape(n_img,nao1,nao1,nao_pair)
        for i in range(n_img):
            eri += eri1[i]
    eri = ao2mo._ao2mo.nr_e2(eri.reshape(-1,nao_pair),
                             numpy.asarray(c, order='F'),
                             (0,nao1,0,nao1), aosym='s2', mosym='s1')
    eri = eri.reshape([nao1]*4)
    def get_veff(molN, dm, *args, **kwargs):
        vj = numpy.einsum('ijkl,ji->kl', eri, dm)
        vk = numpy.einsum('ijkl,li->kj', eri, dm)
        vk += mad * ovlp.dot(dm).dot(ovlp)
        vhf = vj - vk * .5
        lib.logger.debug(unit_cell, 'h1/j/k = %s',
                         ((h1e*dm).sum(), (vj*dm).sum()*.5, (vk*dm).sum()*.5))
        return vhf

# lattice sum for Vnuc = \sum_{A in unit_cell} \sum_a <i^0 | Z_A/r_A | j^a>
    shls_slice = [nimgs*nbas1,(nimgs+1)*nbas1,0,nbasN]
    h1e = molN.intor('int1e_kin', shls_slice=shls_slice[:4]).dot(c)
    for i in range(unit_cell.natm):
        molN.set_rinv_orig(unit_cell.atom_coord(i))
        h1e -= c.T.dot(molN.intor('int1e_rinv')).dot(c) * unit_cell.atom_charge(i)
    ovlp = molN.intor('int1e_ovlp', shls_slice=shls_slice[:4]).dot(c)

    mol = gto.M()
    mol.nelectron = unit_cell.nelectron
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: ovlp
    mf.get_veff = get_veff
    mf.energy_nuc = lambda *args: enuc
    return mf

if __name__ == '__main__':
    R = 3.6
    nimgs = 200
    unit_cell = gto.M(atom='H 0 0 0; H 1.8 0 0', unit='bohr')#, basis='3-21g')
    mf = scf.RHF(unit_cell).run()

    mf = pbc1d_mf(unit_cell, (R,0,0), nimgs).run()

    from pyscf.pbc import gto, scf, tools, df
    cell = gto.M(atom='H 0 0 0; H 1.8 0 0', a=numpy.eye(3)*R, unit='bohr',
                 dimension=1)#, basis='321g')
    mf = scf.KRHF(cell, cell.make_kpts([1,1,1])).density_fit().run()
