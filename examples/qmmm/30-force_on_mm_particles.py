#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
The force from QM region acting on the background MM particles.
'''

import numpy
from pyscf import gto, scf, mp, qmmm

mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
            ''',
            basis='3-21g')

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.1

def force(dm):
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    qm_coords = mol.atom_coords()
    qm_charges = mol.atom_charges()
    dr = qm_coords[:,None,:] - coords
    r = numpy.linalg.norm(dr, axis=2)
    g = numpy.einsum('r,R,rRx,rR->Rx', qm_charges, charges, dr, r**-3)

    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(charges):
        with mol.with_rinv_origin(coords[i]):
            v = mol.intor('int1e_iprinv')
        f =(numpy.einsum('ij,xji->x', dm, v) +
            numpy.einsum('ij,xij->x', dm, v.conj())) * -q
        g[i] += f

    # Force = -d/dR
    return -g

# The force from HF electron density
# Be careful with the unit of the MM particle coordinates. The gradients are
# computed in the atomic unit.
mf = qmmm.mm_charge(scf.RHF(mol), coords, charges, unit='Bohr').run()
e1_mf = mf.e_tot
dm = mf.make_rdm1()
mm_force_mf = force(dm)
print('HF force:')
print(mm_force_mf)

# Verify HF force
coords[0,0] += 1e-3
mf = qmmm.mm_charge(scf.RHF(mol), coords, charges, unit='Bohr').run()
e2_mf = mf.e_tot
print(-(e2_mf-e1_mf)/1e-3, '==', mm_force_mf[0,0])


#
# For post-HF methods, the response of HF orbitals needs to be considered in
# the analytical gradients. It is similar to the gradients code implemented in
# the module pyscf.grad.
#
# Below we use MP2 gradients as example to demonstrate how to include the
# orbital response effects in the force for MM particles.
#

# Based on the grad_elec function in pyscf.grad.mp2
def make_rdm1_with_orbital_response(mp):
    import time
    from pyscf import lib
    from pyscf.grad.mp2 import _response_dm1, _index_frozen_active, _shell_prange
    from pyscf.mp import mp2
    from pyscf.ao2mo import _ao2mo
    log = lib.logger.new_logger(mp)
    time0 = time.clock(), time.time()
    mol = mp.mol

    log.debug('Build mp2 rdm1 intermediates')
    d1 = mp2._gamma1_intermediates(mp, mp.t2)
    doo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)

    with_frozen = not (mp.frozen is None or mp.frozen is 0)
    OA, VA, OF, VF = _index_frozen_active(mp.get_frozen_mask(), mp.mo_occ)
    orbo = mp.mo_coeff[:,OA]
    orbv = mp.mo_coeff[:,VA]
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]

# Partially transform MP2 density matrix and hold it in memory
# The rest transformation are applied during the contraction to ERI integrals
    part_dm2 = _ao2mo.nr_e2(mp.t2.reshape(nocc**2,nvir**2),
                            numpy.asarray(orbv.T, order='F'), (0,nao,0,nao),
                            's1', 's1').reshape(nocc,nocc,nao,nao)
    part_dm2 = (part_dm2.transpose(0,2,3,1) * 4 -
                part_dm2.transpose(0,3,2,1) * 2)

    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    Imat = numpy.zeros((nao,nao))

# 2e AO integrals dot 2pdm
    max_memory = max(0, mp.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
            dm2buf+= lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:,ip0:ip1])
            dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0,1,3,2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2buf[:,:,diagidx] *= .5

            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None
            dm2buf = None
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

# Recompute nocc, nvir to include the frozen orbitals and make contraction for
# the 1-particle quantities, see also the kernel function in ccsd_grad module.
    mo_coeff = mp.mo_coeff
    mo_energy = mp._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mp.mo_occ > 0)
    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mp._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mp._scf.get_veff(mp.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mp, Xvo)

    # Transform to AO basis
    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    dm1 += mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)
    return dm1


# The force from MP2 electron density (including orbital response)
m = mp.MP2(mf).run()
e1_mp2 = m.e_tot
dm = make_rdm1_with_orbital_response(m)
mm_force_mp2 = force(dm)
print('MP2 force:')
print(mm_force_mp2)

# Verify MP2 force
coords[0,0] += 1e-3
mf = qmmm.mm_charge(scf.RHF(mol), coords, charges, unit='Bohr').run()
m = mp.MP2(mf).run()
e2_mp2 = m.e_tot
print(-(e2_mp2-e1_mp2)/1e-3, '==', mm_force_mp2[0,0])
