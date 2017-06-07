#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.df import incore
from pyscf.scf import _vhf

# (ij|L)
def aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=1, hermi=0):
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    ao_loc1 = mol.ao_loc_2c()
    ao_loc2 = auxmol.ao_loc_nr()
    nao = ao_loc1[-1]
    ao_loc = numpy.append(ao_loc1, ao_loc2[1:]+nao)
    out = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice,
                                  comp, aosym, ao_loc=ao_loc)
    return out

# (L|ij)
def aux_e1(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=1, hermi=0):
    raise NotImplementedError


def cholesky_eri(mol, auxbasis='weigend+etb', aosym='s1', verbose=0):
    t0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    auxmol = incore.format_aux_basis(mol, auxbasis)

    j2c = incore.fill_2c2e(mol, auxmol)
    log.debug('size of aux basis %d', j2c.shape[0])
    t1 = log.timer('2c2e', *t0)
    low = scipy.linalg.cholesky(j2c, lower=True)
    j2c = None
    t1 = log.timer('Cholesky 2c2e', *t1)

    j3c_ll = aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym=aosym)
    j3c_ss = aux_e2(mol, auxmol, intor='int3c2e_spsp1_spinor', aosym=aosym)
    t1 = log.timer('3c2e', *t1)
    cderi_ll = scipy.linalg.solve_triangular(low, j3c_ll.T, lower=True,
                                             overwrite_b=True)
    cderi_ss = scipy.linalg.solve_triangular(low, j3c_ss.T, lower=True,
                                             overwrite_b=True)
    # solve_triangular return cderi in Fortran order
    cderi = (lib.transpose(cderi_ll.T), lib.transpose(cderi_ss.T))
    log.timer('cholesky_eri', *t0)
    return cderi



if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    cderi = cholesky_eri(mol, verbose=5)
    n2c = mol.nao_2c()
    c2 = .5 / lib.param.LIGHT_SPEED
    def fjk(mol, dm, *args, **kwargs):
        # dm is 4C density matrix
        cderi_ll = cderi[0].reshape(-1,n2c,n2c)
        cderi_ss = cderi[1].reshape(-1,n2c,n2c)
        vj = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)
        vk = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)

        rho =(numpy.dot(cderi[0], dm[:n2c,:n2c].T.reshape(-1))
            + numpy.dot(cderi[1], dm[n2c:,n2c:].T.reshape(-1)*c2**2))
        vj[:n2c,:n2c] = numpy.dot(rho, cderi[0]).reshape(n2c,n2c)
        vj[n2c:,n2c:] = numpy.dot(rho, cderi[1]).reshape(n2c,n2c) * c2**2

        v1 = numpy.einsum('pij,jk->pik', cderi_ll, dm[:n2c,:n2c])
        vk[:n2c,:n2c] = numpy.einsum('pik,pkj->ij', v1, cderi_ll)
        v1 = numpy.einsum('pij,jk->pik', cderi_ss, dm[n2c:,n2c:])
        vk[n2c:,n2c:] = numpy.einsum('pik,pkj->ij', v1, cderi_ss) * c2**4
        v1 = numpy.einsum('pij,jk->pik', cderi_ll, dm[:n2c,n2c:])
        vk[:n2c,n2c:] = numpy.einsum('pik,pkj->ij', v1, cderi_ss) * c2**2
        vk[n2c:,:n2c] = vk[:n2c,n2c:].T.conj()
        return vj, vk

    mf = scf.DHF(mol)
    mf.get_jk = fjk
    mf.direct_scf = False
    ehf1 = mf.scf()
    print(ehf1, -76.08073868516945)

    cderi = cderi[0].reshape(-1,n2c,n2c)
    print(numpy.allclose(cderi, cderi.transpose(0,2,1).conj()))
