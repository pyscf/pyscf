#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import _ctypes
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
import pyscf.gto
from pyscf.df import incore
from pyscf.scf import _vhf

libri = pyscf.lib.load_library('libri')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libri._handle, name))

# (ij|L)
def aux_e2(mol, auxmol, intor='cint3c2e_spinor', aosym='s1', comp=1, hermi=0):
    atm, bas, env = \
            pyscf.gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                    auxmol._atm, auxmol._bas, auxmol._env)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(mol.natm+auxmol.natm)
    nbas = ctypes.c_int(mol.nbas)

    nao = mol.nao_2c()
    naoaux = auxmol.nao_nr()
    if aosym == 's1':
        eri = numpy.empty((nao*nao,naoaux), dtype=numpy.complex)
        fill = _fpointer('RIfill_r_s1_auxe2')
    else:
        eri = numpy.empty((nao*(nao+1)//2,naoaux), dtype=numpy.complex)
        fill = _fpointer('RIfill_r_s2ij_auxe2')
    fintor = _fpointer(intor)
    cintopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)
    libri.RIr_3c2e_auxe2_drv(fintor, fill,
                             eri.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(0), ctypes.c_int(mol.nbas),
                             ctypes.c_int(mol.nbas), ctypes.c_int(auxmol.nbas),
                             ctypes.c_int(1), cintopt,
                             c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                             c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                             c_env.ctypes.data_as(ctypes.c_void_p))
    return eri

# (L|ij)
def aux_e1(mol, auxmol, intor='cint3c2e_spinor', aosym='s1', comp=1, hermi=0):
    pass


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

    j3c_ll = aux_e2(mol, auxmol, intor='cint3c2e_spinor', aosym=aosym)
    j3c_ss = aux_e2(mol, auxmol, intor='cint3c2e_spsp1_spinor', aosym=aosym)
    t1 = log.timer('3c2e', *t1)
    cderi_ll = scipy.linalg.solve_triangular(low, j3c_ll.T, lower=True,
                                             overwrite_b=True)
    cderi_ss = scipy.linalg.solve_triangular(low, j3c_ss.T, lower=True,
                                             overwrite_b=True)
    # solve_triangular return cderi in Fortran order
    cderi = (pyscf.lib.transpose(cderi_ll.T),
             pyscf.lib.transpose(cderi_ss.T))
    log.timer('cholesky_eri', *t0)
    return cderi



if __name__ == '__main__':
    from pyscf import scf
    mol = pyscf.gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    cderi = cholesky_eri(mol, verbose=5)
    n2c = mol.nao_2c()
    c2 = .5 / mol.light_speed
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
