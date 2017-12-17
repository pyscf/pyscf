#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import lib
from pyscf.cc import ccsd_grad
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

# Only works with canonical orbitals
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           mf_grad=None, verbose=lib.logger.INFO):
    d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris,
                                    fd2intermediate, True)
    return ccsd_grad.kernel(mycc, t1, t2, l1, l2, eris, atmlst, mf_grad,
                            d1, d2, verbose)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf.cc import ccsd_t
    from pyscf import grad
    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda

    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.    )],
            [1   , (0. ,-0.757  ,-0.587)],
            [1   , (0. , 0.757  ,-0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = kernel(mycc, t1, t2, l1, l2, eris=eris, mf_grad=grad.RHF(mf))
    print(g1)
#O      0.0000000000            0.0000000000           -0.0112045345
#H      0.0000000000            0.0234464201            0.0056022672
#H      0.0000000000           -0.0234464201            0.0056022672


    mol = gto.M(
        verbose = 0,
        atom = '''
H         -1.90779510     0.92319522     0.08700656
H         -1.08388168    -1.61405643    -0.07315086
H          2.02822318    -0.61402169     0.09396693
H          0.96345360     1.30488291    -0.10782263
               ''',
        unit='bohr',
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf0+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = kernel(mycc, t1, t2, l1, l2, eris=eris, mf_grad=grad.RHF(mf))
    print(g1)
#CCSD
#H   0.0113620114            0.0664344363            0.0029855587
#H   0.0528858926           -0.0483942979           -0.0033960631
#H   0.0109676543           -0.0827248466            0.0074005299
#H  -0.0752155583            0.0646847082           -0.0069900255
#
#CCSD(T) gradient:
#
#H   0.0112264011            0.0658917731            0.0029936671
#H   0.0525667206           -0.0481602008           -0.0033751503
#H   0.0107197424           -0.0823677005            0.0073804163
#H  -0.0745128642            0.0646361282           -0.0069989330 
