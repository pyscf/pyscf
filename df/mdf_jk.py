#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import copy
import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df import mdf

#
# Split the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

OCCDROP = 1e-12

def density_fit(mf, auxbasis=None, gs=(10,10,10), with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        gs : tuple
            number of grids in each (+)direction
        with_df : MDF object
    '''
    mf_class = mf.__class__

    if with_df is None:
        with_df = mdf.MDF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        with_df.gs = gs

    class XDFHF(mf_class):
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])

        def get_jk(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                return get_jk(self.with_df, mol, dm, hermi, self.opt,
                              True, True)
            else:
                return mf_class.get_jk(self, mol, dm, hermi)

        def get_j(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                return get_jk(self.with_df, mol, dm, hermi, self.opt,
                              True, False)[0]
            else:
                return mf_class.get_j(self, mol, dm, hermi)

        def get_k(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                return get_jk(self.with_df, mol, dm, hermi, self.opt,
                              False, True)[1]
            else:
                return mf_class.get_k(self, mol, dm, hermi)

    return XDFHF()


def get_jk(mydf, mol, dms, hermi=1, vhfopt=None, with_j=True, with_k=True):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer('Init get_jk', *t1)
    auxmol = mydf.auxmol
    naux = auxmol.nao_nr()

    if len(dms) == 0:
        return [], []
    elif isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        nset = 1
        dms = [dms]
    else:
        nset = len(dms)
    nao = dms[0].shape[0]

    def add_j_(vj, dm, pqk, coulG):
        nG = coulG.shape[0]
        pqk = pqk.reshape(-1,nG)
        rho = numpy.dot(dm.T.ravel(), pqk)
        rho *= coulG
        vj += numpy.dot(pqk, rho)
        return vj

    def add_k_(vk, dm, pqk, coulG, buf=None):
        nG = pqk.shape[-1]
        pqk *= numpy.sqrt(coulG)
        ipk = lib.dot(dm, pqk.reshape(nao,-1)).reshape(nao,nao,-1)
        pik = numpy.ndarray((nao,nao,nG), buffer=buf)
        pik[:] = ipk.transpose(1,0,2)
        vk += lib.dot(pqk.reshape(nao,-1), pik.reshape(nao,-1).T)
        return vk

    sublk = min(max(16, int(mydf.max_memory*1e6/16/nao**2)), 8192)
    pikbuf = numpy.empty(nao*nao*sublk)

    vj = numpy.zeros((nset,nao*nao))
    vk = numpy.zeros((nset,nao,nao))
    for pqkR, LkR, pqkI, LkI, coulG \
            in mydf.pw_loop(mol, auxmol, mydf.gs, max_memory=mydf.max_memory):
        for i in range(nset):
            if with_j:
                add_j_(vj[i], dms[i], pqkR, coulG)
                add_j_(vj[i], dms[i], pqkI, coulG)
            if with_k:
                add_k_(vk[i], dms[i], pqkR, coulG, pikbuf)
                add_k_(vk[i], dms[i], pqkI, coulG, pikbuf)
    pqkR = LkR = pqkI = LkI = coulG = pikbuf = None
    t1 = log.timer('pw jk', *t1)

    def short_range_j(dmtril, Lpq, j3c):
        jaux = numpy.einsum('ki,i->k', j3c, dmtril)
        rho_coeff = numpy.einsum('ki,i->k', Lpq, dmtril)
        vj  = numpy.dot(Lpq.T, jaux)
        vj += numpy.dot(j3c.T, rho_coeff)
        return vj

    def short_range_k(dm, Lpq, j3c):
        Lpq = Lpq.reshape(-1,nao)
        j3c = j3c.reshape(-1,nao)
        iLp = lib.dot(dm, Lpq.T).reshape(-1,nao)
        vk  = lib.dot(j3c.T, iLp)
        if hermi:
            vk = vk + vk.T
        else:
            iLp = lib.dot(dm.T, Lpq.T).reshape(-1,nao)
            vk += lib.dot(iLp.T, j3c)
        return vk

    if with_j:
        i = numpy.arange(nao)
        def dm_for_vj_tril(dm):
            dmtril = lib.pack_tril(dm+dm.T.conj())
            dmtril[i*(i+1)//2+i] *= .5
            return dmtril
        dmstril = [dm_for_vj_tril(x) for x in dms]
        vj1 = numpy.zeros((nset,nao*(nao+1)//2))

    for Lpq, j3c in mydf.sr_loop(mol, auxmol, mydf.max_memory):
        if with_j:
            for i in range(nset):
                vj1[i] += short_range_j(dmstril[i], Lpq, j3c)
        if with_k:
            Lpq = lib.unpack_tril(Lpq)
            j3c = lib.unpack_tril(j3c).transpose(1,0,2)
            for i in range(nset):
                vk[i] += short_range_k(dms[i], Lpq, j3c)
    if with_j:
        vj = vj.reshape(-1,nao,nao)
        vj += lib.unpack_tril(vj1)

    if nset == 1:
        vj = vj[0]
        vk = vk[0]
    t1 = log.timer('sr jk', *t1)
    return vj, vk


if __name__ == '__main__':
    from pyscf import scf
    from pyscf.df import addons

    mol = gto.M(
        atom = '''#Fe    1.3    2.       3.
        fe 0 0 0;
        fe 0 0 2
                  #Fe    0    1.3       0.
                  #Fe    1.    1.       1.''',
        basis = 'ccpvdz',
        #verbose=4,
    )
    mf0 = scf.RHF(mol)
    dm = mf0.get_init_guess(mol, mf0.init_guess)
    #dm = numpy.random.random(dm.shape)
    #dm = dm + dm.T
    #mf0.max_cycle = 0
    mf0.kernel()
    dm1 = mf0.make_rdm1()
    #dm1 = numpy.random.random(dm.shape)
    #dm1 = dm1 + dm1.T

    #auxbasis = {'C': gto.expand_etbs([[0, 10, .5, 2.], [1, 7, .3, 2.]])}
    auxbasis = 'weigend'
    #auxbasis = 'sto3g'
    #auxbasis = 'tzp'
    #auxbasis = {'Fe': gto.mole.uncontract(gto.basis.load('tzp', 'Fe'))}
    #auxbasis = mdf._default_basis(mol, addons.aug_etb_for_dfbasis(mol, beta=2.2, start_at=0))
    #auxbasis = addons.aug_etb_for_dfbasis(mol, beta=2.2, start_at=0)
    auxbasis = None
    #auxbasis = addons.aug_etb_for_dfbasis(mol, beta=2.2, start_at=0)
    #auxbasis = mdf._default_basis(mol)
    #mf = scf.density_fit(scf.RHF(mol), auxbasis)
    #mf = with_poisson_(scf.RHF(mol), auxbasis, gs=(50,302))
    mf = mdf.MDF(mol).set(auxbasis=auxbasis, gs=(5,)*3).update_mf(scf.RHF(mol))
    vj0, vk0 = mf0.get_jk(mol, dm)
    vj1, vk1 = mf.get_jk(mol, dm)
    print('J', numpy.einsum('ij,ij->', vj1, dm1), 'ref=2495.9843578661084',
          'exact', numpy.einsum('ij,ij->', vj0, dm1))
    print('K', numpy.einsum('ij,ij->', vk1, dm1), 'ref=426.19988812673307',
          'exact', numpy.einsum('ij,ij->', vk0, dm1))
    print('J-K', numpy.einsum('ij,ij->', vj1-vk1*.5, dm1), 'ref=2282.8844138027462',
          'exact', numpy.einsum('ij,ij->', vj0-vk0*.5, dm1))
    exit()
    print(mf.scf())

