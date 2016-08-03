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


def get_jk(mydf, mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer('Init get_jk', *t1)
    auxmol = mydf.auxmol
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()

    def read_Lpq():
        with mydf.load('Lpq') as Lpq:
            Lpq = lib.unpack_tril(Lpq.value)
        return Lpq.reshape(naux,-1)

    def decompose_dm(dm, ovlp, Lpq):
        e, c = scipy.linalg.eigh(dm, ovlp, type=2)
        pos = e > OCCDROP
        neg = e < -OCCDROP
        cpos = numpy.einsum('ij,j->ij', c[:,pos], numpy.sqrt(e[pos]))
        cpos = numpy.asarray(cpos, order='F')
        cneg = numpy.einsum('ij,j->ij', c[:,neg], numpy.sqrt(-e[neg]))
        cneg = numpy.asarray(cneg, order='F')
        Lpi_pos = lib.dot(Lpq.reshape(-1,nao), cpos).reshape(naux,nao,-1)
        if cneg.shape[1] > 0:
            Lpi_neg = lib.dot(Lpq.reshape(-1,nao), cneg).reshape(naux,nao,-1)
        else:
            Lpi_neg = None
        return (cpos, cneg, Lpi_pos, Lpi_neg)

    def add_j_(vj, jaux, dm, rho_coeff, pqk, Lk, coulG):
        nG = Lk.shape[1]
        pqk = pqk.reshape(-1,nG)
        rho = numpy.dot(dm.T.ravel(), pqk) - numpy.dot(rho_coeff, Lk)
        rho *= coulG
        vj += numpy.dot(pqk, rho)
        jaux += numpy.dot(Lk, rho)
        return vj, jaux

    def short_range_vj(jaux, dm, rho_coeff, Lpq, j3c, j2c):
        nao = dm.shape[0]
        dmtril = lib.pack_tril(dm+dm.T.conj())
        i = numpy.arange(nao)
        dmtril[i*(i+1)//2+i] *= .5
        if isinstance(Lpq, tuple):
            with mydf.load('Lpq') as Lpq:
                Lpq = Lpq.value
        assert(j3c.shape[0] != nao*nao)
        jaux += j2c.dot(rho_coeff)
        jaux = numpy.einsum('ik,i->k', j3c, dmtril) - jaux
        vj = numpy.dot(Lpq.T, jaux)
        if vj.size == nao*nao:
            vj = lib.pack_tril(vj.reshape(nao,nao))
        vj += numpy.einsum('ij,j->i', j3c, rho_coeff)
        return lib.unpack_tril(vj)

    Lpq = read_Lpq()
    rho_coeff = numpy.einsum('kij,ji->k', Lpq.reshape(-1,nao,nao), dm)

    ovlp = mol.intor_symmetric('cint1e_ovlp_sph')
    if with_k and hermi and numpy.einsum('ij,ij->', dm, ovlp) > 0.1:
        Lpq = decompose_dm(dm, ovlp, Lpq)

        def add_k_(vk, dm, pqk, Lk, coulG, Lpq, buf=None):
            nG = Lk.shape[1]
            cpos, cneg, Lpi_pos, Lpi_neg = Lpq
            naux = Lpi_pos.shape[0]
            npos = cpos.shape[1]
            nneg = cneg.shape[1]
            coulG = numpy.sqrt(coulG)

            ipk = lib.dot(cpos.T, pqk.reshape(nao,-1)).reshape(npos,nao,-1)
            pik = numpy.ndarray((nao*npos,nG), buffer=buf)
            pik[:] = ipk.transpose(1,0,2).reshape(nao*npos,-1)
            pik = lib.dot(Lpi_pos.reshape(naux,-1).T, Lk, -1, pik, 1)
            pik *= coulG
            vk += lib.dot(pik.reshape(nao,-1), pik.reshape(nao,-1).T)
            if nneg > 0:
                ipk = lib.dot(cneg.T, pqk.reshape(nao,-1)).reshape(nneg,nao,-1)
                pik = numpy.ndarray((nao*nneg,nG), buffer=buf)
                pik[:] = ipk.transpose(1,0,2).reshape(nao*nneg,-1)
                pik = lib.dot(Lpi_neg.reshape(naux,-1).T, Lk, -1, pik, 1)
                pik *= coulG
                vk -= lib.dot(pik.reshape(nao,-1), pik.reshape(nao,-1).T)
            return vk

        def short_range_vk(dm, Lpq, j3c, j2c, hermi):
            j3c = lib.unpack_tril(j3c, axis=0).reshape(nao,-1)
            cpos, cneg, Lpi_pos, Lpi_neg = Lpq
            #:tmp = numpy.einsum('Lpi,LM->piM', Lpi_pos, j2c)
            #:tmp = numpy.einsum('Lpi,qiL->pq', Lpi_pos, tmp)
            #:vk = numpy.einsum('Lpi,qLi->pq', Lpi_pos, pLi.reshape(nao,naux,-1))*2 - tmp

            pLi = lib.dot(j3c.T, cpos)
            vk  = lib.dot(pLi.reshape(nao,-1),
                          numpy.asarray(Lpi_pos.transpose(1,0,2).reshape(nao,-1), order='C').T)
            tmp = lib.dot(Lpi_pos.reshape(naux,-1).T, j2c)
            kaux = lib.dot(numpy.asarray(Lpi_pos.transpose(1,2,0).reshape(nao,-1),order='C'),
                           tmp.reshape(nao,-1).T)

            if cneg.shape[1] > 0:
                pLi = lib.dot(j3c.T, cneg)
                vk -= lib.dot(pLi.reshape(nao,-1),
                              numpy.asarray(Lpi_neg.transpose(1,0,2).reshape(nao,-1),order='C').T)
                tmp = lib.dot(Lpi_neg.reshape(naux,-1).T, j2c)
                kaux -= lib.dot(numpy.asarray(Lpi_neg.transpose(1,2,0).reshape(nao,-1),order='C'),
                                tmp.reshape(nao,-1).T)
            vk = vk + vk.T - kaux
            return vk

    else:  # hermi == 0

        def add_k_(vk, dm, pqk, Lk, coulG, Lpq, buf=None):
            nG = Lk.shape[1]
            #:with mydf.load('Lpq') as Lpq:
            #:    Lpq = lib.unpack_tril(Lpq.value).reshape(naux,-1)
            #:pqk -= numpy.dot(Lpq.T, Lk)
            #:v4 = reduce(numpy.dot, (pqk*coulG, pqk.T))
            #:v4 = v4.reshape((nao,)*4)
            #:vk += numpy.einsum('pqrs,qr->ps', v4, dm)
            #:return vk
            pqk = numpy.asarray(pqk.reshape(nao,nao,nG), order='C')
            pqk = lib.dot(Lpq.reshape(naux,-1).T, Lk, -1, pqk.reshape(-1,nG), 1)
            pqk *= numpy.sqrt(coulG)
            ipk = lib.dot(dm, pqk.reshape(nao,-1)).reshape(nao,nao,-1)
            pik = numpy.ndarray((nao,nao,nG), buffer=buf)
            pik[:] = ipk.transpose(1,0,2)
            vk += lib.dot(pqk.reshape(nao,-1), pik.reshape(nao,-1).T)
            return vk

        def short_range_vk(dm, Lpq, j3c, j2c, hermi):
            j3c = lib.unpack_tril(j3c, axis=0).reshape(nao,-1)
            #:Lpq = Lpq.reshape(naux,-1)
            #:j3c = j3c.reshape(-1,naux)
            #:v4 = numpy.dot(j3c,Lpq)
            #:v4 = v4 + v4.T
            #:v4 -= reduce(numpy.dot, (Lpq.T, j2c, Lpq))
            #:v4 = v4.reshape((nao,)*4)
            #:vk = numpy.einsum('pqrs,qr->ps', v4, dm)
            #:return vk
            iLp = lib.dot(dm, Lpq.reshape(-1,nao).T).reshape(-1,nao)
            vk  = lib.dot(j3c, iLp)

            Lip = iLp.reshape(nao,naux,nao).transpose(1,0,2)
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1))
            vk1 = lib.dot(tmp.reshape(-1,nao).T,
                          numpy.asarray(Lip.reshape(-1,nao), order='C'))
            Lip = iLp = tmp = None

            if hermi:
                vk = vk + vk.T
            else:
                iLp = lib.dot(dm.T, Lpq.reshape(-1,nao).T).reshape(-1,nao)
                vk += lib.dot(iLp.T, j3c.T)
            vk -= vk1
            return vk

    sublk = min(max(16, int(mydf.max_memory*1e6/16/nao**2)), 8192)
    pikbuf = numpy.empty(nao*nao*sublk)

    vj = 0
    jaux = 0
    vk = 0
    for pqkR, LkR, pqkI, LkI, coulG \
            in mydf.pw_loop(mol, auxmol, mydf.gs, mydf.max_memory):
        if with_j:
            vj, jaux = add_j_(vj, jaux, dm, rho_coeff, pqkR, LkR, coulG)
            vj, jaux = add_j_(vj, jaux, dm, rho_coeff, pqkI, LkI, coulG)
        if with_k:
            vk = add_k_(vk, dm, pqkR, LkR, coulG, Lpq, pikbuf)
            vk = add_k_(vk, dm, pqkI, LkI, coulG, Lpq, pikbuf)
    pqkR = LkR = pqkI = LkI = coulG = pikbuf = None
    t1 = log.timer('pw jk', *t1)

    for j3c, j2c in mydf.sr_loop(mol, auxmol, mydf.max_memory):
        if with_j:
            vj = vj.reshape(nao,nao)
            vj += short_range_vj(jaux, dm, rho_coeff, Lpq, j3c, j2c)
        if with_k:
            vk += short_range_vk(dm, Lpq, j3c, j2c, hermi)
    j3c = j2c = None
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

