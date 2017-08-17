import time
import numpy
import numpy as np

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
from pyscf.cc import uccsd

#einsum = np.einsum
einsum = lib.einsum

# This is unrestricted (U)MP2, i.e. spin-orbital form.

def kernel(mp, mo_coeff, verbose=logger.NOTE):
    eris = mp.ao2mo(mo_coeff)
    mo_e = eris.fock.diagonal()
    nocc = mp.nocc
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t2 = eris.oovv/eijab
    emp2 = 0.25*einsum('ijab,ijab',t2,eris.oovv.conj()).real
    return emp2, t2


class UMP2(pyscf.lib.StreamObject):

    def __init__(self, mf, frozen=[[],[]]):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mf.mo_coeff
        self.mo_occ   = mf.mo_occ
        self.emp2 = None
        self.e_corr = None
        self.t2 = None
        self._nocc = None
        self._nmo = None

        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        nocca, noccb = self.get_nocc()
        return nocca + noccb

    @property
    def nmo(self):
        nmoa, nmob = self.get_nmo()
        return nmoa + nmob

    get_nocc = uccsd.get_nocc
    get_nmo = uccsd.get_nmo

    def kernel(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff is not given.\n'
                     'You may need mf.kernel() to generate it.')
            raise RuntimeError

        self.emp2, self.t2 = \
                kernel(self, mo_coeff, verbose=self.verbose)
        logger.log(self, 'UMP2 energy = %.15g', self.emp2)
        self.e_corr = self.emp2
        return self.emp2, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)


class _ERIS:
    def __init__(self, cc, mo_coeff=None, 
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = uccsd.get_umoidx(cc)
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = [cc.mo_coeff[0][:,moidx[0]], 
                                        cc.mo_coeff[1][:,moidx[1]]]
        else:
            self.mo_coeff = mo_coeff = [mo_coeff[0][:,moidx[0]], 
                                        mo_coeff[1][:,moidx[1]]]

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        self.fock, so_coeff, spin = uccsd.uspatial2spin(cc, moidx, mo_coeff)

        log = logger.Logger(cc.stdout, cc.verbose)

        orbo = so_coeff[:,:nocc]
        orbv = so_coeff[:,nocc:]
        eri = ao2mofn(cc._scf.mol, (orbo,orbv,orbo,orbv), compact=0)
        if mo_coeff[0].dtype == np.float: eri = eri.real
        eri = eri.reshape((nocc,nvir,nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                if spin[i] != spin[nocc+a]:
                    eri[i,a,:,:] = eri[:,:,i,a] = 0.
        eri1 = eri - eri.transpose(0,3,2,1) 
        eri1 = eri1.transpose(0,2,1,3) 

        self.dtype = eri1.dtype
        self.oovv = eri1

        log.timer('MP2 integral transformation', *cput0)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz' 
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    print(mf.scf())

    # Freeze 1s orbitals
    frozen = [[0,1],[0,1]]
    # also acceptable
    #frozen = 4
    pt = UMP2(mf, frozen=frozen)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)
