#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from functools import reduce
import numpy
import pyscf.lib


class TDA(pyscf.lib.StreamObject):
    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.mol = mf.mol
        self.chkfile = mf.chkfile
        self._scf = mf

        self.conv_tol = 1e-9
        self.nstates = 3
        self.singlet = True
        self.lindep = 1e-12
        self.level_shift = 0
        self.max_space = 30
        self.max_cycle = 100
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        # xy = (X,Y), normlized to 1/2: 2(XX-YY) = 1
        # In TDA or TDHF, Y = 0
        self.e = None
        self.xy = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        log.info('nstates = %d', self.nstates)
        if self.singlet:
            log.info('Singlet')
        else:
            log.info('Triplet')
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_space = %d', self.max_space)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscf.lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    # z_{ai} = X_{ai}
    def get_vind(self, z):
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        z = z.reshape(eai.shape)
        dmvo = reduce(numpy.dot, (orbv, z, orbo.T))
        vj, vk = self._scf.get_jk(mol, dmvo, hermi=0)

        if self.singlet:
            v1ao = vj*2 - vk
        else:
            v1ao = -vk

        v1vo = reduce(numpy.dot, (orbv.T, v1ao, orbo))
        v1vo = eai*z + v1vo
        return v1vo.ravel()

    def get_precond(self, eai):
        def precond(x, e, x0):
            diagd = eai.ravel() - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def kernel(self, x0=None):
        '''TDA diagonalization solver
        '''
        self.check_sanity()

        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.copy()

        if x0 is None:
            nroot = min(3, self.nstates)
            x0 = numpy.zeros((nroot, nvir*nocc))
            idx = numpy.argsort(eai.ravel())
            for i in range(nroot):
                x0[i,idx[i]] = 1  # lowest excitations

        precond = self.get_precond(eai)

        w, x1 = pyscf.lib.davidson(self.get_vind, x0, precond,
                                   tol=self.conv_tol,
                                   nroots=self.nstates, lindep=self.lindep,
                                   max_space=self.max_space,
                                   verbose=self.verbose)
        self.e = w
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        if self.nstates == 1:
            self.xy = [(x1.reshape(eai.shape)*numpy.sqrt(.5),0)]
        else:
            self.xy = [(xi.reshape(eai.shape)*numpy.sqrt(.5),0) for xi in x1]
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    # z_{ai} = [(A-B)^{-1/2}(X+Y)]_{ai}
    def get_vind(self, z):
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        z = z.reshape(eai.shape)
        dmvo = reduce(numpy.dot, (orbv, z, orbo.T))

        raise NotImplementedError
        if self.singlet:
            vj, vk = self._scf.get_jk(mol, (dmvo, dmvo.T))

        else: # Triplet
            pass

        return v1vo.ravel()

    def kernel(self, x0=None):
        '''TDHF diagonalization solver
        '''
        w2, x1 = TDA.kernel(self, x0)
        self.e = numpy.sqrt(w2)

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = numpy.sqrt(eai)
        def norm_xy(w, z):
            zp = eai * z[0].reshape(eai.shape)
            zm = w/eai * z[0].reshape(eai.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = 2*(pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return x*norm,y*norm

        self.xy = [norm_xy(self.e[i], z) for i, z in enumerate(x1)]

        return self.e, self.xy



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print td.kernel()[0] * 27.2114
# [ 11.01747918  11.01747918  13.16955056]
