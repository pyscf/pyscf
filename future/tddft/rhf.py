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
from pyscf.tddft import davidson
from pyscf.ao2mo import _ao2mo


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
        self.max_space = 40
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

    def get_vind(self, zs):
        '''Compute Ax'''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))
        vj, vk = self._scf.get_jk(self.mol, dmvo, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk

        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        v1vo = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nvir,0,nocc)).reshape(-1,nvir*nocc)
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        for i, z in enumerate(zs):
            v1vo[i] += eai * z
        return v1vo.reshape(nz,-1)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None:
            nstates = self.nstates
        nvir, nocc = eai.shape
        nroot = min(3, nstates, nvir*nocc)
        x0 = numpy.zeros((nroot, nvir*nocc))
        idx = numpy.argsort(eai.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None):
        '''TDA diagonalization solver
        '''
        self.check_sanity()

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())

        self.e, x1 = pyscf.lib.davidson1(self.get_vind, x0, precond,
                                         tol=self.conv_tol,
                                         nroots=self.nstates, lindep=self.lindep,
                                         max_space=self.max_space,
                                         verbose=self.verbose)
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(eai.shape)*numpy.sqrt(.5),0) for xi in x1]
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def get_vind(self, xys):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(xys)
        dms = numpy.empty((nz*2,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            dmx = reduce(numpy.dot, (orbv, x, orbo.T))
            dmy = reduce(numpy.dot, (orbv, y, orbo.T))
            dms[i   ] = dmx + dmy.T  # AX + BY
            dms[i+nz] = dms[i].T # = dmy + dmx.T  # AY + BX
        vj, vk = self._scf.get_jk(self.mol, dms, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk
        #vhf = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        vhf = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nvir,0,nocc)).reshape(-1,nvir*nocc)
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        for i, z in enumerate(xys):
            x, y = z.reshape(2,-1)
            vhf[i   ] += eai * x  # AX
            vhf[i+nz] += eai * y  # AY
        hx = numpy.hstack((vhf[:nz], -vhf[nz:]))
        return hx.reshape(nz,-1)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            y = x.reshape(2,-1)/diagd
            return y.reshape(-1)
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None:
            nstates = self.nstates
        nov = eai.size
        nroot = min(3, nstates, nov)
        x0 = numpy.zeros((nroot, nov*2))
        idx = numpy.argsort(eai.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        self.check_sanity()

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        nvir = eai.shape[0]

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())

        # We only need positive eigenvalues
        def pickeig(w, v, nroots):
            realidx = numpy.where((w.imag == 0) & (w.real > 0))[0]
            return realidx[w[realidx].real.argsort()[:nroots]]

        w, x1 = davidson.eig(self.get_vind, x0, precond,
                             tol=self.conv_tol,
                             nroots=self.nstates, lindep=self.lindep,
                             max_space=self.max_space, pick=pickeig,
                             verbose=self.verbose)
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nvir,nocc)
            norm = 2*(pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        return self.e, self.xy
RPA = TDHF


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

    td = TDHF(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print td.kernel()[0] * 27.2114
# [ 10.8919234   10.8919234   12.63440705]

