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
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo


class TDA(lib.StreamObject):
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
        self.max_space = 50
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
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    def get_vind(self, mf):
        '''Compute Ax'''
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (mf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()

        def vind(zs):
            nz = len(zs)
            dmvo = numpy.empty((nz,nao,nao))
            for i, z in enumerate(zs):
                dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))
            vj, vk = mf.get_jk(self.mol, dmvo, hermi=0)
            if self.singlet:
                vhf = vj*2 - vk
            else:
                vhf = -vk

            #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
            v1vo = _ao2mo.nr_e2(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
            for i, z in enumerate(zs):
                v1vo[i] += eai * z
            return v1vo.reshape(nz,-1)

        return vind

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None: nstates = self.nstates
        nov = eai.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov))
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
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())
        vind = self.get_vind(self._scf)

        self.e, x1 = lib.davidson1(vind, x0, precond,
                                   tol=self.conv_tol,
                                   nroots=self.nstates, lindep=self.lindep,
                                   max_space=self.max_space,
                                   verbose=self.verbose)[1:]
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(eai.shape)*numpy.sqrt(.5),0) for xi in x1]
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def get_vind(self, mf):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (mf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()

        def vind(xys):
            nz = len(xys)
            dms = numpy.empty((nz,nao,nao))
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                dmx = reduce(numpy.dot, (orbv, x, orbo.T))
                dmy = reduce(numpy.dot, (orbo, y.T, orbv.T))
                dms[i] = dmx + dmy  # AX + BY
            vj, vk = mf.get_jk(self.mol, dms, hermi=0)
            if self.singlet:
                vhf = vj*2 - vk
            else:
                vhf = -vk

            nov = nocc*nvir
            vhfvo = _ao2mo.nr_e2(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nov)
            vhfov = _ao2mo.nr_e2(vhf, mo_coeff, (0,nocc,nocc,nmo))
            vhfov = vhfov.reshape(-1,nocc,nvir).transpose(0,2,1).reshape(-1,nov)
            hx = numpy.empty((nz,nov*2))
            for i, z in enumerate(xys):
                x, y = z.reshape(2,-1)
                hx[i,:nov] = vhfvo[i] + eai * x  # AX
                hx[i,nov:] =-vhfov[i] - eai * y  #-AY
            return hx

        return vind

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            y = x.reshape(2,-1)/diagd
            return y.reshape(-1)
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None: nstates = self.nstates
        nov = eai.size
        nroot = min(nstates, nov)
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
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        nvir = eai.shape[0]

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())
        vind = self.get_vind(self._scf)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < 1e-6) & (w.real > 0))[0]
            idx = realidx[w[realidx].real.argsort()]
            return w[idx].real, v[:,idx].real, idx

        w, x1 = lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=self.nstates, lindep=self.lindep,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=self.verbose)[1:]
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nvir,nocc)
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
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
    td.verbose = 4
    print(td.kernel()[0] * 27.2114)
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 11.01747918  11.01747918  13.16955056]

    td = TDHF(mf)
    td.verbose = 4
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440705]

