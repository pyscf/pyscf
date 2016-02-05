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


class TDHF(pyscf.lib.StreamObject):
    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.mol = mf.mol
        self.chkfile = mf.chkfile
        self._scf = mf

        self.conv_tol = 1e-9
        self.nstates = 5
        self.singlet = True
        self.lindep = 1e-12
        self.level_shift = 0
        self.max_space = 30
        self.max_cycle = 100
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        self.x = None
        self.y = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        log.info('nstates = %d', self.nstates)
        #if self.singlet:
        #    log.info('Singlet')
        #else:
        #    log.info('Triplet')
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
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        z = z.reshape(eai.shape)

        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        dm = reduce(numpy.dot, (orbv, z, orbo.T))
        vj, vk = self._scf.get_jk(mol, dm, hermi=0)
        v1ao = vj*2 - vk

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
        '''TDHF diagonalization solver
        '''
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
        if self.nstates == 1:
            x1 = [x1.reshape(eai.shape)]
        else:
            x1 = [xi.reshape(eai.shape) for xi in x1]

        self.e = w
        self.x = x1
        return w, x1

## ================
## stream functions
## ================
#    def run_(self, *args, **kwargs):
#        args, kwargs = self._format_args(args, kwargs, (('x0', None),))
#        self.set(**kwargs)
#        self.kernel(*args)
#        return self


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
    td = TDHF(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [ 11.90276464  11.90276464  16.86036434  33.88245939  33.88245939]



