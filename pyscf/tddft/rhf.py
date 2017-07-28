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
from pyscf.dft import numint
from pyscf.scf import hf_symm
from pyscf.scf.newton_ah import _gen_rhf_response


def gen_tda_hop(mf, fock_ao=None, singlet=True, wfnsym=None, max_memory=2000):
    '''(A+B)x
    
    Kwargs:
        wfnsym : int
            Point group symmetry for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

    def vind(zs):
        nz = len(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs).reshape(-1,nvir,nocc)
            zs[:,sym_forbid] = 0
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            # *2 for double occupancy
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dmvo)
        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in v1ao])
        v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir,nocc)
        for i, z in enumerate(zs):
            v1vo[i]+= numpy.einsum('ps,sq->pq', fvv, z.reshape(nvir,nocc))
            v1vo[i]-= numpy.einsum('ps,rp->rs', foo, z.reshape(nvir,nocc))
        if wfnsym is not None and mol.symmetry:
            v1vo[:,sym_forbid] = 0
        return v1vo.reshape(nz,-1)

    return vind, hdiag


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
        self.wfnsym = None
        self.lindep = 1e-12
        self.level_shift = 0
        self.max_space = 50
        self.max_cycle = 100
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        # xy = (X,Y), normlized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.e = None
        self.xy = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        if self.singlet:
            log.info('nstates = %d singlet', self.nstates)
        else:
            log.info('nstates = %d triplet', self.nstates)
        log.info('wfnsym = %s', self.wfnsym)
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

    def gen_vind(self, mf):
        '''Compute Ax'''
        return gen_tda_hop(mf, singlet=self.singlet, wfnsym=self.wfnsym,
                           max_memory=self.max_memory)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        eai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])

        if wfnsym is not None and mf.mol.symmetry:
            orbsym = hf_symm.get_orbsym(mf.mol, mf.mo_coeff)
            sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym
            eai[sym_forbid] = 1e99

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
        self.dump_flags()

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)
        self.e, x1 = lib.davidson1(vind, x0, precond,
                                   tol=self.conv_tol,
                                   nroots=self.nstates, lindep=self.lindep,
                                   max_space=self.max_space,
                                   verbose=self.verbose)[1:]

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nvir,nocc)*numpy.sqrt(.5),0) for xi in x1]
        #TODO: analyze CIS wfn point group symmetry
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def gen_vind(self, mf):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        if wfnsym is not None and mol.symmetry:
            orbsym = hf_symm.get_orbsym(mol, mo_coeff)
            sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym

        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        #fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
        #foo = fock[occidx[:,None],occidx]
        #fvv = fock[viridx[:,None],viridx]
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        e_ai = hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        if wfnsym is not None and mol.symmetry:
            hdiag[sym_forbid] = 0
        hdiag = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

        mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
        vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

        def vind(xys):
            nz = len(xys)
            if wfnsym is not None and mol.symmetry:
                # shape(nz,2,nvir,nocc): 2 ~ X,Y
                xys = numpy.copy(zs).reshape(nz,2,nvir,nocc)
                xys[:,:,sym_forbid] = 0
            dms = numpy.empty((nz,nao,nao))
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                # *2 for double occupancy
                dmx = reduce(numpy.dot, (orbv, x*2, orbo.T))
                dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
                dms[i] = dmx + dmy  # AX + BY

            v1ao = vresp(dms)
            v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir,nocc)
            v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0,nocc,nocc,nmo))
            v1ov = v1ov.reshape(-1,nocc,nvir).transpose(0,2,1)
            hx = numpy.empty((nz,2,nvir,nocc), dtype=v1vo.dtype)
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                hx[i,0] = v1vo[i]
                hx[i,0]+= numpy.einsum('ps,sq->pq', fvv, x)  # AX
                hx[i,0]-= numpy.einsum('ps,rp->rs', foo, x)  # AX
                hx[i,1] =-v1ov[i]
                hx[i,1]-= numpy.einsum('ps,sq->pq', fvv, y)  #-AY
                hx[i,1]+= numpy.einsum('ps,rp->rs', foo, y)  #-AY

            if wfnsym is not None and mol.symmetry:
                hx[:,:,sym_forbid] = 0
            return hx.reshape(nz,-1)

        return vind, hdiag

    def init_guess(self, mf, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym)
        y0 = numpy.zeros_like(x0)
        return numpy.hstack((x0,y0))

    def kernel(self, x0=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        self.check_sanity()
        self.dump_flags()

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

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

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
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

    mf = scf.RHF(mol).run()
    td = TDA(mf)
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 11.01747918  11.01747918  13.16955056]

    td = TDHF(mf)
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440705]

