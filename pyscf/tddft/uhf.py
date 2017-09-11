#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.tddft import rhf
from pyscf.scf import uhf_symm
from pyscf.scf.newton_ah import _gen_uhf_response


def gen_tda_hop(mf, fock_ao=None, wfnsym=None, max_memory=2000):
    '''(A+B)x
    
    Kwargs:
        wfnsym : int
            Point group symmetry for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff[0].dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    if wfnsym is not None and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbida = (orbsyma[viridxa].reshape(-1,1) ^ orbsyma[occidxa]) != wfnsym
        sym_forbidb = (orbsymb[viridxb].reshape(-1,1) ^ orbsymb[occidxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ai_a = mo_energy[0][viridxa].reshape(-1,1) - mo_energy[0][occidxa]
    e_ai_b = mo_energy[1][viridxb].reshape(-1,1) - mo_energy[1][occidxb]
    e_ai = hdiag = numpy.hstack((e_ai_a.reshape(-1), e_ai_b.reshape(-1)))
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    mo_a = numpy.asarray(numpy.hstack((orboa,orbva)), order='F')
    mo_b = numpy.asarray(numpy.hstack((orbob,orbvb)), order='F')

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.8-mem_now)
    vresp = _gen_uhf_response(mf, hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0
        dmvo = numpy.empty((2,nz,nao,nao))
        for i, z in enumerate(zs):
            za = z[:nocca*nvira].reshape(nvira,nocca)
            zb = z[nocca*nvira:].reshape(nvirb,noccb)
            dmvo[0,i] = reduce(numpy.dot, (orbva, za, orboa.T))
            dmvo[1,i] = reduce(numpy.dot, (orbvb, zb, orbob.T))

        v1ao = vresp(dmvo)
        v1a = _ao2mo.nr_e2(v1ao[0], mo_a, (nocca,nmo,0,nocca)).reshape(-1,nvira,nocca)
        v1b = _ao2mo.nr_e2(v1ao[1], mo_b, (noccb,nmo,0,noccb)).reshape(-1,nvirb,noccb)
        for i, z in enumerate(zs):
            za = z[:nocca*nvira].reshape(nvira,nocca)
            zb = z[nocca*nvira:].reshape(nvirb,noccb)
            v1a[i] += numpy.einsum('ai,ai->ai', e_ai_a, za)
            v1b[i] += numpy.einsum('ai,ai->ai', e_ai_b, zb)
        hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        if wfnsym is not None and mol.symmetry:
            hx[:,sym_forbid] = 0
        return hx

    return vind, hdiag


class TDA(rhf.TDA):

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        log.info('nstates = %d', self.nstates)
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

    def get_vind(self, mf):
        '''Compute Ax'''
        return gen_tda_hop(mf, wfnsym=self.wfnsym, max_memory=self.max_memory)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        e_ai_a = mo_energy[0][viridxa].reshape(-1,1) - mo_energy[0][occidxa]
        e_ai_b = mo_energy[1][viridxb].reshape(-1,1) - mo_energy[1][occidxb]

        if wfnsym is not None and mol.symmetry:
            orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
            sym_forbida = (orbsyma[viridxa].reshape(-1,1) ^ orbsyma[occidxa]) != wfnsym
            sym_forbidb = (orbsymb[viridxb].reshape(-1,1) ^ orbsymb[occidxb]) != wfnsym
            e_ai_a[sym_forbida] = 1e99
            e_ai_b[sym_forbidb] = 1e99

        eai = numpy.hstack((e_ai_a.ravel(), e_ai_b.ravel()))
        nov = eai.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov))
        idx = numpy.argsort(eai)
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None):
        '''TDA diagonalization solver
        '''
        self.check_sanity()
        self.dump_flags()

        vind, hdiag = self.get_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.e, x1 = lib.davidson1(vind, x0, precond,
                                   tol=self.conv_tol,
                                   nroots=self.nstates, lindep=self.lindep,
                                   max_space=self.max_space,
                                   verbose=self.verbose)[1:]

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nvira,nocca),  # X_alpha
                     xi[nocca*nvira:].reshape(nvirb,noccb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                   for xi in x1]
        #TODO: analyze CIS wfn point group symmetry
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def get_vind(self, mf):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff[0].dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff[0].shape
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]

        if wfnsym is not None and mol.symmetry:
            orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
            sym_forbida = (orbsyma[viridxa].reshape(-1,1) ^ orbsyma[occidxa]) != wfnsym
            sym_forbidb = (orbsymb[viridxb].reshape(-1,1) ^ orbsymb[occidxb]) != wfnsym
            sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

        e_ai_a = mo_energy[0][viridxa].reshape(-1,1) - mo_energy[0][occidxa]
        e_ai_b = mo_energy[1][viridxb].reshape(-1,1) - mo_energy[1][occidxb]
        e_ai = hdiag = numpy.hstack((e_ai_a.reshape(-1), e_ai_b.reshape(-1)))
        if wfnsym is not None and mol.symmetry:
            hdiag[sym_forbid] = 0
        hdiag = numpy.hstack((hdiag.ravel(), hdiag.ravel()))
        mo_a = numpy.asarray(numpy.hstack((orboa,orbva)), order='F')
        mo_b = numpy.asarray(numpy.hstack((orbob,orbvb)), order='F')

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = _gen_uhf_response(mf, hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            if wfnsym is not None and mol.symmetry:
                # shape(nz,2,-1): 2 ~ X,Y
                xys = numpy.copy(zs).reshape(nz,2,-1)
                xys[:,:,:,sym_forbid] = 0
            dms = numpy.empty((2,nz,nao,nao)) # 2 ~ alpha,beta
            for i in range(nz):
                x, y = xys[i].reshape(2,-1)
                xa = x[:nocca*nvira].reshape(nvira,nocca)
                xb = x[nocca*nvira:].reshape(nvirb,noccb)
                ya = y[:nocca*nvira].reshape(nvira,nocca)
                yb = y[nocca*nvira:].reshape(nvirb,noccb)
                dmx = reduce(numpy.dot, (orbva, xa, orboa.T))
                dmy = reduce(numpy.dot, (orboa, ya.T, orbva.T))
                dms[0,i] = dmx + dmy  # AX + BY
                dmx = reduce(numpy.dot, (orbvb, xb, orbob.T))
                dmy = reduce(numpy.dot, (orbob, yb.T, orbvb.T))
                dms[1,i] = dmx + dmy  # AX + BY

            v1ao  = vresp(dms)
            v1avo = _ao2mo.nr_e2(v1ao[0], mo_a, (nocca,nmo,0,nocca))
            v1bvo = _ao2mo.nr_e2(v1ao[1], mo_b, (noccb,nmo,0,noccb))
            v1aov = _ao2mo.nr_e2(v1ao[0], mo_a, (0,nocca,nocca,nmo))
            v1bov = _ao2mo.nr_e2(v1ao[1], mo_b, (0,noccb,noccb,nmo))
            hx = numpy.empty((nz,2,nvira*nocca+nvirb*noccb), dtype=v1avo.dtype)
            for i in range(nz):
                x, y = xys[i].reshape(2,-1)
                hx[i,0,:nvira*nocca] = v1avo[i].ravel()
                hx[i,0,nvira*nocca:] = v1bvo[i].ravel()
                hx[i,0]+= e_ai * x  # AX
                hx[i,1,:nvira*nocca] =-v1aov[i].reshape(nocca,nvira).T.ravel()
                hx[i,1,nvira*nocca:] =-v1bov[i].reshape(noccb,nvirb).T.ravel()
                hx[i,1]-= e_ai * y  #-AY

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

        vind, hdiag = self.get_vind(self._scf)
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

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        e = []
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(w[i])
                xy.append(((x[:nocca*nvira].reshape(nvira,nocca) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(nvirb,noccb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nvira,nocca) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(nvirb,noccb) * norm)))# Y_beta
        self.e = numpy.array(e)
        self.xy = xy
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

    mf = scf.UHF(mol).run()
    td = TDA(mf)
    td.nstates = 5
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.01748568  11.01748568  11.90277134  11.90277134  13.16955369]

    td = TDHF(mf)
    td.nstates = 5
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 10.89192986  10.89192986  11.83487865  11.83487865  12.6344099 ]

    mol.spin = 2
    mf = scf.UHF(mol).run()
    td = TDA(mf)
    td.nstates = 6
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# FIXME:  first state
# [ 0.02231607274  3.32113736  18.55977052  21.01474222  21.61501962  25.0938973 ]

    td = TDHF(mf)
    td.nstates = 6
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# FIXME:  first state
# [ 0.00077090739  3.31267103  18.4954748   20.84935404  21.54808392]

