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
from pyscf.tddft import rhf
from pyscf.pbc.dft import numint
from pyscf.pbc.scf.newton_ah import _gen_rhf_response


class TDA(rhf.TDA):
#FIXME: numerically unstable with small gs?
#TODO: Add a warning message for small gs.
    def __init__(self, mf):
        self.cell = mf.cell
        self.conv_tol = 1e-6
        rhf.TDA.__init__(self, mf)

    def get_vind(self, mf):
        singlet = self.singlet
        cell = mf.cell
        kpts = mf.kpts

        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
        viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]
        eai = _get_eai(mo_energy, mo_occ)
        hdiag = numpy.hstack([x.ravel() for x in eai])

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = _gen_rhf_response(mf, singlet, hermi=0, max_memory=max_memory)

        def vind(zs):
            nz = len(zs)
            z1s = [_unpack(z, mo_occ) for z in zs]
            dmvo = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                # *2 for double occupancy
                dm1 = z1s[i] * 2
                for k in range(nkpts):
                    dmvo[i,k] = reduce(numpy.dot, (orbv[k], dm1[k], orbo[k].T.conj()))

            v1ao = vresp(dmvo)
            v1s = []
            for i in range(nz):
                dm1 = z1s[i]
                for k in range(nkpts):
                    v1vo = reduce(numpy.dot, (orbv[k].T.conj(), v1ao[i,k], orbo[k]))
                    v1vo += eai[k] * dm1[k]
                    v1s.append(v1vo.ravel())
            return lib.asarray(v1s).reshape(nz,-1)
        return vind, hdiag

    def init_guess(self, mf, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        eai = numpy.hstack([x.ravel() for x in _get_eai(mo_energy, mo_occ)])

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

        vind, hdiag = self.get_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)
        self.e, x1 = lib.davidson1(vind, x0, precond,
                                   tol=self.conv_tol,
                                   nroots=self.nstates, lindep=self.lindep,
                                   max_space=self.max_space,
                                   verbose=self.verbose)[1:]

        mo_occ = self._scf.mo_occ
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(_unpack(xi*numpy.sqrt(.5), mo_occ), 0) for xi in x1]
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def get_vind(self, mf):
        '''
        [ A   B ][X]
        [-B* -A*][Y]
        '''
        singlet = self.singlet
        cell = mf.cell
        kpts = mf.kpts

        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
        viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]
        eai = _get_eai(mo_energy, mo_occ)
        hdiag = numpy.hstack([x.ravel() for x in eai])
        tot_x = hdiag.size
        hdiag = numpy.hstack((hdiag, hdiag))

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = _gen_rhf_response(mf, singlet, hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            z1xs = [_unpack(xy[:tot_x], mo_occ) for xy in xys]
            z1ys = [_unpack(xy[tot_x:], mo_occ) for xy in xys]
            dmvo = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                # *2 for double occupancy
                dmx = z1xs[i] * 2
                dmy = z1ys[i] * 2
                for k in range(nkpts):
                    dmvo[i,k] = reduce(numpy.dot, (orbv[k], dmx[k], orbo[k].T.conj()))
                    dmvo[i,k]+= reduce(numpy.dot, (orbo[k], dmy[k].T, orbv[k].T.conj()))

            v1ao = vresp(dmvo)
            v1s = []
            for i in range(nz):
                dmx = z1xs[i]
                dmy = z1ys[i]
                v1xs = []
                v1ys = []
                for k in range(nkpts):
                    v1x = reduce(numpy.dot, (orbv[k].T.conj(), v1ao[i,k], orbo[k]))
                    v1y = reduce(numpy.dot, (orbo[k].T.conj(), v1ao[i,k], orbv[k])).T
                    v1x+= eai[k] * dmx[k]
                    v1y+= eai[k] * dmy[k]
                    v1xs.append(v1x.ravel())
                    v1ys.append(-v1y.ravel())
                v1s += v1xs + v1ys
            return lib.asarray(v1s).reshape(nz,-1)
        return vind, hdiag

    def init_guess(self, mf, nstates=None):
        x0 = TDA.init_guess(self, mf, nstates)
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
            realidx = numpy.where((abs(w.imag) < 1e-4) & (w.real > 0))[0]
            idx = realidx[w[realidx].real.argsort()]
            return w[idx].real, v[:,idx].real, idx

        w, x1 = lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=self.nstates, lindep=self.lindep,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=self.verbose)[1:]
        mo_occ = self._scf.mo_occ
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,-1)
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            x *= norm
            y *= norm
            return _unpack(x, mo_occ), _unpack(y, mo_occ)
        self.xy = [norm_xy(z) for z in x1]

        return self.e, self.xy
RPA = TDHF


def _get_eai(mo_energy, mo_occ):
    eai = []
    for k, occ in enumerate(mo_occ):
        occidx = occ >  0
        viridx = occ == 0
        ai = lib.direct_sum('a-i->ai', mo_energy[k,viridx], mo_energy[k,occidx])
        eai.append(ai)
    return eai

def _unpack(vo, mo_occ):
    nmo = mo_occ.shape[-1]
    nocc = numpy.sum(mo_occ > 0, axis=1)
    z = []
    ip = 0
    for k, no in enumerate(nocc):
        nv = nmo - no
        z.append(vo[ip:ip+nv*no].reshape(nv,no))
        ip += nv * no
    return z


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf.pbc import df
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.        
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [12]*3
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([2,1,1])).set(exxdiv=None)
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.run()
#gs=9  -8.65192427146353
#gs=12 -8.65192352289817
#gs=15 -8.6519235231529
#MDF gs=5 -8.6519301815144

    td = TDA(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=9  [ 6.0073749   6.09315355  6.3479901 ]
#gs=12 [ 6.00253282  6.09317929  6.34799109]
#gs=15 [ 6.00253396  6.09317949  6.34799109]
#MDF gs=5 [ 6.09317489  6.09318265  6.34798637]

#    from pyscf.pbc import tools
#    scell = tools.super_cell(cell, [2,1,1])
#    mf = scf.RHF(scell).run()
#    td = rhf.TDA(mf)
#    td.verbose = 5
#    print(td.kernel()[0] * 27.2114)

    td = TDHF(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=9  [ 6.03860914  6.21664545  8.20305225]
#gs=12 [ 6.03868259  6.03860343  6.2167623 ]
#gs=15 [ 6.03861321  6.03861324  6.21675868]
#MDF gs=5 [ 6.03861693  6.03861775  6.21675694]

