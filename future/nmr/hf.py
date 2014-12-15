#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic NMR shielding tensor
'''


import time
import numpy
import pyscf.lib
import pyscf.lib.logger as logger
import pyscf.lib.parameters as param
import pyscf.scf
import pyscf.scf._vhf as _vhf



class NMR(object):
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        self.shielding_nuc = [i+1 for i in range(self.mol.natm)]
        self.gauge_orig = (0,0,0)
        self.giao = True
        self.cphf = True
        self.max_cycle_cphf = 20
        self.threshold = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys() + ['_keys'])

## ** default method **
#        # RHF: exchange parts
#        if not isinstance(scf_method, pyscf.scf.hf.RHF):
#            raise AttributeError('TODO: UHF')

    def dump_flags(self):
        logger.info(self, '\n')
        logger.info(self, '******** NMR flags ********')
        logger.info(self, 'potential = %s', self._scf.get_veff.__doc__)
        logger.info(self, 'gauge = %s', ('Common gauge','GIAO')[self.giao])
        logger.info(self, 'shielding for atoms %s', str(self.shielding_nuc))
        if self.cphf:
            logger.info(self, 'Solving MO10 eq. with CPHF')
        if not self._scf.converged:
            log.warn(self, 'underneath SCF of NMR not converged')
        logger.info(self, '\n')

    def shielding(self, mo1=None):
        cput0 = t1 = (time.clock(), time.time())
        self.dump_flags()
        self.mol.check_sanity(self)

        if not self.giao:
            self.mol.set_common_origin(self.gauge_orig)

        mf = self._scf
        if mo1 is None:
            self.mo10 = self.solve_mo10()[1]
        else:
            self.mo10 = mo1

        res = self.para(self.mol, self.mo10, mf.mo_coeff, mf.mo_occ)
        fac2ppm = 1e6/param.LIGHTSPEED**2
        msc_para, para_vir, para_occ = [x*fac2ppm for x in res]
        msc_dia = self.dia(self.mol, mf.mo_coeff, mf.mo_occ, mf) * fac2ppm
        e11 = msc_para + msc_dia

        logger.timer(self, 'NMR shielding', *cput0)
        if self.verbose > param.VERBOSE_QUIET:
            for i, atm_id in enumerate(self.shielding_nuc):
                self.write(e11[i], \
                           '\ntotal shielding of atom %d %s' \
                           % (atm_id, self.mol.symbol_of_atm(atm_id-1)))
                self.write(msc_dia[i], 'dia-magnetism')
                self.write(msc_para[i], 'para-magnetism')
                if self.verbose >= param.VERBOSE_INFO:
                    self.write(para_occ[i], 'occ part of para-magnetism')
                    self.write(para_vir[i], 'vir part of para-magnetism')
        self.stdout.flush()
        return e11

    def dia(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if not self.giao:
            self.mol.set_common_origin(self.gauge_orig)

        msc_dia = []
        dm0 = scf0.make_rdm1(mo_coeff, mo_occ)
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            if self.giao:
                h11 = mol.intor('cint1e_giao_a11part_sph', 9)
            else:
                h11 = mol.intor('cint1e_cg_a11part_sph', 9)
            trh11 = -(h11[0] + h11[4] + h11[8])
            h11[0] += trh11
            h11[4] += trh11
            h11[8] += trh11
            if self.giao:
                g11 = mol.intor('cint1e_a01gp_sph', 9)
                # (mu,B) => (B,mu)
                h11 = h11 + g11[numpy.array((0,3,6,1,4,7,2,5,8))]
            a11 = [numpy.einsum('ij,ji', dm0, (x+x.T)*.5) for x in h11]
            msc_dia.append(a11)
            # param.MI_POS XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
            #           => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
        return numpy.array(msc_dia).reshape(-1, 3, 3)

    def para(self, mol, mo10, mo_coeff, mo_occ):
        msc_para = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_vir = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_occ = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            # 1/2(A01 dot p + p dot A01) => (ia01p - c.c.)/2 => <ia01p>
            h01 = mol.intor_asymmetric('cint1e_ia01p_sph', 3)
            # *2 for doubly occupied orbitals
            h01_mo = _mat_ao2mo(h01, mo_coeff, mo_occ) * 2
            for b in range(3):
                for m in range(3):
                    # c10^T * h01 + c.c.
                    p = numpy.einsum('ij,ji->i',h01_mo[m], self.mo10[b].T) * 2
                    msc_para[n,b,m] = p.sum()
                    para_occ[n,b,m] = p[mo_occ>0].sum()
                    para_vir[n,b,m] = msc_para[n,b,m] - para_occ[n,b,m]
        return msc_para, para_vir, para_occ

    @pyscf.lib.omnimethod
    def make_rdm1_1(self, mo1occ, mo0, occ):
        ''' DM^1 = (i * C_occ^1 C_occ^{0,dagger}) + c.c.'''
        mocc = mo0[:,occ>0] * occ[occ>0]
        dm1 = []
        for i in range(3):
            tmp = reduce(numpy.dot, (mo0, mo1occ[i], mocc.T.conj()))
            # note the minus sign due to the phase i
            dm1.append(tmp - tmp.T.conj())
        return numpy.array(dm1)

    def make_h10(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if self.giao:
            # A10_i dot p + p dot A10_i consistents with <p^2 g>
            # A10_j dot p + p dot A10_j consistents with <g p^2>
            # A10_j dot p + p dot A10_j => i/2 (rjxp - pxrj) = irjxp
            h1_ao = .5 * mol.intor('cint1e_giao_irjxp_sph', 3)
            logger.debug(self, 'First-order Fock matrix from GIAOs\n')
            h1 = _mat_ao2mo(h1_ao, mo_coeff, mo_occ)
            h1 += self.make_h10giao(mol, mo_coeff, mo_occ, scf0)
            pyscf.scf.chkfile.dump(self.chkfile, 'nmr/vhf_GIAO', h1)
        else:
            self.mol.set_common_origin(self.gauge_orig)
            h1_ao = .5 * mol.intor('cint1e_cg_irxp_sph', 3)
            h1 = _mat_ao2mo(h1_ao, mo_coeff, mo_occ)
        return h1

    def make_h10giao(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        dm0 = scf0.make_rdm1(mo_coeff, mo_occ)
        vj, vk = _vhf.direct_mapdm('cint2e_ig1_sph',  # (g i,j|k,l)
                                   'CVHFfill_dot_nrs4', # ip1_sph has k>=l,
# fill ij, ip1_sph has anti-symm between i and j
                                   'CVHFunpack_nrblock2trilu_anti',
# funpack fill the entire 2D array for ij, then transpose to kl, fjk ~ nr2sij_...
                                   ('CVHFnrs2ij_ij_s1kl', 'CVHFnrs2ij_il_s1jk'),
                                   dm0, 3, # xyz, 3 components
                                   mol._atm, mol._bas, mol._env)
# J = i[(i i|\mu g\nu) + (i gi|\mu \nu)]
# K = i[(\mu gi|i \nu) + (\mu i|i g\nu)]
#   = (\mu g i|i \nu) - h.c.
# vk ~ vk_{jk} ~ {l, nabla i}, but vj ~ {nabla i, j}
        vk = vk.transpose(0,2,1) - vk
        h1 = vj - .5 * vk

        h1 += mol.intor_asymmetric('cint1e_ignuc_sph', 3)
        h1 += mol.intor('cint1e_igkin_sph', 3)
        return _mat_ao2mo(h1, mo_coeff, mo_occ)

    def make_s10(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if self.giao:
            s1_ao = mol.intor_asymmetric('cint1e_igovlp_sph', 3)
            s1 = _mat_ao2mo(s1_ao, mo_coeff, mo_occ)
        else:
            nmo = mo_coeff.shape[1]
            nocc = (mo_occ>0).sum()
            s1 = numpy.zeros((3,nmo,nocc))
        return s1

    def solve_mo10(self, mol=None, h1=None, scf0=None):
        if mol is None:
            mol = self.mol
        if scf0 is None:
            scf0 = self._scf
        cput1 = (time.clock(), time.time())
        if h1 is None:
            h1 = self.make_h10(mol, scf0.mo_coeff, scf0.mo_occ, scf0)
            cput1 = logger.timer(self, 'first order Fock matrix', *cput1)

        s1 = self.make_s10(mol, scf0.mo_coeff, scf0.mo_occ, scf0)
        mo_e10, mo10 = solve_mo1(scf0.mo_energy, scf0.mo_occ, h1, s1)
        if self.cphf:
            direct_scf_bak, scf0.direct_scf = scf0.direct_scf, False
            mo_e10, mo10 = solve_cphf(self.mol, self.v_ind,
                                      scf0.mo_energy, scf0.mo_occ,
                                      mo_e10, mo10,
                                      self.max_cycle_cphf, self.threshold,
                                      self.verbose)
            scf0.direct_scf = direct_scf_bak
        logger.timer(self, 'solving mo1 eqn', *cput1)
        return mo_e10, mo10

    def v_ind(self, mo1occ):
        '''Induced potential'''
        mol = self.mol
        scf0 = self._scf
        dm1 = self.make_rdm1_1(mo1occ, scf0.mo_coeff, scf0.mo_occ)
        v_ao = self._scf.get_veff(self.mol, dm1, hermi=2)
        return _mat_ao2mo(v_ao, scf0.mo_coeff, scf0.mo_occ)

    def write(self, msc3x3, title):
        self.stdout.write('%s\n' % title)
        self.stdout.write('B_x %s\n' % str(msc3x3[0]))
        self.stdout.write('B_y %s\n' % str(msc3x3[1]))
        self.stdout.write('B_z %s\n' % str(msc3x3[2]))

    def restart(self, h1, cphf_x, cphf_ax):
        pass


def solve_mo1(mo_energy, mo_occ, h1, s1):
    '''uncoupled equation'''
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

    hs = h1 - s1 * e_i

    mo10 = numpy.empty_like(hs)
    mo10[:,mo_occ==0,:] = -hs[:,mo_occ==0,:] * e_ai
    mo10[:,mo_occ>0,:] = -s1[:,mo_occ>0,:] * .5

    e_ji = e_i.reshape(-1,1) - e_i
    mo_e10 = hs[:,mo_occ>0,:] + mo10[:,mo_occ>0,:] * e_ji
    return mo_e10, mo10


# raw_mo_e1 and raw_mo1 are calculated from uncoupled calculation
# raw_mo_e1 is h1_ai / (e_i-e_a)
def solve_cphf(mol, fvind, mo_energy, mo_occ, raw_mo_e1, raw_mo1, \
               max_cycle=20, tol=1e-9, verbose=0):
    log = logger.Logger(mol.stdout, verbose)
    t0 = (time.clock(), time.time())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

# brute force solver
#    mo1 = numpy.copy(raw_mo1)
#    for i in range(20):
#        v_mo = fvind(mo1)
#        # only update the v-o block
#        mo1[:,mo_occ==0,:] = raw_mo1[:,mo_occ==0,:]-v_mo[:,mo_occ==0,:] * e_ai
#    return 0, mo1

    def vind_vo(mo1):
# only update vir-occ block of induced potential, occ-occ block can be
# absorbed in mo_e1
        v_mo = fvind(mo1.reshape(raw_mo1.shape))
        v_mo[:,mo_occ==0,:] *= e_ai
        v_mo[:,mo_occ>0,:] = 0
        return v_mo.ravel()

    mo1 = pyscf.lib.krylov(vind_vo, raw_mo1.ravel(),
                           tol=tol, maxiter=max_cycle, log=log)
    log.timer('krylov solver in CPHF', *t0)

# extra iteration to get mo1 and e1
    v_mo = fvind(mo1.reshape(raw_mo1.shape))
    mo_e1 = raw_mo_e1 - v_mo[:,mo_occ>0,:]
    mo1 = raw_mo1.copy()
    mo1[:,mo_occ==0,:] = raw_mo1[:,mo_occ==0,:] - v_mo[:,mo_occ==0,:] * e_ai
    return mo_e1, mo1

def _mat_ao2mo(mat, mo_coeff, occ):
    '''transform an AO-based matrix to a MO-based matrix. The MO-based
    matrix only has the occupied columns M[:,:nocc]'''
    mo0 = mo_coeff[:,occ>0]
    mat_mo = [reduce(numpy.dot, (mo_coeff.T.conj(),i,mo0)) for i in mat]
    return numpy.array(mat_mo)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    #mol.nucmod = {'F':2}
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.RHF(mol)
    rhf.scf()
    nmr = NMR(rhf)
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    nmr.giao = True
    msc = nmr.shielding() # _xx,_yy = 375.232781, _zz = 483.002149
    print(msc)

