#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic NMR shielding tensor
'''


import time
import numpy
from pyscf import gto
from pyscf.lib import logger as log
from pyscf.lib import parameters as param
from pyscf import lib
from pyscf import scf
from pyscf.future import grad
import pyscf.scf._vhf

CPSCF_MAX_CYCLE = 20
CPSCF_THRESHOLD = 1e-9

# Krylov subspace method
class CPSCF:
# ref: J. A. Pople, R. Krishnan, H. B. Schlegel, and J. S. Binkley,
#      Int. J.  Quantum. Chem.  Symp. 13, 225 (1979).
    def __init__(self, mol, restart=False):
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.restart = restart #TODO
        self.max_cycle = CPSCF_MAX_CYCLE
        self.threshold = CPSCF_THRESHOLD

    def init_guess(self, b, chkfile=None):
        if 0 and self.restart:
            try:
                grp_x, grp_ox = scf.chkfile.load(chkfile, 'CPSCF')
                for x, ox in grp_x, grp_ox:
                    while x.__len__() <= ox.__len__():
                        ox.pop()
                log.info(self, 'Restore %d iterations from chkfile.', \
                         ox.__len__())
            except:
                log.info(self, 'Fail to read DIIS-vector from chkfile.')
        else:
            grp_x  = [[i] for i in b]
            grp_ox = [[] for i in range(3)]
        return grp_x, grp_ox

    def solve(self, operator, b):
        '''use DIIS method to solve CPSCF Eq.  x + operator(x) = b. \n
                expand x wrt op(): x = x[0] + x[1] + ... \n
                x[0]   = b, \n
                x[n+1] = -op(x[n])'''
        log.info(self, 'start DIIS-CPSCF.')

        grp_x, grp_ox = self.init_guess(b)
        grp_x, grp_ox = self.cpscf_iter(operator, grp_x, grp_ox)
        res = []
        # form H, G for eq. Hc=G
        for n in range(3):
            nd = grp_ox[n].__len__()
            H = -numpy.dot(numpy.array(grp_x[n]).conj(),
                           numpy.array(grp_ox[n]).T)
            for j in range(nd):
                H[j,j] += numpy.dot(grp_x[n][j].conj(), grp_x[n][j])
            G = numpy.zeros(nd, grp_x[0][0].dtype)
            G[0] = numpy.dot(grp_x[n][0].conj(), grp_x[n][0])

            # solve  H*x = G
            try:
                c = numpy.linalg.solve(H, G)
            except numpy.linalg.linalg.LinAlgError:
                # damp diagonal elements to prevent singular
                c = numpy.linalg.solve(H+numpy.eye(H.shape[0])*1e-8, G)
            #c = lib.solve_lineq_by_SVD(H, G)
            x = 0
            for j, ci in enumerate(c):
                x = x + grp_x[n][j] * ci
            res.append(x)
        return res

    def cpscf_iter(self, operator, grp_x, grp_ox):
        '''for Eq.  x + operator(x) = b \n
            save x[0] in grp_x, op(x[0]) in grp_ox'''
        nset = grp_x.__len__()
        conv = [False] * nset
        cycle = grp_ox[0].__len__()
        while not all(conv) and cycle < max(1, self.max_cycle):
            # generate op(x[n])
            opx = operator([i[-1] for i in grp_x])
            for i in range(nset):
                # save op(x[n])
                grp_ox[i].append(opx[i])
                # save x[n+1]
                grp_x[i].append(opx[i])
                # Use Schmidt process to project x[0]...x[n] from x[n+1]
                q, r = numpy.linalg.qr(numpy.array(grp_x[i]).T)
                grp_x[i][-1] = q[:,-1] * r[-1,-1]

                if abs(r[-1,-1]) < self.threshold:
                    conv[i] = True
                log.info(self, 'cycle=%d, norm(residue(vec x_%d))=%g;', \
                         cycle+1, i, abs(r[-1,-1]))
            cycle += 1
            #scf.chkfile.dump('CPSCF', (grp_x, grp_ox))

        for i in range(3):
            grp_x[i].pop()
        return grp_x, grp_ox


def solve_ucpscf(mol, scf0, h1, s1):
    '''uncoupled equation'''
    occ = scf0.mo_occ
    e_a = scf0.mo_energy[occ==0]
    e_i = scf0.mo_energy[occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

    hs = h1 - s1 * e_i

    mo10 = numpy.empty_like(hs)
    mo10[:,occ==0,:] = -hs[:,occ==0,:] * e_ai
    mo10[:,occ>0,:] = -s1[:,occ>0,:] * .5

    e_ji = e_i.reshape(-1,1) - e_i
    mo_e10 = hs[:,occ>0,:] + mo10[:,occ>0,:] * e_ji
    return mo_e10, mo10


def solve_cpscf(mol, scf0, f_ind_pot, h1, s1, \
                max_cycle=20, threshold=1e-9):
    occ = scf0.mo_occ
    e_a = scf0.mo_energy[occ==0]
    e_i = scf0.mo_energy[occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

    # initial guess
    mo_e10, mo10 = solve_ucpscf(mol, scf0, h1, s1)
# brute force solver
#    mo1 = numpy.copy(mo10)
#    for i in range(10):
#        v_mo = f_ind_pot(scf0, mo1)
#        # only update the v-o block
#        mo1[:,occ==0,:] = mo10[:,occ==0,:]-v_mo[:,occ==0,:] * e_ai

    def vind_vo(mo1):
        '''update vir-occ block of induced potential'''
        mo1 = numpy.array(mo1).reshape(mo10.shape)
        v_mo = f_ind_pot(scf0, mo1)
        v_mo[:,occ==0,:] = -v_mo[:,occ==0,:] * e_ai
        v_mo[:,occ>0,:] = 0
        return v_mo.reshape((3,-1))

    cpscf = CPSCF(mol)
    cpscf.max_cycle = max_cycle
    cpscf.threshold = threshold
    mo1 = cpscf.solve(vind_vo, mo10.reshape((3,-1)))

    # extra iteration to get mo1 and e1
    v_mo = f_ind_pot(scf0, numpy.array(mo1).reshape(mo10.shape))
    mo_e10 += -v_mo[:,occ>0,:]
    mo10[:,occ==0,:] += -v_mo[:,occ==0,:] * e_ai
    return mo_e10, mo10


class MSC(object):
    __doc__ = 'magnetic shielding constants'
    def __init__(self, scf_method, restart=False):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.chkfile = scf_method.chkfile
        self.restart = restart
        self.scf = grad.hf.redo_scf(self.mol, scf_method)
        self.mo10 = None
        self.mo_e10 = None

        self.shielding_nuc = [i+1 for i in range(self.mol.natm)]
        self.gauge_orig = (0,0,0)
        self.is_giao = True
        self.is_cpscf = True
        self.max_cycle = CPSCF_MAX_CYCLE
        self.threshold = CPSCF_THRESHOLD

## ** default method **
#        # RHF: exchange parts
#        if not isinstance(scf_method, scf.hf.RHF):
#            raise AttributeError('TODO: UHF')

    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** MSC flags ********')
        log.info(self, 'potential = %s', self.scf.get_veff.__doc__)
        log.info(self, 'gauge = %s', ('Common gauge','GIAO')[self.is_giao])
        log.info(self, 'MO10 eq. is %s', ('UCPSCF','CPSCF')[self.is_cpscf])
        #if self.restart:
        #    log.info(self, 'restart from chkfile  %s', self.chkfile)
        #    log.info(self, '    restore %s', self.rec['NMR'].keys())
        log.info(self, '\n')

    def _ab_diagonal(self, a, b):
        ''' p_i = A_ij * B_ji'''
        n = a.shape[0]
        p = []
        for i in range(n):
            p.append(numpy.dot(a[i], b[:,i]))
        return numpy.array(p)

    def msc(self):
        cput0 = (time.clock(), time.time())
        if self.verbose >= param.VERBOSE_INFO:
            self.dump_flags()

        if not self.is_giao:
            self.mol.set_common_origin(self.gauge_orig)

        res = self.para(self.mol, self.scf)
        msc_para, para_vir, para_occ = [x*1e6/param.LIGHTSPEED**2 for x in res]
        msc_dia = self.dia(self.mol, self.scf) * 1e6/param.LIGHTSPEED**2
        e11 = msc_para + msc_dia

        log.timer(self, 'NMR', *cput0)
        if self.verbose > param.VERBOSE_QUITE:
            fout = self.stdout
            for tot,d,p,p0,pvir,atom in \
                    zip(e11, msc_dia, msc_para, para_occ, \
                    para_vir, self.shielding_nuc):
                fout.write('\ntotal MSC of atom %d %s\n' \
                           % (atom, self.mol.symbol_of_atm(atom-1)))
                fout.write('B_x %s\n' % str(tot[0]))
                fout.write('B_y %s\n' % str(tot[1]))
                fout.write('B_z %s\n' % str(tot[2]))
                fout.write('dia-magnetism\n')
                fout.write('B_x %s\n' % str(d[0]))
                fout.write('B_y %s\n' % str(d[1]))
                fout.write('B_z %s\n' % str(d[2]))
                fout.write('para-magnetism\n')
                fout.write('B_x %s\n' % str(p[0]))
                fout.write('B_y %s\n' % str(p[1]))
                fout.write('B_z %s\n' % str(p[2]))
                if self.verbose >= param.VERBOSE_INFO:
                    fout.write('INFO: occ part of para-magnetism\n')
                    fout.write('INFO: B_x %s\n' % str(p0[0]))
                    fout.write('INFO: B_y %s\n' % str(p0[1]))
                    fout.write('INFO: B_z %s\n' % str(p0[2]))
                    fout.write('INFO: vir part of para-magnetism\n')
                    fout.write('INFO: B_x %s\n' % str(pvir[0]))
                    fout.write('INFO: B_y %s\n' % str(pvir[1]))
                    fout.write('INFO: B_z %s\n' % str(pvir[2]))
        self.stdout.flush()
        return e11

    def dia(self, mol, scf0):
        '''Dia-magnetic'''
        msc_dia = []
        dm0 = scf0.calc_den_mat(scf0.mo_coeff, scf0.mo_occ)
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            if self.is_giao:
                h11 = mol.intor('cint1e_giao_a11part_sph', 9)
            else:
                h11 = mol.intor('cint1e_cg_a11part_sph', 9)
            trh11 = -(h11[0] + h11[4] + h11[8])
            h11[0] += trh11
            h11[4] += trh11
            h11[8] += trh11
            if self.is_giao:
                g11 = mol.intor('cint1e_a01gp_sph', 9)
                # (mu,B) => (B,mu)
                h11 = h11 + g11[numpy.array((0,3,6,1,4,7,2,5,8))]
            a11 = [lib.trace_ab(dm0, (x+x.T)*.5) for x in h11]
            msc_dia.append(a11)
            # param.MI_POS XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
            #           => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
        return numpy.array(msc_dia).reshape(self.shielding_nuc.__len__(), 3, 3)

    def para(self, mol, scf0):
        '''Para-magnetism'''
        t0 = (time.clock(), time.time())
        h1, s1 = self.get_h10_s10(mol, scf0)
        t0 = log.timer(self, 'h10', *t0)
        if self.is_cpscf:
            direct_scf_bak, scf0.direct_scf = scf0.direct_scf, False
            self.mo_e10, self.mo10 = solve_cpscf(mol, scf0, self.v_ind, h1,
                                                 s1, self.max_cycle,
                                                 self.threshold)
            scf0.direct_scf = direct_scf_bak
        else:
            self.mo_e10, self.mo10 = solve_ucpscf(mol, scf0, h1, s1)
        t0 = log.timer(self, 'solving CPSCF/UCPSCF', *t0)

        msc_para = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_vir = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_occ = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            # 1/2(A01 dot p + p dot A01) => (ia01p - c.c.)/2 => <ia01p>
            h01 = mol.intor_asymmetric('cint1e_ia01p_sph', 3)
            # *2 for doubly occupied orbitals
            h01_mo = self._mat_ao2mo(h01, scf0.mo_coeff, scf0.mo_occ) * 2
            for b in range(3):
                for m in range(3):
                    # c10^T * h01 + c.c.
                    p = self._ab_diagonal(h01_mo[m], self.mo10[b].T) * 2
                    msc_para[n,b,m] = p.sum()
                    para_occ[n,b,m] = p[scf0.mo_occ>0].sum()
                    para_vir[n,b,m] = msc_para[n,b,m] - para_occ[n,b,m]
        return msc_para, para_vir, para_occ

    @lib.omnimethod
    def calc_den_mat_1(self, mo1, mo0, occ):
        ''' i * DM^1'''
        m = mo0[:,occ>0] * occ[occ>0]
        dm1 = []
        for i in range(3):
            mo1_ao = numpy.dot(mo0, mo1[i])
            tmp = numpy.dot(mo1_ao, m.T.conj())
            dm1.append(tmp - tmp.T.conj())
        return numpy.array(dm1)

    def get_h10_s10(self, mol, scf0):
        if self.is_giao:
            # A10_i dot p + p dot A10_i consistents with <p^2 g>
            # A10_j dot p + p dot A10_j consistents with <g p^2>
            # A10_j dot p + p dot A10_j => i/2 (rjxp - pxrj) = irjxp
            h1_ao = .5 * mol.intor('cint1e_giao_irjxp_sph', 3)
            hg_ao, s1_ao = self.add_giao(mol, scf0)
            h1 = self._mat_ao2mo(h1_ao + hg_ao, scf0.mo_coeff, scf0.mo_occ)
            s1 = self._mat_ao2mo(s1_ao, scf0.mo_coeff, scf0.mo_occ)
        else:
            h1_ao = .5 * mol.intor('cint1e_cg_irxp_sph', 3)
            h1 = self._mat_ao2mo(h1_ao, scf0.mo_coeff, scf0.mo_occ)
            s1 = numpy.zeros_like(h1)
        return h1, s1

    def _mat_ao2mo(self, mat, mo_coeff, occ):
        '''transform an AO-based matrix to a MO-based matrix. The MO-based
        matrix only keep the occupied band: M[:,:nocc]'''
        mo0 = mo_coeff[:,occ>0]
        mat_mo = [reduce(numpy.dot, (mo_coeff.T.conj(),i,mo0)) for i in mat]
        return numpy.array(mat_mo)

    def add_giao(self, mol, scf0):
        '''GIAO'''
        log.info(self, 'First-order Fock matrix / GIAOs\n')

        if False and self.restart:
            h1 = scf.chkfile.load(self.scf.chkfile, 'vhf_GIAO')
            log.info(self, 'Restore vhf_GIAO from chkfile\n')
        else:
            dm0 = scf0.calc_den_mat(scf0.mo_coeff, scf0.mo_occ)
            vj, vk = scf._vhf.direct_mapdm('cint2e_ig1_sph',  # (g i,j|k,l)
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
            #scf.chkfile.dump(self.scf.chkfile, 'vhf_GIAO', h1)

        h1 += mol.intor_asymmetric('cint1e_ignuc_sph', 3)
        h1 += mol.intor('cint1e_igkin_sph', 3)
        s1 = mol.intor_asymmetric('cint1e_igovlp_sph', 3)
        return h1, s1

    def v_ind(self, scf0, mo1):
        '''Induced potential'''
        mol = scf0.mol
        dm1 = self.calc_den_mat_1(mo1, scf0.mo_coeff, scf0.mo_occ)
        v_ao = self.scf.get_veff(mol, dm1, hermi=2)
        return self._mat_ao2mo(v_ao, scf0.mo_coeff, scf0.mo_occ)


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = 'out_hf'

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    #mol.nucmod = {'F':2, 'H':2}
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.RHF(mol)
    rhf.scf()
    nmr = MSC(rhf)
    nmr.is_cpscf = True
    #nmr.gauge_orig = (0,0,0)
    nmr.is_giao = True
    msc = nmr.msc() # _xx,_yy = 375.232781, _zz = 483.002149
    print(msc)

