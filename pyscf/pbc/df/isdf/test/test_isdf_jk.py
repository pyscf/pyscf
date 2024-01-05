from pyscf.pbc.df.isdf import ISDF

import numpy as np
from pyscf import ao2mo
from pyscf.pbc.df import aft
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc import tools
from pyscf.pbc import df
import pyscf.pbc.scf as pscf

from pyscf.pbc.df.isdf.isdf_jk import get_jk_dm, get_jk_mo, _benchmark_time
from pyscf.lib import logger

if __name__ == '__main__':

    for c in [3, 5, 10, 15]:
        for N in [1, 2]:

            print("Testing c = ", c, "N = ", N, "...")

            cell   = pbcgto.Cell()
            boxlen = 3.5668
            cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

            cell.atom = '''
                           C     0.      0.      0.
                           C     0.8917  0.8917  0.8917
                           C     1.7834  1.7834  0.
                           C     2.6751  2.6751  0.8917
                           C     1.7834  0.      1.7834
                           C     2.6751  0.8917  2.6751
                           C     0.      1.7834  1.7834
                           C     0.8917  2.6751  2.6751
                        '''

            cell.basis   = 'gth-szv'
            cell.pseudo  = 'gth-pade'
            cell.verbose = 1

            # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
            cell.ke_cutoff = 128
            cell.max_memory = 800  # 800 Mb
            cell.precision  = 1e-8  # integral precision
            cell.use_particle_mesh_ewald = True

            cell.build()

            cell = tools.super_cell(cell, [N, N, N])

            myisdf = ISDF(cell=cell)
            myisdf.build(c=c)

            #### check whether two methods of get_jk is consistent ####

            nocc = 4
            mo_coeff = np.random.random((cell.nao,cell.nao))
            mo_coeff = np.linalg.qr(mo_coeff)[0]
            dm = np.dot(mo_coeff[:,:nocc],mo_coeff[:,:nocc].T) * 2.0

            vj1, vk1 = get_jk_dm(myisdf, dm, kpt=np.zeros(3), kpts_band=None, exxdiv=None)
            vj2, vk2 = get_jk_mo(myisdf, mo_coeff[:,:nocc], kpt=np.zeros(3), kpts_band=None, exxdiv=None)

            print("vj_dm == vj_mo ?", np.allclose(vj1,vj2))
            print("vk_dm == vk_mo ?", np.allclose(vk1,vk2))

            # print("vj1.shape =", vj1.shape)
            # print("vk1.shape =", vk1.shape)

            #### bench make eri ####

            t1 = (logger.process_clock(), logger.perf_counter())
            eri = myisdf.get_eri(compact=False).reshape((cell.nao,)*4)
            vj  = np.einsum('ijkl,kl->ij', eri, dm)
            vk  = np.einsum('ijkl,jk->il', eri, dm)
            t2 = (logger.process_clock(), logger.perf_counter())

            _benchmark_time(t1, t2, "get_jk benchmark")

            print("vj1 == vj ?", np.allclose(vj1,vj))
            print("vk1 == vk ?", np.allclose(vk1,vk))

            # df get jk

            mydf = df.FFTDF(cell)

            t1 = (logger.process_clock(), logger.perf_counter())

            vj_df, vk_df = mydf.get_jk(dm, kpts=np.zeros(3), kpts_band=None, exxdiv=None)

            t2 = (logger.process_clock(), logger.perf_counter())

            _benchmark_time(t1, t2, "df get_jk benchmark")

            print("vj_dm == vj_df ?", np.allclose(vj1, vj_df))
            print("vk_dm == vk_df ?", np.allclose(vk1, vk_df))

            for i in range(vj1.shape[0]):
                for j in range(vj1.shape[1]):
                    if abs(vj1[i,j] - vj_df[i,j]) > 1e-6:
                        print("vj1[%3d, %3d] = %15.8f != %15.8f, ration = %15.8f" %
                              (i,j,vj1[i,j],vj_df[i,j], vj1[i,j]/vj_df[i,j]))

            for i in range(vk1.shape[0]):
                for j in range(vk1.shape[1]):
                    if abs(vk1[i,j] - vk_df[i,j]) > 1e-6:
                        print("vk1[%3d, %3d] = %15.8f != %15.8f, ration = %15.8f" %
                              (i,j,vk1[i,j],vk_df[i,j], vk1[i,j]/vk_df[i,j]))
