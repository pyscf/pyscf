import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys

import numpy as np
import cupy
#from jax import numpy as jnp
#from jax import config as jconf
#jconf.update('jax_enable_x64', True)
import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf import pbc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import tools
from pyscf.pbc import df
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import multigrid
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band
from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

cell=pbcgto.Cell()

#Molecule
#boxlen=12.4138
boxlen=8.
cell.a=np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
#cell.atom="H2O_64.xyz"
cell.atom = 'LiH.xyz'
#cell.basis='gth-tzv2p'
cell.basis = 'gth-dzvp'
cell.ke_cutoff=100  # kinetic energy cutoff in a.u.
#cell.mesh = [70,]*3
cell.max_memory=80000  # 20 Gb
cell.precision=1e-8  # integral precision
cell.pseudo='gth-pade'
cell.verbose=4
#cell.rcut_by_shell_radius=True # integral screening based on shell radii
cell.use_particle_mesh_ewald = True  # use particle mesh ewald for nuclear repulsion
cell.build()
cell = tools.super_cell(cell, [2,2,2])  # build super cell by replicating unit cell

mf=pbcdft.RKS(cell)
#mf.xc = "LDA, VWN"
mf.xc = "PBE,PBE"
mf.init_guess='atom'  # atom guess is fast
mf.with_df = multigrid.MultiGridFFTDF2(cell)
mf.with_df.ngrids = 4  # number of sets of grid points ? ? ? 
#mf.kernel()

dm1 = mf.get_init_guess(cell, 'atom')
mydf = MultiGridFFTDF2(cell)

def isdf(mydf, dm_kpts, hermi=1, naux=None, c=5, max_iter=100, kpts=np.zeros((1,3)), kpts_band=None, verbose=None):
    grids = mydf.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    log = logger.Logger(sys.stdout, 4)
    cput0 = (logger.process_clock(), logger.perf_counter())
    aoR = mydf._numint.eval_ao(cell, coords)[0]

    cput1 = log.timer('eval_ao', *cput0)
    if naux is None:
        naux = cell.nao * c

    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert nset == 1
    assert nkpts == 1
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

    weight = cell.vol / ngrids
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.flatten()
    assert rhoR.size == ngrids

    cput1 = log.timer('eval_rhoR', *cput1)
    # from cuml.cluster import KMeans
    # from scikit-learn.cluster import KMeans
    from sklearn.cluster import KMeans
    kmeans_float = KMeans(n_clusters=naux,
                          max_iter=max_iter,
                          # max_samples_per_batch=32768*8//naux,
                          # output_type='numpy'
                          )
    kmeans_float.fit(coords, sample_weight=rhoR)
    centers = kmeans_float.cluster_centers_

    cput1 = log.timer('kmeans', *cput1)

    a = cell.lattice_vectors()
    scaled_centers = np.dot(centers, np.linalg.inv(a))

    idx = (np.rint(scaled_centers*mesh[None,:]) + mesh[None,:]) % (mesh[None,:])
    idx = idx[:,2] + idx[:,1]*mesh[2] + idx[:,0]*(mesh[1]*mesh[2])
    idx = idx.astype(int)

    cput1 = log.timer('get idx', *cput1)

    aoRg = lib.device_put(aoR[idx])
    A = cupy.dot(aoRg, aoRg.T) ** 2  # (Naux, Naux)
    #A = lib.device_get(A)
    cput1 = log.timer('get A', *cput1)

    X = np.empty((naux,ngrids))
    blksize = int(10*1e9/8/naux)
    for p0, p1 in lib.prange(0, ngrids, blksize):
        B = cupy.dot(aoRg, lib.device_put(aoR[p0:p1]).T) ** 2
        #B = lib.device_get(B)
        X[:,p0:p1] = lib.device_get(cupy.linalg.lstsq(A, B, rcond=1e-8)[0])
        #X[:,p0:p1] = np.asarray(jnp.linalg.lstsq(A, B, rcond=1e-8)[0])
        B = None
    A = None
    aoRg = lib.device_get(aoRg)

    cput1 = log.timer('least squre fit', *cput1)

    V_R = np.empty((naux,ngrids))
    coulG = tools.get_coulG(cell, mesh=mesh)
    coulG = lib.device_put(coulG)

    blksize1 = int(5*1e9/8/ngrids)
    for p0, p1 in lib.prange(0, naux, blksize1):
        X_freq = cupy.fft.fftn(lib.device_put(X[p0:p1]).reshape(-1,*mesh), axes=(1,2,3)).reshape(-1,ngrids)
        V_G = X_freq * coulG[None,:]
        X_freq = None
        V_R[p0:p1] = lib.device_get(cupy.fft.ifftn(V_G.reshape(-1,*mesh), axes=(1,2,3)).real.reshape(-1,ngrids))
        V_G = None
    coulG = None

    cput1 = log.timer('fft', *cput1)

    W = np.zeros((naux,naux))
    for p0, p1 in lib.prange(0, ngrids, blksize):
        W += lib.device_get(cupy.dot(lib.device_put(X[:,p0:p1]), lib.device_put(V_R[:,p0:p1]).T))

    cput1 = log.timer('get W', *cput1)
    return W, aoRg

def get_k(mydf, dm, W, aoRg):
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)

    if getattr(dm, 'mo_coeff', None) is not None:
        print('has mo')
        mo_coeff = dm.mo_coeff
        mo_occ   = dm.mo_occ
    else:
        mo_coeff = None

    nao = dm.shape[-1]
    dm = dm.reshape(nao,nao)

    weight = (cell.vol/ngrids)

    if mo_coeff is not None:
        mo_coeff = mo_coeff[:,mo_occ>0] * np.sqrt(mo_occ[mo_occ>0])
        ao2 = np.dot(mo_coeff.T, aoRg.T)
        M = np.dot(ao2.T, ao2)
    else:
        M = np.dot(aoRg, np.dot(dm, aoRg.T))

    tmp = W * M
    K = np.dot(aoRg.T, np.dot(tmp, aoRg)) * weight
    return K

def get_jk(mydf, dm, hermi=1, kpts=None, kpts_band=None,
           with_j=True, with_k=True, omega=None, exxdiv=None):
    from pyscf.pbc.df import fft_jk
    log = logger.new_logger(mydf)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if kpts is None:
        if np.all(mydf.kpts == 0):  # Gamma-point J/K by default
            kpts = np.zeros(3)
        else:
            kpts = mydf.kpts
    else:
        kpts = np.asarray(kpts)

    vj = vk = None
    if kpts.shape == (3,):
        if with_j:
            vj = fft_jk.get_j(mydf, dm, hermi, kpts, kpts_band)
            cput1 = log.timer('vj', *cput0)
        if with_k:
            vk = get_k(mydf, dm, W, aoRg)
            log.timer('vk', *cput1)
    else:
        raise NotImplementedError
    del log
    return vj, vk

class ISDF(df.FFTDF):
    get_jk = get_jk


log = logger.Logger(sys.stdout, 4)
cput0 = (logger.process_clock(), logger.perf_counter())
W, aoRg = isdf(mydf, dm1, c=4, max_iter=2000)
log.timer('isdf', *cput0)

mf = pbcscf.RHF(cell, exxdiv=None)
mf.with_df = ISDF(cell)
mf.kernel()
