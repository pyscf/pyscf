#cython: boundscheck=False
#cython: wraparound=False
import numpy
cimport numpy
cimport cython

def restore_full_eri(numpy.ndarray[double, ndim=1, mode='c'] eri, int nao):
    cdef numpy.ndarray[double, ndim=1, mode='c'] eri_full = numpy.empty((nao*nao*nao*nao))
    cdef unsigned int i, j, k, l, ij, kl, ijkl
    cdef unsigned int pijkl, pijlk, pjikl, pjilk, pklij, plkij, pklji, plkji
    cdef unsigned int d1 = nao
    cdef unsigned int d2 = nao * nao
    cdef unsigned int d3 = nao * nao * nao
    for i in range(nao):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    ij = i * (i + 1) / 2 + j
                    kl = k * (k + 1) / 2 + l
                    if kl <= ij:
                        ijkl = ij * (ij + 1) / 2 + kl
                        pijkl = i*d3+j*d2+k*d1+l
                        pijlk = i*d3+j*d2+l*d1+k
                        pjikl = j*d3+i*d2+k*d1+l
                        pjilk = j*d3+i*d2+l*d1+k
                        pklij = k*d3+l*d2+i*d1+j
                        pklji = k*d3+l*d2+j*d1+i
                        plkij = l*d3+k*d2+i*d1+j
                        plkji = l*d3+k*d2+j*d1+i
                        eri_full[pijkl] = eri[ijkl]
                        eri_full[pijlk] = eri[ijkl]
                        eri_full[pjikl] = eri[ijkl]
                        eri_full[pjilk] = eri[ijkl]
                        eri_full[pklij] = eri[ijkl]
                        eri_full[pklji] = eri[ijkl]
                        eri_full[plkij] = eri[ijkl]
                        eri_full[plkji] = eri[ijkl]
    return eri_full.reshape(nao,nao,nao,nao)


cdef extern from "cvhf.h":
    ctypedef struct CVHFOpt:
        double direct_scf_cutoff
    void CVHFunpack(int, double*, double*)
    void nr_vhf_k(int, double *eri, double *dm, double *vk)
    void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                            int *bas, int nbas, double *env)
    void CVHFdel_optimizer(CVHFOpt **opt)
    void nr_vhf_optimizer(CVHFOpt **vhfopt, int *atm, int natm,
                          int *bas, int nbas, double *env)
    void nr_vhf_direct_o4(double *dm, double *vj, double *vk, CVHFOpt *vhfopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
    void nr_vhf_direct_m4(double *dm, double *vj, double *vk, int nset,
                          CVHFOpt *vhfopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
    void nr_vhf_incore_o3(int n, double *eri, double *dm, double *vj, double *vk)
    void nr_vhf_incore_o4(int n, double *eri, double *dm, double *vj, double *vk)

cdef class VHFOpt:
    cdef CVHFOpt *_this
    def __cinit__(self):
        self._this = NULL
    def __dealloc__(self):
        CVHFdel_optimizer(&self._this)
    def init_nr_vhf_direct(self, atm, bas, env):
        cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
        cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
        cdef numpy.ndarray[double] c_env = numpy.array(env)
        cdef int natm = c_atm.shape[0]
        cdef int nbas = c_bas.shape[0]
        nr_vhf_optimizer(&self._this, &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    property direct_scf_threshold:
        def __get__(self): return self._this.direct_scf_cutoff
        def __set__(self, v): self._this.direct_scf_cutoff = v

def vhf_jk_incore_o2(numpy.ndarray[double, ndim=2, mode='c'] eri, \
                     numpy.ndarray[double, ndim=2, mode='c'] dm):
    '''use 4-fold symmetry for eri, ijkl=ijlk=jikl=jilk'''
    cdef int nao = dm.shape[0]
    cdef numpy.ndarray[double, mode='c'] dm0 = numpy.empty(nao*(nao+1)/2)
    cdef int i, j, ij
    ij = 0
    for i in range(nao):
        for j in range(i):
            dm0[ij] = dm[i,j] * 2
            ij += 1
        dm0[ij] = dm[i,i]
        ij += 1
    cdef numpy.ndarray[double,ndim=1,mode='c'] vj0 = numpy.dot(eri, dm0)
    cdef numpy.ndarray[double,ndim=2,mode='c'] vj = numpy.zeros((nao,nao))
    CVHFunpack(nao, <double *>vj0.data, &vj[0,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] vk = numpy.zeros((nao,nao))
    nr_vhf_k(nao, &eri[0,0], &dm[0,0], &vk[0,0])
    for i in range(nao):
        for j in range(i):
            vk[j,i] = vk[i,j]
    return vj, vk

def vhf_jk_incore_o4(numpy.ndarray[double, ndim=1, mode='c'] eri, \
                     numpy.ndarray[double, ndim=2, mode='c'] dm):
    '''use 8-fold symmetry for eri'''
    cdef int nao = dm.shape[0]
    cdef numpy.ndarray[double,ndim=2,mode='c'] vj = numpy.empty((nao,nao))
    cdef numpy.ndarray[double,ndim=2,mode='c'] vk = numpy.zeros((nao,nao))
    #nr_vhf_incore_o3(nao, &eri[0], &dm[0,0], &vj[0,0], &vk[0,0])
    nr_vhf_incore_o4(nao, &eri[0], &dm[0,0], &vj[0,0], &vk[0,0])
    return vj, vk

def vhf_jk_direct_o4(numpy.ndarray[double, ndim=2, mode='c'] dm,
                     atm, bas, env, VHFOpt vhfopt=None):
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]
    cdef int nao = dm.shape[0]
    cdef numpy.ndarray[double,ndim=2,mode='c'] vj = numpy.empty((nao,nao))
    cdef numpy.ndarray[double,ndim=2,mode='c'] vk = numpy.zeros((nao,nao))
    if vhfopt is None:
        nr_vhf_direct_o4(&dm[0,0], &vj[0,0], &vk[0,0], NULL,
                         &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    else:
        nr_vhf_direct_o4(&dm[0,0], &vj[0,0], &vk[0,0], vhfopt._this,
                         &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    return vj, vk

# deal with multiple components of dm
def vhf_jk_direct_m4(numpy.ndarray[double, ndim=3, mode='c'] dm,
                     atm, bas, env, VHFOpt vhfopt=None):
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]
    cdef int nao = dm.shape[1]
    cdef int nset = dm.shape[0]
    cdef numpy.ndarray[double,ndim=3,mode='c'] vj = numpy.empty((nset,nao,nao))
    cdef numpy.ndarray[double,ndim=3,mode='c'] vk = numpy.empty((nset,nao,nao))
    if vhfopt is None:
        nr_vhf_direct_m4(&dm[0,0,0], &vj[0,0,0], &vk[0,0,0], nset, NULL,
                         &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    else:
        nr_vhf_direct_m4(&dm[0,0,0], &vj[0,0,0], &vk[0,0,0], nset, vhfopt._this,
                         &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    return vj, vk
