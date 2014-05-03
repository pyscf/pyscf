#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
import numpy
cimport numpy
cimport cython

cdef extern int CINTtot_cgto_spheric(const int *bas, const int nbas)

cdef extern void int2e_sph_o4(double *eri, int *atm, int natm,
                              int *bas, int nbas, double *env)

def int2e_sph_8fold(atm, bas, env):
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]
    nao = CINTtot_cgto_spheric(&c_bas[0,0], nbas)
    nao_pair = nao*(nao+1)/2
    cdef numpy.ndarray[double,ndim=1] eri = numpy.empty((nao_pair*(nao_pair+1)/2))
    int2e_sph_o4(&eri[0], &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    return eri

cdef extern void ci_misc_half_trans_o2(int nao, int nmo, double *eri, double *c,
                                       double *mat)
cdef extern void ci_misc_half_trans_o3(int nao, int nmo, int pair_id,
                                       double *eri, double *c, double *mat)

def partial_eri_ao2mo_o2(numpy.ndarray[double,ndim=2] eri_ao,
                         numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff):
    cdef int nao = mo_coeff.shape[0]
    cdef int nmo = mo_coeff.shape[1]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nmo_pair = nmo*(nmo+1)/2
    cdef int ij
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nmo_pair))
    for ij in range(nao_pair):
        ci_misc_half_trans_o2(nao, nmo, &eri_ao[ij,0], &mo_coeff[0,0],
                              &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    for ij in range(nmo_pair):
        ci_misc_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_coeff[0,0],
                              &eri2[ij,0])
    return eri2

def partial_eri_ao2mo_ab_o2(numpy.ndarray[double,ndim=2] eri_ao,
                            numpy.ndarray[double,ndim=2,mode='fortran'] mo_a,
                            numpy.ndarray[double,ndim=2,mode='fortran'] mo_b):
    ''' integral (AA|BB) = [AA,BB] in Fortran continues '''
    assert(mo_a.shape[1] == mo_b.shape[1])
    cdef int nao = mo_a.shape[0]
    cdef int nmo = mo_a.shape[1]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nmo_pair = nmo*(nmo+1)/2
    cdef int ij
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nmo_pair))
    # first transform mo_b, because the following mo_a can produce (AA|BB) in
    # Fortran continues
    for ij in range(nao_pair):
        ci_misc_half_trans_o2(nao, nmo, &eri_ao[ij,0], &mo_b[0,0],
                              &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    for ij in range(nmo_pair):
        ci_misc_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_a[0,0],
                              &eri2[ij,0])
    return eri2


def partial_eri_ao2mo_o3(numpy.ndarray[double,ndim=1] eri_ao,
                         numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff):
    cdef int nao = mo_coeff.shape[0]
    cdef int nmo = mo_coeff.shape[1]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nmo_pair = nmo*(nmo+1)/2
    cdef int ij
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nmo_pair))
    for ij in range(nao_pair):
        ci_misc_half_trans_o3(nao, nmo, ij, &eri_ao[0], &mo_coeff[0,0],
                              &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    for ij in range(nmo_pair):
        ci_misc_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_coeff[0,0],
                              &eri2[ij,0])
    return eri2

def partial_eri_ao2mo_ab_o3(numpy.ndarray[double,ndim=1] eri_ao,
                            numpy.ndarray[double,ndim=2,mode='fortran'] mo_a,
                            numpy.ndarray[double,ndim=2,mode='fortran'] mo_b):
    ''' integral (AA|BB) = [AA,BB] in Fortran continues '''
    assert(mo_a.shape[1] == mo_b.shape[1])
    cdef int nao = mo_a.shape[0]
    cdef int nmo_a = mo_a.shape[1]
    cdef int nmo_b = mo_b.shape[1]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nmo_apair = nmo_a*(nmo_a+1)/2
    cdef int nmo_bpair = nmo_b*(nmo_b+1)/2
    cdef int ij
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nmo_bpair))
    # first transform mo_b, because the following mo_a can produce (AA|BB) in
    # Fortran continues
    for ij in range(nao_pair):
        ci_misc_half_trans_o3(nao, nmo_b, ij, &eri_ao[0], &mo_b[0,0],
                              &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_bpair,nmo_apair))
    eri1 = numpy.array(eri1.T,order='C')
    for ij in range(nmo_bpair):
        ci_misc_half_trans_o2(nao, nmo_a, &eri1[ij,0], &mo_a[0,0],
                              &eri2[ij,0])
    return eri2

