#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
import numpy
cimport numpy
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free

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


#######################################
cdef extern void nr_e1_ao2mo_o1(double *eri, double *mo_coeff,
                                int i_start, int i_count, int j_start, int j_count,
                                int *atm, int natm,
                                int *bas, int nbas, double *env)
cdef extern void nr_e2_ao2mo_o1(const int nrow, double *vout, double *vin,
                                double *mo_coeff, int nao,
                                int i_start, int i_count, int j_start, int j_count)
cdef extern void trans_e2_tri_o1(double *vout, double *vin, double *mo_coeff, int nao,
                                 int i_start, int i_count, int j_start, int j_count) nogil
cdef extern void nr_e1_ao2mo_o2(double *eri, double *mo_coeff,
                                int i_start, int i_count, int j_start, int j_count,
                                int *atm, int natm,
                                int *bas, int nbas, double *env)
cdef extern void nr_e2_ao2mo_o2(const int nrow, double *vout, double *vin,
                                double *mo_coeff, int nao,
                                int i_start, int i_count, int j_start, int j_count)
cdef extern void trans_e2_tri_o2(double *vout, double *vin, double *mo_coeff, int nao,
                                 int i_start, int i_count, int j_start, int j_count) nogil

def _count_ij(shape):
    istart, icount, jstart, jcount = shape
    if jstart+jcount <= istart:
        ntri = 0
    else:
        noff = jstart+jcount - (istart + 1)
        ntri = noff*(noff+1)/2
    return icount*jcount - ntri

def nr_eri_e1mo(numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff,
                shape, atm, bas, env):
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]

    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic)
    nao = mo_coeff.shape[0]
    nao_pair = nao*(nao+1)/2

    cdef numpy.ndarray[double,ndim=2] eri = numpy.empty((_count_ij(shape),nao_pair))
    if ic <= jc:
        nr_e1_ao2mo_o2(&eri[0,0], &mo_coeff[0,0], i0, ic, j0, jc,
                       &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    else:
        nr_e1_ao2mo_o1(&eri[0,0], &mo_coeff[0,0], i0, ic, j0, jc,
                       &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    return eri

def nr_e1_ao2mo(numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff,
                shape, atm, bas, env):
    return nr_eri_e1mo(mo_coeff, shape, atm, bas, env)

def nr_ao2mo_tri(numpy.ndarray vin,
                 numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff, shape):
    cdef numpy.ndarray vout = numpy.empty(_count_ij(shape))
    cdef int nao = mo_coeff.shape[0]
    i0, ic, j0, jc = shape
    if ic <= jc:
        trans_e2_tri_o2(<double *>vout.data, <double *>vin.data,
                        &mo_coeff[0,0], nao, i0, ic, j0, jc)
    else:
        trans_e2_tri_o1(<double *>vout.data, <double *>vin.data,
                        &mo_coeff[0,0], nao, i0, ic, j0, jc)
    return vout

# in-place transform AO to MO
def nr_e2_ao2mo(numpy.ndarray[double,ndim=2] eri,
                numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff, shape):
    cdef int nao = mo_coeff.shape[0]

    i0, ic, j0, jc = shape
    assert(j0 <= i0)
    assert(j0+jc <= i0+ic)
    nao = mo_coeff.shape[0]
    nao_pair = nao*(nao+1)/2
    nrow = eri.shape[0]
    nij = _count_ij(shape)

    cdef numpy.ndarray vout
    if nij < nao_pair: # we can reuse memory
        vout = eri.reshape(-1)[:nrow*nij].reshape(nrow,nij)
    else:
        vout = numpy.empty((nrow,nij))

    if ic <= jc:
        nr_e2_ao2mo_o2(nrow, <double *>vout.data, &eri[0,0],
                       &mo_coeff[0,0], nao, i0, ic, j0, jc)
    else:
        nr_e2_ao2mo_o1(nrow, <double *>vout.data, &eri[0,0],
                       &mo_coeff[0,0], nao, i0, ic, j0, jc)
    return vout


#######################################

cdef extern void ao2mo_half_trans_o2(int nao, int nmo, double *eri, double *c,
                                     double *mat) nogil
cdef extern void ao2mo_half_trans_o3(int nao, int nmo, int pair_id,
                                     double *eri, double *c, double *mat) nogil

cdef extern void extract_row_from_tri_eri(double *row, unsigned int row_id,
                                          double *eri, unsigned int npair) nogil

def partial_eri_ao2mo_o2(numpy.ndarray[double,ndim=2] eri_ao,
                         numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff):
    cdef int nao = mo_coeff.shape[0]
    cdef int nmo = mo_coeff.shape[1]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nmo_pair = nmo*(nmo+1)/2
    cdef int ij
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nmo_pair))
    with nogil, parallel():
        for ij in prange(nao_pair, schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo, &eri_ao[ij,0], &mo_coeff[0,0],
                                &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    with nogil, parallel():
        for ij in prange(nmo_pair, schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_coeff[0,0],
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
    with nogil, parallel():
        for ij in prange(nao_pair, schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo, &eri_ao[ij,0], &mo_b[0,0],
                                &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    with nogil, parallel():
        for ij in prange(nmo_pair, schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_a[0,0],
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
    with nogil, parallel():
        for ij in prange(nao_pair, schedule='guided'):
            ao2mo_half_trans_o3(nao, nmo, ij, &eri_ao[0], &mo_coeff[0,0],
                                &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_pair,nmo_pair))
    eri1 = numpy.array(eri1.T,order='C')
    with nogil, parallel():
        for ij in prange(nmo_pair, schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo, &eri1[ij,0], &mo_coeff[0,0],
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
    with nogil, parallel():
        for ij in prange(nao_pair, schedule='guided'):
            ao2mo_half_trans_o3(nao, nmo_b, ij, &eri_ao[0], &mo_b[0,0],
                                &eri1[ij,0])

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2 = numpy.empty((nmo_bpair,nmo_apair))
    eri1 = numpy.array(eri1.T,order='C')
    with nogil, parallel():
        for ij in prange(nmo_bpair,schedule='guided'):
            ao2mo_half_trans_o2(nao, nmo_a, &eri1[ij,0], &mo_a[0,0],
                                &eri2[ij,0])
    return eri2


def nr_e1_ao2mo_incore(numpy.ndarray[double,ndim=1] eri_ao,
                       numpy.ndarray[double,ndim=2,mode='fortran'] mo_coeff, shape):
    cdef int nao = mo_coeff.shape[0]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int ij
    cdef int nij = _count_ij(shape)
    cdef int i0, ic, j0, jc
    i0, ic, j0, jc = shape
    cdef double *buf

    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,nij))
    with nogil, parallel():
        for ij in prange(nao_pair, schedule='guided'):
            buf = <double *>malloc(sizeof(double)*nao_pair)
            extract_row_from_tri_eri(buf, ij, &eri_ao[0], nao_pair)
            if ic <= jc:
                trans_e2_tri_o2(&eri1[ij,0], buf, &mo_coeff[0,0], nao,
                                i0, ic, j0, jc)
            else:
                trans_e2_tri_o1(&eri1[ij,0], buf, &mo_coeff[0,0], nao,
                                i0, ic, j0, jc)
            free(buf)
    return numpy.array(eri1.T,order='C')
