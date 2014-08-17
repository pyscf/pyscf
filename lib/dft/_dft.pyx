#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
import numpy
cimport numpy
cimport cython

cdef extern int MakeAngularGrid(double *Out, int nPoints)

def make_angular_grid(n_angpt):
    cdef numpy.ndarray[double,ndim=2] grid = numpy.empty((n_angpt,4))
    MakeAngularGrid(&grid[0,0], n_angpt)
    return grid[:,:3], grid[:,3]


#######################

cdef extern from "vxc.h":
    double VXChybrid_coeff(int xc_id, int spin)
    double VXCnr_vxc(int x_id, int c_id, int spin, int relativity,
                     double *dm, double *exc, double *v,
                     int num_grids, double *coords, double *weights,
                     int *atm, int natm, int *bas, int nbas, double *env)


# spin = 1, unpolarized; spin = 2, polarized
def hybrid_coeff(xc_id, spin):
    return VXChybrid_coeff(xc_id, spin)

#######################
# for general DM
# hermi = 1 : Hermitian
# hermi = 2 : anti-Hermitian
#######################
def nr_vxc(x_id, c_id, spin, relativity, numpy.ndarray dm, \
           numpy.ndarray[double,ndim=2,mode='c'] coords, numpy.ndarray weights,
           atm, bas, env, hermi=1):
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]
    cdef double exc
    dm = numpy.array(dm, copy=False, order='F')
    cdef numpy.ndarray v = numpy.empty_like(dm, order='F')
    nelec = VXCnr_vxc(x_id, c_id, spin, relativity,
                      <double *>dm.data, &exc, <double *>v.data, 
                      weights.size, <double *>coords.data, <double *>weights.data,
                      &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    cdef int i, j, nd = dm.shape[1]
    if hermi == 1:
        for i in range(nd):
            for j in range(i):
                v[i,j] = v[j,i]
    else:
        raise('anti-Hermitian is not supported')
    return nelec, exc, v
