#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
cimport numpy
cimport cython
import numpy

cdef extern void CVHFunpack(int n, double *vec, double *mat)

def unpack_eri_tril(numpy.ndarray[double,ndim=2,mode='c'] eriblk):
    cdef int row = eriblk.shape[0]
    cdef int nmo = int(numpy.sqrt(2*eriblk.shape[1]))
    cdef int i
    cdef numpy.ndarray[double,ndim=3,mode='c'] eri = numpy.empty((row,nmo,nmo))
    for i in range(row):
        CVHFunpack(nmo, &eriblk[i,0], &eri[i,0,0])
    return eri



#########################################################
#
#########################################################


cdef extern int FCIpopcount_1(unsigned long x)
cdef extern int FCIpopcount_4(unsigned long x)

def parity(unsigned long string0, unsigned long string1):
    cdef unsigned long ss
    if string1 > string0:
        ss = string1 - string0
        # string1&ss gives the number of 1s between two strings
        if FCIpopcount_1(string1&ss) % 2:
            return -1
        else:
            return 1
    elif string1 == string0:
        return 1
    else:
        ss = string0 - string1
        if FCIpopcount_1(string0&ss) % 2:
            return -1
        else:
            return 1

def gen_linkstr_index(orb_list, int nelec, strs):
    cdef int na = len(strs)
    cdef dict strdic = dict(zip(strs,range(na)))
    cdef int norb = len(orb_list)
    cdef int[::1] orb_ids = numpy.array(orb_list, dtype=numpy.int32)
    cdef int nocc = nelec
    cdef int nvir = norb - nocc
    cdef numpy.ndarray[int,ndim=3,mode='c'] tab = \
            numpy.empty((na,nocc*nvir+nocc,4), dtype=numpy.int32)
    cdef int *ptab
    cdef int[::1] occ = numpy.empty(nocc, dtype=numpy.int32)
    cdef int[::1] vir = numpy.empty(nvir, dtype=numpy.int32)
    cdef int tab_id, io, iv, i, a, k
    cdef unsigned long str0, str1
    for tab_id in range(na):
        str0 = strs[tab_id]
        io = 0
        iv = 0
        for i in orb_ids:
            if str0 & (1<<i):
                occ[io] = i
                io += 1
            else:
                vir[iv] = i
                iv += 1
        ptab = &tab[tab_id,0,0]
        for i in range(nocc):
            ptab[i*4+0] = occ[i]
            ptab[i*4+1] = occ[i]
            ptab[i*4+2] = strdic[str0]
            ptab[i*4+3] = 1
        k = nocc
        for i in range(nocc):
            for a in range(nvir):
                str1 = str0 ^ (1<<occ[i]) | (1<<vir[a])
                ptab[k*4+0] = vir[a]
                ptab[k*4+1] = occ[i]
                ptab[k*4+2] = strdic[str1]
                ptab[k*4+3] = parity(str1,str0)
                k += 1

    return tab

#######################

# compress the a, i index, to fit the symmetry of integrals
def reform_linkstr_index(int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef int i, a, ia, j, k, str1, sign0
    cdef int [:,:,::1] link_new = numpy.zeros_like(link_index)
    cdef int [:,::1] tab
    for k in range(na):
        tab = link_index[k]
        for j in range(ntab):
            a = tab[j,0]
            i = tab[j,1]
            str1 = tab[j,2]
            sign = tab[j,3]
            if a > i:
                ia = a*(a+1)/2+i
            else:
                ia = i*(i+1)/2+a
            link_new[k,j,0] = ia
            link_new[k,j,1] = str1
            link_new[k,j,2] = sign
    return link_new

def make_hdiag(double[:,:] h1e, double[:,:,:,:] g2e,
               int norb, int nelec, int[:,:,::1] link_index):
    cdef int nocc = nelec / 2
    cdef int na = link_index.shape[0]
    cdef int ia, ib, j0, k0, j, jk0, jk
    cdef int[:,::1] occslist = numpy.array(link_index[:,:nocc,0])
    cdef int *poccs = &occslist[0,0]
    cdef int *paocc
    cdef int *pbocc
    cdef double e1, e2
    cdef double[:,::1] hdiag = numpy.empty((na,na))
    cdef double[::1] diagj = numpy.einsum('iijj->ij',g2e).flatten()
    cdef double[::1] diagk = numpy.einsum('ijji->ij',g2e).flatten()
    for ia in range(na):
        paocc = poccs + ia * nocc
        for ib in range(na):
            e1 = 0
            e2 = 0
            pbocc = poccs + ib * nocc
            for j0 in range(nocc):
                j = paocc[j0]
                jk0 = j * norb
                e1 += h1e[j,j]
                for k0 in range(nocc):
                    jk = jk0+paocc[k0]
                    e2 += diagj[jk] - diagk[jk]
                for k0 in range(nocc):
                    jk = jk0+pbocc[k0]
                    e2 += diagj[jk] * 2
            for j0 in range(nocc):
                j = pbocc[j0]
                jk0 = j * norb
                e1 += h1e[j,j]
                for k0 in range(nocc):
                    jk = jk0+pbocc[k0]
                    e2 += diagj[jk] - diagk[jk]
            hdiag[ia,ib] = e1 + e2*.5
    return numpy.asarray(hdiag).reshape(-1)


######################################################

def contract_1e_spin0(numpy.ndarray[double,ndim=2] f1e,
                      ci0, int norb, int[:,:,::1] link_index):
    cdef int i, a, ia, j, k, str0, str1, sign
    cdef int na = link_index.shape[0]
    cdef int nnorb = norb*(norb+1)/2
    cdef int ntab = link_index.shape[1]
    cdef double[::1] f1e_tril = numpy.empty(nnorb)
    cdef numpy.ndarray[double,ndim=2] fcivec = ci0.reshape(na,na)
    cdef numpy.ndarray[double,ndim=2] cipart = numpy.zeros((na,na))
    cdef double[:,::1] t1 = numpy.empty((nnorb,na))
    cdef double[::1] t2
    cdef int[:,::1] tab
    cdef double *pt1 = &t1[0,0]
    cdef double *pcp0
    cdef double *pcp1
    cdef int[:,:,::1] link_new = reform_linkstr_index(link_index)
    cdef int *pindex = &link_new[0,0,0]
    cdef int *ptab
    cdef double *pci0 = &fcivec[0,0]
    cdef double *pci1 = &cipart[0,0]
    cdef double tmp

    ia = 0
    for i in range(norb):
        for a in range(i+1):
            f1e_tril[ia] = f1e[i,a]
            ia += 1

    for str0 in range(na):
        ptab = pindex + str0 * ntab*4
        for j in range(ntab):
            ia = ptab[j*4+0]
            str1 = ptab[j*4+1]
            sign = ptab[j*4+2]
            pcp0 = pci0 + str0*na
            pcp1 = pci1 + str1*na
            tmp = sign * f1e_tril[ia]
            for k in range(na):
                pcp1[k] += tmp * pcp0[k]

# contracted hc = cipart + cipart.T
    return cipart

cdef extern void FCIcontract_2e_spin0(double *eri, double *ci0, double *ci1,
                                      int norb, int na, int nov, int *link_index,
                                      int buf_size)

def contract_2e_spin0_omp(numpy.ndarray eri, numpy.ndarray ci0,
                          int norb, int[:,:,::1] link_index, buf_size=1024):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef numpy.ndarray cipart = numpy.zeros((na,na))
    cdef int[:,:,::1] link_new = reform_linkstr_index(link_index)
    FCIcontract_2e_spin0(<double *>eri.data, <double *>ci0.data,
                         <double *>cipart.data, norb, na, ntab,
                         &link_new[0,0,0], buf_size)

# contracted hc = cipart + cipart.T
    return cipart



#########################################################
#
#########################################################

def reorder_rdm(numpy.ndarray[double,ndim=2,mode='c'] rdm1,
                numpy.ndarray[double,ndim=4,mode='c'] rdm2):
    cdef int nmo = rdm1.shape[0]
    cdef int k
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    rdm2 = (rdm2 + rdm2.transpose(2,3,0,1)) * .5
    return rdm1, rdm2

# compress the a, i index, to fit the symmetry of integrals
def reform_linkstr_index_dm(int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef int i, a, ia, j, j1, k, str1, sign0
    cdef int [:,:,::1] link_new = numpy.zeros_like(link_index)
    cdef int [:,::1] tab
    for k in range(na):
        tab = link_index[k]
        j1 = 0
        for j in range(ntab):
            a = tab[j,0]
            i = tab[j,1]
            str1 = tab[j,2]
            sign = tab[j,3]
            if a >= i:
                link_new[k,j1,0] = a*(a+1)/2+i
                link_new[k,j1,1] = str1
                link_new[k,j1,2] = sign
                j1 += 1
        link_new[k,0,3] = j1
    return link_new

def make_rdm1_spin0_o3(fcivec, int norb, int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef int i, a, ia, j, k, str0, str1, sign
    cdef int *pindex = &link_index[0,0,0]
    cdef int *ptab
    cdef double[:,::1] ci0 = fcivec.reshape(na,na)
    cdef double *pci0
    cdef double *pci1
    cdef double ctmp
    cdef numpy.ndarray[double,ndim=2,mode='c'] rdm1 = numpy.zeros((norb,norb))
    cdef double *pdm1 = &rdm1[0,0]
    for str0 in range(na):
        ptab = pindex + str0 * ntab*4
        pci0 = &ci0[str0,0]
        for j in range(ntab):
            a = ptab[j*4+0]
            i = ptab[j*4+1]
            str1 = ptab[j*4+2]
            sign = ptab[j*4+3]
            pci1 = &ci0[str1,0]
            if str1 > str0:
                if sign > 0:
                    for k in range(na):
                        pdm1[a*norb+i] += pci0[k]*pci1[k]*2
                else:
                    for k in range(na):
                        pdm1[a*norb+i] -= pci0[k]*pci1[k]*2
            elif str1 == str0:
                if sign > 0:
                    for k in range(na):
                        pdm1[a*norb+i] += pci0[k]*pci1[k]
                else:
                    for k in range(na):
                        pdm1[a*norb+i] -= pci0[k]*pci1[k]
    return rdm1 + rdm1.T

cdef extern void FCImake_rdm12_spin0_o3(double *rdm1, double *rdm2, double *ci0,
                                        int norb, int na, int nov, int *link_index)

def make_rdm12_spin0_omp(numpy.ndarray fcivec, int norb, int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef numpy.ndarray[double,ndim=2,mode='c'] rdm1 = numpy.zeros((norb,norb))
    cdef numpy.ndarray[double,ndim=4,mode='c'] rdm2 = numpy.zeros((norb,)*4)
    FCImake_rdm12_spin0_o3(<double *>rdm1.data, <double *>rdm2.data,
                           <double *>fcivec.data, norb, na, ntab,
                           &link_index[0,0,0])

    rdm2 = numpy.array(rdm2.transpose(1,0,2,3), order='C')
    return reorder_rdm(rdm1, rdm2)


#######################################################

def trans_rdm1_spin0_o2(cibra, ciket, int norb, int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef int i, a, j, k, str0, str1, sign
    cdef int *pindex = &link_index[0,0,0]
    cdef int *ptab
    cdef double[:,::1] ci0 = ciket.reshape(na,na)
    cdef double[:,::1] ci1 = cibra.reshape(na,na)
    cdef double *pci0
    cdef double *pci1
    cdef double ctmp
    cdef numpy.ndarray[double,ndim=2,mode='c'] rdm1 = numpy.zeros((norb,norb))
    cdef double *pdm1 = &rdm1[0,0]
    for str0 in range(na):
        ptab = pindex + str0 * ntab*4
        for j in range(ntab):
            a = ptab[j*4+0]
            i = ptab[j*4+1]
            str1 = ptab[j*4+2]
            sign = ptab[j*4+3]
            pci0 = &ci0[str1,0]
            pci1 = &ci1[str0,0]
            for k in range(str0+1):
                pdm1[i*norb+a] += sign * pci1[k]*pci0[k]
        for k in range(str0):
            ptab = pindex + k * ntab*4
            ctmp = ci1[str0,k]
            pci0 = &ci0[str0,0]
            for j in range(ntab):
                a = ptab[j*4+0]
                i = ptab[j*4+1]
                str1 = ptab[j*4+2]
                sign = ptab[j*4+3]
                pdm1[i*norb+a] += sign * ctmp*pci0[str1]
    return rdm1 * 2

cdef extern void FCItrans_rdm12_spin0_o3(double *rdm1, double *rdm2,
                                         double *bra, double *ket,
                                         int norb, int na, int nov, int *link_index)

def trans_rdm12_spin0_omp(numpy.ndarray cibra, numpy.ndarray ciket,
                          int norb, int[:,:,::1] link_index):
    cdef int na = link_index.shape[0]
    cdef int ntab = link_index.shape[1]
    cdef numpy.ndarray[double,ndim=2,mode='c'] rdm1 = numpy.zeros((norb,norb))
    cdef numpy.ndarray[double,ndim=4,mode='c'] rdm2 = numpy.zeros((norb,)*4)
    FCItrans_rdm12_spin0_o3(<double *>rdm1.data, <double *>rdm2.data,
                            <double *>cibra.data, <double *>ciket.data,
                            norb, na, ntab, &link_index[0,0,0])

    rdm2 = numpy.array(rdm2.transpose(1,0,2,3), order='C')
    return reorder_rdm(rdm1, rdm2)

