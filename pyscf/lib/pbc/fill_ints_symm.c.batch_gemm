#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>
#include <mkl.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "pbc/optimizer.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2
#define BLKSIZE         175

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define GRP_COUNT 1

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

void sort3c_kks1(double complex *out, double *bufr, double *bufi,
                        int *kptij_idx, int *shls_slice, int *ao_loc,
                        int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                        int msh0, int msh1);

void sort3c_kks2_igtj(double complex *out, double *bufr, double *bufi,
                             int *kptij_idx, int *shls_slice, int *ao_loc,
                             int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                             int msh0, int msh1);


static void kron(double *c, double *a, double *b, int nrowa, int ncola, int nrowb, int ncolb)
{
    //a has dimension of [nrowa, ncola] with lda = nrowa
    //b has dimension of [nrowb, ncolb] with ldb = nrowb
    //c has dimension of [nrowa*nrowb, ncola*ncolb] with ldc = nrowa*nrowb
    int i,j;
    int One = 1;
    double D1 = 1.;
    for (i=0; i<nrowa*nrowb*ncola*ncolb; i++) {
        c[i] = 0;
    }
    double *pc = c, *pa, *pb;
    for (i=0; i<ncola; i++){
        pa = a + (size_t)nrowa*i;
        for (j=0; j<ncolb; j++){
            pb = b + (size_t)nrowb*j;
            dger_(&nrowb, &nrowa, &D1, pb, &One, pa, &One, pc, &nrowb);
            pc += nrowa*nrowb;
        }
    }
}

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
        env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
        env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
        env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

static int get_shltrip_idx(int *shlcen, int *shltrip_cen_idx, int nshltrip)
{
    int idx = -1;
    int i;
    for (i=0; i<nshltrip; i++)
    {
        if (shlcen[0] == shltrip_cen_idx[0+i*3] &&
            shlcen[1] == shltrip_cen_idx[1+i*3] &&
            shlcen[2] == shltrip_cen_idx[2+i*3]){
            idx = i;
            break;
        }
    }
    assert(idx >= 0);
    return idx;
}

static void build_block_diag_mat(double* A, int dA, double* B, int dB, int nc)  
{                                                                               
    int i,j,k,off;                                                              
    for(i=0; i<dA*dA; i++){                                                     
        A[i] = 0.0;                                                             
    }                                                                           
    for(k=0; k<nc; k++){                                                        
        off = k*dA*dB + k*dB;                                                
        for(i=0; i<dB; i++){                                                    
            for(j=0; j<dB; j++){                                                
                A[off+i*dA+j] = B[i*dB+j];                                      
    }}}                                                                         
}            

static void multiply_Dmats(double* Tkji, double *Dmats, int *rot_loc, int rot_mat_size, int nop, int iop, 
                           int di, int dj, int dk, int l_i, int l_j, int l_k,
                           int nci, int ncj, int nck)
{
    double *pDmats, *pTk, *pTj, *pTi;
    pDmats = Dmats+(size_t)rot_mat_size*iop;
    pTk = pDmats + rot_loc[l_k];
    pTj = pDmats + rot_loc[l_j];
    pTi = pDmats + rot_loc[l_i];

    int dim_Tk = dk*nck;
    int dim_Tj = dj*ncj;
    int dim_Ti = di*nci;
    double *Tk, *Tj, *Ti;
    if (nck==1) {Tk = pTk;}
    else {
        Tk = malloc(sizeof(double)*dim_Tk*dim_Tk);
        build_block_diag_mat(Tk, dim_Tk, pTk, dk, nck);
    }
    if (ncj==1) {Tj = pTj;}
    else {
        Tj = malloc(sizeof(double)*dim_Tj*dim_Tj);
        build_block_diag_mat(Tj, dim_Tj, pTj, dj, ncj);
    }
    if (nci==1) {Ti = pTi;}
    else {
        Ti = malloc(sizeof(double)*dim_Ti*dim_Ti);
        build_block_diag_mat(Ti, dim_Ti, pTi, di, nci);
    }

    int dim_Tkj = dim_Tk*dim_Tj;
    double *Tkj = malloc(sizeof(double)*dim_Tkj*dim_Tkj);
    kron(Tkj, Tk, Tj, dim_Tk, dim_Tk, dim_Tj, dim_Tj);

    int dkji = dim_Tkj*dim_Ti;
    kron(Tkji+(size_t)dkji*dkji*iop, Tkj, Ti, dim_Tkj, dim_Tkj, dim_Ti, dim_Ti);

    free(Tkj);
    if (nck>1) free(Tk);
    if (ncj>1) free(Tj);
    if (nci>1) free(Ti);
}

static void multiply_two_Dmats(double* Tji, double *Dmats, int *rot_loc, int rot_mat_size, int nop, int iop,
                               int di, int dj, int l_i, int l_j, int nci, int ncj)
{
    double *pDmats, *pTj, *pTi;
    pDmats = Dmats+(size_t)rot_mat_size*iop;
    pTj = pDmats + rot_loc[l_j];
    pTi = pDmats + rot_loc[l_i];

    int dim_Tj = dj*ncj;
    int dim_Ti = di*nci;
    double *Tj, *Ti;
    if (ncj==1) {Tj = pTj;}
    else {
        Tj = malloc(sizeof(double)*dim_Tj*dim_Tj);
        build_block_diag_mat(Tj, dim_Tj, pTj, dj, ncj);
    }
    if (nci==1) {Ti = pTi;}
    else {
        Ti = malloc(sizeof(double)*dim_Ti*dim_Ti);
        build_block_diag_mat(Ti, dim_Ti, pTi, di, nci);
    }

    int dji = dim_Tj*dim_Ti;
    kron(Tji+(size_t)dji*dji*iop, Tj, Ti, dim_Tj, dim_Tj, dim_Ti, dim_Ti);

    if (ncj>1) free(Tj);
    if (nci>1) free(Ti);
}


static void build_Dmats(double** pD, double *Dmats, int *rot_loc, int rot_mat_size, int nop, int l)
{
    int iop;
    for (iop=0; iop<nop; iop++){
        pD[iop] = Dmats+(size_t)rot_mat_size*iop + rot_loc[l];
    }
}

static void build_Tmats(double* T, double *Dmats, int *rot_loc, int rot_mat_size, int nop, int l, int d, int nc)
{
    int dimT = d*nc;
    double *pDmats, *pT=T;
    int iop;
    for (iop=0; iop<nop; iop++){
        pDmats = Dmats+(size_t)rot_mat_size*iop + rot_loc[l];
        build_block_diag_mat(pT, dimT, pDmats, d, nc);
        pT += dimT*dimT;
    }
}

static void apply_Tmats(double* out, double* ints, double* Ti, double* Tj, double* Tk,
                           int dim_Ti, int dim_Tj, int dim_Tk) 
{                                                                               
    double *tmp1 = malloc(sizeof(double) * dim_Tk*dim_Tj*dim_Ti);
    const char TRANS_N = 'N';                                                   
    const char TRANS_T = 'T';                                                   
    const double D0 = 0;                                                        
    const double D1 = 1;
    int k;
    double *tmp = malloc(sizeof(double) * dim_Tj*dim_Ti);
    for (k=0; k<dim_Tk; k++) {
        double *pint = ints + k*dim_Tj*dim_Ti;
        double *ptmp1 = tmp1 + k*dim_Tj*dim_Ti;
        dgemm_(&TRANS_N, &TRANS_N, &dim_Ti, &dim_Tj, &dim_Ti, 
               &D1, Ti, &dim_Ti, pint, &dim_Ti,
               &D0, tmp, &dim_Ti);
        dgemm_(&TRANS_N, &TRANS_T, &dim_Ti, &dim_Tj, &dim_Tj,                   
               &D1, tmp, &dim_Ti, Tj, &dim_Tj,                                 
               &D0, ptmp1, &dim_Ti); 
    }
    free(tmp);

    int dim_TiTj = dim_Ti * dim_Tj;
    dgemm_(&TRANS_N, &TRANS_T, &dim_TiTj, &dim_Tk, &dim_Tk,                   
           &D1, tmp1, &dim_TiTj, Tk, &dim_Tk,                                  
           &D0, out, &dim_TiTj);                                                                           

    free(tmp1);                                                              
}

static void apply_Dmats(double* out, double* ints, double *Dmats, int *rot_loc, int rot_mat_size, int iop,
                           int di, int dj, int dk, int l_i, int l_j, int l_k,   
                           int nci, int ncj, int nck)                           
{                                                                               
    double *pDmats, *pTk, *pTj, *pTi;                                           
    pDmats = Dmats+(size_t)rot_mat_size*iop;                                    
    pTk = pDmats + rot_loc[l_k];                                                
    pTj = pDmats + rot_loc[l_j];                                                
    pTi = pDmats + rot_loc[l_i];                                                
                                                                                
    int dim_Tk = dk*nck;                                                        
    int dim_Tj = dj*ncj;                                                        
    int dim_Ti = di*nci;                                                        
    double *Tk, *Tj, *Ti;                                                       
    if (nck==1) {Tk = pTk;}                                                     
    else {                                                                      
        Tk = malloc(sizeof(double)*dim_Tk*dim_Tk);                              
        build_block_diag_mat(Tk, dim_Tk, pTk, dk, nck);                         
    }
    if (ncj==1) {Tj = pTj;}                                                     
    else {
        Tj = malloc(sizeof(double)*dim_Tj*dim_Tj);                              
        build_block_diag_mat(Tj, dim_Tj, pTj, dj, ncj);                         
    } 
    if (nci==1) {Ti = pTi;}                                                     
    else {
        Ti = malloc(sizeof(double)*dim_Ti*dim_Ti);                              
        build_block_diag_mat(Ti, dim_Ti, pTi, di, nci);                         
    }

    double *tmp1 = malloc(sizeof(double) * dim_Tk*dim_Tj*dim_Ti);
    const char TRANS_N = 'N';                                                   
    const char TRANS_T = 'T';                                                   
    const double D0 = 0;                                                        
    const double D1 = 1;
    //#pragma omp parallel num_threads(2)
    //{
    int k,i,j;
    double *tmp = malloc(sizeof(double) * dim_Tj*dim_Ti);
    //#pragma omp for schedule(static) 
    for (k=0; k<dim_Tk; k++) {
        double *pint = ints + k*dim_Tj*dim_Ti;
        double *ptmp1 = tmp1 + k*dim_Tj*dim_Ti;
        dgemm_(&TRANS_N, &TRANS_N, &dim_Ti, &dim_Tj, &dim_Ti, 
               &D1, Ti, &dim_Ti, pint, &dim_Ti,
               &D0, tmp, &dim_Ti);
        dgemm_(&TRANS_N, &TRANS_T, &dim_Ti, &dim_Tj, &dim_Tj,                   
               &D1, tmp, &dim_Ti, Tj, &dim_Tj,                                 
               &D0, ptmp1, &dim_Ti); 
        /*
        for (i=0; i<nci; i++){
            dgemm_(&TRANS_N, &TRANS_N, &di, &dim_Tj, &di,                   
                   &D1, Ti, &di, pint+i*di, &dim_Ti,                                 
                   &D0, tmp+i*di, &dim_Ti);
        }
        for (j=0; j<ncj; j++){
            dgemm_(&TRANS_N, &TRANS_T, &dim_Ti, &dj, &dj,                   
                   &D1, tmp+dim_Ti*j*dj, &dim_Ti, Tj, &dj,                                  
                   &D0, ptmp1+dim_Ti*j*dj, &dim_Ti);
        }*/
    }
    free(tmp);
    //}

    int dim_TiTj = dim_Ti * dim_Tj;
    dgemm_(&TRANS_N, &TRANS_T, &dim_TiTj, &dim_Tk, &dim_Tk,                   
           &D1, tmp1, &dim_TiTj, Tk, &dim_Tk,                                  
           &D0, out, &dim_TiTj);                                                                           

    free(tmp1);                                                              
    if (nck>1) free(Tk);                                                        
    if (ncj>1) free(Tj);                                                        
    if (nci>1) free(Ti);                                                        
}

static void apply_Tmats_batch_gemm(double* out, double* Tkji, double* ints, int* op_idx, int* L2_off, int dijk, int nL)
{
    MKL_INT    m[GRP_COUNT] = {dijk};
    MKL_INT    k[GRP_COUNT] = {dijk};
    MKL_INT    n[GRP_COUNT] = {1};

    MKL_INT    lda[GRP_COUNT] = {dijk};
    MKL_INT    ldb[GRP_COUNT] = {dijk};
    MKL_INT    ldc[GRP_COUNT] = {dijk};

    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans};

    double    alpha[GRP_COUNT] = {1.0};
    double    beta[GRP_COUNT] = {0.0};

    MKL_INT    size_per_grp[GRP_COUNT] = {nL};

    const double *a_array[nL], *b_array[nL];
    double *c_array[nL];
    int iL;
    for (iL=0; iL<nL; iL++){
        int iop = op_idx[iL];
        int idx_L2 = L2_off[iL];
        a_array[iL] = Tkji+(size_t)dijk*dijk*iop; 
        b_array[iL] = ints+(size_t)dijk*idx_L2; 
        c_array[iL] = out+(size_t)dijk*iL;
    }

    cblas_dgemm_batch (CblasColMajor, transA, transB, m, n, k,
        alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
        GRP_COUNT, size_per_grp);
}

static void apply_Dmats_batch_gemm(double* out, double* ints, double** Di, double** Dj, double** Dk,
                                   int* op_idx, int* L2_off, int nL, 
                                   int mi, int mj, int mk, int nci, int ncj, int nck)
{
    int di = mi*nci; 
    int dj = mj*ncj; 
    int dk = mk*nck;
    int dij = di*dj;
    int dijk = dij*dk;

    /************
    * apply Di  *
    ************/
    MKL_INT    m1[GRP_COUNT] = {mi};
    MKL_INT    k1[GRP_COUNT] = {mi};
    MKL_INT    n1[GRP_COUNT] = {dj};
    MKL_INT    lda1[GRP_COUNT] = {mi};
    MKL_INT    ldb1[GRP_COUNT] = {di};
    MKL_INT    ldc1[GRP_COUNT] = {di};
    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans};
    double    alpha[GRP_COUNT] = {1.0};
    double    beta[GRP_COUNT] = {0.0};
    MKL_INT    size_per_grp1[GRP_COUNT] = {nL*dk*nci};
    const double *amat1[nL*dk*nci], *bmat1[nL*dk*nci];
    double *cmat1[nL*dk*nci];
    int iL, k, ic, ioff=0;
    //int iop, idx_L2;
    double *pD, *pints;
    double *tmp1 = malloc(sizeof(double)*dijk*nL);
    double *ptmp1;
    for (iL=0; iL<nL; iL++) {
        pD = Di[op_idx[iL]];
        pints = ints + (size_t)dijk*L2_off[iL];
        ptmp1 = tmp1 + (size_t)dijk*iL;
        for (k=0; k<dk; k++) {
            double *pints_ij = pints + (size_t)dij*k;
            double *ptmp1_ij = ptmp1 + (size_t)dij*k;
            for (ic=0; ic<nci; ic++) {
                amat1[ioff] = pD; 
                bmat1[ioff] = pints_ij + (size_t)mi*ic;
                cmat1[ioff] = ptmp1_ij + (size_t)mi*ic;
                ioff++;
            }
        }
    }

    cblas_dgemm_batch(CblasColMajor, transA, transB, m1, n1, k1, 
        alpha, amat1, lda1, bmat1, ldb1, beta, cmat1, ldc1,
        GRP_COUNT, size_per_grp1);

    /************
    * apply Dj  *
    ************/
    MKL_INT    m2[GRP_COUNT] = {di};
    MKL_INT    k2[GRP_COUNT] = {mj};
    MKL_INT    n2[GRP_COUNT] = {mj};
    MKL_INT    lda2[GRP_COUNT] = {di};
    MKL_INT    ldb2[GRP_COUNT] = {mj};
    MKL_INT    ldc2[GRP_COUNT] = {di};
    CBLAS_TRANSPOSE    transB2[GRP_COUNT] = {CblasTrans};
    MKL_INT    size_per_grp2[GRP_COUNT] = {nL*dk*ncj};
    const double *amat2[nL*dk*ncj], *bmat2[nL*dk*ncj];
    double *cmat2[nL*dk*ncj];
    double *tmp2 = malloc(sizeof(double)*dijk*nL);
    double *ptmp2;
    ioff = 0;
    for (iL=0; iL<nL; iL++) {
        pD = Dj[op_idx[iL]];
        pints = tmp1 + (size_t)dijk*iL;
        ptmp2 = tmp2 + (size_t)dijk*iL;
        for (k=0; k<dk; k++) {
            double *pints_ij = pints + (size_t)dij*k;
            double *ptmp2_ij = ptmp2 + (size_t)dij*k;
            for (ic=0; ic<ncj; ic++) {
                amat2[ioff] = pints_ij + (size_t)di*mj*ic;
                bmat2[ioff] = pD;
                cmat2[ioff] = ptmp2_ij + (size_t)di*mj*ic;
                ioff++;
            }
        }
    }

    cblas_dgemm_batch(CblasColMajor, transA, transB2, m2, n2, k2,
        alpha, amat2, lda2, bmat2, ldb2, beta, cmat2, ldc2,
        GRP_COUNT, size_per_grp2);

    free(tmp1);

    /************
    * apply Dk  *
    ************/
    MKL_INT    m3[GRP_COUNT] = {dij};
    MKL_INT    k3[GRP_COUNT] = {mk};
    MKL_INT    n3[GRP_COUNT] = {mk};
    MKL_INT    lda3[GRP_COUNT] = {dij};
    MKL_INT    ldb3[GRP_COUNT] = {mk};
    MKL_INT    ldc3[GRP_COUNT] = {dij};
    MKL_INT    size_per_grp3[GRP_COUNT] = {nL*nck};
    const double *amat3[nL*nck], *bmat3[nL*nck];
    double *cmat3[nL*nck];
    double *pout;
    ioff = 0;
    for (iL=0; iL<nL; iL++) {
        pD = Dk[op_idx[iL]];
        pints = tmp2 + (size_t)dijk*iL;
        pout = out + (size_t)dijk*iL;
        for (ic=0; ic<nck; ic++) {
            amat3[ioff] = pints + (size_t)dij*mk*ic;
            bmat3[ioff] = pD;
            cmat3[ioff] = pout + (size_t)dij*mk*ic;
            ioff++;
        }
    }

    cblas_dgemm_batch(CblasColMajor, transA, transB2, m3, n3, k3,
        alpha, amat3, lda3, bmat3, ldb3, beta, cmat3, ldc3,
        GRP_COUNT, size_per_grp3);

    free(tmp2);
}

static void apply_Tji_Dk_batch_gemm(double* out, double* ints, double* Tji, double** Dk,
                                   int* op_idx, int* L2_off, int nL,
                                   int dij, int mk, int nck) {

    int dk = mk * nck;
    int dijk = dij * dk;
    /************
    * apply Tji *
    ************/
    MKL_INT    m1[GRP_COUNT] = {dij};
    MKL_INT    k1[GRP_COUNT] = {dij};
    MKL_INT    n1[GRP_COUNT] = {1};
    MKL_INT    lda1[GRP_COUNT] = {dij};
    MKL_INT    ldb1[GRP_COUNT] = {dij};
    MKL_INT    ldc1[GRP_COUNT] = {dij};
    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans};
    double    alpha[GRP_COUNT] = {1.0};
    double    beta[GRP_COUNT] = {0.0};
    MKL_INT    size_per_grp1[GRP_COUNT] = {nL*dk};
    const double *amat1[nL*dk], *bmat1[nL*dk];
    double *cmat1[nL*dk];
    int iL, k, ic, ioff=0;
    double *pTji, *pints;
    double *tmp1 = malloc(sizeof(double)*dijk*nL);
    double *ptmp1;
    for (iL=0; iL<nL; iL++) {
        pTji = Tji + op_idx[iL]*dij*dij;
        pints = ints + (size_t)dijk*L2_off[iL];
        ptmp1 = tmp1 + (size_t)dijk*iL;
        for (k=0; k<dk; k++) {
            amat1[ioff] = pTji;
            bmat1[ioff] = pints + (size_t)dij*k;
            cmat1[ioff] = ptmp1 + (size_t)dij*k;
            ioff++;
        }
    }

    cblas_dgemm_batch(CblasColMajor, transA, transB, m1, n1, k1,
        alpha, amat1, lda1, bmat1, ldb1, beta, cmat1, ldc1,
        GRP_COUNT, size_per_grp1);

    /************
    * apply Dk  *
    ************/
    MKL_INT    m3[GRP_COUNT] = {dij};
    MKL_INT    k3[GRP_COUNT] = {mk};
    MKL_INT    n3[GRP_COUNT] = {mk};
    MKL_INT    lda3[GRP_COUNT] = {dij};
    MKL_INT    ldb3[GRP_COUNT] = {mk};
    MKL_INT    ldc3[GRP_COUNT] = {dij};
    MKL_INT    size_per_grp3[GRP_COUNT] = {nL*nck};
    CBLAS_TRANSPOSE    transB2[GRP_COUNT] = {CblasTrans};
    const double *amat3[nL*nck], *bmat3[nL*nck];
    double *cmat3[nL*nck];
    double *pout, *pD;
    ioff = 0;
    for (iL=0; iL<nL; iL++) {
        pD = Dk[op_idx[iL]];
        pints = tmp1 + (size_t)dijk*iL;
        pout = out + (size_t)dijk*iL;
        for (ic=0; ic<nck; ic++) {
            amat3[ioff] = pints + (size_t)dij*mk*ic;
            bmat3[ioff] = pD;
            cmat3[ioff] = pout + (size_t)dij*mk*ic;
            ioff++;
        }
    }

    cblas_dgemm_batch(CblasColMajor, transA, transB2, m3, n3, k3,
        alpha, amat3, lda3, bmat3, ldb3, beta, cmat3, ldc3,
        GRP_COUNT, size_per_grp3);

    free(tmp1);
}


static void _nr3c_fill_symm_kk(int (*intor)(), void (*fsort)(),
                          double complex *out, int nkpts_ij,
                          int nkpts, int comp, int nimgs, int ish, int jsh,
                          double *buf, double *env_loc, double *Ls,
                          double *expkL_r, double *expkL_i, int *kptij_idx,
                          int *shls_slice, int *ao_loc,
                          CINTOpt *cintopt, PBCOpt *pbcopt,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                          int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size, double* t_cpu, double* tD)
{
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];

    const char TRANS_N = 'N';
    const double D0 = 0;
    const double D1 = 1;
    const double ND1 = -1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int l_i = bas[ANG_OF+ish*BAS_SLOTS];
    int l_j = bas[ANG_OF+jsh*BAS_SLOTS];
    int l_k;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    int dkmax;
    int mi = 2 * l_i + 1; //FIXME
    int mj = 2 * l_j + 1; //FIXME
    int mk;
    int nci = bas[NCTR_OF+ish*BAS_SLOTS];
    int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
    int nck;

    int i, dijm, dijmc, dijmk, empty;
    int ksh, iL0, iL, jL, iLcount; 
    int idx_L2, iop;

    int shls[3], shlcen[3];
    double *bufkk_r, *bufkk_i, *bufkL_r, *bufkL_i, *bufL, *pbuf, *cache;
    int (*fprescreen)();
    if (pbcopt != NULL) {
        fprescreen = pbcopt->fprescreen;
    } else {
        fprescreen = PBCnoscreen;
    }

    double *pDi[nop];
    build_Dmats(pDi, Dmats, rot_loc, rot_mat_size, nop, l_i);
    double *pDj[nop];
    build_Dmats(pDj, Dmats, rot_loc, rot_mat_size, nop, l_j);

    shls[0] = ish;
    shls[1] = jsh;
    shlcen[2] = shlcen_atm_idx[jsh];
    shlcen[1] = shlcen_atm_idx[ish];
    for (ksh = ksh0; ksh < ksh1; ksh++) {//loop over shells one by one
        l_k = bas[ANG_OF+ksh*BAS_SLOTS];
        dkmax = ao_loc[ksh+1] - ao_loc[ksh];
        nck = bas[NCTR_OF+ksh*BAS_SLOTS];
        dijm = dij * dkmax;
        dijmc = dijm * comp;
        dijmk = dijmc * nkpts;
        bufkk_r = buf;
        bufkk_i = bufkk_r + (size_t)nkpts * dijmk;
        bufkL_r = bufkk_i + (size_t)nkpts * dijmk;
        bufkL_i = bufkL_r + (size_t)MIN(nimgs,IMGBLK) * dijmk;
        bufL    = bufkL_i + (size_t)MIN(nimgs,IMGBLK) * dijmk;
        cache   = bufL    + (size_t)nimgs * dijmc;
        for (i = 0; i < nkpts*dijmk*OF_CMPLX; i++) {
            bufkk_r[i] = 0;
        }

        shls[2] = ksh;
        bool *int_flags_L2 = malloc(sizeof(bool)*nimgs*nimgs);
        for (i = 0; i < nimgs*nimgs; i++) {
            int_flags_L2[i] = false;
        }
        double *int_ijk_buf = malloc(sizeof(double)*dijmc*nimgs*nimgs);
        double *pint_ijk;

        shlcen[0] = shlcen_atm_idx[ksh];
        int L2iL_off = get_shltrip_idx(shlcen, shltrip_cen_idx, nshltrip) * nimgs * nimgs;
        int *pL2iL = L2iL + (size_t)L2iL_off;
        int *piop = ops + (size_t)L2iL_off;

        //int dkj = dkmax*dj;
        //int dkji = dkj * di;
        mk = 2 * l_k + 1; //FIXME

        double *pDk[nop];
        build_Dmats(pDk, Dmats, rot_loc, rot_mat_size, nop, l_k);

        double *Tkji, *Tji;
        if (dijm <= BLKSIZE){
            Tkji = malloc(sizeof(double)*dijm*dijm*nop);
            //direct product of Wigner D matrices
            for (iop=0; iop<nop; iop++) {
                multiply_Dmats(Tkji, Dmats, rot_loc, rot_mat_size, nop, iop,
                           mi, mj, mk, l_i, l_j, l_k, nci, ncj, nck);
            }
        } else if (dij <= BLKSIZE) {
            Tji = malloc(sizeof(double)*dij*dij*nop);
            for (iop=0; iop<nop; iop++) {
                multiply_two_Dmats(Tji, Dmats, rot_loc, rot_mat_size, nop, iop,
                                   mi, mj, l_i, l_j, nci, ncj);
            }
        }

        int op_idx[nimgs*nimgs];
        int int_idx[nimgs*nimgs];
        int L2_off = 0;

        clock_t CPU_time_1 = clock();
        for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
            iLcount = MIN(IMGBLK, nimgs - iL0);
            for (iL = iL0; iL < iL0+iLcount; iL++) {
                for (jL = 0; jL < nimgs; jL++) {
                    idx_L2 = pL2iL[iL * nimgs + jL];
                    int_idx[L2_off] = idx_L2;
                    op_idx[L2_off] = piop[iL * nimgs + jL];
                    L2_off++;
                    if (int_flags_L2[idx_L2] == false){//build integral
                        pint_ijk = int_ijk_buf+(size_t)dijmc*idx_L2;
                        int_flags_L2[idx_L2] = true;
                        int iL_irr = idx_L2 / nimgs;
                        int jL_irr = idx_L2 % nimgs;
                        shift_bas(env_loc, env, Ls, iptrxyz, iL_irr);
                        shift_bas(env_loc, env, Ls, jptrxyz, jL_irr);
                        if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                            //clock_t CPU_time_1 = clock();
                            if ((*intor)(pint_ijk, NULL, shls, atm, natm, bas, nbas,
                                         env_loc, cintopt, cache)) {
                                empty = 0;
                            }
                            //clock_t CPU_time_2 = clock();
                            //*t_cpu += (double)(CPU_time_2-CPU_time_1)/CLOCKS_PER_SEC;
                        }
                        else {
                            for (i = 0; i < dijmc; i++) {
                                pint_ijk[i] = 0;
                            }
                        }
                    }
                }
            }
        }
        clock_t CPU_time_2 = clock();
        *t_cpu += (double)(CPU_time_2-CPU_time_1)/CLOCKS_PER_SEC;

        for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
            iLcount = MIN(IMGBLK, nimgs - iL0);
            for (iL = iL0; iL < iL0+iLcount; iL++) {
                //bufL = int_ijk_buf + iL*dijmc*nimgs;
                pbuf = bufL;
                clock_t CPU_time_1 = clock();
                if (dijm <= BLKSIZE){
                    //multiply Tmats
                    apply_Tmats_batch_gemm(pbuf, Tkji, int_ijk_buf, op_idx+nimgs*iL, int_idx+nimgs*iL, dijm, nimgs);
                } else if (dij <= BLKSIZE) {
                    apply_Tji_Dk_batch_gemm(pbuf, int_ijk_buf, Tji, pDk,
                                            op_idx+nimgs*iL, int_idx+nimgs*iL, nimgs,
                                            dij, mk, nck);
                } else {
                    //apply Dmats
                    apply_Dmats_batch_gemm(pbuf, int_ijk_buf, pDi, pDj, pDk,
                                           op_idx+nimgs*iL, int_idx+nimgs*iL, nimgs,
                                           mi, mj, mk, nci, ncj, nck);
                }
                clock_t CPU_time_2 = clock();
                *tD += (double)(CPU_time_2-CPU_time_1)/CLOCKS_PER_SEC;

                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
                       &D1, bufL, &dijmc, expkL_r, &nimgs,
                       &D0, bufkL_r+(iL-iL0)*(size_t)dijmk, &dijmc);
                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
                       &D1, bufL, &dijmc, expkL_i, &nimgs,
                       &D0, bufkL_i+(iL-iL0)*(size_t)dijmk, &dijmc);
            } // iL in range(0, nimgs)
            // conj(exp(1j*dot(h,k)))
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_r, &dijmk, expkL_r+iL0, &nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_i, &dijmk, expkL_i+iL0, &nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_i, &dijmk, expkL_r+iL0, &nimgs,
                   &D1, bufkk_i, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &ND1, bufkL_r, &dijmk, expkL_i+iL0, &nimgs,
                   &D1, bufkk_i, &dijmk);
        }
        if (dijm <= BLKSIZE) {
            free(Tkji);
        } else if (dij <= BLKSIZE) {
            free(Tji);
        }
        free(int_flags_L2);
        free(int_ijk_buf);
        (*fsort)(out, bufkk_r, bufkk_i, kptij_idx, shls_slice,
                 ao_loc, nkpts, nkpts_ij, comp, ish, jsh,
                 ksh, ksh+1);
    }
}

void PBCnr3c_fill_symm_kks2(int (*intor)(), double complex *out, int nkpts_ij,
                       int nkpts, int comp, int nimgs, int ish, int jsh,
                       double *buf, double *env_loc, double *Ls,
                       double *expkL_r, double *expkL_i, int *kptij_idx,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, PBCOpt *pbcopt,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                       int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size, double* t, double* t2)
{
    int ip = ish + shls_slice[0];
    int jp = jsh + shls_slice[2] - nbas;
    if (ip > jp) {
        _nr3c_fill_symm_kk(intor, &sort3c_kks2_igtj, out,
                      nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                      buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                      shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                      shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                      L2iL, ops, Dmats, rot_loc, nop, rot_mat_size, t, t2);
    } else if (ip == jp) {
        _nr3c_fill_symm_kk(intor, &sort3c_kks1, out,
                      nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                      buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                      shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                      shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                      L2iL, ops, Dmats, rot_loc, nop, rot_mat_size, t, t2);
    }
}

void PBCnr3c_symm_drv(int (*intor)(), void (*fill)(), double complex *eri,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv,
                 int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                 int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
    double *expkL_i = expkL_r + nimgs*nkpts;
    int i;
    for (i = 0; i < nimgs*nkpts; i++) {
        expkL_r[i] = creal(expkL[i]);
        expkL_i[i] = cimag(expkL[i]);
    }

    assert(comp == 1);
    size_t count;
    if (fill == &PBCnr3c_fill_symm_kks2) {
        int dijk =(GTOmax_shell_dim(ao_loc, shls_slice+0, 1) *
                   GTOmax_shell_dim(ao_loc, shls_slice+2, 1) *
                   GTOmax_shell_dim(ao_loc, shls_slice+4, 1));
        count = nkpts*nkpts * OF_CMPLX +
                nkpts*MIN(nimgs,IMGBLK) * OF_CMPLX + nimgs;
        //MAX(INTBUFMAX, dijk) to ensure buffer is enough for at least one (i,j,k) shell
        count*= MAX(INTBUFMAX, dijk) * comp;
    } else {
        count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
        count+= nimgs * nkpts * OF_CMPLX;
    }
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas, env);

    double* t_cpu = malloc(sizeof(double));
    *t_cpu = 0.0;
    double* t2 = malloc(sizeof(double));
    *t2 = 0.0;

    #pragma omp parallel
    {
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        memcpy(env_loc, env, sizeof(double)*nenv);
        double *buf = malloc(sizeof(double)*(count+cache_size));
        #pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
            ish = ij / njsh;
            jsh = ij % njsh;
            (*fill)(intor, eri, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                    buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                    shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                    shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                    L2iL, ops, Dmats, rot_loc, nop, rot_mat_size, t_cpu, t2);
        }
        free(buf);
        free(env_loc);
    }
    free(expkL_r);
    printf("debug: ints/mat cpu time: %f, %f\n", *t_cpu, *t2);
}
