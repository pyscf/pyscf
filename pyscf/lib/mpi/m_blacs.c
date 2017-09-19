/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2010  Argonne National Laboratory
 *  Please see the accompanying LICENSE file for further information. */

#ifdef PARALLEL
#include <Python.h>
#ifdef GPAW_WITH_SL
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <mpi.h>
#include <structmember.h>
#include "extensions.h"
#include "mympi.h"

// BLACS
#define BLOCK_CYCLIC_2D 1

#ifdef GPAW_NO_UNDERSCORE_CBLACS
#define Cblacs_barrier_    Cblacs_barrier
#define Cblacs_gridexit_   Cblacs_gridexit
#define Cblacs_gridinfo_   Cblacs_gridinfo
#define Cblacs_gridinit_   Cblacs_gridinit
#define Cblacs_pinfo_      Cblacs_pinfo
#define Csys2blacs_handle_ Csys2blacs_handle
#endif

void Cblacs_barrier_(int ConTxt, char *scope);

void Cblacs_gridexit_(int ConTxt);

void Cblacs_gridinfo_(int ConTxt, int* nprow, int* npcol,
                      int* myrow, int* mycol);

void Cblacs_gridinit_(int* ConTxt, char* order, int nprow, int npcol);

void Cblacs_pinfo_(int* mypnum, int* nprocs);

int Csys2blacs_handle_(MPI_Comm SysCtxt);
// End of BLACS

// ScaLAPACK
#ifdef GPAW_NO_UNDERSCORE_SCALAPACK
#define   numroc_  numroc
#define   pdlamch_ pdlamch
#define   pdlaset_ pdlaset
#define   pzlaset_ pzlaset

#define   pdpotrf_ pdpotrf
#define   pzpotrf_ pzpotrf
#define   pzpotri_ pzpotri
#define   pdtrtri_ pdtrtri
#define   pztrtri_ pztrtri

#define   pzgesv_ pzgesv
#define   pdgesv_ pdgesv

#define   pdsyevd_  pdsyevd
#define   pzheevd_  pzheevd
#define   pdsyevx_  pdsyevx
#define   pzheevx_  pzheevx
#define   pdsygvx_  pdsygvx
#define   pzhegvx_  pzhegvx
#define   pdsyngst_ pdsyngst
#define   pzhengst_ pzhengst
#ifdef GPAW_MR3
#define   pdsyevr_  pdsyevr
#define   pzheevr_  pzheevr
#endif // GPAW_MR3

#define   pdtran_  pdtran
#define   pztranc_ pztranc
#define   pdgemm_  pdgemm
#define   pzgemm_  pzgemm
#define   pdgemv_  pdgemv
#define   pzgemv_  pzgemv
#define   pdsyr2k_ pdsyr2k
#define   pzher2k_ pzher2k
#define   pdsyrk_  pdsyrk
#define   pzherk_  pzherk
#define   pdtrsm_  pdtrsm
#define   pztrsm_  pztrsm

#define   pzhemm_  pzhemm
#define   pdsymm_  pdsymm

#endif

#ifdef GPAW_NO_UNDERSCORE_CSCALAPACK
#define   Cpdgemr2d_  Cpdgemr2d
#define   Cpzgemr2d_  Cpzgemr2d
#define   Cpdtrmr2d_  Cpdtrmr2d
#define   Cpztrmr2d_  Cpztrmr2d
#endif

// tools
int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void Cpdgemr2d_(int m, int n,
                double* a, int ia, int ja, int* desca,
                double* b, int ib, int jb, int* descb,
                int gcontext);

void Cpzgemr2d_(int m, int n,
                void* a, int ia, int ja, int* desca,
                void* b, int ib, int jb, int* descb,
                int gcontext);

void Cpdtrmr2d_(char* uplo, char* diag, int m, int n,
                double* a, int ia, int ja, int* desca,
                double* b, int ib, int jb, int* descb,
                int gcontext);

void Cpztrmr2d_(char* uplo, char* diag, int m, int n,
                void* a, int ia, int ja, int* desca,
                void* b, int ib, int jb, int* descb,
                int gcontext);

double pdlamch_(int* ictxt, char* cmach);

void pzpotri_(char* uplo, int* n, void* a, int *ia, int* ja, int* desca, int* info);

void pzgetri_(int* n, void* a,
              int *ia, int* ja, int* desca, int* info);

void pdlaset_(char* uplo, int* m, int* n, double* alpha, double* beta,
              double* a, int* ia, int* ja, int* desca);

void pzlaset_(char* uplo, int* m, int* n, void* alpha, void* beta,
              void* a, int* ia, int* ja, int* desca);

// cholesky
void pdpotrf_(char* uplo, int* n, double* a,
              int* ia, int* ja, int* desca, int* info);

void pzpotrf_(char* uplo, int* n, void* a,
              int* ia, int* ja, int* desca, int* info);

void pzgesv_(int* n, int* nrhs, void* a,
             int* ia, int* ja, int* desca, int* ipiv,
             void* b, int* ib, int* jb, int* descb, int* info);

void pdgesv_(int *n, int *nrhs, void *a,
             int *ia, int *ja, int* desca, int *ipiv,
             void* b, int* ib, int* jb, int* descb, int* info);


void pdtrtri_(char* uplo, char* diag, int* n, double* a,
              int *ia, int* ja, int* desca, int* info);

void pztrtri_(char* uplo, char* diag, int* n, void* a,
              int *ia, int* ja, int* desca, int* info);

// diagonalization
void pdsyevd_(char* jobz, char* uplo, int* n,
              double* a, int* ia, int* ja, int* desca,
              double* w, double* z, int* iz, int* jz,
              int* descz, double* work, int* lwork, int* iwork,
              int* liwork, int* info);

void pzheevd_(char* jobz, char* uplo, int* n,
              void* a, int* ia, int* ja, int* desca,
              double* w, void* z, int* iz, int* jz,
              int* descz, void* work, int* lwork, double* rwork,
              int* lrwork, int* iwork, int* liwork, int* info);

void pdsyevx_(char* jobz, char* range,
              char* uplo, int* n,
              double* a, int* ia, int* ja, int* desca,
              double* vl, double* vu,
              int* il, int* iu, double* abstol,
              int* m, int* nz, double* w, double* orfac,
              double* z, int* iz, int* jz, int* descz,
              double* work, int* lwork, int* iwork, int* liwork,
              int* ifail, int* iclustr, double* gap, int* info);

void pzheevx_(char* jobz, char* range,
              char* uplo, int* n,
              void* a, int* ia, int* ja, int* desca,
              double* vl, double* vu,
              int* il, int* iu, double* abstol,
              int* m, int* nz,  double* w, double* orfac,
              void* z, int* iz, int* jz, int* descz,
              void* work, int* lwork, double* rwork, int* lrwork,
              int* iwork, int* liwork,
              int* ifail, int* iclustr, double* gap, int* info);

void pdsygvx_(int* ibtype, char* jobz, char* range,
              char* uplo, int* n,
              double* a, int* ia, int* ja, int* desca,
              double* b, int *ib, int* jb, int* descb,
              double* vl, double* vu,
              int* il, int* iu, double* abstol,
              int* m, int* nz, double* w, double* orfac,
              double* z, int* iz, int* jz, int* descz,
              double* work, int* lwork, int* iwork, int* liwork,
              int* ifail, int* iclustr, double* gap, int* info);

void pzhegvx_(int* ibtype, char* jobz, char* range,
              char* uplo, int* n,
              void* a, int* ia, int* ja, int* desca,
              void* b, int *ib, int* jb, int* descb,
              double* vl, double* vu,
              int* il, int* iu, double* abstol,
              int* m, int* nz,  double* w, double* orfac,
              void* z, int* iz, int* jz, int* descz,
              void* work, int* lwork, double* rwork, int* lrwork,
              int* iwork, int* liwork,
              int* ifail, int* iclustr, double* gap, int* info);

void pdsyngst_(int* ibtype, char* uplo, int* n,
               double* a, int* ia, int* ja, int* desca,
               double* b, int* ib, int* jb, int* descb,
               double* scale, double* work, int* lwork, int* info);

void pzhengst_(int* ibtype, char* uplo, int* n,
               void* a, int* ia, int* ja, int* desca,
               void* b, int* ib, int* jb, int* descb,
               double* scale, void* work, int* lwork, int* info);

#ifdef GPAW_MR3
void pdsyevr_(char* jobz, char* range,
              char* uplo, int* n,
              double* a, int* ia, int* ja, int* desca,
              double* vl, double* vu,
              int* il, int* iu,
              int* m, int* nz, double* w,
              double* z, int* iz, int* jz, int* descz,
              double* work, int* lwork, int* iwork, int* liwork,
              int* info);

void pzheevr_(char* jobz, char* range,
              char* uplo, int* n,
              void* a, int* ia, int* ja, int* desca,
              double* vl, double* vu,
              int* il, int* iu,
              int* m, int* nz,  double* w,
              void* z, int* iz, int* jz, int* descz,
              void* work, int* lwork, double* rwork, int* lrwork,
              int* iwork, int* liwork,
              int* info);
#endif // GPAW_MR3

// pblas
void pdtran_(int* m, int* n,
             double* alpha,
             double* a, int* ia, int* ja, int* desca,
             double* beta,
             double* c, int* ic, int* jc, int* descc);

void pztranc_(int* m, int* n,
              void* alpha,
              void* a, int* ia, int* ja, int* desca,
              void* beta,
              void* c, int* ic, int* jc, int* descc);

void pdgemm_(char* transa, char* transb, int* m, int* n, int* k,
             double* alpha,
             double* a, int* ia, int* ja, int* desca,
             double* b, int* ib, int* jb, int* descb,
             double* beta,
             double* c, int* ic, int* jc, int* descc);

void pzgemm_(char* transa, char* transb, int* m, int* n, int* k,
             void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* b, int* ib, int* jb, int* descb,
             void* beta,
             void* c, int* ic, int* jc, int* descc);

void pzhemm_(char* side, char* uplo, int* m, int* n,
             void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* b, int* ib, int* jb, int* descb,
             void* beta,
             void* c, int* ic, int* jc, int* descc);

void pdsymm_(char* side, char* uplo, int* m, int* n,
             void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* b, int* ib, int* jb, int* descb,
             void* beta,
             void* c, int* ic, int* jc, int* descc);

void pdgemv_(char* transa, int* m, int* n, double* alpha,
             double* a, int* ia, int* ja, int* desca,
             double* x, int* ix, int* jx, int* descx, int* incx,
             double* beta,
             double* y, int* iy, int* jy, int* descy, int* incy);

void pzgemv_(char* transa, int* m, int* n, void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* x, int* ix, int* jx, int* descx, int* incx,
             void* beta,
             void* y, int* iy, int* jy, int* descy, int* incy);

void pdsyr2k_(char* uplo, char* trans, int* n, int* k,
              double* alpha,
              double* a, int* ia, int* ja, int* desca,
              double* b, int* ib, int* jb, int* descb,
              double* beta,
              double* c, int* ic, int *jc, int* descc);

void pzher2k_(char* uplo, char* trans, int* n, int* k,
              void* alpha,
              void* a, int* ia, int* ja, int* desca,
              void* b, int* ib, int* jb, int* descb,
              void* beta,
              void* c, int* ic, int* jc, int* descc);

void pdsyrk_(char* uplo, char* trans, int* n, int* k,
             double* alpha,
             double* a, int* ia, int* ja, int* desca,
             double* beta,
             double* c, int* ic, int* jc, int* descc);

void pzherk_(char* uplo, char* trans, int* n, int* k,
             void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* beta,
             void* c, int* ic, int* jc, int* descc);

void pdtrsm_(char* side, char* uplo, char* trans, char* diag,
             int* m, int *n, double* alpha,
             double* a, int* ia, int* ja, int* desca,
             double* b, int* ib, int* jb, int* descb);

void pztrsm_(char* side, char* uplo, char* trans, char* diag,
             int* m, int *n, void* alpha,
             void* a, int* ia, int* ja, int* desca,
             void* b, int* ib, int* jb, int* descb);


PyObject* pblas_tran(PyObject *self, PyObject *args)
{
    int m, n;
    Py_complex alpha;
    Py_complex beta;
    PyArrayObject *a, *c;
    PyArrayObject *desca, *descc;
  
    if (!PyArg_ParseTuple(args, "iiDODOOO", &m, &n, &alpha,
                          &a, &beta, &c,
                          &desca, &descc))
        return NULL;
 
    int one = 1;
    if (PyArray_DESCR(c)->type_num == NPY_DOUBLE)
        pdtran_(&m, &n,
                &(alpha.real),
                DOUBLEP(a), &one, &one, INTP(desca),
                &(beta.real),
                DOUBLEP(c), &one, &one, INTP(descc));
    else
        pztranc_(&m, &n,
                 &alpha,
                 (void*)PyArray_DATA(a), &one, &one, INTP(desca),
                 &beta,
                 (void*)PyArray_DATA(c), &one, &one, INTP(descc));
    Py_RETURN_NONE;
}

PyObject* pblas_gemm(PyObject *self, PyObject *args)
{
  char* transa;
  char* transb;
  int m, n, k;
  Py_complex alpha;
  Py_complex beta;
  PyArrayObject *a, *b, *c;
  PyArrayObject *desca, *descb, *descc;
  int one = 1;
  
  if (!PyArg_ParseTuple(args, "iiiDOODOOOOss", &m, &n, &k, &alpha,
                        &a, &b, &beta, &c,
                        &desca, &descb, &descc,
                        &transa, &transb)) {
    return NULL;
  }

  // cdesc
  // int c_ConTxt = INTP(descc)[1];

  // If process not on BLACS grid, then return.
  // if (c_ConTxt == -1) Py_RETURN_NONE;

  if (PyArray_DESCR(c)->type_num == NPY_DOUBLE)
    pdgemm_(transa, transb, &m, &n, &k,
            &(alpha.real),
            DOUBLEP(a), &one, &one, INTP(desca),
            DOUBLEP(b), &one, &one, INTP(descb),
            &(beta.real),
            DOUBLEP(c), &one, &one, INTP(descc));
  else
    pzgemm_(transa, transb, &m, &n, &k,
            &alpha,
            (void*)COMPLEXP(a), &one, &one, INTP(desca),
            (void*)COMPLEXP(b), &one, &one, INTP(descb),
            &beta,
            (void*)COMPLEXP(c), &one, &one, INTP(descc));

  Py_RETURN_NONE;
}


PyObject* pblas_hemm(PyObject *self, PyObject *args)
{
  char* side;
  char* uplo;
  int m, n;
  Py_complex alpha;
  Py_complex beta;
  PyArrayObject *a, *b, *c;
  PyArrayObject *desca, *descb, *descc;
  int one = 1;
  if (!PyArg_ParseTuple(args, "ssiiDOOdOOOO",
                 &side, &uplo, &n, &m,
                 &alpha, &a, &b, &beta,
                 &c, &desca, &descb, &descc)) {
    return NULL;
  }

  if (PyArray_DESCR(c)->type_num == NPY_DOUBLE) {
     pdsymm_(side, uplo, &n, &m, &alpha,
             (void*)DOUBLEP(a), &one, &one, INTP(desca),
             (void*)DOUBLEP(b), &one, &one, INTP(descb),
             &beta,
             (void*)DOUBLEP(c), &one, &one, INTP(descc));
  } else {
     pzhemm_(side, uplo, &n, &m, &alpha,
             (void*)COMPLEXP(a), &one, &one, INTP(desca),
             (void*)COMPLEXP(b), &one, &one, INTP(descb),
             &beta,
             (void*)COMPLEXP(c), &one, &one, INTP(descc));
  }

 Py_RETURN_NONE;
}

PyObject* pblas_gemv(PyObject *self, PyObject *args)
{
  char* transa;
  int m, n;
  Py_complex alpha;
  Py_complex beta;
  PyArrayObject *a, *x, *y;
  int incx = 1, incy = 1; // what should these be?
  PyArrayObject *desca, *descx, *descy;
  int one = 1;
  if (!PyArg_ParseTuple(args, "iiDOODOOOOs",
                        &m, &n, &alpha,
                        &a, &x, &beta, &y,
                        &desca, &descx,
                        &descy, &transa)) {
    return NULL;
  }
  
  // ydesc
  // int y_ConTxt = INTP(descy)[1];

  // If process not on BLACS grid, then return.
  // if (y_ConTxt == -1) Py_RETURN_NONE;

  if (PyArray_DESCR(y)->type_num == NPY_DOUBLE)
    pdgemv_(transa, &m, &n,
            &(alpha.real),
            DOUBLEP(a), &one, &one, INTP(desca),
            DOUBLEP(x), &one, &one, INTP(descx), &incx,
            &(beta.real),
            DOUBLEP(y), &one, &one, INTP(descy), &incy);
  else
    pzgemv_(transa, &m, &n,
            &alpha,
            (void*)COMPLEXP(a), &one, &one, INTP(desca),
            (void*)COMPLEXP(x), &one, &one, INTP(descx), &incx,
            &beta,
            (void*)COMPLEXP(y), &one, &one, INTP(descy), &incy);

  Py_RETURN_NONE;
}

PyObject* pblas_r2k(PyObject *self, PyObject *args)
{
  char* uplo;
  int n, k;
  Py_complex alpha;
  Py_complex beta;
  PyArrayObject *a, *b, *c;
  PyArrayObject *desca, *descb, *descc;
  int one = 1;
  
  if (!PyArg_ParseTuple(args, "iiDOODOOOOs", &n, &k, &alpha,
                        &a, &b, &beta, &c,
                        &desca, &descb, &descc,
                        &uplo)) {
    return NULL;
  }

  // cdesc
  // int c_ConTxt = INTP(descc)[1];

  // If process not on BLACS grid, then return.
  // if (c_ConTxt == -1) Py_RETURN_NONE;

  if (PyArray_DESCR(c)->type_num == NPY_DOUBLE)
    pdsyr2k_(uplo, "T", &n, &k,
             &(alpha.real),
             DOUBLEP(a), &one, &one, INTP(desca),
             DOUBLEP(b), &one, &one, INTP(descb),
             &(beta.real),
             DOUBLEP(c), &one, &one, INTP(descc));
  else
    pzher2k_(uplo, "C", &n, &k,
             &alpha,
             (void*)COMPLEXP(a), &one, &one, INTP(desca),
             (void*)COMPLEXP(b), &one, &one, INTP(descb),
             &beta,
             (void*)COMPLEXP(c), &one, &one, INTP(descc));

  Py_RETURN_NONE;
}

PyObject* pblas_rk(PyObject *self, PyObject *args)
{
  char* uplo;
  int n, k;
  Py_complex alpha;
  Py_complex beta;
  PyArrayObject *a, *c;
  PyArrayObject *desca, *descc;
  int one = 1;
  
  if (!PyArg_ParseTuple(args, "iiDODOOOs", &n, &k, &alpha,
                        &a, &beta, &c,
                        &desca, &descc,
                        &uplo)) {
    return NULL;
  }

  // cdesc
  // int c_ConTxt = INTP(descc)[1];

  // If process not on BLACS grid, then return.
  // if (c_ConTxt == -1) Py_RETURN_NONE;

  if (PyArray_DESCR(c)->type_num == NPY_DOUBLE)
    pdsyrk_(uplo, "T", &n, &k,
            &(alpha.real),
            DOUBLEP(a), &one, &one, INTP(desca),
            &(beta.real),
            DOUBLEP(c), &one, &one, INTP(descc));
  else
    pzherk_(uplo, "C", &n, &k,
            &alpha,
            (void*)COMPLEXP(a), &one, &one, INTP(desca),
            &beta,
            (void*)COMPLEXP(c), &one, &one, INTP(descc));

  Py_RETURN_NONE;
}

PyObject* new_blacs_context(PyObject *self, PyObject *args)
{
  PyObject* comm_obj;
  int nprow, npcol;

  int iam, nprocs;
  int ConTxt;
  char* order;

  if (!PyArg_ParseTuple(args, "Oiis", &comm_obj, &nprow, &npcol, &order)){
    return NULL;
  }

  // Create blacs grid on this communicator
  MPI_Comm comm = ((MPIObject*)comm_obj)->comm;
  
  // Get my id and nprocs. This is for debugging purposes only
  Cblacs_pinfo_(&iam, &nprocs);
  MPI_Comm_size(comm, &nprocs);
  
  // Create blacs grid on this communicator continued
  ConTxt = Csys2blacs_handle_(comm);
  Cblacs_gridinit_(&ConTxt, order, nprow, npcol);
  PyObject* returnvalue = Py_BuildValue("i", ConTxt);
  return returnvalue;
}

PyObject* get_blacs_gridinfo(PyObject *self, PyObject *args)
{
  int ConTxt, nprow, npcol;
  int myrow, mycol;
  
  if (!PyArg_ParseTuple(args, "iii", &ConTxt, &nprow, &npcol)) {
    return NULL;
  }

  Cblacs_gridinfo_(ConTxt, &nprow, &npcol, &myrow, &mycol);
  return Py_BuildValue("(ii)", myrow, mycol);
}


PyObject* get_blacs_local_shape(PyObject *self, PyObject *args)
{
  int ConTxt;
  int m, n, mb, nb, rsrc, csrc;
  int nprow, npcol, myrow, mycol;
  int locM, locN;

  if (!PyArg_ParseTuple(args, "iiiiiii", &ConTxt, &m, &n, &mb,
                        &nb, &rsrc, &csrc)){
    return NULL;
  }

  Cblacs_gridinfo_(ConTxt, &nprow, &npcol, &myrow, &mycol);
  locM = numroc_(&m, &mb, &myrow, &rsrc, &nprow);
  locN = numroc_(&n, &nb, &mycol, &csrc, &npcol);
  return Py_BuildValue("(ii)", locM, locN);
}

PyObject* blacs_destroy(PyObject *self, PyObject *args)
{
  int ConTxt;
  if (!PyArg_ParseTuple(args, "i", &ConTxt))
    return NULL;

  Cblacs_gridexit_(ConTxt);

  Py_RETURN_NONE;
}

PyObject* scalapack_set(PyObject *self, PyObject *args)
{
  PyArrayObject* a; // matrix;
  PyArrayObject* desca; // descriptor
  Py_complex alpha;
  Py_complex beta;
  int m, n;
  int ia, ja;
  char* uplo;

  if (!PyArg_ParseTuple(args, "OODDsiiii", &a, &desca,
                        &alpha, &beta, &uplo,
                        &m, &n, &ia, &ja))
    return NULL;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    pdlaset_(uplo, &m, &n, &(alpha.real), &(beta.real), DOUBLEP(a),
             &ia, &ja, INTP(desca));
  else
    pzlaset_(uplo, &m, &n, &alpha, &beta, (void*)COMPLEXP(a),
             &ia, &ja, INTP(desca));

  Py_RETURN_NONE;
}

PyObject* scalapack_redist(PyObject *self, PyObject *args)
{
  PyArrayObject* a; // source matrix
  PyArrayObject* b; // destination matrix
  PyArrayObject* desca; // source descriptor
  PyArrayObject* descb; // destination descriptor

  char* uplo;
  char diag='N'; // copy the diagonal
  int c_ConTxt;
  int m;
  int n;

  int ia, ja, ib, jb;

  if (!PyArg_ParseTuple(args, "OOOOiiiiiiis",
                        &desca, &descb,
                        &a, &b,
                        &m, &n,
                        &ia, &ja,
                        &ib, &jb,
                        &c_ConTxt,
                        &uplo))
    return NULL;

  if (*uplo == 'G') // General matrix
    {
      if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
        Cpdgemr2d_(m, n,
                   DOUBLEP(a), ia, ja, INTP(desca),
                   DOUBLEP(b), ib, jb, INTP(descb),
                   c_ConTxt);
      else
        Cpzgemr2d_(m, n,
                   (void*)COMPLEXP(a), ia, ja, INTP(desca),
                   (void*)COMPLEXP(b), ib, jb, INTP(descb),
                   c_ConTxt);
    }
  else // Trapezoidal matrix
    {
      if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
        Cpdtrmr2d_(uplo, &diag, m, n,
                   DOUBLEP(a), ia, ja, INTP(desca),
                   DOUBLEP(b), ib, jb, INTP(descb),
                   c_ConTxt);
      else
        Cpztrmr2d_(uplo, &diag, m, n,
                   (void*)COMPLEXP(a), ia, ja, INTP(desca),
                   (void*)COMPLEXP(b), ib, jb, INTP(descb),
                   c_ConTxt);
    }
    
  Py_RETURN_NONE;
}

PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args)
{
  // Standard driver for divide and conquer algorithm
  // Computes all eigenvalues and eigenvectors

  PyArrayObject* a; // symmetric matrix
  PyArrayObject* desca; // symmetric matrix description vector
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int one = 1;

  char jobz = 'V'; // eigenvectors also
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOsOO", &a, &desca, &uplo, &z, &w))
    return NULL;

  // adesc
  // int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // zdesc = adesc; this can be relaxed a bit according to pdsyevd.f

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  // Query part, need to find the optimal size of a number of work arrays
  int info;
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;
  int lrwork;
  int i_work;
  double d_work;
  double_complex c_work;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyevd_(&jobz, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               DOUBLEP(z), &one,  &one, INTP(desca),
               &d_work, &querywork, &i_work, &querywork, &info);
      lwork = (int)(d_work);
    }
  else
    {
      pzheevd_(&jobz, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               (void*)COMPLEXP(z), &one,  &one, INTP(desca),
               (void*)&c_work, &querywork, &d_work, &querywork,
               &i_work, &querywork, &info);
      lwork = (int)(c_work);
      lrwork = (int)(d_work);
    }

  if (info != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "scalapack_diagonalize_dc error in query.");
      return NULL;
    }

  // Computation part
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyevd_(&jobz, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               work, &lwork, iwork, &liwork, &info);
      free(work);
    }
  else
    {
      double_complex *work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzheevd_(&jobz, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork, &info);
      free(rwork);
      free(work);
    }
  free(iwork);

  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

PyObject* scalapack_diagonalize_ex(PyObject *self, PyObject *args)
{
  // Standard driver for bisection and inverse iteration algorithm
  // Computes 'iu' eigenvalues and eigenvectors

  PyArrayObject* a; // Hamiltonian matrix
  PyArrayObject* desca; // Hamintonian matrix descriptor
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int a_mycol = -1;
  int a_myrow = -1;
  int a_nprow, a_npcol;
  int il = 1;  // not used when range = 'A' or 'V'
  int iu;
  int eigvalm, nz;
  int one = 1;

  double vl, vu; // not used when range = 'A' or 'I'

  char jobz = 'V'; // eigenvectors also
  char range = 'I'; // eigenvalues il-th through iu-th
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOsiOO", &a, &desca, &uplo, &iu,
                        &z, &w))
    return NULL;

  // a desc
  int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // zdesc = adesc = bdesc; required by pdsyevx.f

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  Cblacs_gridinfo_(a_ConTxt, &a_nprow, &a_npcol, &a_myrow, &a_mycol);

  // Convergence tolerance
  double abstol = 1.0e-8;
  // char cmach = 'U'; // most orthogonal eigenvectors
  // char cmach = 'S'; // most acccurate eigenvalues
  // double abstol = pdlamch_(&a_ConTxt, &cmach);     // most orthogonal eigenvectors
  // double abstol = 2.0*pdlamch_(&a_ConTxt, &cmach); // most accurate eigenvalues
  
  double orfac = -1.0;

  // Query part, need to find the optimal size of a number of work arrays
  int info;
  int *ifail;
  ifail = GPAW_MALLOC(int, n);
  int *iclustr;
  iclustr = GPAW_MALLOC(int, 2*a_nprow*a_npcol);
  double  *gap;
  gap = GPAW_MALLOC(double, a_nprow*a_npcol);
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;  // workspace size must be at least 3
  int lrwork; // workspace size must be at least 3
  int i_work;
  double d_work[3];
  double_complex c_work;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyevx_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               DOUBLEP(z), &one, &one, INTP(desca),
               d_work, &querywork,  &i_work, &querywork,
               ifail, iclustr, gap, &info);
      lwork = MAX(3, (int)(d_work[0]));
    }
  else
    {
      pzheevx_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)&c_work, &querywork, d_work, &querywork,
               &i_work, &querywork,
               ifail, iclustr, gap, &info);
      lwork = MAX(3, (int)(c_work));
      lrwork = MAX(3, (int)(d_work[0]));
    }

  if (info != 0) {
    printf ("info = %d", info);
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_diagonalize_ex error in query.");
    return NULL;
  }

  // Computation part
  // lwork = lwork + (n-1)*n; // this is a ridiculous amount of workspace
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyevx_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               DOUBLEP(z), &one, &one, INTP(desca),
               work, &lwork, iwork, &liwork,
               ifail, iclustr, gap, &info);
      free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzheevx_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork,
               ifail, iclustr, gap, &info);
      free(rwork);
      free(work);
    }
  free(iwork);
  free(gap);
  free(iclustr);
  free(ifail);
  
  // If this fails, fewer eigenvalues than requested were computed.
  assert (eigvalm == iu);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

#ifdef GPAW_MR3
PyObject* scalapack_diagonalize_mr3(PyObject *self, PyObject *args)
{
  // Standard driver for MRRR algorithm
  // Computes 'iu' eigenvalues and eigenvectors
  // http://icl.cs.utk.edu/lapack-forum/archives/scalapack/msg00159.html

  PyArrayObject* a; // Hamiltonian matrix
  PyArrayObject* desca; // Hamintonian matrix descriptor
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int il = 1;  // not used when range = 'A' or 'V'
  int iu;
  int eigvalm, nz;
  int one = 1;

  double vl, vu; // not used when range = 'A' or 'I'

  char jobz = 'V'; // eigenvectors also
  char range = 'I'; // eigenvalues il-th through iu-th
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOsiOO", &a, &desca, &uplo, &iu,
                        &z, &w))
    return NULL;

  // a desc
  // int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // zdesc = adesc = bdesc; required by pdsyevx.f

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  // Query part, need to find the optimal size of a number of work arrays
  int info;
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;
  int lrwork;
  int i_work;
  double d_work[3];
  double_complex c_work;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyevr_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               d_work, &querywork,  &i_work, &querywork,
               &info);
      lwork = (int)(d_work[0]);
    }
  else
    {
      pzheevr_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)&c_work, &querywork, d_work, &querywork,
               &i_work, &querywork,
               &info);
      lwork = (int)(c_work);
      lrwork = (int)(d_work[0]);
    }
  
  if (info != 0) {
    printf ("info = %d", info);
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_diagonalize_evr error in query.");
    return NULL;
  }
  
  // Computation part
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyevr_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               work, &lwork, iwork, &liwork,
               &info);
      free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzheevr_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork,
               &info);
      free(rwork);
      free(work);
    }
  free(iwork);
  
  // If this fails, fewer eigenvalues than requested were computed.
  assert (eigvalm == iu);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}
#endif

PyObject* scalapack_general_diagonalize_dc(PyObject *self, PyObject *args)
{
  // General driver for divide and conquer algorithm
  // Computes *all* eigenvalues and eigenvectors

  PyArrayObject* a; // Hamiltonian matrix
  PyArrayObject* b; // overlap matrix
  PyArrayObject* desca; // Hamintonian matrix descriptor
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int ibtype  =  1; // Solve H*psi = lambda*S*psi
  int one = 1;

  char jobz = 'V'; // eigenvectors also
  char* uplo;

  double scale;

  if (!PyArg_ParseTuple(args, "OOsOOO", &a, &desca, &uplo,
                        &b, &z, &w))
    return NULL;

  // a desc
  // int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // zdesc = adesc = bdesc can be relaxed a bit according to pdsyevd.f

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  // Cholesky Decomposition
  int info;
  if (PyArray_DESCR(b)->type_num == NPY_DOUBLE)
    pdpotrf_(uplo, &n, DOUBLEP(b), &one, &one, INTP(desca), &info);
  else
    pzpotrf_(uplo, &n, (void*)COMPLEXP(b), &one, &one, INTP(desca), &info);

  if (info != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "scalapack_general_diagonalize_dc error in Cholesky.");
      return NULL;
    }

  // Query variables
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;
  int lrwork;
  int i_work;
  double d_work;
  double_complex c_work;
  // NGST Query
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyngst_(&ibtype, uplo, &n,
                DOUBLEP(a), &one, &one, INTP(desca),
                DOUBLEP(b), &one, &one, INTP(desca),
                &scale, &d_work, &querywork, &info);
      lwork = (int)(d_work);
    }
  else
    {
      pzhengst_(&ibtype, uplo, &n,
                (void*)COMPLEXP(a), &one, &one, INTP(desca),
                (void*)COMPLEXP(b), &one, &one, INTP(desca),
                &scale, (void*)&c_work, &querywork, &info);
      lwork = (int)(c_work);
    }

  if (info != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_dc error in NGST query.");
    return NULL;
  }
  // NGST Compute
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyngst_(&ibtype, uplo, &n,
                DOUBLEP(a), &one, &one, INTP(desca),
                DOUBLEP(b), &one, &one, INTP(desca),
                &scale, work, &lwork, &info);
      free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      pzhengst_(&ibtype, uplo, &n,
                (void*)COMPLEXP(a), &one, &one, INTP(desca),
                (void*)COMPLEXP(b), &one, &one, INTP(desca),
                &scale, (void*)work, &lwork, &info);
      free(work);
    }

  if (info != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_dc error in NGST compute.");
    return NULL;
  }

  // NOTE: Scale is always equal to 1.0 above. In future version of ScaLAPACK, we
  // may need to rescale eigenvalues by scale. This can be accomplised by using
  // the BLAS1 d/zscal. See pdsygvx.f

  // EVD Query
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyevd_(&jobz, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               DOUBLEP(z), &one,  &one, INTP(desca),
               &d_work, &querywork, &i_work, &querywork, &info);
      lwork = (int)(d_work);
    }
  else
    {
      pzheevd_(&jobz, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               (void*)COMPLEXP(z), &one,  &one, INTP(desca),
               (void*)&c_work, &querywork, &d_work, &querywork,
               &i_work, &querywork, &info);
      lwork = (int)(c_work);
      lrwork = (int)(d_work);
    }

  if (info != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "scalapack_general_diagonalize_dc error in EVD query.");
      return NULL;
    }

  // EVD Computation
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyevd_(&jobz, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               work, &lwork, iwork, &liwork, &info);
      free(work);
    }
  else
    {
      double_complex *work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzheevd_(&jobz, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork, &info);
      free(rwork);
      free(work);
    }
  free(iwork);

  // Backtransformation to the original problem
  char trans;
  double d_one = 1.0;
  double_complex c_one = 1.0;

  if (*uplo == 'U')
    trans = 'N';
  else
    trans = 'T';

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    pdtrsm_("L", uplo, &trans, "N", &n, &n, &d_one,
            DOUBLEP(b), &one, &one, INTP(desca),
            DOUBLEP(z), &one, &one, INTP(desca));
  else
    pztrsm_("L", uplo, &trans, "N", &n, &n, (void*)&c_one,
            (void*)COMPLEXP(b), &one, &one, INTP(desca),
            (void*)COMPLEXP(z), &one, &one, INTP(desca));

  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

PyObject* scalapack_general_diagonalize_ex(PyObject *self, PyObject *args)
{
  // General driver for bisection and inverse iteration algorithm
  // Computes 'iu' eigenvalues and eigenvectors

  PyArrayObject* a; // Hamiltonian matrix
  PyArrayObject* b; // overlap matrix
  PyArrayObject* desca; // Hamintonian matrix descriptor
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int ibtype  =  1; // Solve H*psi = lambda*S*psi
  int a_mycol = -1;
  int a_myrow = -1;
  int a_nprow, a_npcol;
  int il = 1;  // not used when range = 'A' or 'V'
  int iu;     //
  int eigvalm, nz;
  int one = 1;

  double vl, vu; // not used when range = 'A' or 'I'

  char jobz = 'V'; // eigenvectors also
  char range = 'I'; // eigenvalues il-th through iu-th
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOsiOOO", &a, &desca, &uplo, &iu,
                        &b, &z, &w))
    return NULL;

  // a desc
  int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // zdesc = adesc = bdesc; required by pdsygvx.f

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  Cblacs_gridinfo_(a_ConTxt, &a_nprow, &a_npcol, &a_myrow, &a_mycol);

  // Convergence tolerance
  double abstol = 1.0e-8;
  // char cmach = 'U'; // most orthogonal eigenvectors
  // char cmach = 'S'; // most acccurate eigenvalues
  // double abstol = pdlamch_(&a_ConTxt, &cmach);     // most orthogonal eigenvectors
  // double abstol = 2.0*pdlamch_(&a_ConTxt, &cmach); // most accurate eigenvalues
  
  double orfac = -1.0;

  // Query part, need to find the optimal size of a number of work arrays
  int info;
  int *ifail;
  ifail = GPAW_MALLOC(int, n);
  int *iclustr;
  iclustr = GPAW_MALLOC(int, 2*a_nprow*a_npcol);
  double  *gap;
  gap = GPAW_MALLOC(double, a_nprow*a_npcol);
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;  // workspace size must be at least 3
  int lrwork; // workspace size must be at least 3
  int i_work;
  double d_work[3];
  double_complex c_work;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsygvx_(&ibtype, &jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(b), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               DOUBLEP(z),  &one, &one, INTP(desca),
               d_work, &querywork, &i_work, &querywork,
               ifail, iclustr, gap, &info);
      lwork = MAX(3, (int)(d_work[0]));
    }
  else
    {
      pzhegvx_(&ibtype, &jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               (void*)COMPLEXP(b), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)&c_work, &querywork, d_work, &querywork,
               &i_work, &querywork,
               ifail, iclustr, gap, &info);
      lwork = MAX(3, (int)(c_work));
      lrwork = MAX(3, (int)(d_work[0]));
    }
  if (info != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_ex error in query.");
    return NULL;
  }
  
  // Computation part
  // lwork = lwork + (n-1)*n; // this is a ridiculous amount of workspace
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsygvx_(&ibtype, &jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               DOUBLEP(b), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               DOUBLEP(z), &one, &one,  INTP(desca),
               work, &lwork,  iwork, &liwork,
               ifail, iclustr, gap, &info);
    free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzhegvx_(&ibtype, &jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               (void*)COMPLEXP(b), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &abstol, &eigvalm,
               &nz, DOUBLEP(w), &orfac,
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork,
               ifail, iclustr, gap, &info);
      free(rwork);
      free(work);
  }
  free(iwork);
  free(gap);
  free(iclustr);
  free(ifail);
  
  // If this fails, fewer eigenvalues than requested were computed.
  assert (eigvalm == iu);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

#ifdef GPAW_MR3
PyObject* scalapack_general_diagonalize_mr3(PyObject *self, PyObject *args)
{
  // General driver for MRRR algorithm
  // Computes 'iu' eigenvalues and eigenvectors
  // http://icl.cs.utk.edu/lapack-forum/archives/scalapack/msg00159.html

  PyArrayObject* a; // Hamiltonian matrix
  PyArrayObject* b; // overlap matrix
  PyArrayObject* desca; // Hamintonian matrix descriptor
  PyArrayObject* z; // eigenvector matrix
  PyArrayObject* w; // eigenvalue array
  int ibtype  =  1; // Solve H*psi = lambda*S*psi
  int il = 1;  // not used when range = 'A' or 'V'
  int iu;
  int eigvalm, nz;
  int one = 1;

  double vl, vu; // not used when range = 'A' or 'I'

  char jobz = 'V'; // eigenvectors also
  char range = 'I'; // eigenvalues il-th through iu-th
  char* uplo;

  double scale;

  if (!PyArg_ParseTuple(args, "OOsiOOO", &a, &desca, &uplo, &iu,
                        &b, &z, &w))
    return NULL;

  // a desc
  // int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;

  // zdesc = adesc = bdesc can be relaxed a bit according to pdsyevd.f

  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  // Cholesky Decomposition
  int info;
  if (PyArray_DESCR(b)->type_num == NPY_DOUBLE)
    pdpotrf_(uplo, &n, DOUBLEP(b), &one, &one, INTP(desca), &info);
  else
    pzpotrf_(uplo, &n, (void*)COMPLEXP(b), &one, &one, INTP(desca), &info);

  if (info != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "scalapack_general_diagonalize_mr3 error in Cholesky.");
      return NULL;
    }

  // Query variables
  int querywork = -1;
  int* iwork;
  int liwork;
  int lwork;
  int lrwork;
  int i_work;
  double d_work[3];
  double_complex c_work;
  // NGST Query
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyngst_(&ibtype, uplo, &n,
                DOUBLEP(a), &one, &one, INTP(desca),
                DOUBLEP(b), &one, &one, INTP(desca),
                &scale, d_work, &querywork, &info);
      lwork = (int)(d_work[0]);
    }
  else
    {
      pzhengst_(&ibtype, uplo, &n,
                (void*)COMPLEXP(a), &one, &one, INTP(desca),
                (void*)COMPLEXP(b), &one, &one, INTP(desca),
                &scale, (void*)&c_work, &querywork, &info);
      lwork = (int)(c_work);
    }

  if (info != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_mr3 error in NGST query.");
    return NULL;
  }

  // NGST Compute
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyngst_(&ibtype, uplo, &n,
                DOUBLEP(a), &one, &one, INTP(desca),
                DOUBLEP(b), &one, &one, INTP(desca),
                &scale, work, &lwork, &info);
      free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      pzhengst_(&ibtype, uplo, &n,
                (void*)COMPLEXP(a), &one, &one, INTP(desca),
                (void*)COMPLEXP(b), &one, &one, INTP(desca),
                &scale, (void*)work, &lwork, &info);
      free(work);
    }

  if (info != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_mr3 error in NGST compute.");
    return NULL;
  }

  // NOTE: Scale is always equal to 1.0 above. In future version of ScaLAPACK, we
  // may need to rescale eigenvalues by scale. This can be accomplised by using
  // the BLAS1 d/zscal. See pdsygvx.f

  // EVR Query
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdsyevr_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               d_work, &querywork,  &i_work, &querywork,
               &info);
      lwork = (int)(d_work[0]);
    }
  else
    {
      pzheevr_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)&c_work, &querywork, d_work, &querywork,
               &i_work, &querywork,
               &info);
      lwork = (int)(c_work);
      lrwork = (int)(d_work[0]);
    }

  if (info != 0) {
    printf ("info = %d", info);
    PyErr_SetString(PyExc_RuntimeError,
                    "scalapack_general_diagonalize_evr error in query.");
    return NULL;
  }

  // EVR Computation
  liwork = i_work;
  iwork = GPAW_MALLOC(int, liwork);
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double* work = GPAW_MALLOC(double, lwork);
      pdsyevr_(&jobz, &range, uplo, &n,
               DOUBLEP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               DOUBLEP(z), &one, &one, INTP(desca),
               work, &lwork, iwork, &liwork,
               &info);
      free(work);
    }
  else
    {
      double_complex* work = GPAW_MALLOC(double_complex, lwork);
      double* rwork = GPAW_MALLOC(double, lrwork);
      pzheevr_(&jobz, &range, uplo, &n,
               (void*)COMPLEXP(a), &one, &one, INTP(desca),
               &vl, &vu, &il, &iu, &eigvalm,
               &nz, DOUBLEP(w),
               (void*)COMPLEXP(z), &one, &one, INTP(desca),
               (void*)work, &lwork, rwork, &lrwork,
               iwork, &liwork,
               &info);
      free(rwork);
      free(work);
    }
  free(iwork);

  // Backtransformation to the original problem
  char trans;
  double d_one = 1.0;
  double_complex c_one = 1.0;

  if (*uplo == 'U')
    trans = 'N';
  else
    trans = 'T';

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    pdtrsm_("L", uplo, &trans, "N", &n, &n, &d_one,
            DOUBLEP(b), &one, &one, INTP(desca),
            DOUBLEP(z), &one, &one, INTP(desca));
  else
    pztrsm_("L", uplo, &trans, "N", &n, &n, (void*)&c_one,
            (void*)COMPLEXP(b), &one, &one, INTP(desca),
            (void*)COMPLEXP(z), &one, &one, INTP(desca));

  // If this fails, fewer eigenvalues than requested were computed.
  assert (eigvalm == iu);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}
#endif

PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args)
{
  // Cholesky plus inverse of triangular matrix

  PyArrayObject* a; // overlap matrix
  PyArrayObject* desca; // symmetric matrix description vector
  int info;
  double d_zero = 0.0;
  double_complex c_zero = 0.0;
  int one = 1;
  int two = 2;

  char diag = 'N'; // non-unit triangular
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOs", &a, &desca, &uplo))
    return NULL;

  // adesc
  // int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];

  // Only square matrices
  assert (a_m == a_n);
  int n = a_n;
  int p = a_n - 1;
  
  // If process not on BLACS grid, then return.
  // if (a_ConTxt == -1) Py_RETURN_NONE;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdpotrf_(uplo, &n, DOUBLEP(a), &one, &one,
               INTP(desca), &info);
      if (info == 0)
        {
          pdtrtri_(uplo, &diag, &n, DOUBLEP(a), &one, &one,
                   INTP(desca), &info);
          if (*uplo == 'L')
            pdlaset_("U", &p, &p, &d_zero, &d_zero, DOUBLEP(a),
                     &one, &two, INTP(desca));
          else
            pdlaset_("L", &p, &p, &d_zero, &d_zero, DOUBLEP(a),
                 &two, &one, INTP(desca));
        }
    }
  else
    {
      pzpotrf_(uplo, &n, (void*)COMPLEXP(a), &one, &one,
               INTP(desca), &info);
      if (info == 0)
        {
          pztrtri_(uplo, &diag, &n, (void*)COMPLEXP(a), &one, &one,
                   INTP(desca), &info);
          if (*uplo == 'L')
            pzlaset_("U", &p, &p, (void*)&c_zero, (void*)&c_zero,
                     (void*)COMPLEXP(a), &one, &two, INTP(desca));
          else
            pzlaset_("L", &p, &p, (void*)&c_zero, (void*)&c_zero,
                     (void*)COMPLEXP(a), &two, &one, INTP(desca));
        }
    }

  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

PyObject* scalapack_inverse(PyObject *self, PyObject *args)
{
  // Inverse of an hermitean matrix
  PyArrayObject* a; // Matrix
  PyArrayObject* desca; // Matrix description vector
  char* uplo;
  int info;
  int one = 1;
  if (!PyArg_ParseTuple(args, "OOs", &a, &desca, &uplo))
    return NULL;

  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];
  // Only square matrices
  assert (a_m == a_n);

  int n = a_n;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
     {
      assert(1==-1);       // No double version implemented
     }
  else
    {
      pzpotrf_(uplo, &n, (void*)COMPLEXP(a), &one, &one, INTP(desca), &info);
      if (info == 0)
      {
        pzpotri_(uplo, &n, (void*)COMPLEXP(a), &one, &one, INTP(desca), &info);
      }
    }
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

/*
PyObject* scalapack_solve(PyObject *self, PyObject *args)
{
  // Solves equation Ax = B, where A is a general matrix
  PyArrayObject* a; // Matrix
  PyArrayObject* desca; // Matrix description vector
  PyArrayObject* b; // Matrix
  PyArrayObject* descb; // Matrix description vector
  char uplo;
  int info;
  int one = 1;
  if (!PyArg_ParseTuple(args, "OOOO", &a, &desca, &b, &descb))
    return NULL;

  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];
  // Only square matrices
  assert (a_m == a_n);

  int b_m      = INTP(descb)[2];
  int b_n      = INTP(descb)[3];
  // Equation valid
  assert (a_n == b_m);

  int n = a_n;
  int nrhs = b_n;

  int* pivot = GPAW_MALLOC(int, a_m+2000); // TODO: How long should this exaclty be?

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
     {
      assert(1==-1);       // No double version implemented
     }
  else
    {
       pzgesv_(&n, &nrhs,(void*)COMPLEXP(a), &one, &one, INTP(desca), pivot,
               (void*)COMPLEXP(b), &one, &one, INTP(descb), &info);
    }
  free(pivot);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}
*/

PyObject* scalapack_solve(PyObject *self, PyObject *args) {
  // Solves equation Ax = B, where A is a general matrix
  PyArrayObject* a; // Matrix
  PyArrayObject* desca; // Matrix description vector
  PyArrayObject* b; // Matrix
  PyArrayObject* descb; // Matrix description vector
  int info;
  int one = 1;
  if (!PyArg_ParseTuple(args, "OOOO", &a, &desca, &b, &descb))
    return NULL;

  int a_ConTxt = INTP(desca)[1];
  int a_m      = INTP(desca)[2];
  int a_n      = INTP(desca)[3];
  int a_mb     = INTP(desca)[4];
  // Only square matrices
  assert (a_m == a_n);

  int b_m      = INTP(descb)[2];
  int b_n      = INTP(descb)[3];
  // Equation valid
  assert (a_n == b_m);

  int n = a_n;
  int nrhs = b_n;

  int nprow, npcol, myrow, mycol, locM;

  Cblacs_gridinfo_(a_ConTxt, &nprow, &npcol, &myrow, &mycol);
  // LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A
  locM = (((a_m/a_mb) + 1)/nprow + 1) * a_mb;

  /*
   *  IPIV    (local output) INTEGER array, dimension ( LOCr(M_A)+MB_A )
   *          This array contains the pivoting information.
   *          IPIV(i) -> The global row local row i was swapped with.
   *          This array is tied to the distributed matrix A.
   
   *  An upper bound for these quantities may be computed by:
   *          LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A
   
   *  M_A    (global) DESCA( M_ )    The number of rows in the global
   *                                 array A.
   
   *  MB_A   (global) DESCA( MB_ )   The blocking factor used to distribute
   *                                 the rows of the array.
   
   *  NPROW   (global input) INTEGER
   *          NPROW specifies the number of process rows in the grid
   *          to be created.
   */

  int* pivot = GPAW_MALLOC(int, locM + a_mb);

  //if (a->descr->type_num == PyArray_DOUBLE)
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      pdgesv_(&n, &nrhs,(double*)DOUBLEP(a), &one, &one, INTP(desca), pivot,
              (double*)DOUBLEP(b), &one, &one, INTP(descb), &info);
    }
  else
    {
      pzgesv_(&n, &nrhs,(void*)COMPLEXP(a), &one, &one, INTP(desca), pivot,
              (void*)COMPLEXP(b), &one, &one, INTP(descb), &info);
    }
  free(pivot);
  PyObject* returnvalue = Py_BuildValue("i", info);
  return returnvalue;
}

#endif
#endif // PARALLEL
