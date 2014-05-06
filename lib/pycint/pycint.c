/*
 * File: pycint.c
 *
 * interface for c/python
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * rewrite from swig + numpy.i
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "swig_numpy.h"
#include "cint.h"

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define PyArray_NewFArray(nd, dims, typenum) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, \
                        NULL, NULL, 0, NPY_FORTRANORDER, NULL)
#define PyArray_NewCArray(nd, dims, typenum) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, \
                        NULL, NULL, 0, NPY_CORDER, NULL)

#define NPY_COMPLEX     NPY_CDOUBLE

typedef int (*INT1E_t)(double *, int *, int *, int, int *, int, double *);
typedef void (*VHF_t)(double *, double *, double *, int, int, int,
                      int *, int, int *, int, double *);

// mat[m,n] -> mat_T[n,m]
static void transpose(double *mat_t, double *mat, int m, int n, int nset,
                      const int typecode)
{
        int k;
        for (k = 0; k < nset; k++) {
                if (typecode == NPY_DOUBLE) {
                        CINTdmat_transpose(mat_t+k*m*n, mat+k*m*n, m, n);
                } else {
                        CINTzmat_transpose(mat_t+k*m*n*2, mat+k*m*n*2, m, n);
                }
        }
}

static char doc_vhf_common[] =
"Function signature:\n"
"  vj, vk = vhf_name(dm, atm, bas, env)\n\n"
"Required arguments:\n"
"  vhf_name : one of\n"
"        nr_vhf_o3, \n"
"        nr_vhf_igiao_o2, \n"
"        rkb_vhf_coul_o3, \n"
"        rkb_vhf_coul_grad_o3, \n"
"  dm  : (nset,nd,nd) or (nd,nd) array('d') or array('z')\n"
"  nset: dims of output array\n"
"  atm : rank-2 array('i')\n"
"  bas : rank-2 array('i')\n"
"  env : rank-1 array('d')\n"
"Return objects:\n"
"  vj : rank-3 array('d')/('z') with bounds (nset,ndim,ndim)\n"
"    or rank-2 array('d')/('z') with bounds (ndim,ndim)\n"
"  vk : rank-3 array('d')/('z') with bounds (nset,ndim,ndim)\n"
"    or rank-2 array('d')/('z') with bounds (ndim,ndim)";
PyObject *pycint_vhf_common(const char *func_format, const int nset,
                VHF_t intor, const int typecode, PyObject *args)
{
        PyObject * dm_py = Py_None;
        PyArrayObject *dm_np = NULL;
        double *dm;
        int nset_dm = 1;
        int nao = -1;

        PyObject *atm_py = Py_None;
        PyArrayObject *atm_np = NULL;
        int is_new_atm = 0;
        int *atm;
        int natm;

        PyObject *bas_py = Py_None;
        PyArrayObject *bas_np = NULL;
        int is_new_bas = 0;
        int *bas;
        int nbas;

        PyObject *env_py = Py_None;
        PyArrayObject *env_np = NULL;
        int is_new_env = 0;
        double *env;

        npy_intp dims[] = {-1, -1, -1};
        PyArrayObject *vj_np, *vk_np;
        double *vj, *vk;
        PyObject *resultobj;

        if (!PyArg_ParseTuple(args, func_format,
                              &dm_py, &atm_py, &bas_py, &env_py))
                goto fail;

        // extract atm
        atm_np = obj_to_array_contiguous_allow_conversion(atm_py, NPY_INT,
                                                          &is_new_atm);
        if (!atm_np || atm_np->nd < 2) goto fail;
        atm  = (int *)(atm_np->data);
        natm = (int)(atm_np->dimensions[0]);

        // extract bas
        bas_np = obj_to_array_contiguous_allow_conversion(bas_py, NPY_INT,
                                                          &is_new_bas);
        if (!bas_np || bas_np->nd < 2) goto fail;
        bas  = (int *)(bas_np->data);
        nbas = (int) (bas_np->dimensions[0]);

        // extract env
        env_np = obj_to_array_contiguous_allow_conversion(env_py, NPY_DOUBLE,
                                                          &is_new_env);
        if (!env_np) goto fail;
        env = (double *)(env_np->data);

        // extract dm in C order
        dm_np = (PyArrayObject *)dm_py;
        if (dm_np->nd < 2) goto fail;
        if (dm_np->nd > 2) {
                nset_dm = (int) (dm_np->dimensions[0]);
        } else {
                nset_dm = 1;
        }
        nao = dm_np->dimensions[1];
        dm = (double *)malloc(sizeof(double) * nset_dm * nao * nao
                              * (typecode==NPY_DOUBLE ? 1 : 2));
        //dm = (double *)(dm_np->data);
        transpose(dm, (double *)(dm_np->data), nao, nao, nset_dm, typecode);

        // new vj and vk
        dims[0] = (npy_intp) nset * nset_dm;
        dims[1] = (npy_intp) nao;
        dims[2] = (npy_intp) nao;
        if (dims[0] > 1) {
                vj_np = (PyArrayObject *)PyArray_NewCArray(3, dims, typecode);
                vk_np = (PyArrayObject *)PyArray_NewCArray(3, dims, typecode);
        } else {
                vj_np = (PyArrayObject *)PyArray_NewCArray(2, dims+1, typecode);
                vk_np = (PyArrayObject *)PyArray_NewCArray(2, dims+1, typecode);
        }
        vj = (double *)malloc(sizeof(double) * dims[0] * nao * nao
                              * (typecode==NPY_DOUBLE ? 1 : 2));
        vk = (double *)malloc(sizeof(double) * dims[0] * nao * nao
                              * (typecode==NPY_DOUBLE ? 1 : 2));
        //vj = (double *)(((PyArrayObject *)vj_np)->data);
        //vk = (double *)(((PyArrayObject *)vk_np)->data);

        (*intor)(dm, vj, vk, nao, nset, nset_dm, atm, natm, bas, nbas, env);
        transpose((double *)(vj_np->data), vj, nao, nao, dims[0], typecode);
        transpose((double *)(vk_np->data), vk, nao, nao, dims[0], typecode);
        free(dm);
        free(vj);
        free(vk);

        resultobj = Py_BuildValue("NN", vj_np, vk_np);

        if (is_new_atm && atm_np) { Py_DECREF(atm_np); }
        if (is_new_bas && bas_np) { Py_DECREF(bas_np); }
        if (is_new_env && env_np) { Py_DECREF(env_np); }
        return resultobj;
fail:
        if (is_new_atm && atm_np) { Py_DECREF(atm_np); }
        if (is_new_bas && bas_np) { Py_DECREF(bas_np); }
        if (is_new_env && env_np) { Py_DECREF(env_np); }
        return NULL;
}


/*
 * c/python interface for vhf_name(...)
 */
static char doc_cint1e_common[] =
"Function signature:\n"
"  block = cint1e...(shls, atm, bas, env)\n";
PyObject *pycint_cint1e_common(const char *func_format, const int nset,
                INT1E_t intor, const int typecode, PyObject *args)
{
        PyObject *shls_py = Py_None;
        PyArrayObject *shls_np = NULL;
        int is_new_shls = 0;
        int *shls;

        PyObject *atm_py = Py_None;
        PyArrayObject *atm_np = NULL;
        int is_new_atm = 0;
        int *atm;
        int natm;

        PyObject *bas_py = Py_None;
        PyArrayObject *bas_np = NULL;
        int is_new_bas = 0;
        int *bas;
        int nbas;

        PyObject *env_py = Py_None;
        PyArrayObject *env_np = NULL;
        int is_new_env = 0;
        double *env;

        npy_intp dims[] = {-1, -1, -1};
        PyArrayObject *mat_np;
        double *mat;
        PyObject *resultobj;

        if (!PyArg_ParseTuple(args, func_format,
                &shls_py, &atm_py, &bas_py, &env_py))
                goto fail;

        // extract shls
        shls_np = obj_to_array_contiguous_allow_conversion(shls_py, NPY_INT,
                                                           &is_new_shls);
        if (!shls_np || shls_np->dimensions[0] < 2) goto fail;
        shls = (int *)(shls_np->data);

        // extract atm
        atm_np = obj_to_array_contiguous_allow_conversion(atm_py, NPY_INT,
                                                          &is_new_atm);
        if (!atm_np || atm_np->nd < 2) goto fail;
        atm = (int *)(atm_np->data);
        natm = (int)(atm_np->dimensions[0]);

        // extract bas
        bas_np = obj_to_array_contiguous_allow_conversion(bas_py, NPY_INT,
                                                          &is_new_bas);
        if (!bas_np || bas_np->nd < 2) goto fail;
        bas = (int *)(bas_np->data);
        nbas = (int) (bas_np->dimensions[0]);

        // extract env
        env_np = obj_to_array_contiguous_allow_conversion(env_py, NPY_DOUBLE,
                                                          &is_new_env);
        if (!env_np) goto fail;
        env = (double *)(env_np->data);

        // new matrix block: mat
        dims[0] = (npy_intp) nset;
        if (typecode == NPY_DOUBLE) {
                dims[1] = (npy_intp) (bas(ANG_OF, shls[0]) * 2 + 1) * bas(NCTR_OF, shls[0]);
                dims[2] = (npy_intp) (bas(ANG_OF, shls[1]) * 2 + 1) * bas(NCTR_OF, shls[1]);
        } else if (typecode == NPY_COMPLEX) {
                dims[1] = (npy_intp) CINTlen_spinor(shls[0], bas) * bas(NCTR_OF, shls[0]);
                dims[2] = (npy_intp) CINTlen_spinor(shls[1], bas) * bas(NCTR_OF, shls[1]);
        } else {
                goto fail;
        }
        if (nset > 1) {
                mat_np = (PyArrayObject *)PyArray_NewCArray(3, dims, typecode);
                mat = (double *)malloc(sizeof(double) * nset*dims[1]*dims[2]
                                       * (typecode==NPY_DOUBLE ? 1 : 2));
        } else {
                mat_np = (PyArrayObject *)PyArray_NewCArray(2, dims+1, typecode);
                mat = (double *)malloc(sizeof(double) * dims[1]*dims[2]
                                       * (typecode==NPY_DOUBLE ? 1 : 2));
        }
        (*intor)(mat, shls, atm, natm, bas, nbas, env);
        transpose((double *)(mat_np->data), mat, dims[1], dims[2], nset, typecode);
        free(mat);
        resultobj = Py_BuildValue("N", mat_np);

        if (is_new_shls && shls_np) { Py_DECREF(shls_np); }
        if (is_new_atm && atm_np) { Py_DECREF(atm_np); }
        if (is_new_bas && bas_np) { Py_DECREF(bas_np); }
        if (is_new_env && env_np) { Py_DECREF(env_np); }
        return resultobj;
fail:
        if (is_new_shls && shls_np) { Py_DECREF(shls_np); }
        if (is_new_atm && atm_np) { Py_DECREF(atm_np); }
        if (is_new_bas && bas_np) { Py_DECREF(bas_np); }
        if (is_new_env && env_np) { Py_DECREF(env_np); }
        return NULL;
}


#include "pycint_reg.c"


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  Init method
 * use swig and numpy.i to generate init method if PY_VERSION_HEX >= 0x03000000
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
static char doc_pycint_module[] = "\
Functions:\n"
"  block = cint1e...(shls, atm, bas, env)\n"
"  vj, vk = vhf_name(dm, atm, bas, env)\n\n"
"Required arguments:\n"
"  vhf_name : one of\n"
"        nr_vhf_o3, \n"
"        nr_vhf_igiao_o2, \n"
"        rkb_vhf_coul_o3, \n"
"        rkb_vhf_coul_grad_o3, \n"
"  dm  : (nset,nd,nd) or (nd,nd) array('d') or array('z')\n"
"  nset: dims of output array\n"
"  atm : rank-2 array('i')\n"
"  bas : rank-2 array('i')\n"
"  env : rank-1 array('d')\n"
"Return objects:\n"
"  vj : rank-3 array('d')/('z') with bounds (nset,ndim,ndim)\n"
"    or rank-2 array('d')/('z') with bounds (ndim,ndim)\n"
"  vk : rank-3 array('d')/('z') with bounds (nset,ndim,ndim)\n"
"    or rank-2 array('d')/('z') with bounds (ndim,ndim)";
void initpycint(void)
{
  PyObject *m, *d, *s;

  m = Py_InitModule("pycint", pycintMethods);
  import_array();

  if (PyErr_Occurred()) {
          PyErr_SetString(PyExc_ImportError, "failed to import pycint");
          return;
  }

  d = PyModule_GetDict(m);
//  s = PyString_FromString("$Revision: 0.1$");
//  PyDict_SetItemString(d, "__version__", s);
//  Py_DECREF(s);
  s = PyString_FromString(doc_pycint_module);
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);

//  SWIG_InitializeModule(0);
//  SWIG_InstallConstants(d, swig_const_table);

  return;
}

