/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2005       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#ifndef H_EXTENSIONS
#define H_EXTENSIONS


#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>

/* If strict ANSI, then some useful macros are not defined */
#if defined(__STRICT_ANSI__) && !defined(__DARWIN_UNIX03)
# define M_PI           3.14159265358979323846  /* pi */
#endif

#ifndef DOUBLECOMPLEXDEFINED
#  define DOUBLECOMPLEXDEFINED 1
#  include <complex.h>
   typedef double complex double_complex;
#endif

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION < 4
#  define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#define INLINE inline

static INLINE void* gpaw_malloc(size_t n)
{
  void* p = malloc(n);
  assert(p != NULL);
  return p;
}

#ifdef GPAW_BGP
#define GPAW_MALLOC(T, n) (gpaw_malloc((n) * sizeof(T)))
#else
#ifdef GPAW_AIX
#define GPAW_MALLOC(T, n) (malloc((n) * sizeof(T)))
#else
#define GPAW_MALLOC(T, n) (gpaw_malloc((n) * sizeof(T)))
#endif
#endif
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define INTP(a) ((int*)PyArray_DATA(a))
#define LONGP(a) ((long*)PyArray_DATA(a))
#define DOUBLEP(a) ((double*)PyArray_DATA(a))
#define COMPLEXP(a) ((double_complex*)PyArray_DATA(a))

#endif //H_EXTENSIONS
