#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>

/********************************************************************/
/* This is a Python extension module that calculates the fuzzy      */
/* Hui-Walter log likelihood and gradient. The extension function   */
/* John.log_like returns a tuple (f, D) where f is the (scalar)     */
/* result of the log likelihood function and D is the log           */
/* likelihood gradient function.                                    */
/*                                                                  */
/* Version 0.1                                                      */
/* Daniel Smith PhD                                                 */
/* December 23, 2012                                                */
/********************************************************************/


static char module_docstring[] = 
  "This module provides a C-implementation of the fuzzy Hui-Walter \
likelihood and likelihood gradient functions. The extension function \
John.log_like(P[2],Sj[4],c[2],b[2],data[4][N])  returns a tuple (f, D) \
where f is the (scalar) result of the log likelihood function and D is the \
log likelihood gradient function. The data input is organized as: data[0][:] \
population 1 test 1; data[1][:] pop1 test2; data[2][:] pop2 test1; data[3][:] \
pop2 test2.";

static char log_like_docstring[] = 
  "The extension function John.log_like(P[2],Sj[4],c[2],b[2],data[4][N]) \
returns a tuple (f, D) where f is the (scalar) result of the log likelihood \
function and D is the log likelihood gradient function. The data input is \
organized as: data[0][:] population 1 test 1; data[1][:] pop1 test2; \
data[2][:] pop2 test1; data[3][:] pop2 test2.";


double classify(double x, double c, double b);
void like_sums(double* P, double* c, double* b, int N, double** data, 
	       double sums[][2]);
double log_like(double* P, double* Sj, double sums[][2]);
void grad_log_like(double* P, double* Sj, double sums[][2], double ret[]);

double grad_terms(int i, int j, double P, double S);
double fl(int i, double x);
double like_term(int K, double P, double* Sj);
double neg(int i);

static PyObject* John_hello(PyObject* self, PyObject* args);
static PyObject* John_log_like(PyObject* self, PyObject* args);

static PyObject* John_hello(PyObject* self, PyObject* args) {
  return Py_BuildValue("s", "Hello, World!");
}

static PyObject* John_log_like(PyObject* self, PyObject* args) {
  double P1, P2, Se1, Se2, Sp1, Sp2, c1, c2, b1, b2;
  PyObject *x_obj;
  int i;

  if (!PyArg_ParseTuple(args, "ddddddddddO", &P1, &P2, &Se1, &Se2, &Sp1, &Sp2,
			&c1, &c2, &b1, &b2, &x_obj))
    return NULL;

  double P[2], Sj[4], c[2], b[2];
  P[0] = P1; P[1] = P2;
  Sj[0] = Se1; Sj[1] = Se2; Sj[2] = Sp1; Sj[3] = Sp2;
  c[0] = c1; c[1] = c2;
  b[0] = b1; b[1] = b2;

  /*
  PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  if (x_array == NULL) {
    Py_XDECREF(x_array);
    return NULL;
  }
  Py_INCREF(x_obj);*/

  int nd = PyArray_NDIM(x_obj);
  if (nd != 2) {
    //Py_XDECREF(x_array);
    PyErr_SetString(PyExc_TypeError,"Array must be 2-d array!\n");
    return NULL;
  }

  npy_intp N = PyArray_DIM(x_obj, 0);
  npy_intp M = PyArray_DIM(x_obj, 1);
  if (N != 4) {
    //Py_XDECREF(x_array);
    PyErr_SetString(PyExc_TypeError,"Array is wrong shape!\n");
    return NULL;
  }

  double **x;
  npy_intp *xshape = PyArray_DIMS(x_obj);

  if (PyArray_AsCArray(&x_obj, (void *) &x, xshape, nd, 
		       PyArray_DescrFromType(PyArray_DOUBLE)) < 0) {
    PyErr_SetString(PyExc_TypeError, "Error obtaining C array.\n");
    // Py_XDECREF(x_array);
    return NULL;
  }

  npy_intp shape[1];
  shape[0] = 6;
  PyArrayObject *vecout = (PyArrayObject *) PyArray_SimpleNew(1, shape, 
							      PyArray_DOUBLE);
  double *OUT = (double *) PyArray_DATA(vecout);

  double sums[4][2];
  like_sums(P, c, b, M, x, sums);

  double ret = log_like(P, Sj, sums);
  grad_log_like(P, Sj, sums, OUT);

  //  PyObject_FREE(x_array);
  // Py_DECREF(x_array);
  Py_DECREF(x_obj);
  free(x);

  return Py_BuildValue("dN",ret,vecout);
}

static PyMethodDef John_methods[] =
  {
    {"hello", John_hello, METH_VARARGS, "John, say hello!"},
    {"log_like", John_log_like, METH_VARARGS, log_like_docstring},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC initJohn(void)
{
  PyObject *m = Py_InitModule3("John", John_methods, module_docstring);
  if (m==NULL) 
    return;

  import_array();
}

double classify(double x, double c, double b) {
  if (x<(c-b))
    return 0;
  else if (x>(c+b))
    return 1;
  else {
    if (b>1E-7)
      return (x-c+b)/2/b;
    else
      return 0.5;
  } 
}

double like_term(int K, double P, double* Sj) {
  switch(K) {
  case 0:
    return P*Sj[0]*Sj[1]+(1-P)*(1-Sj[2])*(1-Sj[3]);
    break;
  case 1:
    return P*(1-Sj[0])*Sj[1]+(1-P)*Sj[2]*(1-Sj[3]);
    break;
  case 2:
    return P*Sj[0]*(1-Sj[1])+(1-P)*(1-Sj[2])*Sj[3];
    break;
  case 3:
    return P*(1-Sj[0])*(1-Sj[1])+(1-P)*Sj[2]*Sj[3];
    break;
  default:
    return -1;
  }
}

void like_sums(double* P, double* c, double* b, int N, double** data, 
	       double sums[][2]) {
  double cla[4][N];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<N; j++) {
      cla[i][j] = classify(data[i][j], c[i%2], b[i%2]);
    }
    for (j=0; j<2; j++) {
      sums[i][j] = 0;
    }
  }

  for (i=0; i<2; i++) {
    for (j=0; j<N; j++) {
      sums[0][i] += cla[2*i][j]*cla[2*i+1][j];
      sums[1][i] += (1-cla[2*i][j])*cla[2*i+1][j];
      sums[2][i] += cla[2*i][j]*(1-cla[2*i+1][j]);
      sums[3][i] += (1-cla[2*i][j])*(1-cla[2*i+1][j]);
    }
  }
}

double log_like(double* P, double* Sj, double sums[][2]) {
  double ret = 0;
  int i, j;

  for (i=0; i<2; i++) {
    for (j=0; j<4; j++) {
      ret += log(like_term(j, P[i], Sj))*sums[j][i];
    }
  }
  return ret;
}

double fl(int i, double x) {
  if (i % 2 == 0) 
    return x;
  else
    return (1-x);
}

double neg(int i) {
  if (i % 2 == 0) 
    return 1;
  else
    return -1;
}

double grad_terms(int i, int j, double P, double S) {
  // i is the return index, 
  // j is the sum index
  // i.e. ret[i] += ...grad_terms(i, j, P[k], Sj[i])*sums[j][k]...

  if (i % 2 == 0) 
    return neg(j+i/2)*fl(i/2, P)*fl(j/2+i/2, S);
  else
    return neg(j/2+i/2)*fl(i/2, P)*fl(j%2+i/2, S);
}

void grad_log_like(double* P, double* Sj, double sums[][2], double ret[]) {
  int i, j, k;

  for (i=0; i<6; i++) ret[i] = 0;

  for (i=0; i<2; i++) {
    for (j=0; j<4; j++) {
      ret[i] += sums[j][i]*(fl(j, Sj[0])*fl(j/2, Sj[1])
					   -fl(j+1, Sj[2])*fl(j/2+1, Sj[3]))
					   /like_term(j, P[i], Sj);
    }
  }
  
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      for (k=0; k<2; k++) {
	ret[i+2] += sums[j][k]*grad_terms(i, j, P[k], Sj[2*(i/2)+1-i%2])
	  /like_term(j, P[k], Sj);
      }
    }
  }
}
