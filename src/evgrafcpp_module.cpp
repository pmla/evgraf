#include <Python.h>
#include <ndarraytypes.h>
#include <arrayobject.h>
#include <cstring>
#include <cassert>
#include <vector>
#include "crystalline.h"


#ifdef __cplusplus
extern "C" {
#endif

static PyObject* error(PyObject* type, const char* msg)
{
	PyErr_SetString(type, msg);
	return NULL;
}

static PyObject* bipartite_matching(PyObject* self, PyObject* args, PyObject* kwargs)
{
	(void)self;
	int num_cells = 0;
	PyObject* obj_pos = NULL;
	PyObject* obj_nbrpos = NULL;
	PyObject* obj_poscont = NULL;
	PyObject* obj_nbrposcont = NULL;
	if (!PyArg_ParseTuple(args, "OiO", &obj_pos, &num_cells, &obj_nbrpos))
		return NULL;

	if (num_cells <= 0)
		return error(PyExc_TypeError, "num_cells must be a positive integer");

	obj_poscont = PyArray_ContiguousFromAny(obj_pos, NPY_DOUBLE, 1, 2);
	if (obj_poscont == NULL)
		return error(PyExc_TypeError, "Invalid input data: pos");

	obj_nbrposcont = PyArray_ContiguousFromAny(obj_nbrpos, NPY_DOUBLE, 1, 2);
	if (obj_nbrposcont == NULL)
		return error(PyExc_TypeError, "Invalid input data: nbrpos");

	if (PyArray_NDIM(obj_poscont) != 2)
		return error(PyExc_TypeError, "pos must have shape N x 3");

	if (PyArray_NDIM(obj_nbrposcont) != 2)
		return error(PyExc_TypeError, "pos must have shape num_cells x N x 3");

	if (PyArray_DIM(obj_poscont, 1) != 3)
		return error(PyExc_TypeError, "pos must contain three-dimensional coordinates");

	if (PyArray_DIM(obj_nbrposcont, 1) != 3)
		return error(PyExc_TypeError, "nbrpos must contain three-dimensional coordinates");

	if (PyArray_DIM(obj_nbrposcont, 0) != num_cells * PyArray_DIM(obj_poscont, 0))
		return error(PyExc_TypeError, "nbrpos contains wrong number of entries");

	int num_atoms = PyArray_DIM(obj_poscont, 0);
	double* P = (double*)PyArray_DATA((PyArrayObject*)obj_poscont);
	double* Q = (double*)PyArray_DATA((PyArrayObject*)obj_nbrposcont);

	npy_intp dim[1] = {num_atoms};
	PyObject* obj_permutation = PyArray_SimpleNew(1, dim, NPY_INT);
	int* permutation = (int*)PyArray_DATA((PyArrayObject*)obj_permutation);

	double cost = INFINITY;
	int res = crystalline_bipartite_matching(num_atoms, num_cells, P, Q, &cost, permutation);
	if (res != 0)
		return error(PyExc_RuntimeError, "bipartite matching failed");

	PyObject* result = Py_BuildValue("dO", cost, obj_permutation);
	Py_DECREF(obj_poscont);
	Py_DECREF(obj_nbrposcont);
	Py_DECREF(obj_permutation);
	return result;
}

static PyMethodDef evgrafcpp_methods[] = {
	{
		"bipartite_matching",
		(PyCFunction)bipartite_matching,
		METH_VARARGS,
		"Description."
	},
	{NULL}
};

static struct PyModuleDef evgrafcpp_definition = {
	PyModuleDef_HEAD_INIT,
	"evgrafcpp",
	"evgraf C++ module.",
	-1,
	evgrafcpp_methods,
	NULL,
	NULL,
	NULL,
	NULL,
};

PyMODINIT_FUNC PyInit_evgrafcpp(void)
{
	Py_Initialize();
	import_array();
	return PyModule_Create(&evgrafcpp_definition);
}

#ifdef __cplusplus
}
#endif

