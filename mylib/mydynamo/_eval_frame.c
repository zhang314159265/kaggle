#include <Python.h>
#include <frameobject.h>

PyObject* registered_callback;
static PyObject* set_eval_frame(PyObject *new_callback);

static inline PyObject *call_callback(PyObject *callable, PyObject *frame) {
  PyObject *args = Py_BuildValue("(O)", frame);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

PyObject *eval_custom_code(PyThreadState *tstate, PyFrameObject *frame, PyCodeObject *newcode, int throw_flag) {
  Py_ssize_t nlocals_new = newcode->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;
  assert(nlocals_new >= nlocals_old);
  assert(newcode->co_flags & CO_NOFREE);

  PyFrameObject *newframe = PyFrame_New(tstate, newcode, frame->f_globals, NULL);
  assert(newframe);

  // setup newframe->f_localsplus
  PyObject **fastlocals_old = frame->f_localsplus;
  PyObject **fastlocals_new = newframe->f_localsplus;

  for (Py_ssize_t i = 0; i < nlocals_old; ++i) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  PyObject *result = _PyEval_EvalFrameDefault(tstate, newframe, throw_flag);
  Py_DECREF(newframe);
  return result;
}

static PyObject *custom_eval_func(PyThreadState *tstate, PyFrameObject *frame, int throw_flag) {
  assert(registered_callback != Py_None);

  PyObject* callback = registered_callback;
  // make sure we don't call the callback itself when evaluate the frame for
  // the callback. That will cause infinite loop.
  set_eval_frame(Py_None);
  PyObject* result = call_callback(callback, (PyObject*) frame);
  set_eval_frame(callback);

  assert(result != NULL && "Call back fails");
 
  if (result != Py_None) {
    // call the original code/frame
    PyCodeObject* newcode = (PyCodeObject*) PyObject_GetAttrString(result, "code");
    assert(newcode != NULL);
    Py_DECREF(result);
    PyObject* ret = eval_custom_code(tstate, frame, newcode, throw_flag);
    return ret;
  } else {
    // use the default eval function
    Py_DECREF(result);
    return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
  }
}

static void set_default_eval_func() {
  _PyInterpreterState_SetEvalFrameFunc(PyThreadState_GET()->interp,
    &_PyEval_EvalFrameDefault);
}

static void set_custom_eval_func() {
  _PyInterpreterState_SetEvalFrameFunc(PyThreadState_GET()->interp,
    &custom_eval_func);
}

static PyObject* set_eval_frame(PyObject *new_callback) {
  PyObject* old_callback = registered_callback;
  if (old_callback == Py_None && new_callback != Py_None) {
    // register custom eval function
    set_custom_eval_func();
  } else if (old_callback != Py_None && new_callback == Py_None) {
    // recover the default eval function
    set_default_eval_func();
  }

  // the ref count for old_callback is transferred to the caller
  // inc the new callback
  Py_INCREF(new_callback);

  registered_callback = new_callback;
  return old_callback;
}

static PyObject* set_eval_frame_py(PyObject* dummy, PyObject *args) {
  PyObject* callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    return NULL;
  }
  if (callback != Py_None && !PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "expected a callbale or None");
    return NULL;
  }
  PyObject* old_callback = set_eval_frame(callback);
  return old_callback;
}

static PyMethodDef _methods[] = {
  {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _module = {
  PyModuleDef_HEAD_INIT, "_eval_frame",
  "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit__eval_frame(void) {
  registered_callback = Py_None;
  return PyModule_Create(&_module); 
}
