#include <Python.h>

PyObject* registered_callback;
static PyObject* set_eval_frame(PyObject *new_callback);

static inline PyObject *call_callback(PyObject *callable, PyObject *frame) {
  PyObject *args = Py_BuildValue("(O)", frame);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

static PyObject *custom_eval_func(PyThreadState *tstate, PyFrameObject *frame, int throw_flag) {
  assert(registered_callback != Py_None);

  PyObject* callback = registered_callback;
  // make sure we don't call the callback itself when evaluate the frame for
  // the callback. That will cause infinite loop.
  set_eval_frame(Py_None);
  call_callback(callback, (PyObject*) frame);
  set_eval_frame(callback);

  PyObject* ret = _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
  return ret;
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
