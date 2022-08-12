# Coverage

[Coverage](https://coverage.readthedocs.io/en/6.4.3/) is a tool based on the Python trace module to analysize the code coverage. Basic steps
- coverage run -m {your module and its arguments}
  - the coverage information is written to .coverage file in the current dir
- coverage reprot -m
  - report the coverage information for the python files executed. Point out the missing lines.
- coverage html
  - show the report in HTML form

# Reference
- [livepython: watch your python run like a movie.](https://github.com/agermanidis/livepython). I have similar idea. But the tool does not work well. Exit at the beginning beforing running the python scripts.
  - use sys.settrace/threading.settrace to do tracing.
- [Doc for CPython trace module](https://docs.python.org/3/library/trace.html)

# Trace Module

In the trace module sys.settrace/threading.settrace is called to register the trace function.

## Useful commands
- `python -m trace -tg {your script file and its arguments}`: Display lines that are executed prefixed by the timestamp since the beginning of execution. A plain 'movie' mode.
- `python -m trace --count -C {out-coverage-dir} -m -s {your script and its args}`: print annotated files with execution count for each line. Not executed lines are marked. A summary of coverage for each file is printed at the end.

# DONE

Roughly understand how sys.settrace is implemented. Most of the magic happens in the eval function (`_PyEval_EvalFrameDefault`).
- when starting evaluating a frame, the interpreter calls the global trace function (or call it frame tracer. It's registered by sys.settrace) with the frame object and the reason of tracing (a int).
- the global trace function returns a local trace function (or None to disable local tracing)
- the local trace function (or line tracer) is called for each instruction processed (with some filtering?). The local tracer returns
  - itself to keep tracing
  - None to disable line tracer
  - or even another line tracer (even though I'm not sure when this will be needed).
