import argparse
import sys
import os
import threading
import re
import linecache

class MyTracer:
    def __init__(self, opts):
        self.opts = opts
        self.path_filter_re = re.compile(self.opts.path_filter)

    def runpath(self, progname):
        sys.argv = [progname]
        sys.path[0] = os.path.dirname(progname)

        with open(progname, "rb") as fp:
            code = compile(fp.read(), progname, "exec")
        globs = {
            "__file__": progname,
            "__name__": "__main__",
            "__package__": None,
            "__cached__": None,
        }
        self.runctx(code, globs, globs)

    def runctx(self, cmd, globals=None, locals=None):
        threading.settrace(self.globaltrace)
        sys.settrace(self.globaltrace)
        try:
            exec(cmd, globals, locals)
        finally:
            threading.settrace(None)
            sys.settrace(None)

    def globaltrace(self, frame, why, arg):
        if why == "call":
            code = frame.f_code
            filename = frame.f_globals.get("__file__", None)
            if filename:
                if self.path_filter_re.search(filename):
                    print(f"-- Trace file {filename}, func {code.co_name}")
                    return self.localtrace

    def localtrace(self, frame, why, arg):
        if why == "line":
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            bname = os.path.basename(filename)
            print(f"{bname}({lineno}): {linecache.getline(filename, lineno)}", end="")
        return self.localtrace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-filter", default=".*", help="Only python files matching this regular expression will be traced")
    parser.add_argument("progname", nargs="?", help="file to run as main program")
    opts = parser.parse_args()

    if opts.progname is None:
        parser.error("progname is missing")

    t = MyTracer(opts)
    t.runpath(opts.progname)
    print("bye")

if __name__ == "__main__":
    main()
