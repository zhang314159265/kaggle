from contextlib import contextmanager
import os

@contextmanager
def chdir(newdir):
    origdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(origdir)
