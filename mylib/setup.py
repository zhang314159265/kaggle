#!/usr/bin/env python
import sys
from setuptools import setup, find_packages, Extension

package_include = [
]

setup(
  name = "mydynamo",
  packages=find_packages(include=package_include),
  ext_modules=[
    Extension(
      "mydynamo._eval_frame",
      ["mydynamo/_eval_frame.c"],
    ),
  ],
)
