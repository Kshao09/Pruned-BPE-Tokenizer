from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "bpe_fast_core_parallel.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "initializedcheck": False,
        },
    )
)