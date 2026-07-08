from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="bpe_fast_core",
        sources=["bpe_fast_core.pyx"],
        language="c++",
        extra_compile_args=["/wd4551"],
    )
]

setup(
    name="bpe_fast_core",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": "3",
        },
    ),
)
