"""Only the C++ extension lives here - all metadata is in pyproject.toml.

setuptools cannot declare ext_modules in pyproject.toml, so this shim exists
purely to compile src/ames/pdb_contacts.cpp into the installed `ames` package.
"""

from setuptools import setup, Extension
import pybind11

setup(
    ext_modules=[
        Extension(
            'ames.pdb_contacts',          # lands next to the .py modules
            ['src/ames/pdb_contacts.cpp'],
            include_dirs=[pybind11.get_include()],
            language='c++',
            extra_compile_args=['-O3', '-std=c++17'],
        ),
    ],
)
