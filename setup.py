from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'bin/pdb_contacts',
        ['src/pdb_contacts.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17'],
    ),
]


setup(
    name='pdb_contacts',
    ext_modules=ext_modules,
)