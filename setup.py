from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'pdb_contacts',
        ['src/pdb_contacts.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17'],
    ),
]

setup(
    name='ames',
    version='1.0.0',
    ext_modules=ext_modules,
    # This makes pip install it properly to site-packages
    zip_safe=False,
)