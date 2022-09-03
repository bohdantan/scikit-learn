import os
from os.path import join

from sklearn._build_utils import gen_from_templates


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("utils", parent_package, top_path)

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config.add_extension(
        "sparsefuncs_fast", sources=["sparsefuncs_fast.pyx"], libraries=libraries
    )

    config.add_extension(
        "_openmp_helpers", sources=["_openmp_helpers.pyx"], libraries=libraries
    )

    config.add_extension(
        "_readonly_array_wrapper",
        sources=["_readonly_array_wrapper.pyx"],
        libraries=libraries,
    )

    config.add_extension(
        "_typedefs",
        sources=["_typedefs.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
