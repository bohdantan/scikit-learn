# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause
import os

import numpy


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("cluster", parent_package, top_path)

    config.add_extension(
        "_dbscan_inner",
        sources=["_dbscan_inner.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
