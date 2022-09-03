import os


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("preprocessing", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    return config
