
from setuptools import setup, Extension
from Cython.Build import cythonize
# from Cython.Distutils import build_ex

import os
import os.path as osp

execfile(osp.join(osp.dirname(osp.abspath(__file__)), "sbs", "version.py"))

# cutils = Extension("cutils",
        # sources=["sbs/utils/cutils.pyx"])

setup(
        name="SpikeBasedSampling",
        version=".".join(map(str, __version__)),
        # install_requires=["PyYAML>=3.10"],
        packages=["sbs", "sbs/db"],
        # url="",
        # license="MIT",
        # entry_points = {
            # "console_scripts" : [
                    # "sbs = sbs.main:main_loop"
                # ]
            # },
        ext_modules=cythonize("sbs/cutils.pyx"),
        zip_safe=True,
    )
