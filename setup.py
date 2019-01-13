
from setuptools import setup
from Cython.Build import cythonize
# from Cython.Distutils import build_ex

import os.path as osp

versionfile = osp.join(osp.dirname(osp.abspath(__file__)), "sbs", "version.py")
with open(versionfile) as f:
    code = compile(f.read(), versionfile, 'exec')
    exec(code, globals(), locals())

setup(
        name="SpikeBasedSampling",
        version=".".join(map(str, __version__)),  # noqa: F821
        packages=["sbs", "sbs/db"],
        ext_modules=cythonize("sbs/cutils.pyx"),
        zip_safe=True,
    )
