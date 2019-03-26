
from setuptools import setup
from Cython.Build import cythonize

import os.path as osp
import numpy as np

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
        include_dirs=[np.get_include()],
    )
