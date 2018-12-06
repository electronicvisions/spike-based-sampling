#!/bin/bash

set -euo pipefail

source /opt/spack/share/spack/setup-env.sh || exit 1

PATH_INSTALL="${PWD}/installed"
PATH_INSTALL_FULL="${PATH_INSTALL}/lib/python2.7/site-packages"

export PYTHONPATH="${PATH_INSTALL_FULL}${PYTHONPATH:+:${PYTHONPATH}}"
export CPATH="$(spack location -i py-numpy)/lib/python2.7/site-packages/numpy/core/include${CPATH:+:${CPATH}}"
