#!/bin/bash

set -euo pipefail

BASEFOLDER="$(realpath -m "$(dirname "${BASH_SOURCE[0]}")/../..")"

source /opt/spack/share/spack/setup-env.sh || exit 1

PATH_INSTALL="$(realpath -m "${BASEFOLDER}/installed")"
PATH_INSTALL_VISIONARY_NEST="$(realpath -m "${BASEFOLDER}/model-visionary-nest/installed")"
PATH_INSTALL_FULL="${PATH_INSTALL}/lib/python2.7/site-packages"

export PYTHONPATH="${PATH_INSTALL_FULL}${PYTHONPATH:+:${PYTHONPATH}}"
CPATH="$(spack location -i py-numpy '^python@:2.999.999')/lib/python2.7/site-packages/numpy/core/include${CPATH:+:${CPATH}}"
export CPATH
export LD_LIBRARY_PATH="${PATH_INSTALL_VISIONARY_NEST}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NEST_MODULES="visionarymodule"
