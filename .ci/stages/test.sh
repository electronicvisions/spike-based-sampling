#!/bin/bash
set -Eeuxo pipefail

SCRIPTPATH="$(realpath -P "${BASH_SOURCE[0]}")"
source "$(dirname "${SCRIPTPATH}")/commons.sh"

# make sure py-sbs is not loaded for test execution
source <(spack module tcl loads -r -x py-sbs -x visionary-nest \
         visionary-simulation~dev "^python@:2.999.999")

# default python installation changed to py3 -> need to set PYTHONHOME to avoid
# conflict of system libraries
export PYTHONHOME=/opt/spack_views/visionary-simulation
source <(python2 $(which spack) module tcl loads -r py-nose "^python@:2.999.999")


# assert that visionarymodule can be loaded
(unset NEST_MODULES; python -c "import nest; nest.Install('visionarymodule')")

# propagate OVERWRITE_DEFAULT_CONTAINER_IMAGE
if [ -n "${OVERWRITE_DEFAULT_CONTAINER_IMAGE:-}" ]; then
    export SBS_CONTAINER_IMAGE=${OVERWRITE_DEFAULT_CONTAINER_IMAGE}
fi
pushd tests
nosetests --with-xunit --xunit-file=test_results.xml . || exit 0
popd
