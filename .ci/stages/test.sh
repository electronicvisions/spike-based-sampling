#!/bin/bash
set -Eeuxo pipefail

SCRIPTPATH="$(realpath -P "${BASH_SOURCE[0]}")"
source "$(dirname "${SCRIPTPATH}")/commons.sh"

# make sure py-sbs is not loaded for test execution
source <(spack module loads -r -x py-sbs visionary-simulation~dev)

pushd tests
nosetests --with-xunit --xunit-file=test_results.xml . || exit 0
popd
