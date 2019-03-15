#!/bin/bash
set -Eeuxo pipefail

SCRIPTPATH="$(realpath -m "${BASH_SOURCE[0]}")"
source "$(dirname "${SCRIPTPATH}")/commons.sh"

mkdir -p "${PATH_INSTALL_FULL}"

python setup.py install "--prefix=${PATH_INSTALL}"
