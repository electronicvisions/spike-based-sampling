#!/bin/bash
#
# Check out model-visionary-nest and checkout possible Depends-On in current
# commit.

set -euo pipefail

SCRIPTPATH="$(realpath -P "${BASH_SOURCE[0]}")"
source "$(dirname "${SCRIPTPATH}")/commons.sh"

if [ -z "${GERRIT_USERNAME:-}" ]; then
    GERRIT_USERNAME="hudson"
fi

if [ -z "${GERRIT_PORT:-}" ]; then
    GERRIT_PORT=29418
fi

if [ -z "${GERRIT_HOSTNAME:-}" ]; then
    GERRIT_HOSTNAME="brainscales-r.kip.uni-heidelberg.de"
fi

if [ -z "${GERRIT_BASE_URL:-}" ]; then
    export GERRIT_BASE_URL="ssh://${GERRIT_USERNAME}@${GERRIT_HOSTNAME}:${GERRIT_PORT}"
fi

MY_GERRIT_URL="${GERRIT_BASE_URL}/model-visionary-nest"

git clone ${MY_GERRIT_URL} -b master

# check if commit message
VISIONARY_NEST_GERRIT_CHANGE="$(git log -1 --pretty=%B \
    | awk '$1 ~ "Depends-On:" { $1 = ""; print $0 }' | tr '\n' ',' \
    | tr -d \[:space:\])"

if [ -n "${VISIONARY_NEST_GERRIT_CHANGE}" ]; then
    pushd "model-visionary-nest"

    gerrit_query="$(mktemp)"

    for change in ${VISIONARY_NEST_GERRIT_CHANGE//,/ }; do
        ssh -p ${GERRIT_PORT} \
               ${GERRIT_USERNAME}@${GERRIT_HOSTNAME} gerrit query \
               --current-patch-set "${change}" > "${gerrit_query}"

        # check that the change corresponds to a model-visoinary-nest change
        # and extract refspec
        VISIONARY_NEST_GERRIT_REFSPEC="$(awk '
            $1 ~ "project:" && $2 ~ "model-visionary-nest" { project_found=1 }
            $1 ~ "ref:" && project_found { print $2 }' "${gerrit_query}" )"

        if [ -n "${VISIONARY_NEST_GERRIT_REFSPEC}" ]; then
            # we found the correct change
            if [ "$(awk '$1 ~ "status:" { print $2 }' "${gerrit_query}")" \
                    == "NEW" ]; then
                git fetch ${MY_GERRIT_URL} "${VISIONARY_NEST_GERRIT_REFSPEC}"
                git checkout FETCH_HEAD
            fi
        break
        fi
    done

    rm "${gerrit_query}"

    popd
fi
