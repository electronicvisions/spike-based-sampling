#!/bin/bash

cd $(dirname $(readlink -f "$0"))
python -m unittest all_tests
