#!/bin/bash

## This is an example file for automated loading of modules
## usage is $ . ./env.sh
##

echo "Adding Olli's modules to modulepath"

export MODULEPATH=\
/wang/users/obreitwi/cluster_home/opt/modules/versions:$MODULEPATH

module load gcc/4.9.3
module load numpy/1.9.3
module load docopt
module load lazyarray/0.2.8
module load quantities/0.10.1
module load neo/0.3.3
module load pynn/0.8.1-10-gccadf1f
module load nest/obreitwi-v2.10.0-876-ge09dfda
module load matplotlib/1.4.3
module load boost/1.57.0
module load nest-semf/1.1.0
module load cython/0.24.0
module load h5py/2.6.0

echo "Now loaded modules: "
module list

