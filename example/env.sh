#!/bin/bash

## This is an example file for automated loading of modules
## usage is $ . ./env.sh
##
## unloads all loaded modules and then loading useful and required modules
## especially the last two are required for sbs to work properly

echo 'List before doing anything:'
module list

echo "Adding Olli's modules to modulepath"
export MODULEPATH=/home/obreitwi/usr/opt/modules/versions/:$MODULEPATH

echo "Unloading"
module purge

module load mongo
module load peewee
module load neuron/7.3
module load numpy/1.9.3
module load lazyarray/0.2.2
module load quantities/0.10.1
module load neo/0.3.1
module load scipy/0.13.1
module load tornado/3.1.1
module load matplotlib/1.4.2
module load hdf5/1.8.13
module load h5py/2.3.1
module load pynn/0.8beta1-143-gffb0cb1
module load nest/2.4.2_custom_semf

echo "List completed: "
module list


#echo "Changing to Ising folder"
#cd ~/master/master/PP/program/02_pyNN/03_Ising

