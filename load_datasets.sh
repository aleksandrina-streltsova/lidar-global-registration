#!/usr/bin/env bash
mkdir -p data
pushd data || exit
mkdir -p tmp
# Stanford Bunny
if [ ! -d "bun" ]
  then
    if [ ! -f "bunny.tar.gz" ]
      then
        wget -q --show-progress http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
    fi
    tar -xf bunny.tar.gz -C tmp
    python ../convert_datasets.py stanford tmp/bunny/data
fi
# Happy Buddha
if [ ! -d "happyStandRight" ]
  then
    if [ ! -f "happy_stand.tar.gz" ]
      then
        wget -q --show-progress http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_stand.tar.gz
    fi
    tar -xf happy_stand.tar.gz -C tmp
    python ../convert_datasets.py stanford tmp/happy_stand
fi
# Dragon
if [ ! -d "dragonStandRight" ]
  then
    if [ ! -f "dragon_stand.tar.gz" ]
      then
        wget -q --show-progress http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_stand.tar.gz
    fi
    tar -xf dragon_stand.tar.gz -C tmp
    python ../convert_datasets.py stanford tmp/dragon_stand
fi
# Armadillo
if [ ! -d "ArmadilloBack" ]
  then
    if [ ! -f "Armadillo_scans.tar.gz" ]
      then
        wget -q --show-progress http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo_scans.tar.gz
    fi
    tar -xf Armadillo_scans.tar.gz -C tmp
    python ../convert_datasets.py stanford tmp/Armadillo_scans
fi
# Apartment
if [ ! -d "Hokuyo" ]
  then
    if [ ! -f "Hokuyo.tar.gz" ]
      then
        wget -q --show-progress -O Hokuyo.tar.gz http://robotics.ethz.ch/~asl-datasets/apartment_03-Dec-2011-18_13_33/csv_local/local_frame.tar.gz
    fi
    mkdir -p tmp/Hokuyo
    tar -xf Hokuyo.tar.gz -C tmp/Hokuyo
    python ../convert_datasets.py eth tmp/Hokuyo
fi
rm -rf tmp
popd || exit