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
    python3 process_datasets.py stanford tmp/bunny/data
fi
rm -rf tmp
popd || exit