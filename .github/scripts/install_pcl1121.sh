#!/bin/bash

wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.12.1/source.zip
unzip source.zip
cd pcl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_apps=OFF \
  -DBUILD_benchmarks=OFF \
  -DBUILD_cuda=OFF \
  -DBUILD_ml=OFF \
  -DBUILD_outofcore=OFF \
  -DBUILD_people=OFF \
  -DBUILD_recognition=OFF \
  -DBUILD_segmentation=OFF \
  -DBUILD_simulation=OFF \
  -DBUILD_stereo=OFF \
  -DBUILD_tools=OFF \
  -DBUILD_tracking=OFF \
  -DBUILD_visualization=OFF \
  -DWITH_QT=OFF \
  -DWITH_VTK=OFF \
  -DWITH_PNG=OFF \
  -DWITH_FZAPI=OFF \
  -DWITH_LIBUSB=OFF \
  -DWITH_OPENNI=OFF \
  -DWITH_OPENNI2=OFF \
  -DWITH_PCAP=OFF \
  -DWITH_PXCAPI=OFF \
  -DWITH_QHULL=OFF \
  -DWITH_OPENGL=FALSE \
  ..
make -j4
make install
make clean