#!/bin/bash

make clean -C TSPComponent/
make clean -C KPComponent/
make clean 

make -C TSPComponent/
make -C KPComponent/
make
