#!/bin/bash
module add intel openmpi/1.8.4-icc

make -f lomonosov_makefile

module rm intel openmpi/1.8.4-icc
