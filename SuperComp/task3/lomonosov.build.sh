#!/bin/bash
module add intel openmpi cuda

make -f lomonosov_makefile

module rm intel openmpi cuda
