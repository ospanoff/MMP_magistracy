#!/bin/bash
module add intel openmpi

make -f lomonosov_makefile

module rm intel openmpi
