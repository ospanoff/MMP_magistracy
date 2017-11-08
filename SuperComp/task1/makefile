HOST_CC=mpic++
FLAGS=-fopenmp -lboost_system -lboost_mpi -lboost_serialization -lboost_graph_parallel -std=c++11 -O3
SOURCE=code.cpp
OUT=run.out

all:
	$(HOST_CC) $(FLAGS) -o $(OUT) $(SOURCE)

generator:
	g++ generator.cpp -O3 -o generator.out -fopenmp
