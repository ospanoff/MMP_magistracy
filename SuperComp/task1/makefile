HOST_CC=mpicxx

FLAGS=-O3
INCLUDES=-I$(HOME)/libs/boost/include
LIBS=-L$(HOME)/libs/boost/lib -lboost_system -lboost_mpi -lboost_serialization -lboost_graph_parallel

SOURCE=code.cpp
OUT=run

all:
	$(HOST_CC) $(FLAGS) $(INCLUDES) -o $(OUT) $(SOURCE) $(LIBS)

generator:
	xlc++_r generator.cpp -O3 -o generator -qsmp=omp
