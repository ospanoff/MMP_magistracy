CC=mpicxx
CFLAGS=-O3 -fopenmp
LDFLAGS=-fopenmp
SOURCES=main.cpp MPIHelper.cpp Grid.cpp Func2D.cpp ConjugateGradientMethod.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=dirichlet
ifeq (${OMP},true)
	CFLAGS += -D"USE_OMP"
	EXECUTABLE = dirichlet_omp
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
