CC=mpicxx
CFLAGS=-O3 -fopenmp
LDFLAGS=-fopenmp
SOURCES=code.cpp MPIHelper.cpp ConjugateGradientMethod.cpp CGMHelper.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=dirichlet
ifeq (${OMP},true)
	CFLAGS += -D"USE_OMP"
	EXECUTABLE = dirichlet_omp
endif

all: clean $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
