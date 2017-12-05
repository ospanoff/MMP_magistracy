CUDA_CC=nvcc
CUDA_ARCH=sm_35

SOURCES=mgrid.cu
OBJECTS=$(SOURCES:.cu=.o)

CFLAGS=-O3
LDFLAGS=

EXECUTABLE=mgrid

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CUDA_CC) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.cu
	$(CUDA_CC) -c -arch=$(CUDA_ARCH) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
