NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp" -Xcompiler "-pthread" -Xcompiler "-Wall" -Xcompiler "-Wextra" -Xcompiler "-march=native"
LDFLAGS  := -lm -lgomp

EXES     := cuda

all: $(EXES)

clean:
	rm -f $(EXES)

cuda: cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?