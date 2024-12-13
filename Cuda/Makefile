NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp" -Xcompiler "-pthread" -Xcompiler "-Wall" -Xcompiler "-Wextra" -Xcompiler "-march=native"
LDFLAGS  := -lm -lgomp

EXES     := cuda cuda_shmem

all: $(EXES)

clean:
	rm -f $(EXES)

cuda: cuda.cu 
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

cuda_shmem: cuda_shmem.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?