CC = clang
CXX = clang++
FLAGS = -fopenmp -pthread -Wall -Wextra -march=native -Ofast
CXXFLAGS = $(FLAGS)
CFLAGS = -lm $(FLAGS)

NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp -pthread -Wall -Wextra -march=native"
LDFLAGS  := -lm

EXES     := cuda

alls: $(EXES)

clean:
	rm -f $(EXES)

cuda: cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
