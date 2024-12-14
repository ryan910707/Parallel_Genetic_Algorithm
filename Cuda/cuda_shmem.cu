#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <omp.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TIME_CHECK(call) \
do { \
    struct timeval start_time, end_time; \
    gettimeofday(&start_time, NULL); \
    call; \
    gettimeofday(&end_time, NULL); \
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + \
                          (end_time.tv_usec - start_time.tv_usec) / 1000000.0; \
    printf("Elapsed time: %.6f seconds\n", elapsed_time); \
} while(0)

int threads_per_block = 1024;
int POP_SIZE;
int ITEMS_NUM;
int GENERATIONS;
float MUTATION_RATE;
float CROSSOVER_RATE;
int KNAPSACK_CAPACITY;

int *h_weights;  // Host weights
int *h_values;   // Host values
int *h_fitness;  // Host fitness
int *h_genes;    // Host genes

int *d_weights;  // Device weights
int *d_values;   // Device values
int *d_fitness;  // Device fitness
int *d_genes;    // Device genes
curandState *d_rand_states;

// CUDA kernel for initializing random states
__global__ void setup_random_states(curandState *states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// CUDA kernel for population initialization
__global__ void initialize_population(int *genes, curandState *states, int pop_size, int items_num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size * items_num) {
        genes[tid] = curand(&states[tid]) % 2;
    }
}

// CUDA kernel for population evaluation
// Modified evaluate_population kernel with shared memory usage
__global__ void evaluate_population(
    int *genes, int *fitness, int *weights, int *values, 
    int pop_size, int items_num, int knapsack_capacity
) {
    // Use extern shared memory:
    // Layout: first 'items_num' integers for s_weights, next 'items_num' integers for s_values
    extern __shared__ int s_data[];

    int *s_weights = s_data;
    int *s_values = &s_data[items_num];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load weights and values into shared memory
    // We do this in a loop in case 'items_num' > blockDim.x
    #pragma unroll
    for (int i = threadIdx.x; i < items_num; i += blockDim.x) {
        s_weights[i] = weights[i];
        s_values[i] = values[i];
    }

    __syncthreads();  // Ensure all weights and values are loaded

    if (tid < pop_size) {
        int total_weight = 0;
        int total_value = 0;        

        // Compute fitness using shared memory
        #pragma unroll
        for (int j = 0; j < items_num; j++) {
            int gene = genes[tid * items_num + j];
            // Accumulate weight and value if gene == 1
            total_weight += s_weights[j] * gene;
            total_value += s_values[j] * gene;
        }

        // Assign fitness only if weight is within capacity
        fitness[tid] = (total_weight <= knapsack_capacity) ? total_value : 0;
    }
}


// CUDA kernel for selection
__global__ void selection(int *old_genes, int *new_genes, int *fitness, 
                          curandState *states, int pop_size, int items_num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size) {
        curandState localState = states[tid];
        
        // Tournament selection
        int parent1 = curand(&localState) % pop_size;
        int parent2 = curand(&localState) % pop_size;
        
        int *selected_parent = (fitness[parent1] > fitness[parent2]) ? 
                                &old_genes[parent1 * items_num] : 
                                &old_genes[parent2 * items_num];
        
        // Copy selected parent's genes, since parent are selected randomly, can't parallel
        #pragma unroll
        for (int j = 0; j < items_num; j++) {
            new_genes[tid * items_num + j] = selected_parent[j];
        }
        
        states[tid] = localState;
    }
}

// CUDA kernel for crossover
__global__ void crossover(int *genes, curandState *states, int pop_size, 
                          int items_num, float crossover_rate) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size) {
        curandState localState = states[tid];
        
        // Check if crossover should occur
        if (curand_uniform(&localState) < crossover_rate) {
            int crossover_point = curand(&localState) % items_num;
            
            // Perform crossover only if this is part of a pair
            if (tid % 2 == 0 && tid + 1 < pop_size) {
                int *parent1 = &genes[tid * items_num];
                int *parent2 = &genes[(tid + 1) * items_num];
                int *child1 = parent1;
                int *child2 = parent2;
                
                // Swap genes after crossover point
                #pragma unroll 
                for (int i = crossover_point; i < items_num; i++) {
                    int temp = child1[i];
                    child1[i] = child2[i];
                    child2[i] = temp;
                }
            }
        }
        
        states[tid] = localState;
    }
}

// CUDA kernel for mutation
__global__ void mutate(int *genes, curandState *states, int pop_size, 
                       int items_num, float mutation_rate) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size * items_num) {   
        curandState localState = states[tid];
        
        float rand_val = curand_uniform(&localState);
        // flip genes bit 
        genes[tid] ^= (rand_val < mutation_rate);
        
        states[tid] = localState;
    }
}

void input(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file\n");
        exit(1);
    }
    
    fscanf(file, "%d", &KNAPSACK_CAPACITY);
    fscanf(file, "%d", &ITEMS_NUM);
    fscanf(file, "%d", &GENERATIONS);
    fscanf(file, "%d", &POP_SIZE);
    fscanf(file, "%f", &MUTATION_RATE);
    fscanf(file, "%f", &CROSSOVER_RATE);

    h_weights = (int*)malloc(ITEMS_NUM * sizeof(int));
    h_values = (int*)malloc(ITEMS_NUM * sizeof(int));

    for (int i = 0; i < ITEMS_NUM; i++) {
        fscanf(file, "%d %d", &h_weights[i], &h_values[i]);
    }
    
    fclose(file);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    clock_t start_time = clock();
    
    // Read input
    input(argv[1]);
    
    // Set up CUDA grid and block dimensions
    int blocks_for_pop = (POP_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_for_genes = ((POP_SIZE * ITEMS_NUM) + threads_per_block - 1) / threads_per_block;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights, ITEMS_NUM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, ITEMS_NUM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_genes, POP_SIZE * ITEMS_NUM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fitness, POP_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, (POP_SIZE * ITEMS_NUM) * sizeof(curandState)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights, ITEMS_NUM * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_values, h_values, ITEMS_NUM * sizeof(int), cudaMemcpyHostToDevice));

    // Setup random states
    setup_random_states<<<blocks_for_genes, threads_per_block>>>(d_rand_states, 0);

    // Initialize population
    initialize_population<<<blocks_for_genes, threads_per_block>>>(
        d_genes, d_rand_states, POP_SIZE, ITEMS_NUM
    );

    // Allocate host fitness array for result retrieval
    h_fitness = (int*)malloc(POP_SIZE * sizeof(int));

    // Main genetic algorithm loop
    for (int generation = 0; generation < GENERATIONS; generation++) {
        // Evaluate population
        evaluate_population<<<blocks_for_pop, threads_per_block, 2*ITEMS_NUM*sizeof(int)>>>(
            d_genes, d_fitness, d_weights, d_values, 
            POP_SIZE, ITEMS_NUM, KNAPSACK_CAPACITY
        );

        // Perform selection
        int *d_temp_genes;
        CUDA_CHECK(cudaMalloc(&d_temp_genes, POP_SIZE * ITEMS_NUM * sizeof(int)));
        selection<<<blocks_for_pop, threads_per_block>>>(
            d_genes, d_temp_genes, d_fitness, 
            d_rand_states, POP_SIZE, ITEMS_NUM
        );

        // Perform crossover
        crossover<<<blocks_for_pop, threads_per_block>>>(
            d_temp_genes, d_rand_states, POP_SIZE, 
            ITEMS_NUM, CROSSOVER_RATE
        );

        // Perform mutation
        mutate<<<blocks_for_genes, threads_per_block>>>(
            d_temp_genes, d_rand_states, POP_SIZE, 
            ITEMS_NUM, MUTATION_RATE
        );

        // Copy back to main population
        CUDA_CHECK(cudaMemcpy(d_genes, d_temp_genes, 
                   POP_SIZE * ITEMS_NUM * sizeof(int), 
                   cudaMemcpyDeviceToDevice));

        // Free temporary genes memory
        CUDA_CHECK(cudaFree(d_temp_genes));
    }

    // Copy fitness back to host
    CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, 
               POP_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Find best solution
    int best_fitness = h_fitness[0];
    int best_index = 0;

    //omp is slower
    for (int i = 1; i < POP_SIZE; i++) {
        if (h_fitness[i] > best_fitness) {
            best_fitness = h_fitness[i];
            best_index = i;
        }
    }

    // Allocate host memory to retrieve best solution
    int *h_best_genes = (int*)malloc(ITEMS_NUM * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_best_genes, 
               &d_genes[best_index * ITEMS_NUM], 
               ITEMS_NUM * sizeof(int), 
               cudaMemcpyDeviceToHost));

    // Timer ends
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print results
    if (best_fitness == 0) {
        printf("No solution found\n");
    } else {
        printf("Best solution found in generation %d with fitness %d:\n", 
               GENERATIONS, best_fitness);
        printf("Items included (binary): ");
        for (int i = 0; i < ITEMS_NUM; i++) {
            printf("%d ", h_best_genes[i]);
        }
        printf("\n");

        // Calculate total weight and value for the best solution
        int total_weight = 0, total_value = 0;
        //omp is slower
        for (int i = 0; i < ITEMS_NUM; i++) {
            if (h_best_genes[i] == 1) {
                total_weight += h_weights[i];
                total_value += h_values[i];
            }
        }

        printf("Total weight: %d, Total value: %d\n", total_weight, total_value);
        printf("Capacity: %d\n", KNAPSACK_CAPACITY);
    }
    printf("Execution time: %.2f seconds\n", elapsed_time);

    // Free host memory
    free(h_weights);
    free(h_values);
    free(h_fitness);
    free(h_best_genes);

    // Free device memory
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_genes));
    CUDA_CHECK(cudaFree(d_fitness));
    CUDA_CHECK(cudaFree(d_rand_states));

    return 0;
}