#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <cstring>
#include <limits.h>

int POP_SIZE;
int ITEMS_NUM;
int GENERATIONS;
float MUTATION_RATE;
float CROSSOVER_RATE;
int KNAPSACK_CAPACITY;

int *weights;
int *values;

typedef struct {
    int *genes;
    int fitness;
} Individual;

void input(const char* filename);
void initialize_population(Individual population[]);
void evaluate_population(Individual population[]);
int calculate_fitness(Individual individual);
void crossover_array(int *genes1, int *genes2, int *newgenes1, int *newgenes2);
void mutate_array(int *genes);
int get_total_weight(Individual individual);
int get_total_value(Individual individual);
void selection(Individual population[], Individual new_population[]);

double get_elapsed_time(struct timespec start, struct timespec end) {
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp.tv_sec + (double) temp.tv_nsec / 1000000000;
}

int main(int argc, char** argv) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0) {
        input(argv[1]);
    }

    // Broadcast global parameters
    MPI_Bcast(&KNAPSACK_CAPACITY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ITEMS_NUM, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GENERATIONS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&POP_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MUTATION_RATE, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&CROSSOVER_RATE, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int base_size = 2 * floor(POP_SIZE / (2 * mpi_size));
    int remainder = POP_SIZE - base_size * mpi_size;
    int local_size = base_size + (mpi_rank < (remainder / 2) ? 2 : 0);

    int *recv_counts = (int *)malloc(mpi_size * sizeof(int));
    int *displacements = (int *)malloc(mpi_size * sizeof(int));
    int offset = 0;

    for (int i = 0; i < mpi_size; i++) {
        recv_counts[i] = (base_size + (i < (remainder / 2) ? 2 : 0)) * ITEMS_NUM;
        displacements[i] = offset;
        offset += recv_counts[i];
    }

    int *global_genes = NULL;
    int *local_genes = (int *)malloc(local_size * ITEMS_NUM * sizeof(int));
    int *local_new_genes = (int *)malloc(local_size * ITEMS_NUM * sizeof(int));

    Individual *population = (Individual *)malloc(POP_SIZE * sizeof(Individual));
    Individual *new_population = (Individual *)malloc(POP_SIZE * sizeof(Individual));
    if (mpi_rank == 0) { // master process
        global_genes = (int *)malloc(POP_SIZE * ITEMS_NUM * sizeof(int));

        initialize_population(population);
        evaluate_population(population);


        int num_threads = omp_get_max_threads();

        // Allocate memory for the new population genes
        #pragma omp parallel for
        for (int i = 0; i < POP_SIZE; i++) {
            new_population[i].genes = (int*)malloc(sizeof(int) * ITEMS_NUM);
        }

        for (int generation = 0; generation < GENERATIONS; generation++) {
            // Selection phase
            #pragma omp parallel for
            for (int i = 0; i < POP_SIZE; i++) {
                unsigned int local_seed = omp_get_thread_num();
                int parent1 = rand_r(&local_seed) % POP_SIZE;
                int parent2 = rand_r(&local_seed) % POP_SIZE;
                Individual selected = (population[parent1].fitness > population[parent2].fitness) ? population[parent1] : population[parent2];

                for (int j = 0; j < ITEMS_NUM; j++) {
                    new_population[i].genes[j] = selected.genes[j];
                }
                new_population[i].fitness = selected.fitness;
            }

            for (int i = 0; i < POP_SIZE; i++) {
                memcpy(global_genes + i * ITEMS_NUM, new_population[i].genes, ITEMS_NUM * sizeof(int));
            }
            MPI_Scatterv(global_genes, recv_counts, displacements, MPI_INT,
                        local_genes, local_size * ITEMS_NUM, MPI_INT, 0, MPI_COMM_WORLD);

            #pragma omp parallel for
            for (int i = 0; i < local_size; i += 2) {
                // crossover_array(local_genes + i * ITEMS_NUM, local_genes + (i + 1) * ITEMS_NUM, local_new_genes + i * ITEMS_NUM, local_new_genes + (i + 1) * ITEMS_NUM);
                // mutate_array(local_new_genes + i * ITEMS_NUM);
                // mutate_array(local_new_genes + (i + 1) * ITEMS_NUM);

                // Crossover and mutation phase
                int *genes1 = local_genes + i * ITEMS_NUM;
                int *genes2 = local_genes + (i + 1) * ITEMS_NUM;
                int *newgenes1 = local_new_genes + i * ITEMS_NUM;
                int *newgenes2 = local_new_genes + (i + 1) * ITEMS_NUM;
                unsigned int local_seed = omp_get_thread_num();
                if ((rand_r(&local_seed) / (float)RAND_MAX) < CROSSOVER_RATE) {
                    int crossover_point = rand_r(&local_seed) % ITEMS_NUM;
                    for (int i = 0; i < crossover_point; i++) {
                        newgenes1[i] = genes1[i];
                        newgenes2[i] = genes2[i];
                    }
                    for (int i = crossover_point; i < ITEMS_NUM; i++) {
                        newgenes1[i] = genes2[i];
                        newgenes2[i] = genes1[i];
                    }
                } else {
                    for (int i = 0; i < ITEMS_NUM; i++) {
                        newgenes1[i] = genes1[i];
                        newgenes2[i] = genes2[i];
                    }
                }

                // Mutation
                for (int j = 0; j < ITEMS_NUM; j++) {
                    if ((rand_r(&local_seed) / (float)RAND_MAX) < MUTATION_RATE) {
                        newgenes1[j] = 1 - newgenes1[j];
                    }
                    if ((rand_r(&local_seed) / (float)RAND_MAX) < MUTATION_RATE) {
                        newgenes2[j] = 1 - newgenes2[j];;
                    }
                }
            }

            MPI_Gatherv(local_new_genes, local_size * ITEMS_NUM, MPI_INT,
                        global_genes, recv_counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

            for (int i = 0; i < POP_SIZE; i++) {
                population[i].genes = global_genes + i * ITEMS_NUM;
            }

            // Evaluate population fitness
            #pragma omp parallel for
            for (int i = 0; i < POP_SIZE; i++) {
                population[i].fitness = calculate_fitness(population[i]);
            }
        }
    }
    else { // slave process
        for (int generation = 0; generation < GENERATIONS; generation++) {
            MPI_Scatterv(global_genes, recv_counts, displacements, MPI_INT,
                        local_genes, local_size * ITEMS_NUM, MPI_INT, 0, MPI_COMM_WORLD);

            #pragma omp parallel for
            for (int i = 0; i < local_size; i += 2) {
                // Crossover and mutation phase
                int *genes1 = local_genes + i * ITEMS_NUM;
                int *genes2 = local_genes + (i + 1) * ITEMS_NUM;
                int *newgenes1 = local_new_genes + i * ITEMS_NUM;
                int *newgenes2 = local_new_genes + (i + 1) * ITEMS_NUM;
                unsigned int local_seed = omp_get_thread_num();
                if ((rand_r(&local_seed) / (float)RAND_MAX) < CROSSOVER_RATE) {
                    int crossover_point = rand_r(&local_seed) % ITEMS_NUM;
                    for (int i = 0; i < crossover_point; i++) {
                        newgenes1[i] = genes1[i];
                        newgenes2[i] = genes2[i];
                    }
                    for (int i = crossover_point; i < ITEMS_NUM; i++) {
                        newgenes1[i] = genes2[i];
                        newgenes2[i] = genes1[i];
                    }
                } else {
                    for (int i = 0; i < ITEMS_NUM; i++) {
                        newgenes1[i] = genes1[i];
                        newgenes2[i] = genes2[i];
                    }
                }

                // Mutation
                for (int j = 0; j < ITEMS_NUM; j++) {
                    if ((rand_r(&local_seed) / (float)RAND_MAX) < MUTATION_RATE) {
                        newgenes1[j] = 1 - newgenes1[j];
                    }
                    if ((rand_r(&local_seed) / (float)RAND_MAX) < MUTATION_RATE) {
                        newgenes2[j] = 1 - newgenes2[j];;
                    }
                }
            }

            MPI_Gatherv(local_new_genes, local_size * ITEMS_NUM, MPI_INT,
                        global_genes, recv_counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }


    if (mpi_rank == 0) {
            int best_fitness = INT_MIN;
            int best_index = -1;

            #pragma omp parallel for
            for (int i = 0; i < POP_SIZE; i++) {
                if (population[i].fitness > best_fitness) {
                    best_fitness = population[i].fitness;
                    best_index = i;
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &end_time);
            double elapsed_time = get_elapsed_time(start_time, end_time);

            if (population[best_index].fitness == 0) {
                printf("No solution found\n");
                printf("Execution time: %.2f seconds\n", elapsed_time);
                return 0;
            }

            printf("Best solution found in generation %d with fitness %d:\n", GENERATIONS, best_fitness);
            printf("Items included (binary): ");
            for (int i = 0; i < ITEMS_NUM; i++) {
                printf("%d ", population[best_index].genes[i]);
            }
            printf("\n");

            printf("Total weight: %d, Total value: %d\n", get_total_weight(population[best_index]), get_total_value(population[best_index]));
            printf("Capacity: %d\n", KNAPSACK_CAPACITY);

            printf("Execution time: %.2f seconds\n", elapsed_time);

            // free(population);
        }

    // if (mpi_rank == 0) {
    //     free(global_genes);
    // }
    // free(local_genes);
    // free(recv_counts);
    // free(displacements);

    MPI_Finalize();
    return 0;
}

void input(const char* filename) {
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d", &KNAPSACK_CAPACITY);
    fscanf(file, "%d", &ITEMS_NUM);
    fscanf(file, "%d", &GENERATIONS);
    fscanf(file, "%d", &POP_SIZE);
    fscanf(file, "%f", &MUTATION_RATE);
    fscanf(file, "%f", &CROSSOVER_RATE);

    weights = (int *)malloc(ITEMS_NUM * sizeof(int));
    values = (int *)malloc(ITEMS_NUM * sizeof(int));

    for (int i = 0; i < ITEMS_NUM; i++) {
        fscanf(file, "%d %d", &weights[i], &values[i]);
    }

    fclose(file);
}

void initialize_population(Individual population[]) {
    #pragma parallel for
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].genes = (int *)malloc(ITEMS_NUM * sizeof(int));
        for (int j = 0; j < ITEMS_NUM; j++) {
            population[i].genes[j] = rand() % 2;
        }
    }
}

void evaluate_population(Individual population[]) {
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].fitness = calculate_fitness(population[i]);
    }
}

int calculate_fitness(Individual individual) {
    int total_value = get_total_value(individual);
    int total_weight = get_total_weight(individual);

    // If the weight exceeds the capacity, the fitness is 0 (invalid solution)
    if (total_weight > KNAPSACK_CAPACITY) {
        return 0;
    }
    return total_value;  // Maximizing the value
}

int get_total_weight(Individual individual) {
    int total_weight = 0;
    for (int i = 0; i < ITEMS_NUM; i++) {
        if (individual.genes[i] == 1) {
            total_weight += weights[i];
        }
    }
    return total_weight;
}

int get_total_value(Individual individual) {
    int total_value = 0;
    for (int i = 0; i < ITEMS_NUM; i++) {
        if (individual.genes[i] == 1) {
            total_value += values[i];
        }
    }
    return total_value;
}

void selection(Individual population[], Individual new_population[]) {
    for (int i = 0; i < POP_SIZE; i++) {
        // Tournament selection: pick two random individuals and select the best one
        int parent1 = rand() % POP_SIZE;
        int parent2 = rand() % POP_SIZE;
        
        // Select the better parent
        Individual selected = (population[parent1].fitness > population[parent2].fitness) ? population[parent1] : population[parent2];
        
        // Allocate memory for the new individual's genes
        new_population[i].genes = (int*)malloc(sizeof(int) * ITEMS_NUM);
        
        // Copy the genes from the selected parent to the new individual
        for (int j = 0; j < ITEMS_NUM; j++) {
            new_population[i].genes[j] = selected.genes[j];
        }
        
        // Copy the fitness value
        new_population[i].fitness = selected.fitness;
    }
}

void crossover_array(int *genes1, int *genes2, int *newgenes1, int *newgenes2) {
    if ((rand() / (float)RAND_MAX) < CROSSOVER_RATE) {
        int crossover_point = rand() % ITEMS_NUM;
        for (int i = 0; i < crossover_point; i++) {
            newgenes1[i] = genes1[i];
            newgenes2[i] = genes2[i];
        }
        for (int i = crossover_point; i < ITEMS_NUM; i++) {
            newgenes1[i] = genes2[i];
            newgenes2[i] = genes1[i];
        }
    }
    else {
        for (int i = 0; i < ITEMS_NUM; i++) {
            newgenes1[i] = genes1[i];
            newgenes2[i] = genes2[i];
        }
    }
}

void mutate_array(int *genes) {
    for (int i = 0; i < ITEMS_NUM; i++) {
        if ((rand() / (float)RAND_MAX) < MUTATION_RATE) {
            genes[i] = 1 - genes[i];
        }
    }
}
