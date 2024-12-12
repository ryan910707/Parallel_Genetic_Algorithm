#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

int POP_SIZE;
int ITEMS_NUM;// Number of items
int GENERATIONS;
int MUTATION_RATE;
int CROSSOVER_RATE;
int KNAPSACK_CAPACITY;

typedef struct {
    int *genes;  // Binary representation of the knapsack selection
    int fitness;
} Individual;
Individual *population, *new_population;


int *weights;  // Weights of the items
int *values;
int cpu_cnt;

pthread_barrier_t barrier;

void* cal(void* thread_id);
void input(const char* filename);
void initialize_population(Individual population[]);
void evaluate_population(Individual population[]);
int calculate_fitness(Individual individual);
void selection(Individual population[], Individual new_population[]);
void crossover(Individual parent1, Individual parent2, Individual *child1, Individual *child2);
void mutate(Individual *individual);
int get_total_weight(Individual individual);
int get_total_value(Individual individual);

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

    // Start time
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // set cpu_cnt
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_cnt = CPU_COUNT(&cpu_set);

    printf("core: %d\n", cpu_cnt);

    input(argv[1]);

    population = (Individual *) malloc(sizeof(Individual) * POP_SIZE);
    new_population = (Individual *) malloc(sizeof(Individual) * POP_SIZE);

    initialize_population(population);
    evaluate_population(population);

    pthread_t threads[cpu_cnt];
    int ID[cpu_cnt];
    pthread_barrier_init(&barrier, NULL, cpu_cnt);

    for (int i = 0; i < cpu_cnt; ++i) {
        ID[i] = i;
        pthread_create(&threads[i], NULL, cal, (void*)&ID[i]);
    }
    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Output the best solution found
    int best_fitness = population[0].fitness;
    int best_index = 0;
    for (int i = 1; i < POP_SIZE; i++) {
        if (population[i].fitness > best_fitness) {
            best_fitness = population[i].fitness;
            best_index = i;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = get_elapsed_time(start_time, end_time);

    printf("Best solution found in generation %d with fitness %d:\n", GENERATIONS, best_fitness);

    printf("Items included (binary): ");
    for (int i = 0; i < ITEMS_NUM; i++) {
        printf("%d ", population[best_index].genes[i]);
    }
    printf("\n");

    printf("Total weight: %d, Total value: %d\n", get_total_weight(population[best_index]), get_total_value(population[best_index]));
    printf("Capacity: %d\n", KNAPSACK_CAPACITY);

    printf("Execution time: %.2f seconds\n", elapsed_time);

    return 0;
}

void* cal(void* thread_id) {
    int generation = 0;
    int tid = *(int*)thread_id;
    int total_value, total_weight, parent1, parent2;
    unsigned int local_seed = tid;

    while (generation < GENERATIONS){
        // selection
        for (int i = tid; i < POP_SIZE; i += cpu_cnt) {
            parent1 = rand_r(&local_seed)  % POP_SIZE;
            parent2 = rand_r(&local_seed)  % POP_SIZE;
            new_population[i] = population[parent1].fitness > population[parent2].fitness ? population[parent1] : population[parent2];
        }
        pthread_barrier_wait(&barrier);

        // crossover, mutate, mutate
        for (int i = tid*2; i < POP_SIZE; i += cpu_cnt*2) {
            // crossover
            if ((rand_r(&local_seed)  / (float)RAND_MAX) < CROSSOVER_RATE) {
                int crossover_point = rand_r(&local_seed)  % ITEMS_NUM;
                for (int j = 0; j < crossover_point; j++) {
                    population[i].genes[j] = new_population[i].genes[j];
                    population[i+1].genes[j] = new_population[i+1].genes[j];
                }
                for (int j = crossover_point; j < ITEMS_NUM; j++) {
                    population[i].genes[j] = new_population[i+1].genes[j];
                    population[i+1].genes[j] = new_population[i].genes[j];
                }
            } else {
                population[i] = new_population[i];
                population[i+1] = new_population[i+1];
            }

            for (int i = 0; i < ITEMS_NUM; i++) {
                if ((rand_r(&local_seed) / (float)RAND_MAX) < MUTATION_RATE) {
                    population[i].genes[i] = 1 - population[i].genes[i];  // Flip the gene (0 to 1, or 1 to 0)
                }
            }
            for (int i = 0; i < ITEMS_NUM; i++) {
                if ((rand_r(&local_seed)  / (float)RAND_MAX) < MUTATION_RATE) {
                    population[i+1].genes[i] = 1 - population[i+1].genes[i];  // Flip the gene (0 to 1, or 1 to 0)
                }
            }
        }
        pthread_barrier_wait(&barrier);
        
        // evaluate population
        for (int i = tid; i < POP_SIZE; i += cpu_cnt) {
            total_value = get_total_value(population[i]);
            total_weight = get_total_weight(population[i]);
            population[i].fitness = (total_weight > KNAPSACK_CAPACITY) ? 0 : total_value;
        }

        generation++;
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void input(const char* filename) {
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d", &KNAPSACK_CAPACITY);
    fscanf(file, "%d", &ITEMS_NUM);
    fscanf(file, "%d", &GENERATIONS);
    fscanf(file, "%d", &POP_SIZE);
    fscanf(file, "%f", &MUTATION_RATE);
    fscanf(file, "%f", &CROSSOVER_RATE);


    weights = (int*)malloc(ITEMS_NUM * sizeof(int));
    values = (int*)malloc(ITEMS_NUM * sizeof(int));

    for (int i = 0; i < ITEMS_NUM; i++) {
        fscanf(file, "%d %d", &weights[i], &values[i]);
    }
}

void initialize_population(Individual population[]) {
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].genes = (int*)malloc(sizeof(int) * ITEMS_NUM);
        for (int j = 0; j < ITEMS_NUM; j++) {
            population[i].genes[j] = rand() % 2;  // Random 0 or 1
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
        new_population[i] = population[parent1].fitness > population[parent2].fitness ? population[parent1] : population[parent2];
    }
}

void crossover(Individual parent1, Individual parent2, Individual *child1, Individual *child2) {
    if ((rand() / (float)RAND_MAX) < CROSSOVER_RATE) {
        int crossover_point = rand() % ITEMS_NUM;
        for (int i = 0; i < crossover_point; i++) {
            child1->genes[i] = parent1.genes[i];
            child2->genes[i] = parent2.genes[i];
        }
        for (int i = crossover_point; i < ITEMS_NUM; i++) {
            child1->genes[i] = parent2.genes[i];
            child2->genes[i] = parent1.genes[i];
        }
    } else {
        *child1 = parent1;
        *child2 = parent2;
    }
}

void mutate(Individual *individual) {
    for (int i = 0; i < ITEMS_NUM; i++) {
        if ((rand() / (float)RAND_MAX) < MUTATION_RATE) {
            individual->genes[i] = 1 - individual->genes[i];  // Flip the gene (0 to 1, or 1 to 0)
        }
    }
}