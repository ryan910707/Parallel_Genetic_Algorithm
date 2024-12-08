#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int POP_SIZE;
int ITEMS_NUM;// Number of items
int GENERATIONS;
float MUTATION_RATE;
float CROSSOVER_RATE;
int KNAPSACK_CAPACITY;

int *weights;  // Weights of the items
int *values;
int *fitness;  // 1D array to store fitness values
int **genes;   // 2D array to store population genes

void input(const char* filename);
void initialize_population();
void evaluate_population();
void selection(int **new_population);
void crossover(int *parent1, int *parent2, int *child1, int *child2);
void mutate(int *individual);

int main(int argc, char** argv) {
    clock_t start_time = clock();
    srand(1);
    input(argv[1]);

    // Allocate memory for genes and fitness
    genes = (int**)malloc(POP_SIZE * sizeof(int*));
    fitness = (int*)malloc(POP_SIZE * sizeof(int));
    for (int i = 0; i < POP_SIZE; i++) {
        genes[i] = (int*)malloc(ITEMS_NUM * sizeof(int));
    }

    int **new_population = (int**)malloc(POP_SIZE * sizeof(int*));
    for (int i = 0; i < POP_SIZE; i++) {
        new_population[i] = (int*)malloc(ITEMS_NUM * sizeof(int));
    }

    initialize_population();
    evaluate_population();

    int generation = 0;
    while (generation < GENERATIONS) {
        selection(new_population);
        for (int i = 0; i < POP_SIZE; i += 2) {
            crossover(new_population[i], new_population[i + 1], genes[i], genes[i + 1]);
            mutate(genes[i]);
            mutate(genes[i + 1]);
        }
        evaluate_population();
        generation++;
    }

    // Output the best solution found
    int best_fitness = fitness[0];
    int best_index = 0;
    for (int i = 1; i < POP_SIZE; i++) {
        if (fitness[i] > best_fitness) {
            best_fitness = fitness[i];
            best_index = i;
        }
    }

    // Timer ends
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    if(best_fitness == 0){
        printf("No solution found\n");
        printf("Execution time: %.2f seconds\n", elapsed_time);
        return 0;
    }
    printf("Best solution found in generation %d with fitness %d:\n", GENERATIONS, best_fitness);
    printf("Items included (binary): ");
    for (int i = 0; i < ITEMS_NUM; i++) {
        printf("%d ", genes[best_index][i]);
    }
    printf("\n");

    // Calculate total weight and value for the best solution
    int total_weight = 0, total_value = 0;
    for (int i = 0; i < ITEMS_NUM; i++) {
        if (genes[best_index][i] == 1) {
            total_weight += weights[i];
            total_value += values[i];
        }
    }
    printf("Total weight: %d, Total value: %d\n", total_weight, total_value);
    printf("capacity: %d\n", KNAPSACK_CAPACITY);

    printf("Execution time: %.2f seconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < POP_SIZE; i++) {
        free(genes[i]);
        free(new_population[i]);
    }
    free(genes);
    free(fitness);
    free(new_population);
    free(weights);
    free(values);

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

    weights = (int*)malloc(ITEMS_NUM * sizeof(int));
    values = (int*)malloc(ITEMS_NUM * sizeof(int));

    for (int i = 0; i < ITEMS_NUM; i++) {
        fscanf(file, "%d %d", &weights[i], &values[i]);
    }
}

void initialize_population() {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < ITEMS_NUM; j++) {
            genes[i][j] = rand() % 2;  // Random 0 or 1
        }
    }
}

void evaluate_population() {
    for (int i = 0; i < POP_SIZE; i++) {
        // Calculate total weight and value
        int total_weight = 0, total_value = 0;
        for (int j = 0; j < ITEMS_NUM; j++) {
            if (genes[i][j] == 1) {
                total_weight += weights[j];
                total_value += values[j];
            }
        }
        
        // Determine fitness: if weight exceeds capacity, fitness is 0
        fitness[i] = (total_weight > KNAPSACK_CAPACITY) ? 0 : total_value;
    }
}

void selection(int **new_population) {
    for (int i = 0; i < POP_SIZE; i++) {
        // Tournament selection: pick two random individuals and select the best one
        int parent1 = rand() % POP_SIZE;
        int parent2 = rand() % POP_SIZE;
        
        // Select the better parent
        int *selected = (fitness[parent1] > fitness[parent2]) ? genes[parent1] : genes[parent2];
        
        // Copy the genes from the selected parent to the new individual
        for (int j = 0; j < ITEMS_NUM; j++) {
            new_population[i][j] = selected[j];
        }
    }
}

void crossover(int *parent1, int *parent2, int *child1, int *child2) {
    if ((rand() / (float)RAND_MAX) < CROSSOVER_RATE) {
        int crossover_point = rand() % ITEMS_NUM;
        for (int i = 0; i < crossover_point; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
        for (int i = crossover_point; i < ITEMS_NUM; i++) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    } else {
        for (int i = 0; i < ITEMS_NUM; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    }
}

void mutate(int *individual) {
    for (int i = 0; i < ITEMS_NUM; i++) {
        if ((rand() / (float)RAND_MAX) < MUTATION_RATE) {
            individual[i] = 1 - individual[i];  // Flip the gene (0 to 1, or 1 to 0)
        }
    }
}