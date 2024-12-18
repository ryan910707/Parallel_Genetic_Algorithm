#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

int POP_SIZE;
int ITEMS_NUM; // Number of items
int GENERATIONS;
float MUTATION_RATE;
float CROSSOVER_RATE;
int KNAPSACK_CAPACITY;

int *weights;  // Weights of the items
int *values;   // Values of the items
int **genes;   // Binary representation of the knapsack selection
int *fitness;  // Fitness of each individual

void input(const char* filename);
void initialize_population();
void evaluate_population();
int calculate_fitness(int individual);
void selection();
void crossover(int parent1, int parent2, int *child1, int *child2);
void mutate(int *individual);
int get_total_weight(int individual);
int get_total_value(int individual);

int main(int argc, char** argv) {
    clock_t start_time = clock();
    srand(1);
    input(argv[1]);
    
    // Allocate memory for genes and fitness
    genes = (int**)malloc(POP_SIZE * sizeof(int*));
    for (int i = 0; i < POP_SIZE; i++) {
        genes[i] = (int*)malloc(ITEMS_NUM * sizeof(int));  // Allocate memory for each individual's genes (columns)
    }

    fitness = (int*)malloc(POP_SIZE * sizeof(int));  // Allocate memory for fitness array

    initialize_population();
    evaluate_population();

    int generation = 0;
    while (generation < GENERATIONS) {
        selection();
        for (int i = 0; i < POP_SIZE; i += 2) {
            int child1[ITEMS_NUM], child2[ITEMS_NUM];
            crossover(i, i + 1, child1, child2);
            mutate(child1);
            mutate(child2);
            
            // Copy children back into genes array
            for (int j = 0; j < ITEMS_NUM; j++) {
                genes[i][j] = child1[j];
                genes[i + 1][j] = child2[j];
            }
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

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    if (fitness[best_index] == 0) {
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

    printf("Total weight: %d, Total value: %d\n", get_total_weight(best_index), get_total_value(best_index));
    printf("Capacity: %d\n", KNAPSACK_CAPACITY);
    printf("Execution time: %.2f seconds\n", elapsed_time);

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

    fclose(file);
}

void initialize_population() {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < ITEMS_NUM; j++) {
            genes[i][j] = rand() % 2;  // Randomly assign 0 or 1 to each gene
        }
    }
}

void evaluate_population() {
    #pragma acc parallel loop copyin(genes[0:POP_SIZE][0:ITEMS_NUM], weights[ITEMS_NUM], values[ITEMS_NUM]) copy(fitness[0:POP_SIZE]) 
    for (int k = 0; k < POP_SIZE; k++) {
        fitness[k] = 0;  // Reset fitness
        int total_weight = 0;
        int total_value = 0;
        for (int i = 0; i < ITEMS_NUM; i++) {
            if (genes[k][i] == 1) {
                total_weight += weights[i];
                total_value += values[i];
            }
        }
        // Assign fitness if within the knapsack capacity
        fitness[k] = total_weight > KNAPSACK_CAPACITY ? 0 : total_value;
    }
}

void selection() {
    int new_population[POP_SIZE][ITEMS_NUM];
    for (int i = 0; i < POP_SIZE; i++) {
        // Tournament selection: pick two random individuals and select the best one
        int parent1 = rand() % POP_SIZE;
        int parent2 = rand() % POP_SIZE;
        
        // Select the better parent
        int selected = (fitness[parent1] > fitness[parent2]) ? parent1 : parent2;
        
        // Copy the genes of the selected parent to the new population
        for (int j = 0; j < ITEMS_NUM; j++) {
            new_population[i][j] = genes[selected][j];
        }
    }

    // Copy the new population back to genes
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < ITEMS_NUM; j++) {
            genes[i][j] = new_population[i][j];
        }
    }
}

void crossover(int parent1, int parent2, int *child1, int *child2) {
    if ((rand() / (float)RAND_MAX) < CROSSOVER_RATE) {
        int crossover_point = rand() % ITEMS_NUM;
        for (int i = 0; i < crossover_point; i++) {
            child1[i] = genes[parent1][i];
            child2[i] = genes[parent2][i];
        }
        for (int i = crossover_point; i < ITEMS_NUM; i++) {
            child1[i] = genes[parent2][i];
            child2[i] = genes[parent1][i];
        }
    } else {
        for (int i = 0; i < ITEMS_NUM; i++) {
            child1[i] = genes[parent1][i];
            child2[i] = genes[parent2][i];
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

int get_total_weight(int individual) {
    int total_weight = 0;
    for (int i = 0; i < ITEMS_NUM; i++) {
        if (genes[individual][i] == 1) {
            total_weight += weights[i];
        }
    }
    return total_weight;
}

int get_total_value(int individual) {
    int total_value = 0;
    for (int i = 0; i < ITEMS_NUM; i++) {
        if (genes[individual][i] == 1) {
            total_value += values[i];
        }
    }
    return total_value;
}
