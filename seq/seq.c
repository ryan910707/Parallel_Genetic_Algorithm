#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int POP_SIZE = 1000;
int ITEMS_NUM;// Number of items
int GENERATIONS;
float MUTATION_RATE = 0.05;
float CROSSOVER_RATE = 0.7;
int KNAPSACK_CAPACITY;

typedef struct {
    int *genes;  // Binary representation of the knapsack selection
    int fitness;
} Individual;

int *weights;  // Weights of the items
int *values;

void input(const char* filename);
void initialize_population(Individual population[]);
void evaluate_population(Individual population[]);
int calculate_fitness(Individual individual);
void selection(Individual population[], Individual new_population[]);
void crossover(Individual parent1, Individual parent2, Individual *child1, Individual *child2);
void mutate(Individual *individual);
int get_total_weight(Individual individual);
int get_total_value(Individual individual);

int main(int argc, char** argv) {
    clock_t start_time = clock();
    srand(1);
    Individual population[POP_SIZE], new_population[POP_SIZE];
    int generation = 0;

    input(argv[1]);

    initialize_population(population);
    evaluate_population(population);

    while (generation < GENERATIONS) {
        selection(population, new_population);
        for (int i = 0; i < POP_SIZE; i += 2) {
            crossover(new_population[i], new_population[i + 1], &population[i], &population[i + 1]);
            mutate(&population[i]);
            mutate(&population[i + 1]);
        }
        evaluate_population(population);
        generation++;
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

    // Timer ends
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Best solution found in generation %d with fitness %d:\n", GENERATIONS, best_fitness);
    printf("Items included (binary): ");
    for (int i = 0; i < ITEMS_NUM; i++) {
        printf("%d ", population[best_index].genes[i]);
    }
    printf("\n");

    printf("Total weight: %d, Total value: %d\n", get_total_weight(population[best_index]), get_total_value(population[best_index]));
    printf("capacity: %d\n", KNAPSACK_CAPACITY);

    printf("Execution time: %.2f seconds\n", elapsed_time);

    return 0;
}

void input(const char* filename) {
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d", &KNAPSACK_CAPACITY);
    fscanf(file, "%d", &ITEMS_NUM);
    fscanf(file, "%d", &GENERATIONS);

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