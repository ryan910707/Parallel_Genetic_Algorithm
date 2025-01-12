# Parallel Genetic Algorithm for 0/1 knapsack problem

### Summary of Work
* Implemented Parallel Genetic Algorithms using CUDA, OMP, Pthread, MPI, MPI+OMP.
* Addressed challenges like random function overhead efficiently.
### Key Findings
* CUDA achieved the highest speedup (~23x), excelling for small-medium workloads.
* CPU solutions provided stable scalability but faced overhead at larger scales.
* Linear execution time observed with ITEMS_NUM and POP_SIZE.

##### for more detail please reference [our presentation](https://docs.google.com/presentation/d/1DuW7L3bttCXZouUEf8UQYybLsrJLiVCzYDVq7Kd1CdY/edit#slide=id.g31e972243bc_2_52) and [demo video](https://www.youtube.com/watch?v=yhGBLl74hiE&ab_channel=Ryank)

