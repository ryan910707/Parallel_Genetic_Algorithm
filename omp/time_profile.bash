#!/bin/bash

# Loop from 1 to 12 to execute the commands with different core values
for cores in $(seq 4 6 8 10 12); do
    # Prepare the command to be run
    cmd="srun -N1 -n1 -c${cores} ./omp ../testcases/n1000.txt"

    # Display the command to the terminal
    echo "Running: $cmd"

    # Run the command directly in the terminal
    eval $cmd
done
