```
module load nvhpc-nompi/24.9
srun -n 1 --gres=gpu:1 ./OpenAcc ../testcase/{testcase}
```
