# lpt4py
Massively parallel GPU-enabled initial conditions and Lagrangian perturbation theory in Python using [jax.]numpy.

## Installation
1. git clone https://github.com/marcelo-alvarez/lpt4py.git
2. cd lpt4py
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/marcelo-alvarez/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate and convolve white noise for a 2048^3 grid, i.e.:
```
# on Perlmutter at NERSC
% module use /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20230615-0.0.1/modulefiles/
% module load xgsmenv
% salloc -N 2 -C gpu
% export XGSMENV_NGPUS=8
% srun -n 8 python scripts/example.py --N 2048 --seed 13579
srun -n 8 python /global/cfs/cdirs/mp107/exgal/users/malvarez/lpt4py/scripts/example.py --N 2048 --seed 13579
MPI process 7: 2.537362 sec for noise generation
MPI process 6: 2.538222 sec for noise generation
MPI process 1: 2.541720 sec for noise generation
MPI process 2: 2.540831 sec for noise generation
MPI process 4: 2.542431 sec for noise generation
MPI process 0: 2.543452 sec for noise generation
[0] shape of cube.noise: (512, 64, 512)
MPI process 3: 2.541479 sec for noise generation
MPI process 5: 2.545826 sec for noise generation
[7] noise[-1,-1,-1]=0.47201836109161377
[0] noise[0,0,0]=-1.6708143949508667
MPI process 0: 3.486865 sec for noise convolution
MPI process 1: 3.488407 sec for noise convolution
MPI process 6: 3.492072 sec for noise convolution
MPI process 3: 3.486915 sec for noise convolution
MPI process 2: 3.487583 sec for noise convolution
MPI process 7: 3.493012 sec for noise convolution
[0] shape of cube.delta: (512, 64, 512)
MPI process 4: 3.487171 sec for noise convolution
MPI process 5: 3.484784 sec for noise convolution
[0] delta[0,0,0]=-1.670815110206604
[7] delta[-1,-1,-1]=0.4720168709754944
```
