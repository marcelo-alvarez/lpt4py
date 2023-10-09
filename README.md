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
% srun -n 4 python /global/cfs/cdirs/mp107/exgal/users/malvarez/lpt4py/scripts/example.py --N 768 --seed 13579
[3] noise[-1,-1,-1]=-1.2530277967453003
[0] shape of noise is (768, 192, 768)
[0] noise[0,0,0]=-1.6708143949508667
MPI process 0: 1.338984 sec for noise generation

[0] shape of delta is (768, 192, 768)
[0] delta[0,0,0]=-1.6708143949508667
[3] delta[-1,-1,-1]=-1.253028154373169
MPI process 0: 2.544447 sec for noise convolution

[0] sx1[0,0,0]=5.488100766589582
[0] sy1[0,0,0]=6.761213949848777
[0] sz1[0,0,0]=4.207604505677209
[0] sx2[0,0,0]=0.8204334739641417
[0] sy2[0,0,0]=-0.9351435481275336
[0] sz2[0,0,0]=0.4561066162034884
MPI process 0: 9.779000 sec for 2LPT
```
