# lpt4py
Massively parallel GPU-enabled initial conditions and Lagrangian perturbation theory in Python using [jax.]numpy.

## Installation
1. git clone https://github.com/marcelo-alvarez/lpt4py.git
2. cd lpt4py
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/exgalsky/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate/convolve white noise and calculate 2LPT displacement from an external density contrast (websky), all at 768^3, i.e.:
```
# on Perlmutter at NERSC with Nnodes = [2, 32, 256]
% module use /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20231013-0.0.0/modulefiles/
% module load xgsmenv
% salloc -N Nnodes -C gpu
% export XGSMENV_NGPUS=4*Nnodes
% srun -n   8  python lpt4py/scripts/example.py --N 2048 --seed 13579 --ityp delta
 4.9 sec for noise generation
12.4 sec for noise convolution
91.2 sec for 2LPT

% srun -n 128  python lpt4py/scripts/example.py --N 4096 --seed 13579 --ityp delta
 5.9 sec for noise generation
18.2 sec for noise convolution
96.5 sec for 2LPT

% srun -n 1024 python lpt4py/scripts/example.py --N 8192 --seed 13579 --ityp delta
 28.3 sec for noise generation 
 80.0 sec for noise convolution
155.7 sec for 2LPT
```
implying the following scalings:
```
===============================================================
    N  Nodes  GPUs Wall time  GPU hours  Node hours  Node hours 
                       (sec)                         (N/8192)^3
===============================================================

 2048      2     8       109       0.24        0.06         3.9
 4096     32   128       121        4.3         1.1         8.6
 8192    256  1024       264         75          19          19      
```
