# lpt4py
Massively parallel GPU-enabled initial conditions and Lagrangian perturbation theory in Python using [jax.]numpy.

## Installation
1. git clone https://github.com/marcelo-alvarez/lpt4py.git
2. cd lpt4py
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/marcelo-alvarez/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate white noise for a 512^3 grid.

```
import jax
import lpt4py as lpt
from mpi4py import MPI

parallel = False
nproc    = MPI.COMM_WORLD.Get_size()
mpiproc  = MPI.COMM_WORLD.Get_rank()
if MPI.COMM_WORLD.Get_size() > 1: parallel = True

if not parallel:
    grid = lpt.Grid(N=512,partype=None)
else:
    jax.distributed.initialize()
    grid = lpt.Grid(N=512)


wn = grid.generate_noise(seed=12345)

if mpiproc==0:
    print(f"[{mpiproc}] wn[0,0,0]={wn[0,0,0]}")
if mpiproc==nproc-1:
    print(f"[{mpiproc}] wn[-1,-1,-1]={wn[-1,-1,-1]}")
```
i.e.:
```
# on Perlmutter at NERSC
% salloc -N 2 -C gpu
% srun -n 8 python -u scripts/example.py
[7] wn[-1,-1,-1]=1.010379433631897
[0] wn[0,0,0]=1.0800635814666748
% python -u scripts/example.py
[0] wn[0,0,0]=1.0800635814666748
[0] wn[-1,-1,-1]=1.010379433631897
```
