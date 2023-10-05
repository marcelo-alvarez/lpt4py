# lpt4py
Initial conditions and Lagrangian perturbation theory in Python with mpi4py.

## Installation
1. git clone https://github.com/marcelo-alvarez/lpt4py.git
2. cd lpt4py
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/marcelo-alvarez/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate white noise for a 512^3 grid.

```
import lpt4py as lpt

# create grid object
grid = lpt.Grid(N=512)

# create white noise
wn = grid.generate_noise(noisetype='white')

print('wn',wn)
print('wn shape:',wn.shape)
print('wn mean:',wn.mean())
```