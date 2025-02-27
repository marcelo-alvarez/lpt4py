# lpt4py
Massively parallel GPU-enabled initial conditions and Lagrangian perturbation theory in Python using [jax.]numpy.

## Installation
pip install git+https://github.com/marcelo-alvarez/lpt4py@fft-rand-update

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate a random density field and calculate 2LPT displacement coefficients i.e.:
```
% python example.py
0.817988 sec for initialization
3.890242 sec for 2LPT
```