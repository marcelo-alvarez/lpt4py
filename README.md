# lpt4py
Massively parallel GPU-enabled initial conditions and Lagrangian perturbation theory in Python using [jax.]numpy.

## Installation
pip install git+https://github.com/marcelo-alvarez/lpt4py

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/lpt4py/blob/master/scripts/example.py) will generate a random density field and calculate 2LPT displacement coefficients.

E.g. on Sherlock:
```
% python example.py
1.377224 sec for initialization
6.567176 sec for first 2LPT
2.534652 sec for 2LPT
2.467021 sec for 2LPT
2.533598 sec for 2LPT
2.608560 sec for 2LPT
2.541626 sec for 2LPT
```
and on Marlowe:
```
% srun -N 1 -n 4 -A marlowe-m000079 -G 4 -p preempt python example.py
7.280852 sec for initialization
5.194542 sec for first 2LPT
0.734118 sec for 2LPT
0.720596 sec for 2LPT
0.728951 sec for 2LPT
0.731462 sec for 2LPT
```
