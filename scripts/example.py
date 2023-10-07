import jax
import lpt4py as lpt
from mpi4py import MPI
import argparse
import sys
from time import time

jax.config.update("jax_enable_x64", True)

def myprint(*args,**kwargs):
    print("".join(map(str,args)),**kwargs);  sys.stdout.flush()

def _profiletime(task_tag, step, times):
    dt = time() - times['t0']
    myprint(f'{task_tag}: {dt:.6f} sec for {step}')
    if step in times.keys():
        times[step] += dt
    else:
        times[step] = dt
    times['t0'] = time()
    return times

parser = argparse.ArgumentParser(description='Commandline interface to lpt4py example')
parser.add_argument('--N',     type=int, help='grid dimention [default = 512]', default=512)
parser.add_argument('--seed',  type=int, help='random seed [default = 13579]',  default=13579)
args = parser.parse_args()

N    = args.N
seed = args.seed

parallel = False
nproc    = MPI.COMM_WORLD.Get_size()
mpiproc  = MPI.COMM_WORLD.Get_rank()
if MPI.COMM_WORLD.Get_size() > 1: parallel = True

if not parallel:
    cube = lpt.Cube(N=N,partype=None)  
else:
    jax.distributed.initialize()
    cube = lpt.Cube(N=N)

tgridmap0 = time()
overalltimes = {}
times = {}
overalltimes={'t0' : time()}
times={'t0' : time()}

task_tag = "MPI process "+str(mpiproc)

cube.generate_noise(seed=seed)

if mpiproc==0:
    myprint(f"[{mpiproc}] shape of noise is {cube.delta.shape}")
    myprint(f"[{mpiproc}] noise[0,0,0]={cube.delta[0,0,0]}")
if mpiproc==nproc-1:
    myprint(f"[{mpiproc}] noise[-1,-1,-1]={cube.delta[-1,-1,-1]}")

MPI.COMM_WORLD.Barrier()
if mpiproc==0: 
    times = _profiletime(task_tag, 'noise generation', times)
    myprint("")
    
cube.noise2delta()

if mpiproc==0:
    myprint(f"[{mpiproc}] shape of delta is {cube.delta.shape}")
    myprint(f"[{mpiproc}] delta[0,0,0]={cube.delta[0,0,0]}")
if mpiproc==nproc-1:
    myprint(f"[{mpiproc}] delta[-1,-1,-1]={cube.delta[-1,-1,-1]}")

MPI.COMM_WORLD.Barrier()
if mpiproc==0:
    times = _profiletime(task_tag, 'noise convolution', times)
    myprint("")