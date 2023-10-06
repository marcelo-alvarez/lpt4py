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
