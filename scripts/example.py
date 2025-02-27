from time import time
times={'t0' : time()}

import jax
import lpt4py as lpt
import argparse
import sys
from jax.experimental.multihost_utils import sync_global_devices

jax.config.update("jax_enable_x64", True)

def myprint(*args,**kwargs):
    print("".join(map(str,args)),**kwargs);  sys.stdout.flush()

def _profiletime(task_tag, step, times, parallel=False, host_id=0):
    if parallel:
        sync_global_devices('profiletime')

    dt = time() - times['t0']
    if step in times.keys():
        times[step] += dt
    else:
        times[step] = dt
    times['t0'] = time()

    if host_id!=0:
        return times

    if task_tag is not None:
        myprint(f'{task_tag}: {dt:.6f} sec for {step}')
    else:
        myprint(f'{dt:.6f} sec for {step}')

    return times

dN = 128
dseed = 13579
dinfile = "__noise__"
parser = argparse.ArgumentParser(description='Commandline interface to lpt4py example')
parser.add_argument('--N',     type=int, help=f'grid dimention [default = {dN}]', default=dN)
parser.add_argument('--seed',  type=int, help=f'noise with random seed when infile = "{dinfile}" [default = {dseed}]',  default=dseed)
parser.add_argument('--infile',type=str, help=f'lpt input filename [default = "{dinfile}"]',  default=dinfile)
parser.add_argument('--parallel', action=argparse.BooleanOptionalAction)
parser.set_defaults(parallel=True)

args = parser.parse_args()

N      = args.N
seed   = args.seed
infile = args.infile

parallel = args.parallel
if parallel:
    jax.distributed.initialize()
    host_id = jax.process_index()
else:
    host_id = 0
print(f"parallel: {parallel}")
times = _profiletime(None, 'initialization', times, parallel, host_id)

# LPT displacements
nrun=5
box = lpt.Box(N=N,parallel=parallel,delta=None,seeds=[(seed+i) for i in range(nrun)])
box.slpt()
times = _profiletime(None, 'first 2LPT', times, parallel, host_id)
for i in range(5):
    box.slpt()
    times = _profiletime(None, '2LPT', times, parallel, host_id)

# LPT displacements are now in
#   cube.s1x
#   cube.s1y
#   cube.s1z
# and
#   cube.s2x
#   cube.s2y
#   cube.s2z