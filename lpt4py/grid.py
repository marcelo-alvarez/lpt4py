import jax
import sys
import os

import scaleran as sr

import jax.numpy as jnp 
import jax.random as rnd


class Grid:
    '''Grid'''
    def __init__(self, **kwargs):

        self.N       = kwargs.get('N',512)
        self.partype = kwargs.get('partype','jaxshard')

    def _generate_sharded_noise(self, N, noisetype, seed, nsub):
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding            
        ngpus = int(os.environ.get("XGSMENV_NGPUS"))
        host_id = jax.process_index()
        start = host_id * N // ngpus
        end = (host_id + 1) * N // ngpus
        #jax.distributed.initialize()

        stream = sr.Stream(seed=seed,nsub=nsub)
        noise = stream.generate(start=start*N**2,size=(end-start)*N**2)
        noise = jnp.reshape(noise,(end-start,N,N))
        return noise

    def generate_noise(self, noisetype='white', nsub=1024**2, seed=13579):

        N = self.N

        noise = None
        if self.partype is None:
            stream = sr.Stream(seed=seed,nsub=nsub)
            noise = stream.generate(start=0,size=N**3)
            noise = jnp.reshape(noise,(N,N,N))
            return noise
        elif self.partype == 'jaxshard':
            noise = self._generate_sharded_noise(N, noisetype, seed, nsub)

        return noise







