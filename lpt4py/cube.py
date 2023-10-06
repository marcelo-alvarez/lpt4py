import jax
import sys
import os

import scaleran as sr

import jax.numpy as jnp 
import jax.random as rnd

class Cube:
    '''Cube'''
    def __init__(self, **kwargs):

        self.N       = kwargs.get('N',512)
        self.partype = kwargs.get('partype','jaxshard')

        self.noise = None
        self.delta = None
        self.s1lpt = None
        self.s2lpt = None

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        if self.partype == 'jaxshard':
            self.ngpus   = int(os.environ.get("XGSMENV_NGPUS"))
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)          

    def _generate_sharded_noise(self, N, noisetype, seed, nsub):           
        ngpus   = self.ngpus
        host_id = self.host_id
        start   = self.start
        end     = self.end

        stream = sr.Stream(seed=seed,nsub=nsub)
        noise = stream.generate(start=start*N**2,size=(end-start)*N**2)
        noise = jnp.reshape(noise,(end-start,N,N))
        return jnp.transpose(noise,(1,0,2)) 

    def _generate_serial_noise(self, N, noisetype, seed, nsub):
        stream = sr.Stream(seed=seed,nsub=nsub)
        noise = stream.generate(start=0,size=N**3)
        noise = jnp.reshape(noise,(N,N,N))
        return jnp.transpose(noise,(1,0,2))

    def _get_grid_transfer_function(self):
        # currently identity transfer function as a placeholder
        return jnp.zeros(self.cshape_local)+1.0

    def _fft(self,x_np,direction='r2c'):
        
        from . import multihost_rfft
        from jax import jit
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding 
        
        num_gpus = self.ngpus
        if direction=='r2c':
            global_shape = self.rshape
        else:
            global_shape = self.cshape

        devices = mesh_utils.create_device_mesh((num_gpus,))
        mesh = Mesh(devices, axis_names=('gpus',))
        with mesh:
            x_single = jax.device_put(x_np)
            xshard = jax.make_array_from_single_device_arrays(
                global_shape,
                NamedSharding(mesh, P(None, "gpus")),
                [x_single])

            rfftn_jit = jit(
                multihost_rfft.rfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
            irfftn_jit = jit(
                multihost_rfft.irfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
            sync_global_devices("wait for compiler output")

            with jax.spmd_mode('allow_all'):

                if direction=='r2c':
                    rfftn_jit(xshard).block_until_ready()
                    out_jit: jax.Array = rfftn_jit(xshard).block_until_ready()
                else:
                    irfftn_jit(xshard).block_until_ready()
                    out_jit: jax.Array = irfftn_jit(xshard).block_until_ready()
                sync_global_devices("loop")
                local_out_subset = out_jit.addressable_data(0)
        return local_out_subset

    def generate_noise(self, noisetype='white', nsub=1024**3, seed=13579):

        N = self.N

        noise = None
        if self.partype is None:
            self.noise = self._generate_serial_noise(N, noisetype, seed, nsub)
        elif self.partype == 'jaxshard':
            self.noise = self._generate_sharded_noise(N, noisetype, seed, nsub)

    def convolve_noise(self):
        self.delta = self._fft(self.noise)
        transfer = self._get_grid_transfer_function()
        self.delta = transfer * self.delta
        self.delta = self._fft(self.delta,direction='c2r')
