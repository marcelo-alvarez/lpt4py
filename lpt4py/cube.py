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
        self.Lbox    = kwargs.get('Lbox',7700.0)
        self.partype = kwargs.get('partype','jaxshard')

        self.delta = None
        self.s1lpt = None
        self.s2lpt = None

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        self.start = 0
        self.end   = self.N

        if self.partype == 'jaxshard':
            self.ngpus   = int(os.environ.get("XGSMENV_NGPUS"))
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)

        k0 = 2*jnp.pi/self.Lbox*self.N
        self.kx = jnp.fft.fftfreq(self.N) * k0
        self.ky = jnp.fft.fftfreq(self.N) * k0
        self.ky = self.ky[self.start:self.end]
        self.kz = jnp.fft.rfftfreq(self.N) * k0

        kxa,kya,kza = jnp.meshgrid(self.kx,self.ky,self.kz,indexing='ij')

        self.k2 = kxa**2+kya**2+kza**2
        self.kx = self.kx.at[self.N//2].set(0.0)
        self.kz = self.kz.at[-1].set(0.0)

        if self.start <= self.N//2 and self.end > self.N//2: 
            self.ky = self.ky.at[self.N//2-self.start].set(0.0)

        self.kx = self.kx[:,None,None]
        self.ky = self.ky[None,:,None]
        self.kz = self.kz[None,None,:]

        self.index0 = jnp.nonzero(self.k2==0.0)

    def _generate_sharded_noise(self, N, noisetype, seed, nsub):           
        ngpus   = self.ngpus
        host_id = self.host_id
        start   = self.start
        end     = self.end

        stream = sr.Stream(seed=seed,nsub=nsub)
        noise = stream.generate(start=start*N**2,size=(end-start)*N**2).astype(jnp.float32)
        noise = jnp.reshape(noise,(end-start,N,N))
        return jnp.transpose(noise,(1,0,2)) 

    def _generate_serial_noise(self, N, noisetype, seed, nsub):
        stream = sr.Stream(seed=seed,nsub=nsub)
        noise = stream.generate(start=0,size=N**3).astype(jnp.float32)
        noise = jnp.reshape(noise,(N,N,N))
        return jnp.transpose(noise,(1,0,2))

    def _get_grid_transfer_function(self):
        # currently identity transfer function as a placeholder
        return (jnp.zeros(self.cshape_local)+1.0).astype(jnp.float32)

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

    def noise2delta(self):
        self.delta = self._fft(self.noise)
        self.delta = self._get_grid_transfer_function()*self.delta
        self.delta = self._fft(self.delta,direction='c2r')

    def slpt(self, infield='noise'):

        def _get_shear_factor(ki,kj):
            arr = ki*kj/self.k2*self.delta
            if self.host_id == 0: 
                arr = arr.at[self.index0].set(0.0+0.0j)
            arr = self._fft(arr,direction='c2r')
            return arr

        def _delta_to_s2(ki,delta):
            arr = (0+1j)*ki/self.k2*delta
            if self.host_id == 0: 
                arr = arr.at[self.index0].set(0.0+0.0j)
            arr = self._fft(arr,direction='c2r')
            return arr

        if infield == 'noise':
            # FT of delta from noise
            self.delta = self._fft(self.noise)*self._get_grid_transfer_function()
            del self.noise
        elif infield == 'delta':
            # FT of delta
            self.delta = self._fft(self.delta)
        else:
            import numpy as np
            # delta from external file
            self.delta = jnp.asarray(np.fromfile(infield,dtype=jnp.float32,count=self.N*self.N*self.N))
            self.delta = jnp.reshape(self.delta,self.rshape)
            self.delta = self.delta[:,self.start:self.end,:]
            # FT of delta
            self.delta = self._fft(self.delta)
    
        # calculate delta2
        self.sxx = _get_shear_factor(self.kx,self.kx)
        self.syy = _get_shear_factor(self.ky,self.ky)
        self.delta2  = - self.sxx * self.syy 

        self.szz = _get_shear_factor(self.kz,self.kz)
        self.delta2 -= self.sxx * self.szz ; del self.sxx 
        self.delta2 -= self.syy * self.szz ; del self.syy ; del self.szz 

        self.sxy = _get_shear_factor(self.kx,self.ky)
        self.delta2 += self.sxy * self.sxy ; del self.sxy

        self.sxz = _get_shear_factor(self.kx,self.kz)
        self.delta2 += self.sxz * self.sxz ; del self.sxz

        self.syz = _get_shear_factor(self.ky,self.kz)
        self.delta2 += self.syz * self.syz ; del self.syz

        # FT delta2
        self.delta2 = self._fft(self.delta2)

        # 1st order displacements
        self.sx1 = _delta_to_s2(self.kx,self.delta)
        self.sy1 = _delta_to_s2(self.ky,self.delta)
        self.sz1 = _delta_to_s2(self.kz,self.delta)

        # 2nd order displacements
        self.sx2 = _delta_to_s2(self.kx,self.delta2)
        self.sy2 = _delta_to_s2(self.ky,self.delta2)
        self.sz2 = _delta_to_s2(self.kz,self.delta2)
    

