import jax
import sys
import os
import gc

import scaleran as sr

import jax.numpy as jnp 
import jax.random as rnd

class Cube:
    '''Cube'''
    def __init__(self, **kwargs):

        self.N       = kwargs.get('N',512)
        self.Lbox    = kwargs.get('Lbox',7700.0)
        self.partype = kwargs.get('partype','jaxshard')

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

    def _apply_grid_transfer_function(self,field):
        # currently identity transfer function as a placeholder
        return field*(jnp.zeros(self.cshape_local)+1.0).astype(jnp.float32)

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
            x_single = jax.device_put(x_np).block_until_ready()
            del x_np ; gc.collect()
            xshard = jax.make_array_from_single_device_arrays(
                global_shape,
                NamedSharding(mesh, P(None, "gpus")),
                [x_single]).block_until_ready()
            del x_single ; gc.collect()
            if direction=='r2c':
                rfftn_jit = jit(
                    multihost_rfft.rfftn,
                    in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                    out_shardings=(NamedSharding(mesh, P(None, "gpus")))
                )
            else:
                irfftn_jit = jit(
                    multihost_rfft.irfftn,
                    in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                    out_shardings=(NamedSharding(mesh, P(None, "gpus")))
                )
            sync_global_devices("wait for compiler output")

            with jax.spmd_mode('allow_all'):

                if direction=='r2c':
                    out_jit: jax.Array = rfftn_jit(xshard).block_until_ready()
                else:
                    out_jit: jax.Array = irfftn_jit(xshard).block_until_ready()
                sync_global_devices("loop")
                local_out_subset = out_jit.addressable_data(0)
        return local_out_subset

    def generate_noise(self, noisetype='white', nsub=1024**3, seed=13579):

        N = self.N

        noise = None
        if self.partype is None:
            noise = self._generate_serial_noise(N, noisetype, seed, nsub)
        elif self.partype == 'jaxshard':
            noise = self._generate_sharded_noise(N, noisetype, seed, nsub)
        return noise

    def noise2delta(self,delta):
        return self._fft(
                    self._apply_grid_transfer_function(self._fft(delta)),
                    direction='c2r')

    def slpt(self, infield='noise', delta=None, mode='lean'):

        k0 = 2*jnp.pi/self.Lbox*self.N
        kx = (jnp.fft.fftfreq(self.N) * k0).astype(jnp.float32)
        ky = (jnp.fft.fftfreq(self.N) * k0).astype(jnp.float32)
        ky = (ky[self.start:self.end]).astype(jnp.float32)
        kz = (jnp.fft.rfftfreq(self.N) * k0).astype(jnp.float32)

        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')

        k2 = (kxa**2+kya**2+kza**2).astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()
        kx = kx.at[self.N//2].set(0.0)
        kz = kz.at[-1].set(0.0)

        if self.start <= self.N//2 and self.end > self.N//2:
            ky = ky.at[self.N//2-self.start].set(0.0)

        kx = kx[:,None,None]
        ky = ky[None,:,None]
        kz = kz[None,None,:]

        index0 = jnp.nonzero(k2==0.0)

        def _get_shear_factor(ki,kj,delta):
            arr = ki*kj/k2*delta
            if self.host_id == 0: 
                arr = arr.at[index0].set(0.0+0.0j)
            return self._fft(arr,direction='c2r')

        def _delta_to_s(ki,delta):
            # convention:
            #   Y_k = Sum_j=0^n-1 [ X_j * e^(- 2pi * sqrt(-1) * j * k / n)]
            # where
            #   Y_k is complex transform of real X_j
            arr = (0+1j)*ki/k2*delta
            if self.host_id == 0: 
                arr = arr.at[index0].set(0.0+0.0j)
            arr = self._fft(arr,direction='c2r')
            return arr

        if infield == 'noise':
            # FT of delta from noise
            delta = self._apply_grid_transfer_function(self._fft(delta))
        elif infield == 'delta':
            # FT of delta
            delta = self._fft(delta)
        else:
            import numpy as np
            # delta from external file
            delta = jnp.asarray(np.fromfile(infield,dtype=jnp.float32,count=self.N*self.N*self.N))
            delta = jnp.reshape(delta,self.rshape)
            delta = delta[:,self.start:self.end,:]
            # FT of delta
            delta = self._fft(delta)
    
        # Definitions used for LPT
        #   grad.S^(n) = - delta^(n)
        # where
        #   delta^(1) = linear density contrast
        #   delta^(2) = Sum [ dSi/dqi * dSj/dqj - (dSi/dqj)^2]
        #   x(q) = q + D * S^(1) + f * D^2 * S^(2)
        # with
        #   f = + 3/7 Omegam_m^(-1/143)
        # being a good approximation for a flat universe

        if mode == 'fast':
            # minimize operations
            sxx = _get_shear_factor(kx,kx,delta)
            syy = _get_shear_factor(ky,ky,delta)
            delta2  = sxx * syy

            szz = _get_shear_factor(kz,kz,delta)
            delta2 += sxx * szz ; del sxx; gc.collect()
            delta2 += syy * szz ; del syy ; del szz; gc.collect()

            sxy = _get_shear_factor(kx,ky,delta)
            delta2 -= sxy * sxy ; del sxy; gc.collect()

            sxz = _get_shear_factor(kx,kz,delta)
            delta2 -= sxz * sxz ; del sxz; gc.collect()

            syz = _get_shear_factor(ky,kz,delta)
            delta2 -= syz * syz ; del syz; gc.collect()

        else:
            # minimize memory footprint
            delta2  = self._fft(
                    _get_shear_factor(kx,kx,delta)*_get_shear_factor(ky,ky,delta)
                  + _get_shear_factor(kx,kx,delta)*_get_shear_factor(kz,kz,delta)
                  + _get_shear_factor(ky,ky,delta)*_get_shear_factor(kz,kz,delta)
                  - _get_shear_factor(kx,ky,delta)*_get_shear_factor(kx,ky,delta)
                  - _get_shear_factor(kx,kz,delta)*_get_shear_factor(kx,kz,delta)
                  - _get_shear_factor(ky,kz,delta)*_get_shear_factor(ky,kz,delta))

        # 2nd order displacements
        self.s2x = _delta_to_s(kx,delta2)
        self.s2y = _delta_to_s(ky,delta2)
        self.s2z = _delta_to_s(kz,delta2)

        del delta2; gc.collect()

        # 1st order displacements
        self.s1x = _delta_to_s(kx,delta)
        self.s1y = _delta_to_s(ky,delta)
        self.s1z = _delta_to_s(kz,delta)

        del delta; gc.collect()


    

