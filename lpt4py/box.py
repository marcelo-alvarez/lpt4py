import jax
import sys
import os
import gc

import jax.numpy as jnp 
import jax.random as rnd

from . import multihost_fft as mfft

class Box:
    '''Box'''
    def __init__(self, **kwargs):

        self.N        = kwargs.get('N',512)
        self.Lbox     = kwargs.get('Lbox',7700.0)
        self.parallel = kwargs.get('parallel',False)
        self.nlpt     = kwargs.get('nlpt',2)
        self.seeds    = kwargs.get('seeds',[13579])
        self.delta    = kwargs.get('delta',None)

        self.iter = 0

        self.dk  = 2*jnp.pi/self.Lbox
        self.d3k = self.dk * self.dk * self.dk

        self.s1lpt = None
        self.s2lpt = None

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        self.start = 0
        self.end   = self.N

        # needed for running on CPU with a single process
        self.ngpus   = 1        
        self.host_id = 0

        if self.parallel:
            self.ngpus   = jax.device_count()
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)

    def k_axis(self, r=False, slab_axis=False):
        if r: 
            k_i = (jnp.fft.rfftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        else:
            k_i = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        if slab_axis: return (k_i[self.start:self.end]).astype(jnp.float32)
        return k_i
    
    def k_square(self, kx, ky, kz):
        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')
        del kx, ky, kz ; gc.collect()

        k2 = (kxa**2+kya**2+kza**2).astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()

        return k2
    
    def interp2kgrid(self, k_1d, f_1d):
        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        interp_fcn = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        del kx, ky, kz ; gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, self.cshape_local).astype(jnp.float32)

    def slpt(self, mode='lean'):

        if self.nlpt <= 0: return

        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        k2 = self.k_square(kx, ky, kz)
        
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
            return mfft.fft(arr,direction='c2r')

        def _delta_to_s(ki,delta):
            # convention:
            #   Y_k = Sum_j=0^n-1 [ X_j * e^(- 2pi * sqrt(-1) * j * k / n)]
            # where
            #   Y_k is complex transform of real X_j
            arr = (0+1j)*ki/k2*delta
            if self.host_id == 0: 
                arr = arr.at[index0].set(0.0+0.0j)
            arr = mfft.fft(arr,direction='c2r')
            return arr

        # if type of delta is None or string, generate or read in, respectively, delta in 
        # config space; otherwise assume delta already contains delta in config space
        if self.delta is None:
            # by default use random white noise
            seed = self.seeds[self.iter%len(self.seeds)]
            self.iter += 1
            delta = rnd.normal(rnd.PRNGKey(seed), dtype=jnp.float32, shape=self.rshape_local)
        elif isinstance(self.delta, str):
            import numpy as np
            # delta from external file
            delta = jnp.asarray(np.fromfile(self.delta,dtype=jnp.float32,count=self.N*self.N*self.N))
            delta = jnp.reshape(delta,self.rshape)
            delta = delta[:,self.start:self.end,:]
        else:
            delta = jnp.copy(self.delta)

        # FT of delta
        delta = mfft.fft(delta)

        # Definitions used for LPT
        #   grad.S^(n) = - delta^(n)
        # where
        #   delta^(1) = linear density contrast
        #   delta^(2) = Sum [ dSi/dqi * dSj/dqj - (dSi/dqj)^2]
        #   x(q) = q + D * S^(1) + f * D^2 * S^(2)
        # with
        #   f = + 3/7 Omegam_m^(-1/143)
        # being a good approximation for a flat universe

        if mode == 'fast' and self.nlpt > 1:
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

            delta2 = mfft.fft(delta2)

        elif self.nlpt > 1:
            # minimize memory footprint
            delta2  = mfft.fft(
                    _get_shear_factor(kx,kx,delta)*_get_shear_factor(ky,ky,delta)
                  + _get_shear_factor(kx,kx,delta)*_get_shear_factor(kz,kz,delta)
                  + _get_shear_factor(ky,ky,delta)*_get_shear_factor(kz,kz,delta)
                  - _get_shear_factor(kx,ky,delta)*_get_shear_factor(kx,ky,delta)
                  - _get_shear_factor(kx,kz,delta)*_get_shear_factor(kx,kz,delta)
                  - _get_shear_factor(ky,kz,delta)*_get_shear_factor(ky,kz,delta))

        if self.nlpt > 1:
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


    

