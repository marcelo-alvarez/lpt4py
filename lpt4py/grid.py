import jax
import sys
import os

import scaleran as sr

import jax.numpy as jnp 
import jax.random as rnd


class Grid:
    '''Grid'''
    def __init__(self, **kwargs):

        self.N = kwargs.get('N',512)

    def generate_noise(self, noisetype='white', nsub=1024**3, seed=13579):

        N = self.N
        stream = sr.Stream(seed=seed,nsub=nsub)

        noise = stream.generate(start=0,size=N**3)
        noise = jnp.reshape(noise,(N,N,N))

        return noise







