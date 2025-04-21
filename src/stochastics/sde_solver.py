import jax
import jax.numpy as jnp
import jax.random as jrandom
from collections.abc import Callable
from jaxtyping import Array, PyTree
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from src.stochastics.sde import SDE
from functools import partial
# class SDESolver(ABC):

class SDESolver(ABC):
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def from_sde(self, sde: SDE, x0: jnp.ndarray, dt: float, total_time: float, batch_size: int, x0_list: Optional[jnp.ndarray] = None, debug_mode: bool = False) -> 'SDESolver':
        pass

class EulerMaruyama(SDESolver):
    def __init__(self, 
                 drift_fn: Callable[[jnp.ndarray, float, Optional[jnp.ndarray]], jnp.ndarray],
                 diffusion_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dt: float,
                 total_time: float,
                 noise_size: int,
                 dim: int,
                 condition_x: Optional[jnp.ndarray] = None,
                 debug_mode: bool = False,
                 reversed: bool = False):
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.noise_size = noise_size
        self.condition_x = condition_x
        self.debug_mode = debug_mode
        self.dim = dim
        self.reversed = reversed
    def solve(self, x0: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
        def step(carry: Tuple[jnp.ndarray, jnp.ndarray], t: float):
            x, key = carry
            key, subkey = jrandom.split(key)
            subkey = jrandom.split(subkey, self.noise_size) # noise size normally is same as the x0 resolution, but not necessarily
            # print("subkey.shape: ", subkey.shape)
            # dW = jax.vmap(lambda key: jrandom.normal(key, (self.dim,)) * jnp.sqrt(self.dt), in_axes=(0))(subkey)
            dW = jax.vmap(lambda key: jrandom.normal(key, (self.dim,)) * jnp.sqrt(self.dt), in_axes=(0))(subkey)
            # dW = dW.reshape(self.noise_size, self.dim)
            # dW = jrandom.multivariate_normal(subkey, jnp.zeros(self.dim), jnp.eye(self.dim)) * jnp.sqrt(self.dt)
            # dW = jrandom.multivariate_normal(subkey[0], jnp.zeros(self.dim), jnp.eye(self.dim)) * jnp.sqrt(self.dt)
            # check the dimension of x, if x is 2D manifold, then we need to reshape x to 3D
            if self.condition_x is not None:
                drift = self.drift_fn(x,t, self.condition_x)    
            else:
                drift = self.drift_fn(x, t)
            diffusion = self.diffusion_fn(x, t)
            if x.ndim == 3:
                x_next = x + drift * self.dt + jnp.einsum('ijk,kl->ijl', diffusion, dW)
                # x_next = x + drift * self.dt + diffusion * dW.reshape(x.shape)
            elif x.ndim == 2:
                x_next = x + drift * self.dt + jnp.einsum('ij,jk->ik', diffusion, dW)
            if self.debug_mode:
                jax.debug.print("t: {t}", t=t)
                jax.debug.print("dt: {dt}", dt=self.dt)
                jax.debug.print("num_steps: {num_steps}", num_steps=self.num_steps)
                jax.debug.print("drift: {drift}", drift=drift)
                jax.debug.print("diffusion: {diffusion}", diffusion=diffusion)
                jax.debug.print("dW: {dW}", dW=dW)
                jax.debug.print("x_next: {x_next}", x_next=x_next)
            return (x_next, key), (x_next, diffusion)

        times = jnp.linspace(0, self.total_time, self.num_steps + 1)
        # times = jnp.linspace(0, self.total_time, self.num_steps)
        _, (trajectory, diffusion_history) = jax.lax.scan(step, (x0, rng_key), times[:-1])
        return jnp.concatenate([x0[None, ...], trajectory], axis=0), diffusion_history
    


    @staticmethod
    def from_sde(sde, dt: float, total_time: float, dim: int, condition_x: Optional[jnp.ndarray] = None, debug_mode: bool = False, reversed: bool = False) -> 'EulerMaruyama':
        return EulerMaruyama(sde.drift_fn, sde.diffusion_fn, dt, total_time, sde.noise_size, dim,  condition_x, debug_mode, reversed)