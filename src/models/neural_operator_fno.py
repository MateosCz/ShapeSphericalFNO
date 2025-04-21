
from dataclasses import field
from flax import linen as nn

from .blocks_fno import *
import math



class CTUNO2D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
                # check the x's shape, if x is 1D manifold, then we need to reshape x to 2D
        if x.ndim == 2:
            sqrt_n = int(math.sqrt(x.shape[0]))
            assert sqrt_n * sqrt_n == x.shape[0], "x.shape[0] is not a square number"
            x = jnp.reshape(x, (sqrt_n, sqrt_n, x.shape[1]))
            flatten = True
        else:
            flatten = False
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Conv(
            features=self.lifting_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = CTUNOBlock2D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        
        x = nn.Conv(
            features=self.out_co_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)
        # flatten the x
        print("out x shape: ", x.shape)
        if flatten:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        return x