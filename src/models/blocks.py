import jax
import jax.numpy as jnp
from flax import linen as nn
import s2fft
from src.utils.sht_helper import zero_pad_flm

def get_activation(activation):
    if activation == "gelu":
        return nn.gelu
    elif activation == "relu":
        return nn.relu
    elif activation == "swish":
        return nn.swish
    elif activation == "elu":
        return nn.elu
    elif activation == "leaky_relu":
        return nn.leaky_relu
    else:
        raise ValueError(f"Activation function {activation} not supported")

def get_sinusoidal_embedding(t, dim):
    embedding = jnp.zeros((t.shape[0], dim))
    embedding = jnp.sin(jnp.pi * t)
    embedding = jnp.concatenate([embedding, jnp.cos(jnp.pi * t)], axis=-1)
    return embedding

class SphericalSpectralTimeConv(nn.Module):
    in_channels: int
    out_channels: int
    L_freq_used: int
    L_out_spatial: int
    sampling: str = "mw"
    path: str = "down"

    

    @nn.compact
    def __call__(self, x, t_emb):
        '''
        x:  if the sampling is mw, shape (L_in, 2*L_in-1, in_channels)
        t_emb: shape (time_emb_dim,)

        output:
        if the sampling is mw, shape (L_out, 2*L_out-1, out_channels)
        '''
        # ensure the input and output L are not larger than the maximum L

        # if self.path == "up":
        #     x = jax.image.resize(x, (self.L_freq_used, 2 * self.L_freq_used - 1, x.shape[2]), method="bilinear")

        x_sht = jax.vmap(lambda x: s2fft.forward(x, x.shape[0], method="jax", sampling=self.sampling), in_axes=(2))(x)
        # x_sht: shape (in_channels, L_out, 2*L_out-1)
        x_sht = x_sht.transpose(1, 2, 0)
        x_sht = jax.image.resize(x_sht, (self.L_freq_used, 2 * self.L_freq_used - 1, x_sht.shape[2]), method="bilinear")
        
        # x_sht: shape (L_out, 2*L_out-1, in_channels), complex matrix

        weight_shape = (self.L_freq_used, self.in_channels, self.out_channels) # weight shared within one L
        weight_real = self.param("weight_real", nn.initializers.normal(stddev=0.02), weight_shape)
        weight_imag = self.param("weight_imag", nn.initializers.normal(stddev=0.02), weight_shape)

        weight = weight_real + 1j * weight_imag
        # weight: shape (L_out, in_channels, out_channels), complex matrix
        
        t_emb_real = nn.Dense(features=self.L_freq_used)(t_emb)
        t_emb_imag = nn.Dense(features=self.L_freq_used)(t_emb)

        t_emb = t_emb_real + 1j * t_emb_imag
        # t_emb: shape (L_out, 1) , complex vector

        weight = jnp.einsum("l,lio->lio", t_emb, weight)
        # weight: shape (L_out, in_channels, out_channels), complex matrix

        x_sht = jnp.einsum("lmi,lio->lmo", x_sht, weight)
        # x_sht: shape (L_out, 2*L_out-1, out_channels), complex matrix

        padded_x_sht = zero_pad_flm(x_sht, self.L_out_spatial)
        x_spatial_out = jax.vmap(lambda x: s2fft.inverse(x, self.L_out_spatial, method="jax", sampling=self.sampling), in_axes=(2))(padded_x_sht)
        x_spatial_out = x_spatial_out.transpose(1,2,0)
        x_out = jnp.real(x_spatial_out)
        # x_out: shape (L_out, 2*L_out-1, out_channels)

        return x_out
        

class SpatialTimeConv(nn.Module):
    out_channels: int
    L_out_spatial: int
    path: str = "down"
    sampling: str = "mw"
    '''
    This is a time-invariant spatial convolution on the sphere.

    '''

    @nn.compact
    def __call__(self, x, t_emb):
        '''
        x: shape (L_in, 2*L_in-1, in_channels)
        t_emb: shape (time_emb_dim,)
        if the path is down, L_out = L_in / 2
        if the path is up, L_out = L_in * 2
        if the path is middle, L_out = L_in

        output:
        if the sampling is mw, shape (L_out, 2*L_out-1, out_channels)
        '''
        in_channels = x.shape[2]



        t = nn.Dense(features=self.out_channels * 2)(t_emb)

        w_t, b_t = jnp.split(t, 2, axis=-1) # w_t: shape (time_emb_dim, out_channels), b_t: shape (out_channels,)

        weight_shape = (in_channels, self.out_channels)

        weight = self.param("weight", nn.initializers.normal(stddev=0.02), weight_shape)

        weight = jnp.einsum("o,io->io", w_t, weight)

        x = jnp.einsum("lmi,io->lmo", x, weight)

        if self.path == "down":
            # downsampling  
            x = jax.image.resize(x, (self.L_out_spatial, 2 * self.L_out_spatial - 1, x.shape[2]), method="bilinear")
            x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            # x: shape (L_out, 2*L_out-1, out_channels)
        elif self.path == "up":
            # upsampling
            x = jax.image.resize(x, (self.L_out_spatial, 2 * self.L_out_spatial - 1, x.shape[2]), method="bilinear")
            x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            # x: shape (L_out, 2*L_out-1, out_channels)
        else:
            x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            # x: shape (L_out, 2*L_out-1, out_channels)
        
        x = x + b_t

        return x

class CTSFNOBlock(nn.Module):
    '''
    Input grid resolution agnostic block. So there is no L_in.
    '''
    L_freq_used: int
    L_out_spatial: int
    in_channels: int
    out_channels: int
    path: str = "down"
    sampling: str = "mw"
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x, t_emb):
        x_spectral = SphericalSpectralTimeConv(in_channels=self.in_channels, out_channels=self.out_channels, L_freq_used=self.L_freq_used, L_out_spatial=self.L_out_spatial, path=self.path, sampling=self.sampling)(x, t_emb)

        x_spatial = SpatialTimeConv(out_channels=self.out_channels, L_out_spatial=self.L_out_spatial, path=self.path, sampling=self.sampling)(x, t_emb)

        x = x_spectral + x_spatial

        x = get_activation(self.activation)(x)

        x = nn.LayerNorm()(x)

        return x

class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    t_emb_dim: int
    scaling: float = 100.0
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, t):
        """ t shape: (,) """
        pe = jnp.empty((self.t_emb_dim,))
        factor = self.scaling * t * jnp.exp(jnp.arange(0, self.t_emb_dim, 2) * -(jnp.log(self.max_period) / self.t_emb_dim))
        pe = pe.at[0::2].set(jnp.sin(factor))
        pe = pe.at[1::2].set(jnp.cos(factor))
        return pe