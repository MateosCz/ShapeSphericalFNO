import jax
import jax.numpy as jnp
from flax import linen as nn
import s2fft
from src.utils.sht_helper import resize_flm, infer_L_from_shape, pad_inverse_output, get_sampling_grid

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
def normal_initializer(input_co_dim: int):
    return nn.initializers.normal(stddev=jnp.sqrt(1.0/(2.0*input_co_dim)))


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
        # forward_L detection
        L_in = infer_L_from_shape(x, self.sampling)
        x_sht = jax.vmap(lambda x: s2fft.forward(x, L_in, method="jax", spin=0,sampling=self.sampling,reality=True), in_axes=(2))(x)
        # x_sht: shape (in_channels, L_out, 2*L_out-1)
        x_sht = x_sht.transpose(1, 2, 0)
        x_sht = resize_flm(x_sht, self.L_freq_used)
        

        # conv_x_sht = nn.Conv(features=self.out_channels, kernel_size=(1, 1), padding="VALID")(x_sht)
        

        
        # x_sht: shape (L_out, 2*L_out-1, in_channels), complex matrix

        A_real = self.param("A_real", nn.initializers.normal(0.01), (self.L_freq_used, 2 * self.L_freq_used - 1, t_emb.shape[0])) 
        A_imag = self.param("A_imag", nn.initializers.normal(0.01), (self.L_freq_used, 2 * self.L_freq_used - 1, t_emb.shape[0]))

        t_emb_exp = t_emb[None, None, :]

        t_emb_real = jnp.einsum("lmt,lmt->lm", t_emb_exp, A_real)
        t_emb_imag = jnp.einsum("lmt,lmt->lm", t_emb_exp, A_imag)

        t_emb_complex = t_emb_real + 1j * t_emb_imag # shape (L_out, 2*L_out-1)

        # weight_shape = (self.L_freq_used, self.in_channels, self.out_channels) # weight shared within one L
        # weight_real = self.param("weight_real", normal_initializer(self.in_channels), weight_shape)
        # weight_imag = self.param("weight_imag", normal_initializer(self.in_channels), weight_shape)

        R_real = self.param("R_real", nn.initializers.normal(0.01), (self.L_freq_used,2 * self.L_freq_used - 1, self.out_channels, self.in_channels))
        R_imag = self.param("R_imag", nn.initializers.normal(0.01), (self.L_freq_used,2 * self.L_freq_used - 1, self.out_channels, self.in_channels))
        R_complex = R_real + 1j * R_imag
        R_t_complex = jnp.einsum("lm,lmoi->loi", t_emb_complex, R_complex)
        x_sht_out = jnp.einsum("lmi,loi->lmo", x_sht, R_t_complex)
        # def freq_multiply(args):
        #     """ This mulitplication shall be vectorized over b and modes dimensions
        #     """
        #     x_slice, phi_val, Rr, Ri = args # x_slice (d_u, ), phi_val (,), Rr & Ri (d_v, d_u)
        #     xr, xi = x_slice.real, x_slice.imag
        #     # first multiply on the fourier base R F(v)
        #     # complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        #     real_tmp = Rr @ xr - Ri @ xi    # (d_v, )
        #     imag_tmp = Rr @ xi + Ri @ xr    # (d_v, )
            
        #     # then multiply by the time information
        #     phi_real, phi_imag = phi_val.real, phi_val.imag
        #     real_out = real_tmp * phi_real - imag_tmp * phi_imag    # (d_v, )
        #     imag_out = real_tmp * phi_imag + imag_tmp * phi_real    # (d_v, )
        #     out = real_out + 1j * imag_out
        #     return out
        
        # lm = self.L_freq_used * (2 * self.L_freq_used - 1)

        # x_sht_flat = x_sht.reshape(lm, x.shape[-1])
        # t_emb_flat = t_emb_complex.reshape(lm,)
        # Rr_flat = R_real.reshape(lm, self.out_channels, self.in_channels)
        # Ri_flat = R_imag.reshape(lm, self.out_channels, self.in_channels)

        # freq_args = (x_sht_flat, t_emb_flat, Rr_flat, Ri_flat)

        # x_sht_flat = jax.vmap(freq_multiply)(freq_args)
        # x_sht_out = x_sht_flat.reshape(self.L_freq_used, 2 * self.L_freq_used - 1, self.out_channels)
        # print("x_sht_flat.shape", x_sht_flat.shape)

        # R_t_complex = jnp.einsum("lm,loi->loi", t_emb_complex, R_complex)

        # x_sht = jnp.einsum("lmi,loi->lmo", x_sht, R_t_complex)



        padded_x_sht = resize_flm(x_sht_out, self.L_out_spatial)
        x_spatial_out = jax.vmap(lambda x: s2fft.inverse(x, self.L_out_spatial, method="jax", spin=0, sampling=self.sampling,reality=True), in_axes=(2))(padded_x_sht)
        x_spatial_out = x_spatial_out.transpose(1,2,0)
        x_spatial_out = jnp.real(x_spatial_out)

        return x_spatial_out
        
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

        L_in = infer_L_from_shape(x, self.sampling)
        # x = nn.Conv(features=self.out_channels, kernel_size=(1, 1), padding="SAME")(x)

        # t = nn.Dense(features=self.out_channels * 2)(t_emb)

        W = self.param("W", nn.initializers.normal(0.01), (x.shape[-1], self.out_channels))
        B = self.param("B", nn.initializers.normal(0.01), (self.out_channels, t_emb.shape[0]))

        # w_t, b_t = jnp.split(t, 2, axis=-1) # w_t: shape (time_emb_dim, out_channels), b_t: shape (out_channels,)

        def build_Wt(psi_t_b):
            """ This function shall be vectorized over b dimension
            """
            diag_vec = B @ psi_t_b          # (d_v, )
            diag_mat = jnp.diag(diag_vec)   # (d_v, d_v)
            return W @ diag_mat             # (d_u, d_v)
        
        Wt = build_Wt(t_emb)

        def apply_Wt(args):
            x, wt = args # x: shape (L_in, 2*L_in-1, in_channels), wt: shape (in_channels, out_channels)
            return jnp.einsum("lmi,io->lmo", x, wt) # x: shape (L_in, 2*L_in-1, in_channels), wt: shape (in_channels, out_channels)
        x = apply_Wt((x, Wt))


        # forward_L detection
        

        if self.path == "down":
            # downsampling  
            x = jax.vmap(lambda x: s2fft.forward(x, L_in, method="jax", spin=0,sampling=self.sampling,reality=True), in_axes=(2))(x)
            x = jnp.transpose(x,(1,2,0))
            x = resize_flm(x, self.L_out_spatial)
            x = jax.vmap(lambda x: s2fft.inverse(x, self.L_out_spatial, method="jax", spin=0, sampling=self.sampling,reality=True), in_axes=(2))(x)
            x = jnp.transpose(x,(1,2,0))

            # x: shape (L_out, 2*L_out-1, out_channels)
        elif self.path == "up":
            # upsampling
            x = jax.vmap(lambda x: s2fft.forward(x, L_in, method="jax", spin=0,sampling=self.sampling,reality=True), in_axes=(2))(x)
            x = jnp.transpose(x,(1,2,0))
            x = resize_flm(x, self.L_out_spatial)
            x = jax.vmap(lambda x: s2fft.inverse(x, self.L_out_spatial, method="jax", spin=0, sampling=self.sampling,reality=True), in_axes=(2))(x)
            x = jnp.transpose(x,(1,2,0))

            # x: shape (L_out, 2*L_out-1, out_channels)
        else:

            x = x
            # x: shape (L_out, 2*L_out-1, out_channels)

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
    grid_embedding: bool = False

    @nn.compact
    def __call__(self, x, t_emb_phi, t_emb_psi):
        x_spectral = SphericalSpectralTimeConv(in_channels=self.in_channels, out_channels=self.out_channels, L_freq_used=self.L_freq_used, L_out_spatial=self.L_out_spatial, path=self.path, sampling=self.sampling)(x, t_emb_phi)
        x_spatial = SpatialTimeConv(out_channels=self.out_channels, L_out_spatial=self.L_out_spatial, path=self.path, sampling=self.sampling)(x, t_emb_psi)

        if self.grid_embedding:
            theta_grid, phi_grid = get_sampling_grid(self.L_out_spatial, self.sampling)
            # print("theta_grid.shape", theta_grid.shape)
            # print("phi_grid.shape", phi_grid.shape)
            x_grid = jnp.stack([theta_grid, phi_grid], axis=-1)
            x_grid = nn.Dense(features=self.out_channels)(x_grid)
            # print("x_grid.shape", x_grid.shape)
            # x = x_spectral + x_spatial + x_grid
            x = x_spectral + x_spatial
        else:
            x = x_spectral + x_spatial
        x = get_activation(self.activation)(x)
        # x = x_spectral

        # x = nn.LayerNorm()(x)

        return x

class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    c: int                      # time embedding dimension
    s: float = 100.0            # time scaling
    min_freq: float = 1.0       # minimal frequency
    max_freq: float = 10000.0      # maximal frequency

    @nn.compact
    def __call__(self, t):
        """ t shape: (b,) """
        num_freqs = self.c // 2
        freqs = 2.0 * jnp.pi * jnp.exp(
            jnp.linspace(
                jnp.log(self.min_freq),
                jnp.log(self.max_freq),
                num_freqs
            )
        )
        freqs = jax.lax.stop_gradient(freqs)
        arg = t * freqs
        sin_emb = jnp.sin(arg)
        cos_emb = jnp.cos(arg)
        embedding = jnp.stack([sin_emb, cos_emb], axis=-1)
        embedding = embedding.reshape(-1, self.c)
        # flatten the embedding
        embedding = embedding.reshape(-1)
        return embedding
    
class Encoder_MLP(nn.Module):
    in_channels: int
    out_channels: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.out_channels)(x)
        x = get_activation(self.activation)(x)
        return x
class SpectralTimeModulatedSFNO(nn.Module):
    in_channels: int
    out_channels: int
    L_freq_used: int
    time_embed_dim: int
    L_out_spatial: int
    sampling: str = "mw"

    @nn.compact
    def __call__(self, x, t_emb):
        L = x.shape[0]
        phi_dim = x.shape[1]

        x_sht = jax.vmap(
            lambda c: s2fft.forward(x[..., c], L, spin=0, sampling=self.sampling, method="jax"),
            in_axes=0
        )(jnp.arange(self.in_channels))
        x_sht = jnp.transpose(x_sht, (1, 2, 0))  # (L, M, Cin)

        x_sht = resize_flm(x_sht, self.L_freq_used)

        A_real = self.param("A_real", nn.initializers.normal(0.01), (self.L_freq_used, self.time_embed_dim))
        A_imag = self.param("A_imag", nn.initializers.normal(0.01), (self.L_freq_used, self.time_embed_dim))

        # Ensure t_emb shape is (1, C) or (C,) for broadcasting
        if t_emb.ndim == 1:
            t_emb = t_emb[None, :]  # (1, C)

        phi_real = jnp.einsum('lc,bc->bl', A_real, t_emb)  # (1, L)
        phi_imag = jnp.einsum('lc,bc->bl', A_imag, t_emb)  # (1, L)
        phi_complex = phi_real + 1j * phi_imag             # (1, L)
        phi_complex = phi_complex[0]  # remove batch dim → (L,)

        R_real = self.param("R_real", nn.initializers.normal(0.01), (self.L_freq_used, self.out_channels, self.in_channels))
        R_imag = self.param("R_imag", nn.initializers.normal(0.01), (self.L_freq_used, self.out_channels, self.in_channels))

        def per_freq_op(x_lm, phi):
            def per_l_op(args):
                x_l, Rr, Ri, p = args
                xr, xi = x_l.real, x_l.imag
                real_tmp = jnp.dot(Rr, xr) - jnp.dot(Ri, xi)
                imag_tmp = jnp.dot(Rr, xi) + jnp.dot(Ri, xr)
                return real_tmp * p.real - imag_tmp * p.imag + 1j * (real_tmp * p.imag + imag_tmp * p.real)

            return jax.vmap(per_l_op)((x_lm, R_real, R_imag, phi))

        outputs = []
        for m in range(x_sht.shape[1]):
            out_l = per_freq_op(x_sht[:, m, :], phi_complex)  # (L, Cout)
            outputs.append(out_l[:, None, :])

        x_sht_out = jnp.concatenate(outputs, axis=1)  # (L_out, M, Cout)
        x_sht_out = resize_flm(x_sht_out, self.L_out_spatial)

        x_out = jax.vmap(
            lambda c: s2fft.inverse(x_sht_out[..., c], self.L_out_spatial, spin=0, sampling=self.sampling, method="jax"),
            in_axes=0
        )(jnp.arange(self.out_channels))
        x_out = jnp.transpose(x_out, (1, 2, 0))  # (L_out, M_out, Cout)
        return jnp.real(x_out)
    
